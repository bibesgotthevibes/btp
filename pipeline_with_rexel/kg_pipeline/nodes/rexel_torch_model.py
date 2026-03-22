from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# Interior words that indicate a span is a verb phrase/clause, not a named entity.
# Copulas, auxiliaries, and the most common entity-separating prepositions.
_INTERIOR_VERB_STOPWORDS = frozenset({
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had",
    "do", "does", "did",
    "in", "at",
})


@dataclass
class REXELTorchConfig:
    model_name: str = "roberta-base"
    max_length: int = 512
    max_span_width: int = 4
    span_threshold: float = 0.35
    relation_threshold: float = 0.35
    top_k_spans: int = 64
    projection_dim: int = 128


class REXELTorchModel(nn.Module):
    """
    Joint DocIE model with shared encoder and task heads.

    Heads:
    - mention/span detection
    - entity type classification
    - relation classification
    - mention coreference
    - entity linking embedding projection
    """

    def __init__(self, num_entity_types: int, num_relations: int, config: Optional[REXELTorchConfig] = None):
        super().__init__()
        self.config = config or REXELTorchConfig()
        self.encoder = AutoModel.from_pretrained(self.config.model_name)
        hidden = self.encoder.config.hidden_size

        self.span_head = nn.Linear(hidden * 2, 1)
        self.type_head = nn.Linear(hidden, num_entity_types)
        self.relation_head = nn.Linear(hidden * 2, num_relations)
        self.coref_head = nn.Linear(hidden * 2, 1)
        self.link_projector = nn.Linear(hidden, self.config.projection_dim)

    def span_score(self, start_vec: torch.Tensor, end_vec: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.span_head(torch.cat([start_vec, end_vec], dim=-1)))

    def span_logits(self, start_vec: torch.Tensor, end_vec: torch.Tensor) -> torch.Tensor:
        """Raw pre-sigmoid logits — use with F.binary_cross_entropy_with_logits."""
        return self.span_head(torch.cat([start_vec, end_vec], dim=-1)).squeeze(-1)

    def type_logits(self, entity_vec: torch.Tensor) -> torch.Tensor:
        return self.type_head(entity_vec)

    def relation_logits(self, head_vec: torch.Tensor, tail_vec: torch.Tensor) -> torch.Tensor:
        return self.relation_head(torch.cat([head_vec, tail_vec], dim=-1))

    def coref_score(self, m1_vec: torch.Tensor, m2_vec: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.coref_head(torch.cat([m1_vec, m2_vec], dim=-1)))

    def coref_logits(self, m1_vec: torch.Tensor, m2_vec: torch.Tensor) -> torch.Tensor:
        """Raw pre-sigmoid logits — use with F.binary_cross_entropy_with_logits."""
        return self.coref_head(torch.cat([m1_vec, m2_vec], dim=-1)).squeeze(-1)

    def link_embedding(self, entity_vec: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.link_projector(entity_vec), p=2, dim=-1)


class REXELTorchPredictor:
    """
    Inference wrapper for a trained REXELTorchModel checkpoint.
    """

    def __init__(self, checkpoint: Dict, device: Optional[str] = None):
        cfg = REXELTorchConfig(**checkpoint["config"])
        if device:
            resolved_device = device
        elif torch.cuda.is_available():
            resolved_device = "cuda"
        elif torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"

        self.device = resolved_device
        self.config = cfg

        self.entity_type_to_id = checkpoint["entity_type_to_id"]
        self.id_to_entity_type = {v: k for k, v in self.entity_type_to_id.items()}

        self.relation_to_id = checkpoint["relation_to_id"]
        self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}

        self.model = REXELTorchModel(
            num_entity_types=len(self.entity_type_to_id),
            num_relations=len(self.relation_to_id),
            config=cfg,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

    @classmethod
    def from_checkpoint_path(
        cls,
        checkpoint_path: str,
        device: Optional[str] = None,
        span_threshold: Optional[float] = None,
        relation_threshold: Optional[float] = None,
        max_span_width: Optional[int] = None,
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        predictor = cls(checkpoint, device=device)
        if span_threshold is not None:
            predictor.config.span_threshold = span_threshold
        if relation_threshold is not None:
            predictor.config.relation_threshold = relation_threshold
        # Cap span width at inference regardless of training config — the model
        # was trained with max_span_width=8 but true named-entity spans are ≤4
        # words. Wider spans mainly capture phrases/clauses, not entity names.
        predictor.config.max_span_width = max_span_width if max_span_width is not None else min(
            predictor.config.max_span_width, 4
        )
        return predictor

    def _tokenize_words(self, words: List[str]):
        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )
        word_ids = enc.word_ids(batch_index=0)
        return enc, word_ids

    def _basic_word_tokenize(self, text: str) -> List[str]:
        # Split punctuation from words to avoid spans like "Rochester.".
        return re.findall(r"[A-Za-z0-9]+|[^\w\s]", text)

    def _normalize_entity_key(self, text: str) -> str:
        key = text.lower().strip()
        key = re.sub(r"[^a-z0-9\s]", " ", key)
        key = re.sub(r"\s+", " ", key).strip()
        return key

    def _is_valid_span_text(self, span_text: str) -> bool:
        if not span_text:
            return False
        words = span_text.split()
        if len(words) == 0 or len(words) > self.config.max_span_width:
            return False

        # Must contain at least one alphanumeric token.
        if not any(re.search(r"[A-Za-z0-9]", w) for w in words):
            return False

        # Reject spans that cross sentence boundaries — these are phrases, not
        # entity names. The tokenizer splits punctuation into its own token so
        # a period inside span_text means the span straddles two sentences.
        if re.search(r'[.!?]', span_text):
            return False

        # Reject punctuation-heavy spans.
        punct = sum(1 for c in span_text if not c.isalnum() and not c.isspace())
        if punct > max(1, len(span_text) // 5):
            return False

        # Reject spans whose interior words (not first/last) are clear verbal or
        # locative stop words — those mark clause/phrase boundaries, not entity
        # boundaries (e.g. "Evelyn Reed is a cardiologist", "Clinic in Rochester").
        if len(words) > 2:
            interior = [w.lower() for w in words[1:-1]]
            if any(w in _INTERIOR_VERB_STOPWORDS for w in interior):
                return False

        return True

    def _suppress_overlapping_mentions(self, mentions: List[Dict]) -> List[Dict]:
        # Keep higher-score spans first. Suppress a candidate if it is
        # substantially *contained* within (or contains) an already-kept span.
        # Using containment ratio instead of IoU prevents fragment spans like
        # "Corp" from surviving alongside "Farma Corp".
        kept = []
        for cand in sorted(mentions, key=lambda x: x["score"], reverse=True):
            overlap = False
            for k in kept:
                inter = max(0, min(cand["end"], k["end"]) - max(cand["start"], k["start"]) + 1)
                if inter == 0:
                    continue
                cand_len = cand["end"] - cand["start"] + 1
                k_len = k["end"] - k["start"] + 1
                # Suppress if either span is more than half covered by the other.
                if inter / cand_len > 0.5 or inter / k_len > 0.5:
                    overlap = True
                    break
            if not overlap:
                kept.append(cand)
            if len(kept) >= self.config.top_k_spans:
                break
        return kept

    def _word_to_subtoken_bounds(self, word_ids: List[Optional[int]], num_words: int) -> Dict[int, Tuple[int, int]]:
        mapping: Dict[int, List[int]] = {}
        for i, widx in enumerate(word_ids):
            if widx is None or widx >= num_words:
                continue
            mapping.setdefault(widx, []).append(i)

        bounds = {}
        for widx, subtoks in mapping.items():
            bounds[widx] = (subtoks[0], subtoks[-1])
        return bounds

    def _span_embedding(self, token_states: torch.Tensor, start_subtok: int, end_subtok: int) -> torch.Tensor:
        span_states = token_states[start_subtok : end_subtok + 1]
        return span_states.mean(dim=0)

    @torch.no_grad()
    def extract(self, text: str) -> Dict:
        words = self._basic_word_tokenize(text)
        if not words:
            return {"mentions": [], "entity_clusters": [], "triples": []}

        enc, word_ids = self._tokenize_words(words)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model.encoder(**enc)
        token_states = out.last_hidden_state[0]

        bounds = self._word_to_subtoken_bounds(word_ids, len(words))
        if not bounds:
            return {"mentions": [], "entity_clusters": [], "triples": []}

        # Mention proposals via span head
        scored_mentions = []
        max_w = self.config.max_span_width
        for s in range(len(words)):
            for e in range(s, min(s + max_w, len(words))):
                if s not in bounds or e not in bounds:
                    continue
                s_sub = bounds[s][0]
                e_sub = bounds[e][1]
                score = self.model.span_score(token_states[s_sub], token_states[e_sub]).item()
                if score >= self.config.span_threshold:
                    span_text = " ".join(words[s : e + 1]).strip()
                    if not self._is_valid_span_text(span_text):
                        continue
                    span_vec = self._span_embedding(token_states, s_sub, e_sub)
                    type_id = int(torch.argmax(self.model.type_logits(span_vec)).item())
                    scored_mentions.append(
                        {
                            "start": s,
                            "end": e,
                            "text": span_text,
                            "score": score,
                            "type": self.id_to_entity_type.get(type_id, "CONCEPT"),
                            "vec": span_vec,
                        }
                    )

        scored_mentions = self._suppress_overlapping_mentions(scored_mentions)

        # --- Step 1: exact-key grouping (punctuation/case variants) ---
        clusters_by_key: Dict[str, Dict] = {}
        for m in scored_mentions:
            key = self._normalize_entity_key(m["text"])
            if not key:
                continue
            if key not in clusters_by_key:
                clusters_by_key[key] = {
                    "canonical": m["text"],
                    "types": [m["type"]],
                    "mentions": [m["text"]],
                    "vectors": [m["vec"]],
                    "best_score": m["score"],
                    "wikidata_id": None,
                    "wikipedia_name": None,
                }
            else:
                clusters_by_key[key]["mentions"].append(m["text"])
                clusters_by_key[key]["vectors"].append(m["vec"])
                if m["score"] > clusters_by_key[key]["best_score"]:
                    clusters_by_key[key]["best_score"] = m["score"]
                    clusters_by_key[key]["canonical"] = m["text"]

        # Build per-cluster averaged vector before coref merging.
        proto_clusters = []
        for cluster in clusters_by_key.values():
            vec = torch.stack(cluster["vectors"]).mean(dim=0)
            proto_clusters.append({"cluster": cluster, "vec": vec})

        # --- Step 2: coref-head merging (REXEL paper §3.3) ---
        # Use both string-level containment and the trained coref head to decide
        # whether two proto-clusters refer to the same entity.
        #
        # Two clusters are merged when EITHER condition holds:
        #   a) One canonical is a proper substring of the other (e.g. "Prosserman"
        #      inside "Jeff Prosserman") — deterministic, never wrong.
        #   b) The coref head scores them very highly (>= 0.85) AND they share at
        #      least one word token — guards against the head over-firing on
        #      semantically unrelated entities.
        COREF_NEURAL_THRESHOLD = 0.85
        n = len(proto_clusters)
        # Union-Find for efficient merging
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            # Merge lower-score root into higher-score root
            sa = proto_clusters[ra]["cluster"]["best_score"]
            sb = proto_clusters[rb]["cluster"]["best_score"]
            if sa >= sb:
                parent[rb] = ra
            else:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                ci = proto_clusters[i]["cluster"]
                cj = proto_clusters[j]["cluster"]
                ki = self._normalize_entity_key(ci["canonical"])
                kj = self._normalize_entity_key(cj["canonical"])

                # (a) String-level: one is a proper substring of the other
                if (ki and kj) and (ki in kj or kj in ki) and ki != kj:
                    union(i, j)
                    continue

                # (b) Neural coref head — high threshold + shared token guard
                tokens_i = set(ki.split())
                tokens_j = set(kj.split())
                if tokens_i & tokens_j:  # must share at least one word
                    vi = proto_clusters[i]["vec"]
                    vj = proto_clusters[j]["vec"]
                    coref_score = self.model.coref_score(vi, vj).item()
                    if coref_score >= COREF_NEURAL_THRESHOLD:
                        union(i, j)

        # Aggregate merged clusters
        merged: Dict[int, Dict] = {}
        for i, pc in enumerate(proto_clusters):
            root = find(i)
            if root not in merged:
                merged[root] = {
                    "canonical": pc["cluster"]["canonical"],
                    "types": list(pc["cluster"]["types"]),
                    "mentions": list(pc["cluster"]["mentions"]),
                    "vectors": [pc["vec"]],
                    "best_score": pc["cluster"]["best_score"],
                }
            else:
                # Fold into existing root cluster
                m_root = merged[root]
                m_root["mentions"].extend(pc["cluster"]["mentions"])
                m_root["vectors"].append(pc["vec"])
                m_root["types"].extend(pc["cluster"]["types"])
                if pc["cluster"]["best_score"] > m_root["best_score"]:
                    m_root["best_score"] = pc["cluster"]["best_score"]
                    m_root["canonical"] = pc["cluster"]["canonical"]

        entity_clusters = []
        entity_vecs = []
        for mc in merged.values():
            uniq_mentions = sorted(set(mc["mentions"]))
            vec = torch.stack(mc["vectors"]).mean(dim=0)
            entity_vecs.append(vec)
            entity_clusters.append(
                {
                    "canonical": mc["canonical"],
                    "types": mc["types"],
                    "mentions": uniq_mentions,
                    "wikidata_id": None,
                    "wikipedia_name": None,
                }
            )

        # Relation prediction over entity pairs
        triples = []
        for i in range(len(entity_clusters)):
            for j in range(len(entity_clusters)):
                if i == j:
                    continue

                # Drop trivial self-like aliases.
                h_key = self._normalize_entity_key(entity_clusters[i]["canonical"])
                t_key = self._normalize_entity_key(entity_clusters[j]["canonical"])
                if not h_key or not t_key or h_key == t_key:
                    continue

                logits = self.model.relation_logits(entity_vecs[i], entity_vecs[j])
                probs = torch.softmax(logits, dim=-1)
                rel_id = int(torch.argmax(probs).item())
                rel = self.id_to_relation.get(rel_id, "NA")
                conf = float(torch.max(probs).item())
                if rel != "NA" and conf >= self.config.relation_threshold:
                    triples.append(
                        {
                            "head": entity_clusters[i]["canonical"],
                            "relation": rel,
                            "tail": entity_clusters[j]["canonical"],
                        }
                    )

        mentions = []
        for m in scored_mentions:
            mentions.append(
                {
                    "text": m["text"],
                    "types": [m["type"]],
                    "score": {m["type"]: m["score"]},
                }
            )

        # De-duplicate triples
        uniq = []
        seen = set()
        for t in triples:
            key = (t["head"], t["relation"], t["tail"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(t)

        return {"mentions": mentions, "entity_clusters": entity_clusters, "triples": uniq}
