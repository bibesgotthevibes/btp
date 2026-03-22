import argparse
import contextlib
import json
import math
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from kg_pipeline.nodes.rexel_torch_model import REXELTorchConfig, REXELTorchModel


def _autocast_ctx(device: str):
    """Return a mixed-precision context for the given device.

    CUDA: forward pass runs in float16 for faster matmuls.
    MPS:  float16 autocast triggers Metal kernel assertion bugs on Apple Silicon
          (MPS GEMV padding overflow). MPS is already GPU-accelerated in float32,
          so no autocast is used — we get the hardware speedup without instability.
    CPU:  no-op (float32 throughout).
    """
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


# Progressive loss schedule: (span, type, relation, coref, link) weights per epoch.
# Span weight stays high throughout — span detection is the main recall bottleneck.
# Relation weight grows gradually but never dominates at the cost of span quality.
LOSS_SCHEDULE = {
    1: (0.45, 0.20, 0.15, 0.10, 0.10),
    2: (0.35, 0.15, 0.30, 0.10, 0.10),
    3: (0.30, 0.12, 0.40, 0.08, 0.10),
    4: (0.25, 0.10, 0.50, 0.07, 0.08),
    5: (0.25, 0.10, 0.50, 0.07, 0.08),
}


def flatten_sents(sents: List[List[str]]) -> Tuple[List[str], List[int]]:
    words = []
    offsets = []
    cursor = 0
    for sent in sents:
        offsets.append(cursor)
        words.extend(sent)
        cursor += len(sent)
    return words, offsets


def build_vocab(train_docs: List[Dict]) -> Tuple[Dict[str, int], Dict[str, int]]:
    type_counter = Counter()
    rel_counter = Counter()

    for doc in train_docs:
        for entity in doc.get("vertexSet", []):
            for mention in entity:
                etype = mention.get("type", "CONCEPT")
                type_counter[etype] += 1

        for lab in doc.get("labels", []):
            rel_counter[lab.get("r", "NA")] += 1

    entity_type_to_id = {t: i for i, t in enumerate(sorted(type_counter.keys()))}
    if "CONCEPT" not in entity_type_to_id:
        entity_type_to_id["CONCEPT"] = len(entity_type_to_id)

    relation_to_id = {"NA": 0}
    for rel in sorted(rel_counter.keys()):
        if rel == "NA":
            continue
        relation_to_id[rel] = len(relation_to_id)

    return entity_type_to_id, relation_to_id


def word_to_subtoken_bounds(word_ids: List[Optional[int]], max_word: int) -> Dict[int, Tuple[int, int]]:
    mapping: Dict[int, List[int]] = {}
    for i, widx in enumerate(word_ids):
        if widx is None or widx >= max_word:
            continue
        mapping.setdefault(widx, []).append(i)

    bounds = {}
    for widx, subtoks in mapping.items():
        bounds[widx] = (subtoks[0], subtoks[-1])
    return bounds


def mention_global_span(mention: Dict, offsets: List[int]) -> Optional[Tuple[int, int]]:
    sid = mention.get("sent_id")
    pos = mention.get("pos", [])
    if sid is None or len(pos) != 2 or sid >= len(offsets):
        return None
    start = offsets[sid] + int(pos[0])
    end = offsets[sid] + int(pos[1]) - 1
    if end < start:
        return None
    return start, end


def sample_negative_spans(num_words: int, positives: set, k: int, max_span_width: int) -> List[Tuple[int, int]]:
    out = []
    attempts = 0
    max_attempts = k * 20
    while len(out) < k and attempts < max_attempts:
        attempts += 1
        s = random.randint(0, num_words - 1)
        e = random.randint(s, min(s + max_span_width - 1, num_words - 1))
        if (s, e) in positives:
            continue
        out.append((s, e))
    return out


def pretokenize_docs(docs: List[Dict], tokenizer, max_length: int) -> Dict[int, Dict]:
    """CPU-side tokenization for all docs, cached by id(doc) so shuffle-safe."""
    cache = {}
    for doc in docs:
        words, offsets = flatten_sents(doc["sents"])
        if not words:
            continue
        enc = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        word_ids = enc.word_ids(batch_index=0)
        bounds = word_to_subtoken_bounds(word_ids, len(words))
        if not bounds:
            continue
        cache[id(doc)] = {"enc": enc, "bounds": bounds, "offsets": offsets}
    return cache


def train_one_doc(
    model: REXELTorchModel,
    tokenizer,
    doc: Dict,
    entity_type_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
    device: str,
    epoch: int = 1,
    relation_class_weights: Optional[torch.Tensor] = None,
    neg_ratio: int = 4,
    cached: Optional[Dict] = None,
) -> torch.Tensor:
    if cached is not None:
        enc_cpu = cached["enc"]
        bounds  = cached["bounds"]
        offsets = cached["offsets"]
        words, _ = flatten_sents(doc["sents"])
    else:
        words, offsets = flatten_sents(doc["sents"])
        if not words:
            return torch.tensor(0.0, device=device)
        enc_cpu = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=model.config.max_length,
        )
        word_ids = enc_cpu.word_ids(batch_index=0)
        bounds = word_to_subtoken_bounds(word_ids, len(words))
    if not bounds or not words:
        return torch.tensor(0.0, device=device)

    enc = {k: v.to(device) for k, v in enc_cpu.items()}
    token_states = model.encoder(**enc).last_hidden_state[0]

    mention_items = []
    entity_reprs = []
    entity_types = []
    entity_qcodes = []
    original_to_compact = {}

    for eidx, entity in enumerate(doc.get("vertexSet", [])):
        mention_vecs = []
        entity_type = "CONCEPT"
        qcode = None

        for mention in entity:
            span = mention_global_span(mention, offsets)
            if not span:
                continue
            s, e = span
            if s not in bounds or e not in bounds:
                continue

            s_sub = bounds[s][0]
            e_sub = bounds[e][1]
            span_vec = token_states[s_sub : e_sub + 1].mean(dim=0)
            mention_vecs.append(span_vec)
            mention_items.append((eidx, s, e, s_sub, e_sub, span_vec))

            entity_type = mention.get("type", entity_type)
            qcode = mention.get("wikidata_qcode", qcode)

        if not mention_vecs:
            continue

        compact_idx = len(entity_reprs)
        original_to_compact[eidx] = compact_idx
        evec = torch.stack(mention_vecs).mean(dim=0)
        entity_reprs.append(evec)
        entity_types.append(entity_type)
        entity_qcodes.append(qcode)

    if not entity_reprs:
        return torch.tensor(0.0, device=device)

    # 1) Mention/span detection loss
    positives = {(m[1], m[2]) for m in mention_items}
    span_losses = []
    for (_, _, _, s_sub, e_sub, _) in mention_items:
        logit = model.span_logits(token_states[s_sub], token_states[e_sub])
        span_losses.append(F.binary_cross_entropy_with_logits(logit, torch.tensor(1.0, device=device)))

    neg_spans = sample_negative_spans(
        num_words=len(words),
        positives=positives,
        k=max(1, len(mention_items)),
        max_span_width=model.config.max_span_width,
    )
    for s, e in neg_spans:
        if s not in bounds or e not in bounds:
            continue
        logit = model.span_logits(token_states[bounds[s][0]], token_states[bounds[e][1]])
        span_losses.append(F.binary_cross_entropy_with_logits(logit, torch.tensor(0.0, device=device)))

    span_loss = torch.stack(span_losses).mean() if span_losses else torch.tensor(0.0, device=device)

    # 2) Entity typing loss
    type_losses = []
    for evec, etype in zip(entity_reprs, entity_types):
        logits = model.type_logits(evec)
        target = torch.tensor(entity_type_to_id.get(etype, entity_type_to_id["CONCEPT"]), device=device)
        type_losses.append(F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0)))

    type_loss = torch.stack(type_losses).mean() if type_losses else torch.tensor(0.0, device=device)

    # 3) Relation classification loss with negatives
    gold_rel = {}
    for lab in doc.get("labels", []):
        h_orig = int(lab.get("h", -1))
        t_orig = int(lab.get("t", -1))
        r = lab.get("r", "NA")
        if h_orig not in original_to_compact or t_orig not in original_to_compact:
            continue
        h = original_to_compact[h_orig]
        t = original_to_compact[t_orig]
        gold_rel[(h, t)] = r

    relation_losses = []
    n_entities = len(entity_reprs)

    # Separate positive and negative entity pairs to control class imbalance.
    # With a 31:1 NA ratio in DocRED, processing all pairs floods the loss with
    # NA gradients even when NA is down-weighted. Cap negatives at neg_ratio x
    # the number of positives in this document.
    pos_pairs = [(i, j) for (i, j) in gold_rel if i != j]
    all_neg_pairs = [
        (i, j)
        for i in range(n_entities)
        for j in range(n_entities)
        if i != j and (i, j) not in gold_rel
    ]
    random.shuffle(all_neg_pairs)
    neg_pairs = all_neg_pairs[: max(1, neg_ratio * len(pos_pairs))]

    for i, j in pos_pairs + neg_pairs:
        logits = model.relation_logits(entity_reprs[i], entity_reprs[j])
        rel = gold_rel.get((i, j), "NA")
        target = torch.tensor(relation_to_id.get(rel, 0), device=device)
        relation_losses.append(
            F.cross_entropy(
                logits.unsqueeze(0),
                target.unsqueeze(0),
                weight=relation_class_weights,
            )
        )

    relation_loss = torch.stack(relation_losses).mean() if relation_losses else torch.tensor(0.0, device=device)

    # 4) Mention coreference loss — sample negatives to avoid O(n²) pairs.
    # All positive (same-entity) pairs + up to 2x negatives.
    coref_losses = []
    pos_coref = []
    neg_coref = []
    for a in range(len(mention_items)):
        for b in range(a + 1, len(mention_items)):
            if mention_items[a][0] == mention_items[b][0]:
                pos_coref.append((a, b))
            else:
                neg_coref.append((a, b))
    random.shuffle(neg_coref)
    sampled_neg_coref = neg_coref[: max(1, 2 * len(pos_coref))]
    for a, b in pos_coref + sampled_neg_coref:
        ent_a = mention_items[a][0]
        ent_b = mention_items[b][0]
        label = 1.0 if ent_a == ent_b else 0.0
        logit = model.coref_logits(mention_items[a][5], mention_items[b][5])
        coref_losses.append(F.binary_cross_entropy_with_logits(logit, torch.tensor(label, device=device)))

    coref_loss = torch.stack(coref_losses).mean() if coref_losses else torch.tensor(0.0, device=device)

    # 5) Linking loss (contrastive on entity embeddings using qcodes)
    link_losses = []
    link_embs = [model.link_embedding(evec) for evec in entity_reprs]
    for i in range(len(link_embs)):
        for j in range(i + 1, len(link_embs)):
            qi = entity_qcodes[i]
            qj = entity_qcodes[j]
            if not qi or not qj:
                continue
            sim = torch.clamp(torch.dot(link_embs[i], link_embs[j]), -1.0, 1.0)
            if qi == qj:
                link_losses.append(1.0 - sim)
            else:
                margin = 0.3
                link_losses.append(torch.relu(sim - margin))

    link_loss = torch.stack(link_losses).mean() if link_losses else torch.tensor(0.0, device=device)

    w_span, w_type, w_rel, w_coref, w_link = LOSS_SCHEDULE.get(epoch, LOSS_SCHEDULE[3])

    # Weighted joint loss (progressive schedule)
    total_loss = (
        w_span * span_loss
        + w_type * type_loss
        + w_rel * relation_loss
        + w_coref * coref_loss
        + w_link * link_loss
    )
    return total_loss


def main():
    parser = argparse.ArgumentParser(description="Train REXEL-style joint DocIE model")
    parser.add_argument("--train-file", type=str, default="data/train_joint_qcode.json")
    parser.add_argument("--dev-file", type=str, default="data/dev_joint_qcode.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.06,
                        help="Fraction of total steps for linear LR warmup")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (simulates larger batch)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-docs", type=int, default=0,
                        help="0 = use the full training file")
    parser.add_argument("--save-path", type=str, default="checkpoints/rexel_joint.pt")
    parser.add_argument("--na-class-weight", type=float, default=0.15,
                        help="Loss weight for the NA (no-relation) class")
    parser.add_argument("--neg-ratio", type=int, default=4,
                        help="Max negative entity-pairs per positive for relation loss")
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.train_file, "r", encoding="utf-8") as f:
        train_docs = json.load(f)

    if args.max_docs > 0:
        train_docs = train_docs[: args.max_docs]

    with open(args.dev_file, "r", encoding="utf-8") as f:
        dev_docs = json.load(f)

    entity_type_to_id, relation_to_id = build_vocab(train_docs)

    # Match max_span_width to inference cap (4) so the span head learns
    # representations that are actually used at test time.
    cfg = REXELTorchConfig(max_span_width=4)
    model = REXELTorchModel(
        num_entity_types=len(entity_type_to_id),
        num_relations=len(relation_to_id),
        config=cfg,
    )

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # MPS-specific settings: allow full GPU heap.
    # CPU fallback for unsupported ops is enabled by default in PyTorch 2.x.
    if device == "mps":
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        print("MPS device selected — float16 autocast enabled for ~1.5x speedup")

    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    if args.resume_checkpoint:
        print(f"Resuming from: {args.resume_checkpoint}")
        ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print("Checkpoint weights loaded.")

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = math.ceil(len(train_docs) / args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    relation_class_weights = torch.ones(len(relation_to_id), device=device)
    relation_class_weights[relation_to_id["NA"]] = args.na_class_weight

    print(f"Training docs: {len(train_docs)}, Dev docs: {len(dev_docs)}")
    print(f"Entity types: {len(entity_type_to_id)}")
    print(f"Relations (incl NA): {len(relation_to_id)}")
    print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
    print(f"Grad accum: {args.grad_accum}, Neg ratio: {args.neg_ratio}")

    # Pre-tokenize all docs on CPU once — eliminates tokenizer overhead from the
    # GPU-side training loop and improves GPU utilization.
    print("Pre-tokenizing train docs...", flush=True)
    train_cache = pretokenize_docs(train_docs, tokenizer, cfg.max_length)
    print("Pre-tokenizing dev docs...", flush=True)
    dev_cache = pretokenize_docs(dev_docs, tokenizer, cfg.max_length)
    print("Tokenization done.", flush=True)

    best_dev_f1 = -1.0
    best_ckpt_path = args.save_path.replace(".pt", "_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_docs)

        w_span, w_type, w_rel, w_coref, w_link = LOSS_SCHEDULE.get(epoch, LOSS_SCHEDULE[5])
        print(
            f"Epoch {epoch} loss weights: "
            f"span={w_span}, type={w_type}, relation={w_rel}, coref={w_coref}, link={w_link}"
        )

        total = 0.0
        steps = 0
        accum_loss = torch.tensor(0.0, device=device)
        accum_count = 0

        for doc_idx, doc in enumerate(train_docs):
            # Wrap the entire forward+loss in mixed precision.
            # Backward and optimizer step stay outside so gradients accumulate
            # correctly in float32 parameter space.
            with _autocast_ctx(device):
                loss = train_one_doc(
                    model=model,
                    tokenizer=tokenizer,
                    doc=doc,
                    entity_type_to_id=entity_type_to_id,
                    relation_to_id=relation_to_id,
                    device=device,
                    epoch=epoch,
                    relation_class_weights=relation_class_weights,
                    neg_ratio=args.neg_ratio,
                    cached=train_cache.get(id(doc)),
                )

            if float(loss.item()) == 0.0:
                continue

            accum_loss = accum_loss + loss / args.grad_accum
            accum_count += 1

            if accum_count == args.grad_accum or doc_idx == len(train_docs) - 1:
                optim.zero_grad()
                accum_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                scheduler.step()

                total += float(accum_loss.item())
                steps += 1
                accum_loss = torch.tensor(0.0, device=device)
                accum_count = 0

            if steps > 0 and steps % 50 == 0:
                print(f"Epoch {epoch} | step {steps} | avg loss {total / steps:.4f} "
                      f"| lr {scheduler.get_last_lr()[0]:.2e}")

        avg = (total / steps) if steps else 0.0
        print(f"Epoch {epoch} complete | steps {steps} | avg loss {avg:.4f}")

        # --- Validation: relation F1 on dev set ---
        dev_f1 = evaluate_dev(model, tokenizer, dev_docs, entity_type_to_id,
                               relation_to_id, device, cfg, dev_cache=dev_cache)
        print(f"Epoch {epoch} | dev relation F1 = {dev_f1:.4f}")

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            payload = {
                "config": cfg.__dict__,
                "entity_type_to_id": entity_type_to_id,
                "relation_to_id": relation_to_id,
                "model_state": model.state_dict(),
            }
            os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
            torch.save(payload, best_ckpt_path)
            print(f"  ** New best dev F1 — saved to {best_ckpt_path}")

    # Always save the final checkpoint too.
    payload = {
        "config": cfg.__dict__,
        "entity_type_to_id": entity_type_to_id,
        "relation_to_id": relation_to_id,
        "model_state": model.state_dict(),
    }
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(payload, args.save_path)
    print(f"Saved final checkpoint to {args.save_path}")
    print(f"Best dev F1: {best_dev_f1:.4f} — weights in {best_ckpt_path}")


# ---------------------------------------------------------------------------
# Dev-set evaluation (relation F1 using gold entity spans)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_dev(
    model: REXELTorchModel,
    tokenizer,
    dev_docs: List[Dict],
    entity_type_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
    device: str,
    cfg: REXELTorchConfig,
    dev_cache: Optional[Dict] = None,
    span_threshold: float = 0.30,
    relation_threshold: float = 0.30,
) -> float:
    """Compute combined span+relation F1 on dev using PREDICTED spans.

    Using predicted spans means checkpoint selection rewards span detection
    quality, not just relation classification — aligning dev metric with
    actual end-to-end test performance.
    Combined score = 0.4 * span_F1 + 0.6 * relation_F1.
    """
    id_to_relation = {v: k for k, v in relation_to_id.items()}
    model.eval()
    rel_tp = rel_fp = rel_fn = 0
    span_tp = span_fp = span_fn = 0

    for doc in dev_docs:
        if dev_cache is not None and id(doc) in dev_cache:
            cached = dev_cache[id(doc)]
            enc_cpu = cached["enc"]
            bounds  = cached["bounds"]
            offsets = cached["offsets"]
            words, _ = flatten_sents(doc["sents"])
            if not words or not bounds:
                continue
            enc = {k: v.to(device) for k, v in enc_cpu.items()}
        else:
            words, offsets = flatten_sents(doc["sents"])
            if not words:
                continue
            enc = tokenizer(
                words,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.max_length,
            )
            word_ids = enc.word_ids(batch_index=0)
            bounds = word_to_subtoken_bounds(word_ids, len(words))
            if not bounds:
                continue
            enc = {k: v.to(device) for k, v in enc.items()}

        with _autocast_ctx(device):
            token_states = model.encoder(**enc).last_hidden_state[0]
        token_states = token_states.float()

        # --- Predicted spans (scored above threshold) ---
        num_words = len(words)
        pred_spans: List[Tuple[int, int]] = []  # (s_word, e_word)
        pred_span_vecs: List[torch.Tensor] = []
        for s in range(num_words):
            for e in range(s, min(s + cfg.max_span_width, num_words)):
                if s not in bounds or e not in bounds:
                    continue
                s_sub, e_sub = bounds[s][0], bounds[e][1]
                score = float(model.span_score(token_states[s_sub], token_states[e_sub]).item())
                if score >= span_threshold:
                    pred_spans.append((s, e))
                    pred_span_vecs.append(token_states[s_sub: e_sub + 1].mean(dim=0))

        # --- Gold spans ---
        gold_spans: set = set()
        for entity in doc.get("vertexSet", []):
            for mention in entity:
                span = mention_global_span(mention, offsets)
                if span:
                    gold_spans.add(span)

        pred_span_set = set(pred_spans)
        span_tp += len(pred_span_set & gold_spans)
        span_fp += len(pred_span_set - gold_spans)
        span_fn += len(gold_spans - pred_span_set)

        if not pred_span_vecs:
            # missed all entities → count all gold relations as FN
            for lab in doc.get("labels", []):
                rel_fn += 1
            continue

        # --- Cluster predicted spans into entities via span overlap with gold ---
        # Match each predicted span to a gold entity index (if any).
        gold_span_to_entity: Dict[Tuple[int, int], int] = {}
        for eidx, entity in enumerate(doc.get("vertexSet", [])):
            for mention in entity:
                span = mention_global_span(mention, offsets)
                if span:
                    gold_span_to_entity[span] = eidx

        # Build entity representations from predicted spans grouped by gold entity.
        entity_to_vecs: Dict[int, List[torch.Tensor]] = {}
        unmatched_vecs: List[torch.Tensor] = []
        for (s, e), vec in zip(pred_spans, pred_span_vecs):
            eidx = gold_span_to_entity.get((s, e))
            if eidx is not None:
                entity_to_vecs.setdefault(eidx, []).append(vec)
            else:
                unmatched_vecs.append(vec)

        # Compact entity list: matched gold entities first, then unmatched spans.
        entity_reprs: List[torch.Tensor] = []
        compact_gold_map: Dict[int, int] = {}
        for eidx, vecs in entity_to_vecs.items():
            compact_gold_map[eidx] = len(entity_reprs)
            entity_reprs.append(torch.stack(vecs).mean(dim=0))
        for vec in unmatched_vecs:
            entity_reprs.append(vec)

        if not entity_reprs:
            continue

        # --- Gold relation set (mapped to compact indices) ---
        gold_rel_set: set = set()
        for lab in doc.get("labels", []):
            h, t, r = int(lab["h"]), int(lab["t"]), lab["r"]
            if h in compact_gold_map and t in compact_gold_map:
                gold_rel_set.add((compact_gold_map[h], compact_gold_map[t], r))

        # --- Predicted relation set ---
        pred_rel_set: set = set()
        n = len(entity_reprs)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                logits = model.relation_logits(entity_reprs[i], entity_reprs[j])
                probs = torch.softmax(logits, dim=-1)
                best_id = int(torch.argmax(probs).item())
                best_prob = float(probs[best_id].item())
                rel = id_to_relation.get(best_id, "NA")
                if rel != "NA" and best_prob >= relation_threshold:
                    pred_rel_set.add((i, j, rel))

        rel_tp += len(pred_rel_set & gold_rel_set)
        rel_fp += len(pred_rel_set - gold_rel_set)
        rel_fn += len(gold_rel_set - pred_rel_set)

    model.train()

    sp = span_tp / (span_tp + span_fp) if (span_tp + span_fp) else 0.0
    sr = span_tp / (span_tp + span_fn) if (span_tp + span_fn) else 0.0
    span_f1 = (2 * sp * sr / (sp + sr)) if (sp + sr) else 0.0

    rp = rel_tp / (rel_tp + rel_fp) if (rel_tp + rel_fp) else 0.0
    rr = rel_tp / (rel_tp + rel_fn) if (rel_tp + rel_fn) else 0.0
    rel_f1 = (2 * rp * rr / (rp + rr)) if (rp + rr) else 0.0

    print(f"  Dev span  P={sp:.3f} R={sr:.3f} F1={span_f1:.4f}")
    print(f"  Dev rel   P={rp:.3f} R={rr:.3f} F1={rel_f1:.4f}")

    # Combined score: span quality matters but relation is primary objective.
    return 0.4 * span_f1 + 0.6 * rel_f1


if __name__ == "__main__":
    main()
