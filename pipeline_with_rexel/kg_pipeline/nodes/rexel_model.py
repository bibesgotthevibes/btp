import json
import os
import re
from typing import Dict, List, Optional

from kg_pipeline.utils.ollama import call_ollama
from kg_pipeline.utils.refined_el import get_wikidata_ids


_ALLOWED_ENTITY_TYPES = {
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "DATE",
    "PRODUCT",
    "EVENT",
    "CONCEPT",
}

_ALLOWED_RELATIONS = {
    "affiliated_with",
    "located_in",
    "developed_by",
    "published_in",
    "conducted_in",
    "partner_with",
    "has_ceo",
    "ceo_of",
}


class REXELJointModel:
    """
    Practical REXEL-style joint extractor.

    This implementation performs document-level mention/entity/relation extraction
    in one model call, then runs optional entity linking.
    """

    def __init__(
        self,
        relation_schema: Optional[List[str]] = None,
        backend: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):
        self.relation_schema = relation_schema or sorted(_ALLOWED_RELATIONS)
        self.backend = (backend or os.getenv("REXEL_BACKEND", "llm")).strip().lower()
        self._torch_predictor = None

        if self.backend == "torch":
            ckpt = checkpoint_path or os.getenv("REXEL_CHECKPOINT", "checkpoints/rexel_joint.pt")
            span_thr_raw = os.getenv("REXEL_SPAN_THRESHOLD")
            rel_thr_raw = os.getenv("REXEL_RELATION_THRESHOLD")
            span_width_raw = os.getenv("REXEL_MAX_SPAN_WIDTH")
            span_thr = float(span_thr_raw) if span_thr_raw else None
            rel_thr = float(rel_thr_raw) if rel_thr_raw else None
            span_width = int(span_width_raw) if span_width_raw else None  # defaults to min(ckpt_width, 4)
            try:
                from kg_pipeline.nodes.rexel_torch_model import REXELTorchPredictor

                self._torch_predictor = REXELTorchPredictor.from_checkpoint_path(
                    ckpt,
                    span_threshold=span_thr,
                    relation_threshold=rel_thr,
                    max_span_width=span_width,
                )
                print(
                    f"Loaded REXEL torch backend from {ckpt} "
                    f"(span_threshold={self._torch_predictor.config.span_threshold}, "
                    f"relation_threshold={self._torch_predictor.config.relation_threshold}, "
                    f"max_span_width={self._torch_predictor.config.max_span_width})"
                )
            except Exception as err:
                print(f"Failed to load torch backend ({err}); falling back to LLM backend.")
                self.backend = "llm"

    def _safe_json_parse(self, response: str) -> Dict:
        try:
            parsed = json.loads(response)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if not match:
                return {}
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}

    def _normalize_entities(self, raw_entities: List[Dict]) -> List[Dict]:
        entities = []
        seen = set()

        for item in raw_entities:
            canonical = str(item.get("canonical", "")).strip()
            if not canonical:
                continue

            mentions = item.get("mentions", [])
            if not isinstance(mentions, list) or not mentions:
                mentions = [canonical]

            mentions = [str(m).strip() for m in mentions if str(m).strip()]
            if not mentions:
                mentions = [canonical]

            etype = str(item.get("type", "CONCEPT")).strip().upper()
            if etype not in _ALLOWED_ENTITY_TYPES:
                etype = "CONCEPT"

            key = canonical.lower()
            if key in seen:
                continue
            seen.add(key)

            entities.append(
                {
                    "canonical": canonical,
                    "type": etype,
                    "mentions": sorted(set(mentions)),
                }
            )

        return entities

    def _normalize_triples(self, raw_triples: List[Dict], valid_entities: set) -> List[Dict]:
        triples = []
        seen = set()

        for item in raw_triples:
            head = str(item.get("head", "")).strip()
            relation = str(item.get("relation", "")).strip().lower()
            tail = str(item.get("tail", "")).strip()

            if not head or not relation or not tail:
                continue
            if head not in valid_entities or tail not in valid_entities:
                continue

            # Map common noisy variant to schema.
            relation = relation.replace(" ", "_")
            if relation == "is_ceo_of":
                relation = "ceo_of"

            if relation not in _ALLOWED_RELATIONS:
                continue

            key = (head, relation, tail)
            if key in seen:
                continue
            seen.add(key)
            triples.append({"head": head, "relation": relation, "tail": tail})

        return triples

    def _build_prompt(self, text: str) -> str:
        relation_schema = ", ".join(self.relation_schema)
        return f"""
You are a document-level information extraction model.

Perform these tasks JOINTLY in a single pass over the document:
1) Mention/entity detection with typing
2) Coreference merge to canonical entities
3) Relation extraction between canonical entities

Document:
{text}

Return STRICT JSON only (no markdown):
{{
  "entities": [
    {{
      "canonical": "...",
      "type": "PERSON|ORGANIZATION|LOCATION|DATE|PRODUCT|EVENT|CONCEPT",
      "mentions": ["...", "..."]
    }}
  ],
  "triples": [
    {{
      "head": "canonical entity",
      "relation": "one of: {relation_schema}",
      "tail": "canonical entity"
    }}
  ]
}}

Rules:
- Keep entity names faithful to the text.
- Merge aliases and pronouns to one canonical entity.
- Only output relations directly supported by the document.
- If no facts are present, return empty arrays.
"""

    def extract(self, text: str) -> Dict:
        if self.backend == "torch" and self._torch_predictor is not None:
            result = self._torch_predictor.extract(text)

            link_map = get_wikidata_ids(text, result.get("entity_clusters", [])) if result.get("entity_clusters") else {}
            for cluster in result.get("entity_clusters", []):
                linked = link_map.get(cluster["canonical"], {})
                cluster["wikidata_id"] = linked.get("wikidata_id")
                cluster["wikipedia_name"] = linked.get("wikipedia_name")

            return {
                "mentions": result.get("mentions", []),
                "entity_clusters": result.get("entity_clusters", []),
                "triples": result.get("triples", []),
            }

        prompt = self._build_prompt(text)
        response = call_ollama(prompt)
        parsed = self._safe_json_parse(response)

        entities = self._normalize_entities(parsed.get("entities", []))
        valid_entities = {e["canonical"] for e in entities}
        triples = self._normalize_triples(parsed.get("triples", []), valid_entities)

        # Convert entities to existing pipeline structure.
        entity_clusters = []
        for ent in entities:
            entity_clusters.append(
                {
                    "canonical": ent["canonical"],
                    "types": [ent["type"]],
                    "mentions": ent["mentions"],
                    "wikidata_id": None,
                    "wikipedia_name": None,
                }
            )

        link_map = get_wikidata_ids(text, entity_clusters) if entity_clusters else {}
        for cluster in entity_clusters:
            linked = link_map.get(cluster["canonical"], {})
            cluster["wikidata_id"] = linked.get("wikidata_id")
            cluster["wikipedia_name"] = linked.get("wikipedia_name")

        mentions = []
        for cluster in entity_clusters:
            for mention in cluster["mentions"]:
                mentions.append(
                    {
                        "text": mention,
                        "types": cluster["types"],
                        "score": {cluster["types"][0]: 1.0},
                    }
                )

        return {
            "mentions": mentions,
            "entity_clusters": entity_clusters,
            "triples": triples,
        }
