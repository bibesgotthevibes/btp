from kg_pipeline.state import KGState
from kg_pipeline.utils.ollama import call_ollama
import json, re


def safe_json_parse(response):
    try:
        return json.loads(response)
    except:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                return {}
        return {}


def joint_reason(state: KGState) -> KGState:
    print("---(JOINT REASONING: REXEL SIMULATION)---")

    text = state["denoised_text"]

    prompt = f"""
You are constructing a knowledge graph from noisy text.

Perform ALL tasks jointly:
1. Detect entity mentions
2. Resolve coreferences (merge same entities)
3. Assign entity types
4. Normalize entity names (canonical form)
5. Extract relations between entities

Text:
{text}

Return STRICT JSON:

{{
  "entities": [
    {{
      "canonical": "...",
      "type": "...",
      "mentions": ["...", "..."]
    }}
  ],
  "triples": [
    {{
      "head": "...",
      "relation": "...",
      "tail": "..."
    }}
  ]
}}

Rules:
- Merge "Doctor Reed" and "Doctor Evelyn Reed"
- Normalize names (e.g., "knew deli" → "New Delhi")
- Only include meaningful relations:
  affiliated_with, located_in, developed_by, published_in, conducted_in, partner_with
- Do NOT include vague relations like "mentioned"
- Ensure consistency across entities and triples
- Be exhaustive but only include facts clearly supported by the text

IMPORTANT OUTPUT RULES:
- Output ONLY valid JSON
- Do NOT include explanations
- Do NOT include any text before or after JSON
- Do NOT use markdown (no ```json)
- The response must start with '{{' and end with '}}'

If you cannot find entities, return:
{{
  "entities": [],
  "triples": []
}}
"""

    response = call_ollama(prompt)
    parsed = safe_json_parse(response)

    entities = parsed.get("entities", [])
    triples = parsed.get("triples", [])

    # Convert to your format
    entity_clusters = []
    for e in entities:
        entity_clusters.append({
            "canonical": e["canonical"],
            "types": [e["type"]],
            "mentions": e["mentions"],
            "wikidata_id": e["canonical"].lower().replace(" ", "_")
        })

    state["entity_clusters"] = entity_clusters
    state["draft_triples"] = triples

    print(f"Entities: {len(entity_clusters)}, Triples: {len(triples)}")

    return state