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

def joint_refine(state: KGState) -> KGState:
    print("---(JOINT REFINEMENT: NER ↔ RE ↔ EL)---")

    text = state["denoised_text"]
    triples = state["draft_triples"]

    prompt = f"""
You are refining a knowledge graph.

Text:
{text}

Initial triples:
{json.dumps(triples, indent=2)}

Tasks:
1. Normalize entity names (merge duplicates)
2. Remove invalid entities (like "the study", "stock", etc.)
3. Fix relation types to standard schema:
   - affiliated_with
   - located_in
   - developed_by
   - published_in
   - conducted_in
   - partner_with
4. Remove incorrect or unsupported triples
5. Ensure consistency across triples

Return JSON:
{{
  "entities": [
    {{"name": "...", "type": "..."}}
  ],
  "triples": [
    {{"head": "...", "relation": "...", "tail": "..."}}
  ]
}}
"""

    response = call_ollama(prompt)
    parsed = safe_json_parse(response)

    state["entity_clusters"] = [
        {
            "canonical": e["name"],
            "types": [e["type"]],
            "mentions": [e["name"]],
            "wikidata_id": e["name"].lower().replace(" ", "_")
        }
        for e in parsed.get("entities", [])
    ]

    state["draft_triples"] = parsed.get("triples", [])

    return state