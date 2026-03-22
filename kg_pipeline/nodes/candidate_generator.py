from kg_pipeline.state import KGState
from kg_pipeline.utils.ollama import call_ollama
import json, re

def safe_json_parse(response):
    try:
        return json.loads(response)
    except:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                return []
        return []

def generate_candidates(state: KGState) -> KGState:
    print("---(CANDIDATE GENERATION: HIGH RECALL)---")

    text = state["denoised_text"]

    prompt = f"""
Extract ALL possible entities and relations from the text.

Be exhaustive — include even uncertain ones.

Text:
{text}

Return JSON:

[
  {{
    "head": "...",
    "relation": "...",
    "tail": "...",
    "confidence": "high/medium/low"
  }}
]

Rules:
- Do NOT filter aggressively
- Capture maximum possible knowledge
"""

    response = call_ollama(prompt)
    triples = safe_json_parse(response)

    state["draft_triples"] = triples
    return state