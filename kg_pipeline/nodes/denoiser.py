from kg_pipeline.state import KGState
from kg_pipeline.utils.ollama import call_ollama


def sentence_filter(text: str, entity_list: list[str]) -> str:
    sentences = text.split(".")
    return ". ".join([
        sentence for sentence in sentences
        if any(entity.lower() in sentence.lower() for entity in entity_list)
    ])

def denoise_text(state: KGState) -> KGState:
    print("---(1) DENOISING TEXT (ENTITY-AWARE)---")

    raw_text = state["raw_text"]
    mentions = state.get("mentions", [])

    # If no entities yet → do LIGHT cleaning only
    if not mentions:
        prompt = f"""
Clean the following text by:
- Removing advertisements, promotional content
- Removing repeated filler speech (uh, um, etc.)
- Fixing sentence structure slightly

DO NOT remove factual content.
DO NOT summarize.

Return only cleaned text.

Text:
{raw_text}
"""
        cleaned = call_ollama(prompt)
        state["denoised_text"] = cleaned
        return state

    # If entities exist → ENTITY-CENTRIC filtering
    entity_list = list(set([m["text"] for m in mentions]))

    filtered = sentence_filter(raw_text, entity_list)
    candidate_text = filtered if filtered.strip() else raw_text

    prompt = f"""
You are cleaning a document for knowledge graph extraction.

Entities detected in the document:
{entity_list}

Your task:
1. Keep ONLY sentences relevant to these entities
2. Remove advertisements, promotional text, irrelevant sections
3. Preserve all factual relationships between entities
4. Do NOT summarize or paraphrase too much
5. Keep original meaning intact

Return ONLY cleaned text.

Document:
{candidate_text}
"""

    cleaned = call_ollama(prompt)
    state["denoised_text"] = cleaned
    return state
