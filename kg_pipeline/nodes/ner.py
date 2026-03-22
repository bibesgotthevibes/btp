from kg_pipeline.state import KGState
from kg_pipeline.utils.gliner_ner import extract_mentions

def detect_and_type_mentions(state: KGState) -> KGState:
    """
    Detects and types mentions in the denoised text using GLiNER.
    """
    print("---(2) DETECTING AND TYPING MENTIONS---")
    denoised_text = state["denoised_text"]

    assert denoised_text and denoised_text != state["raw_text"], "Denoised text is empty or same as raw text."

    labels = ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "PRODUCT", "EVENT", "CONCEPT"]

    mentions = extract_mentions(denoised_text, labels, threshold=0.5)

    state["mentions"] = mentions
    print(f"Detected {len(mentions)} mentions.")
    return state
