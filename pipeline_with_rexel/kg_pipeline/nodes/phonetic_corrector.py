from kg_pipeline.state import KGState
from kg_pipeline.utils.ollama import call_ollama
import json


def _sanitize_corrected_text(raw_output: str, fallback_text: str) -> str:
    """Keep only plain corrected passage and avoid commentary-heavy responses."""
    if not raw_output or not raw_output.strip():
        return fallback_text

    text = raw_output.strip()
    lowered = text.lower()

    # Remove common wrappers the model sometimes prepends.
    wrappers = [
        "corrected text:",
        "here is the cleaned text:",
        "here is the corrected text:",
        "cleaned text:",
    ]
    for prefix in wrappers:
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip()
            lowered = text.lower()
            break

    # If the output is mostly reflective commentary, keep original text.
    commentary_markers = [
        "it seems like",
        "more likely",
        "probably",
        "however",
        "as per surrounding context",
    ]
    if any(marker in lowered for marker in commentary_markers):
        return fallback_text

    return text if text else fallback_text

def correct_phonetics(state: KGState) -> KGState:
    """
    Corrects ASR/OCR phonetic errors using entity-centric context.
    Follows the ECTD approach from GraphJudge paper:
    Step A — rough entity extraction from noisy text
    Step B — use those entities as anchors to correct phonetic variants
    """
    print("---(2) CORRECTING PHONETIC ERRORS---")
    denoised_text = state["denoised_text"]

    # Step A — rough entity extraction from the noisy text
    step_a_prompt = f"""Read the following text and list every proper noun you can find.
This includes: people names, company names, drug names, place names, organization names.
The text may contain speech transcription errors so spellings may be wrong.
List them exactly as they appear in the text, do not correct them yet.

Return ONLY a JSON array of strings. No explanation. No markdown.
Example: ["farma corp", "doctor evelyn reed", "knew deli", "cardia care"]

Text:
{denoised_text}"""

    raw_entities_response = call_ollama(step_a_prompt)
    
    try:
        raw_entities = json.loads(raw_entities_response)
    except json.JSONDecodeError:
        print(f"Step A parse failed, using empty list. Raw: {raw_entities_response}")
        raw_entities = []

    print(f"  Step A found {len(raw_entities)} rough entities: {raw_entities}")

    # Step B — use those rough entities + surrounding context to correct spellings
    step_b_prompt = f"""You are correcting a speech transcription that contains phonetic errors.

I found these potential entity names in the text (they may be misspelled):
{json.dumps(raw_entities, indent=2)}

For each misspelled entity, use the surrounding context in the text to figure out 
what the correct spelling should be. 

Rules:
1. Use context clues — what words appear near the entity? What domain is the text about?
2. If an entity appears near words like "stock", "traded", "drug", "developed" 
   it is likely a company or product name — correct it to proper capitalization and spelling
3. If an entity sounds like a real place name when read aloud, correct it
   Example: "knew deli" sounds like "New Delhi" → correct it
   Example: "P H C" spelled out letter by letter → correct to "PHC"
   Example: "A T one receptor" → "AT1 receptor"
4. Do NOT invent new facts or change the meaning
5. Return the full corrected text only, no explanation

Original text:
{denoised_text}"""

    corrected_text = call_ollama(step_b_prompt)

    sanitized_text = _sanitize_corrected_text(corrected_text, denoised_text)
    if sanitized_text == denoised_text and corrected_text.strip() != denoised_text.strip():
        print("Phonetic correction output looked commentary-heavy; using original denoised text.")

    state["denoised_text"] = sanitized_text
    print("Phonetic correction complete.")
    return state
