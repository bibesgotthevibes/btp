from gliner import GLiNER
from typing import List, Dict
from collections import defaultdict

# Load model once at module level
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

def extract_mentions(text: str, labels: List[str], threshold: float = 0.5) -> List[Dict]:
    """
    Runs GLiNER model to extract mentions and their types from text.
    GLiNER returns one prediction per entity span per label above threshold.
    We group by span so one mention can carry multiple types.
    """
    predictions = model.predict_entities(text, labels, threshold=threshold)

    # Group by span — same text span can have multiple labels
    span_map = defaultdict(lambda: {"types": [], "score": {}, "start": None, "end": None})

    for entity in predictions:
        key = (entity["start"], entity["end"])
        span_map[key]["text"] = entity["text"]
        span_map[key]["start"] = entity["start"]
        span_map[key]["end"] = entity["end"]
        span_map[key]["types"].append(entity["label"])
        span_map[key]["score"][entity["label"]] = entity["score"]

    mentions = list(span_map.values())
    return mentions