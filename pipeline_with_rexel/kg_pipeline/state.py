from typing import TypedDict, List, Dict, Optional

class KGState(TypedDict):
    raw_text: str
    denoised_text: str
    mentions: List[Dict]
    entity_clusters: List[Dict]
    draft_triples: List[Dict]
    verified_triples: List[Dict]
    kg_summary: Dict
    error: str
