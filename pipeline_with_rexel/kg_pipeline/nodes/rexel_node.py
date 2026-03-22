import os

from kg_pipeline.state import KGState
from kg_pipeline.nodes.rexel_model import REXELJointModel


_model = None


def _get_model() -> REXELJointModel:
    global _model
    if _model is None:
        _model = REXELJointModel(
            backend=os.getenv("REXEL_BACKEND", "llm"),
            checkpoint_path=os.getenv("REXEL_CHECKPOINT", "checkpoints/rexel_joint.pt"),
        )
    return _model


def rexel_extract(state: KGState) -> KGState:
    """
    Phase 2 joint modeling node: NER + RE + Linking in one pass.
    """
    print("---(3) REXEL JOINT EXTRACTION: NER + RE + LINKING---")

    text = state.get("denoised_text", "")
    if not text.strip():
        state["error"] = "Empty input passed to REXEL node"
        state["draft_triples"] = []
        state["entity_clusters"] = []
        state["mentions"] = []
        return state

    result = _get_model().extract(text)

    state["mentions"] = result["mentions"]
    state["entity_clusters"] = result["entity_clusters"]
    state["draft_triples"] = result["triples"]

    print(
        f"REXEL produced {len(state['entity_clusters'])} entities and "
        f"{len(state['draft_triples'])} draft triples."
    )

    if state["draft_triples"]:
        print("Sample REXEL relations (up to 10):")
        for triple in state["draft_triples"][:10]:
            print(f"  - {triple.get('head')} | {triple.get('relation')} | {triple.get('tail')}")

    return state
