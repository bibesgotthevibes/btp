from kg_pipeline.state import KGState
from kg_pipeline.nodes.rexel_model import KGModel

model = KGModel()

def rexel_extract(state: KGState) -> KGState:
    print("--- REXEL MODEL ---")

    text = state["denoised_text"]
    triples = model.extract(text)

    state["draft_triples"] = triples
    state["verified_triples"] = triples  # for now

    return state