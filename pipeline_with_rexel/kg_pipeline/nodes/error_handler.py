from kg_pipeline.state import KGState

def handle_error(state: KGState, message: str) -> KGState:
    """
    Handles errors in the pipeline.
    """
    print(f"---ERROR---")
    print(message)
    state["error"] = message
    return state
