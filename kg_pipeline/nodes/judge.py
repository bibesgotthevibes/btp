from kg_pipeline.state import KGState
from kg_pipeline.utils.ollama import call_ollama


def _build_graphjudge_prompt(instruction: str, context_text: str) -> str:
    return f"""
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{context_text}
### Response:
"""


def _is_true_from_generated_response(response_text: str, limit: int = 100) -> bool:
    response_lower = response_text.strip().lower()
    window = response_lower[:limit]
    if "no" in window or "false" in window:
        return False
    return True


def _judge_backend(instruction: str, context_text: str) -> str:
    prompt = _build_graphjudge_prompt(instruction, context_text)
    return call_ollama(prompt)


def judge_triples(state: KGState) -> KGState:
    """
    GraphJudge-style triple verification.
    Uses instruction form from prepare_KGCom.ipynb:
    "Is this true: head relation tail?"
    Then applies the same filtering rule used in notebook scripts:
    if generated response starts with / contains "no" or "false" in early window -> reject.
    """
    print("---(5) GRAPH JUDGE VERIFICATION---")

    draft_triples = state.get("draft_triples", [])
    denoised_text = state.get("denoised_text", "")

    if not draft_triples:
        print("No draft triples to verify.")
        state["verified_triples"] = []
        return state

    verified_triples = []

    for triple in draft_triples:
        head = triple.get("head")
        relation = triple.get("relation")
        tail = triple.get("tail")

        if not head or not relation or not tail:
            continue

        instruction = f"Is this true: {head} {relation} {tail}?"

        try:
            response = _judge_backend(instruction, denoised_text)
        except Exception as exc:
            print(f"Judge call failed for triple {triple}: {exc}")
            continue

        if _is_true_from_generated_response(response, limit=100):
            verified_triples.append({
                "head": head,
                "relation": relation,
                "tail": tail,
            })
        else:
            print(f"Rejected: {head} - {relation} - {tail}")

    state["verified_triples"] = verified_triples
    print(f"Verified {len(verified_triples)} triples out of {len(draft_triples)} drafts.")
    return state