import argparse
import os
from pathlib import Path

from langgraph.graph import StateGraph, END
from kg_pipeline.state import KGState
from kg_pipeline.nodes.denoiser import denoise_text
from kg_pipeline.nodes.judge import judge_triples
from kg_pipeline.nodes.neo4j_writer import store_to_neo4j
from kg_pipeline.nodes.error_handler import handle_error
from kg_pipeline.nodes.phonetic_corrector import correct_phonetics
from kg_pipeline.nodes.rexel_node import rexel_extract
from kg_pipeline.utils.ollama import check_ollama_server


def parse_args():
    parser = argparse.ArgumentParser(description="Run KG pipeline")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Raw input text for the pipeline",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to a UTF-8 text file with input content",
    )
    parser.add_argument(
        "--rexel-backend",
        type=str,
        choices=["llm", "torch"],
        default="llm",
        help="REXEL backend to use for Phase 2 joint modeling",
    )
    parser.add_argument(
        "--rexel-checkpoint",
        type=str,
        default="checkpoints/rexel_joint.pt",
        help="Path to trained torch REXEL checkpoint (used when --rexel-backend torch)",
    )
    parser.add_argument(
        "--rexel-span-threshold",
        type=float,
        default=None,
        help="Optional torch inference threshold for mention/span proposals",
    )
    parser.add_argument(
        "--rexel-relation-threshold",
        type=float,
        default=None,
        help="Optional torch inference threshold for relation predictions",
    )
    return parser.parse_args()


def resolve_input_text(args) -> str:
    if args.text:
        return args.text

    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists() or not input_path.is_file():
            raise ValueError(f"Input file not found: {args.input_file}")
        return input_path.read_text(encoding="utf-8")

    # Fallback sample text so current behavior remains unchanged
    return """
uh so basically doctor evelyn reed she is uh a cardiologist from the mayo clinic
in rochester minnesota she was presenting her research on this new drug called
cardia care um its made by farma corp i think thats how you spell it
the drug showed like a thirty percent reduction in heart related incidents
or something like that

FLASH SALE THIS WEEK ONLY GET 50 PERCENT OFF ALL SUPPLEMENTS VISIT WWW DOT HEALTHDEALS DOT COM

so the study had over five thousand patients and it was published in the
new england journal of medicine on may first twenty twenty four
farma corp stock which trades as P H C jumped fifteen percent after the announcement

doctor reed also mentioned that cardia care works differently from
existing beta blockers she said it targets a specific receptor called
the A T one receptor which existing drugs dont focus on

buy one get one free on all vitamins and minerals limited time offer

the mayo clinic and farma corp are now planning a phase three trial
which will be conducted across twelve hospitals in the united states
doctor reed said the results were very promising
the trial is expected to begin in knew deli india and london england next year
"""


def should_continue(state: KGState):
    if state.get("error"):
        return "handle_error"
    if not state.get("denoised_text"):
        return "handle_error"
    # if not state.get("mentions"):
    #     return "handle_error"
    # if not state.get("draft_triples"):
    #     return "handle_error"
    if state.get("verified_triples") is None:
        return "handle_error"
    return "store_to_neo4j"

def main():
    args = parse_args()

    os.environ["REXEL_BACKEND"] = args.rexel_backend
    os.environ["REXEL_CHECKPOINT"] = args.rexel_checkpoint
    if args.rexel_span_threshold is not None:
        os.environ["REXEL_SPAN_THRESHOLD"] = str(args.rexel_span_threshold)
    if args.rexel_relation_threshold is not None:
        os.environ["REXEL_RELATION_THRESHOLD"] = str(args.rexel_relation_threshold)

    if not check_ollama_server():
        print("Ollama server is not running. Please start it and try again.")
        return

    workflow = StateGraph(KGState)

    workflow.add_node("denoise_text", denoise_text)
    workflow.add_node("correct_phonetics", correct_phonetics)
    workflow.add_node("rexel_extract", rexel_extract)
    workflow.add_node("judge_triples", judge_triples)
    workflow.add_node("store_to_neo4j", store_to_neo4j)
    workflow.add_node(
        "handle_error",
        lambda state: handle_error(state, "Pipeline stopped")
    )

    workflow.set_entry_point("denoise_text")

    workflow.add_edge("denoise_text", "correct_phonetics")
    workflow.add_edge("correct_phonetics", "rexel_extract")
    workflow.add_edge("rexel_extract", "judge_triples")

    workflow.add_conditional_edges(
        "judge_triples",
        should_continue,
        {
            "store_to_neo4j": "store_to_neo4j",
            "handle_error": "handle_error",
        },
    )

    workflow.add_edge("store_to_neo4j", END)
    workflow.add_edge("handle_error", END)
    app = workflow.compile()

    try:
        input_text = resolve_input_text(args)
    except ValueError as err:
        print(err)
        return

    inputs = {"raw_text": input_text}
    final_state = app.invoke(inputs)

    if "kg_summary" in final_state and final_state["kg_summary"]:
        print("\n---FINAL KG SUMMARY---")
        print(final_state["kg_summary"])
    elif "error" in final_state and final_state["error"]:
        print(f"\nPipeline finished with an error: {final_state['error']}")

if __name__ == "__main__":
    main()
