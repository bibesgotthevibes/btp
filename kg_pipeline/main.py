from langgraph.graph import StateGraph, END
from kg_pipeline.state import KGState
from kg_pipeline.nodes.denoiser import denoise_text
from kg_pipeline.nodes.ner import detect_and_type_mentions
from kg_pipeline.nodes.coref import resolve_coreference
from kg_pipeline.nodes.extractor import extract_relations
from kg_pipeline.nodes.judge import judge_triples
from kg_pipeline.nodes.neo4j_writer import store_to_neo4j
from kg_pipeline.nodes.error_handler import handle_error
from kg_pipeline.nodes.phonetic_corrector import correct_phonetics
from kg_pipeline.nodes.entity_disambiguator import disambiguate_entities
from kg_pipeline.nodes.candidate_generator import generate_candidates
from kg_pipeline.nodes.joint_refiner import joint_refine
from kg_pipeline.utils.ollama import check_ollama_server
from kg_pipeline.nodes.joint_reasoner import joint_reason
from kg_pipeline.nodes.rexel_node import rexel_extract


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
    if not check_ollama_server():
        print("Ollama server is not running. Please start it and try again.")
        return

    workflow = StateGraph(KGState)

    workflow.add_node("denoise_text", denoise_text)
    workflow.add_node("denoise_text_entity_aware", denoise_text)
    workflow.add_node("correct_phonetics", correct_phonetics)
    workflow.add_node("detect_and_type_mentions", detect_and_type_mentions)
    workflow.add_node("resolve_coreference", resolve_coreference)
    workflow.add_node("disambiguate_entities", disambiguate_entities)
    workflow.add_node("extract_relations", extract_relations)
    workflow.add_node("judge_triples", judge_triples)
    workflow.add_node("store_to_neo4j", store_to_neo4j)
    workflow.add_node("handle_error", lambda state: handle_error(state, "Pipeline stopped due to error or empty results."))
    workflow.add_node("generate_candidates", generate_candidates)
    workflow.add_node("joint_refine", joint_refine)

    workflow.set_entry_point("denoise_text")
    workflow.add_edge("denoise_text", "correct_phonetics")
    # workflow.add_edge("correct_phonetics", "detect_and_type_mentions")
    workflow.add_edge("correct_phonetics", "generate_candidates")
    # workflow.add_edge("detect_and_type_mentions", "denoise_text_entity_aware")
    workflow.add_edge("generate_candidates", "joint_refine")
    workflow.add_edge("joint_refine", "judge_triples")
    workflow.add_edge("denoise_text_entity_aware", "resolve_coreference")
    workflow.add_edge("resolve_coreference", "disambiguate_entities")
    workflow.add_edge("disambiguate_entities", "extract_relations")
    workflow.add_edge("extract_relations", "judge_triples")
    workflow.add_conditional_edges(
        "judge_triples",
        should_continue,
        {
            "store_to_neo4j": "store_to_neo4j",
            "handle_error": "handle_error"
        }
    )
    workflow.add_edge("store_to_neo4j", END)
    workflow.add_edge("handle_error", END)

    app = workflow.compile()

    # Hardcoded test string
    test_text = """
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

    inputs = {"raw_text": test_text}
    final_state = app.invoke(inputs)

    if "kg_summary" in final_state and final_state["kg_summary"]:
        print("\n---FINAL KG SUMMARY---")
        print(final_state["kg_summary"])
    elif "error" in final_state and final_state["error"]:
        print(f"\nPipeline finished with an error: {final_state['error']}")

if __name__ == "__main__":
    main()
