import json
from kg_pipeline.state import KGState
def store_to_neo4j(state: KGState) -> KGState:
    print("---(6) STORING TRIPLES---")

    verified_triples = state.get("verified_triples", [])
    entity_clusters = state.get("entity_clusters", [])

    output = {
        "entity_clusters": entity_clusters,
        "verified_triples": verified_triples
    }

    output_path = "kg_output.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    summary = {
        "nodes_written": len(entity_clusters),
        "edges_written": len(verified_triples)
    }

    state["kg_summary"] = summary

    print(f"Wrote {summary['nodes_written']} entities and {summary['edges_written']} triples")

    return state