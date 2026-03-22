from kg_pipeline.state import KGState
import re

def normalize_entity(name: str) -> str:
    """
    Creates a stable canonical ID for an entity.
    """
    name = name.lower()
    name = re.sub(r'[^a-z0-9 ]', '', name)
    name = name.strip().replace(" ", "_")
    return name


def disambiguate_entities(state: KGState) -> KGState:
    """
    Lightweight entity linking (no external models).
    """
    print("---(5) LIGHTWEIGHT ENTITY LINKING---")

    entity_clusters = state["entity_clusters"]

    for cluster in entity_clusters:
        canonical = cluster["canonical"]

        # Create stable ID
        entity_id = normalize_entity(canonical)

        cluster["wikidata_id"] = entity_id
        cluster["wikipedia_name"] = canonical

        print(f"  {canonical} → {entity_id}")

    state["entity_clusters"] = entity_clusters
    print(f"Linked {len(entity_clusters)} entities (lightweight).")

    return state