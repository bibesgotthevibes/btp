from typing import List, Dict

_refined = None


def _empty_result(entity_clusters: List[Dict]) -> Dict:
    result = {}
    for cluster in entity_clusters:
        canonical = cluster["canonical"]
        result[canonical] = {
            "wikidata_id": None,
            "wikipedia_name": None
        }
    return result


def _get_refined():
    global _refined
    if _refined is not None:
        return _refined

    from refined.inference.processor import Refined

    _refined = Refined.from_pretrained(
        model_name='wikipedia_model_with_numbers',
        entity_set='wikidata'
    )
    return _refined

def get_wikidata_ids(text: str, entity_clusters: List[Dict]) -> Dict:
    """
    Maps canonical entity names to Wikidata IDs using ReFinED.
    Returns dict: {"PharmaCorp": {"wikidata_id": "Q47117876", "wikipedia_name": "..."}}
    """
    try:
        refined = _get_refined()
        spans = refined.process_text(text)
    except OSError as err:
        print(f"ReFinED unavailable (likely disk space issue): {err}")
        return _empty_result(entity_clusters)
    except Exception as err:
        print(f"ReFinED failed, continuing without Wikidata IDs: {err}")
        return _empty_result(entity_clusters)

    # Build lookup from surface text → wikidata info
    surface_to_wikidata = {}
    for span in spans:
        if span.predicted_entity and span.predicted_entity.wikidata_entity_id:
            surface_to_wikidata[span.text.lower()] = {
                "wikidata_id": span.predicted_entity.wikidata_entity_id,
                "wikipedia_name": span.predicted_entity.wikipedia_entity_title
            }

    # Match against our canonical cluster names
    result = {}
    for cluster in entity_clusters:
        canonical = cluster["canonical"]
        # Try exact match first, then try each mention
        match = surface_to_wikidata.get(canonical.lower())
        if not match:
            for mention in cluster["mentions"]:
                match = surface_to_wikidata.get(mention.lower())
                if match:
                    break

        result[canonical] = match if match else {
            "wikidata_id": None,
            "wikipedia_name": None
        }

    return result
