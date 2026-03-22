import json
import re
from kg_pipeline.state import KGState
from kg_pipeline.utils.ollama import call_ollama

def safe_json_parse(response):
    try:
        return json.loads(response)
    except:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                return []
        return []

def deduplicate(triples):
    seen = set()
    unique = []

    for t in triples:
        key = (t["head"], t["relation"], t["tail"])
        if key not in seen:
            seen.add(key)
            unique.append(t)

    return unique

def clean_triples(triples, entity_clusters):
    valid_entities = set([c["canonical"] for c in entity_clusters])

    clean = []
    for t in triples:
        if t["head"] not in valid_entities:
            continue
        if t["tail"] not in valid_entities:
            continue
        if len(t["relation"].split()) > 4:
            continue
        if "Entity A" in t["head"] or "Entity B" in t["tail"]:
            continue

        clean.append(t)

    return clean
def extract_relations(state: KGState) -> KGState:
    """
    Extracts relations between entity clusters.
    """
    print("---(4) EXTRACTING RELATIONS---")
    denoised_text = state["denoised_text"]
    entity_clusters = state["entity_clusters"]

    # Stage A: Coarse filter
    sentences = denoised_text.split('.')
    cluster_sentence_map = []
    for cluster in entity_clusters:
        sentence_indices = set()
        for mention_text in cluster["mentions"]:
            for i, sentence in enumerate(sentences):
                if mention_text in sentence:
                    sentence_indices.add(i)
        cluster_sentence_map.append(sentence_indices)

    surviving_pairs = []
    for i in range(len(entity_clusters)):
        for j in range(i + 1, len(entity_clusters)):
            sent_indices_i = cluster_sentence_map[i]
            sent_indices_j = cluster_sentence_map[j]

            # Check for overlap within a window of 3 sentences
            for s_i in sent_indices_i:
                for s_j in sent_indices_j:
                    if abs(s_i - s_j) <= 3:
                        surviving_pairs.append((entity_clusters[i], entity_clusters[j]))
                        break
                else:
                    continue
                break

    print(f"Coarse filtering: {len(surviving_pairs)} pairs survived out of {len(entity_clusters) * (len(entity_clusters) - 1) // 2} total pairs.")

    # Stage B: Fine extraction
    draft_triples = []
    with open("kg_pipeline/prompts/extract.txt", "r") as f:
        prompt_template = f.read()

    prompt = f"""
        Extract all relations from the text.

        Text:
        {denoised_text}

        Entities:
        {[c["canonical"] for c in entity_clusters]}

        Return ONLY JSON array:
        [
        {{"head": "...", "relation": "...", "tail": "..."}}
        ]
        """

    response = call_ollama(prompt)

    try:
        relations = safe_json_parse(response)

        valid_entities = set([c["canonical"] for c in entity_clusters])

        clean_relations = []
        for r in relations:
            if r["head"] not in valid_entities:
                continue
            if r["tail"] not in valid_entities:
                continue
            clean_relations.append(r)

        if clean_relations:
            draft_triples.extend(clean_relations)

    except json.JSONDecodeError:
            print(f"Failed to parse JSON from Ollama response: {response}")

    draft_triples = clean_triples(draft_triples, entity_clusters)
    draft_triples = deduplicate(draft_triples)
    state["draft_triples"] = draft_triples
    print(f"Extracted {len(draft_triples)} draft triples.")
    return state
