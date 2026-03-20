from kg_pipeline.state import KGState
from rapidfuzz.fuzz import token_sort_ratio
from collections import defaultdict

def resolve_coreference(state: KGState) -> KGState:
    """
    Resolves coreferences by clustering mentions.
    """
    print("---(3) RESOLVING COREFERENCES---")
    mentions = state["mentions"]

    # Create a graph where nodes are mentions and edges are similarities
    mention_graph = defaultdict(list)
    for i in range(len(mentions)):
        for j in range(i + 1, len(mentions)):
            m1 = mentions[i]
            m2 = mentions[j]

            similarity = token_sort_ratio(m1["text"], m2["text"])

            # Check for shared type
            has_shared_type = any(t in m2["types"] for t in m1["types"])

            if similarity > 85 and has_shared_type:
                mention_graph[i].append(j)
                mention_graph[j].append(i)

    # Find connected components (clusters) using BFS/DFS
    visited = set()
    clusters = []
    for i in range(len(mentions)):
        if i not in visited:
            cluster_indices = []
            q = [i]
            visited.add(i)
            while q:
                node_idx = q.pop(0)
                cluster_indices.append(node_idx)
                for neighbor_idx in mention_graph[node_idx]:
                    if neighbor_idx not in visited:
                        visited.add(neighbor_idx)
                        q.append(neighbor_idx)
            clusters.append(cluster_indices)

    # Format clusters
    entity_clusters = []
    for cluster_indices in clusters:
        cluster_mentions = [mentions[i] for i in cluster_indices]

        canonical = max(cluster_mentions, key=lambda m: len(m["text"]))["text"]

        all_types = set()
        for m in cluster_mentions:
            all_types.update(m["types"])

        # Average scores per type
        type_scores = defaultdict(list)
        for m in cluster_mentions:
            for t in all_types:
                if t in m["score"]:
                    type_scores[t].append(m["score"][t])

        avg_type_scores = {t: sum(scores)/len(scores) for t, scores in type_scores.items()}

        entity_clusters.append({
            "canonical": canonical,
            "types": list(all_types),
            "type_scores": avg_type_scores,
            "mentions": [m["text"] for m in cluster_mentions]
        })

    state["entity_clusters"] = entity_clusters
    print(f"Resolved {len(mentions)} mentions into {len(entity_clusters)} clusters.")
    return state
