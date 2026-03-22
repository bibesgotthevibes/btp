import argparse
import json
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from kg_pipeline.nodes.rexel_torch_model import REXELTorchPredictor


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def flatten_doc_text(doc: Dict) -> str:
    return " ".join(" ".join(sent) for sent in doc.get("sents", []))


def build_name_to_entity_ids(doc: Dict) -> Dict[str, Set[int]]:
    name_map: Dict[str, Set[int]] = defaultdict(set)
    for idx, entity in enumerate(doc.get("vertexSet", [])):
        for mention in entity:
            name = mention.get("name", "")
            key = normalize(name)
            if key:
                name_map[key].add(idx)
    return name_map


def gold_relation_set(doc: Dict) -> Set[Tuple[int, int, str]]:
    gold = set()
    for lab in doc.get("labels", []):
        try:
            h = int(lab.get("h", -1))
            t = int(lab.get("t", -1))
        except Exception:
            continue
        r = str(lab.get("r", "")).strip()
        if h >= 0 and t >= 0 and r:
            gold.add((h, t, r))
    return gold


def predicted_relation_set(
    pred_triples: List[Dict],
    name_to_ids: Dict[str, Set[int]],
) -> Tuple[Set[Tuple[int, int, str]], int]:
    pred = set()
    unmatched = 0

    for triple in pred_triples:
        head = normalize(str(triple.get("head", "")))
        tail = normalize(str(triple.get("tail", "")))
        rel = str(triple.get("relation", "")).strip()

        if not head or not tail or not rel:
            continue

        head_ids = name_to_ids.get(head, set())
        tail_ids = name_to_ids.get(tail, set())

        if not head_ids or not tail_ids:
            unmatched += 1
            continue

        for h in head_ids:
            for t in tail_ids:
                if h != t:
                    pred.add((h, t, rel))

    return pred, unmatched


def main():
    parser = argparse.ArgumentParser(description="Evaluate REXEL checkpoint on test split and export graph")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-file", type=str, default="data/test_joint_qcode.json")
    parser.add_argument("--span-threshold", type=float, default=0.30)
    parser.add_argument("--relation-threshold", type=float, default=0.10)
    parser.add_argument("--max-docs", type=int, default=0, help="0 means full test file")
    parser.add_argument("--graph-output", type=str, default="kg_test_graph.json")
    args = parser.parse_args()

    with open(args.test_file, "r", encoding="utf-8") as f:
        docs = json.load(f)

    if args.max_docs > 0:
        docs = docs[: args.max_docs]

    predictor = REXELTorchPredictor.from_checkpoint_path(
        args.checkpoint,
        span_threshold=args.span_threshold,
        relation_threshold=args.relation_threshold,
    )

    tp = 0
    fp = 0
    fn = 0
    total_pred = 0
    total_gold = 0
    unmatched_pred_triples = 0

    node_counts: Dict[str, int] = defaultdict(int)
    edge_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)

    for idx, doc in enumerate(docs, start=1):
        text = flatten_doc_text(doc)
        pred_result = predictor.extract(text)

        pred_triples = pred_result.get("triples", [])
        entities = pred_result.get("entity_clusters", [])

        for ent in entities:
            canonical = str(ent.get("canonical", "")).strip()
            if canonical:
                node_counts[canonical] += 1

        for tri in pred_triples:
            h = str(tri.get("head", "")).strip()
            r = str(tri.get("relation", "")).strip()
            t = str(tri.get("tail", "")).strip()
            if h and r and t:
                edge_counts[(h, r, t)] += 1

        name_to_ids = build_name_to_entity_ids(doc)
        gold = gold_relation_set(doc)
        pred, unmatched = predicted_relation_set(pred_triples, name_to_ids)

        total_pred += len(pred)
        total_gold += len(gold)
        unmatched_pred_triples += unmatched

        tp_doc = len(pred & gold)
        fp_doc = len(pred - gold)
        fn_doc = len(gold - pred)

        tp += tp_doc
        fp += fp_doc
        fn += fn_doc

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(docs)} docs")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    summary = {
        "docs_evaluated": len(docs),
        "checkpoint": args.checkpoint,
        "span_threshold": args.span_threshold,
        "relation_threshold": args.relation_threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_relations_mapped": total_pred,
        "gold_relations": total_gold,
        "unmatched_pred_triples": unmatched_pred_triples,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "graph_nodes": len(node_counts),
        "graph_edges": len(edge_counts),
    }

    graph_output = {
        "summary": summary,
        "nodes": [{"id": k, "count": v} for k, v in sorted(node_counts.items())],
        "edges": [
            {"head": h, "relation": r, "tail": t, "count": c}
            for (h, r, t), c in sorted(edge_counts.items())
        ],
    }

    with open(args.graph_output, "w", encoding="utf-8") as f:
        json.dump(graph_output, f, indent=2)

    print("\n=== REXEL TEST EVAL SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nGraph output written to: {args.graph_output}")


if __name__ == "__main__":
    main()
