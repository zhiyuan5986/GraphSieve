"""Evaluate path-selector beam-search retrieval results.

The retrieved triple set for a sample is the union of all triples appearing in
its decoded paths. Metrics are therefore reported once for the beam-search
result, with no top-K or threshold configuration.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def resolve_gpt_triple_path(dataset, data_dir):
    candidates = [
        Path(data_dir) / "data_files" / dataset / "gpt_triples.pth",
        Path("data_files") / dataset / "gpt_triples.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Cannot find gpt_triples.pth. Tried: " + ", ".join(str(p) for p in candidates))


def path_triple_set(pred):
    triples = set()
    for path in pred.get("paths", []):
        for step in path.get("path_triples", []):
            triple = step.get("triple") if isinstance(step, dict) else step
            if triple is not None:
                triples.add(tuple(triple))
    return triples


def save_triple_count_summary(triple_counts, result_path):
    counts = np.asarray(triple_counts, dtype=np.float64)
    if counts.size == 0:
        return
    summary_path = Path(result_path).resolve().parent / "path_retrieved_triple_count_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["num_questions", "mean", "std", "median", "min", "max", "p05", "p95"])
        writer.writeheader()
        writer.writerow({
            "num_questions": int(counts.size),
            "mean": float(counts.mean()),
            "std": float(counts.std()),
            "median": float(np.median(counts)),
            "min": float(counts.min()),
            "max": float(counts.max()),
            "p05": float(np.percentile(counts, 5)),
            "p95": float(np.percentile(counts, 95)),
        })
    print(f"Saved triple-count summary to {summary_path}")


def print_path_length_distribution(path_lengths):
    """Print the length distribution of every decoded path."""
    if not path_lengths:
        print("Path length distribution: no retrieved paths found.")
        return

    counts = Counter(path_lengths)
    total_paths = len(path_lengths)
    rows = [
        {
            "path_length": length,
            "num_paths": count,
            "percentage": round(count / total_paths * 100, 2),
        }
        for length, count in sorted(counts.items())
    ]
    print("\nPath length distribution (all retrieved paths):")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Total retrieved paths: {total_paths}")


def main(args):
    pred_dict = torch.load(args.path, map_location="cpu", weights_only=False)
    gpt_triple_dict = torch.load(resolve_gpt_triple_path(args.dataset, args.data_dir), map_location="cpu", weights_only=False)

    metrics = {"ans_recall": [], "shortest_path_triple_recall": [], "gpt_triple_recall": []}
    triple_counts = []
    path_lengths = []
    for sample_id, pred in pred_dict.items():
        path_lengths.extend(len(path.get("path_triples", [])) for path in pred.get("paths", []))
        triples = path_triple_set(pred)
        triple_counts.append(len(triples))
        retrieved_entities = {entity for h, _, t in triples for entity in (h, t)}

        answers = set(pred.get("a_entity_in_graph", []))
        if answers:
            metrics["ans_recall"].append(len(answers & retrieved_entities) / len(answers))

        shortest = set(tuple(x) for x in pred.get("target_relevant_triples", []))
        if shortest:
            metrics["shortest_path_triple_recall"].append(len(shortest & triples) / len(shortest))

        gpt_triples = set(tuple(x) for x in gpt_triple_dict.get(sample_id, []))
        if gpt_triples:
            metrics["gpt_triple_recall"].append(len(gpt_triples & triples) / len(gpt_triples))

    row = {
        metric: round(float(np.mean(values)), 3) if values else 0.0
        for metric, values in metrics.items()
    }
    row["avg_triples"] = round(float(np.mean(triple_counts)), 3) if triple_counts else 0.0
    row["num_questions"] = len(triple_counts)
    print(pd.DataFrame([row]).to_string(index=False))
    print_path_length_distribution(path_lengths)
    save_triple_count_summary(triple_counts, args.path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate path-selector beam-search retrieval results")
    parser.add_argument("-d", "--dataset", required=True, choices=["webqsp", "cwq"], help="Dataset name")
    parser.add_argument("-p", "--path", required=True, help="Path to path-selector retrieval result")
    parser.add_argument("--data_dir", default="retrieve")
    main(parser.parse_args())
