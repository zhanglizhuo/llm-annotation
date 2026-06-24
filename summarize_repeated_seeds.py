from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple


def load_results(results_root: Path) -> Dict[Tuple[str, str, str], List[dict]]:
    grouped: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for result_path in sorted(results_root.glob("seed_*/*_result.json")):
        with open(result_path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        key = (
            record["dataset"],
            record["pseudo_strategy"],
            record["mode"],
        )
        record["result_path"] = str(result_path)
        grouped[key].append(record)
    return grouped


def summarize_group(records: List[dict]) -> dict:
    accuracies = [float(record["best_val_acc"]) for record in records]
    seeds = [int(record.get("seed", -1)) for record in records]
    best_epochs = [int(record["best_epoch"]) for record in records]
    return {
        "n_runs": len(records),
        "seeds": " ".join(str(seed) for seed in sorted(seeds)),
        "mean_val_acc": mean(accuracies),
        "std_val_acc": pstdev(accuracies) if len(accuracies) > 1 else 0.0,
        "min_val_acc": min(accuracies),
        "max_val_acc": max(accuracies),
        "mean_best_epoch": mean(best_epochs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    grouped = load_results(results_root)
    rows = []
    for (dataset, pseudo_strategy, mode), records in sorted(grouped.items()):
        summary = summarize_group(records)
        rows.append(
            {
                "dataset": dataset,
                "pseudo_strategy": pseudo_strategy,
                "mode": mode,
                **summary,
            }
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "pseudo_strategy",
        "mode",
        "n_runs",
        "seeds",
        "mean_val_acc",
        "std_val_acc",
        "min_val_acc",
        "max_val_acc",
        "mean_best_epoch",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} summary rows to {out_path}")


if __name__ == "__main__":
    main()