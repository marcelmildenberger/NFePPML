
#!/usr/bin/env python3
"""Aggregate all experiment_results/*/metrics.json files into a CSV."""

import argparse
import csv
import json
from pathlib import Path


def serialize(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return value


def collect_rows(results_dir):
    rows, columns = [], set()
    for metrics_path in sorted(results_dir.rglob("metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        row = {
            "experiment": str(metrics_path.parent.relative_to(results_dir)),
            "metrics_path": str(metrics_path.relative_to(results_dir)),
        }
        row.update({k: serialize(v) for k, v in data.items()})
        rows.append(row)
        columns.update(row.keys())

    return rows, columns

def main():
    parser = argparse.ArgumentParser(
        description="Extract every metrics.json in experiment_results into a CSV file."
    )
    parser.add_argument(
        "--results-dir",
        default="experiment_results",
        help="Root directory that contains experiment subfolders (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="aggregated_metrics.csv",
        help="CSV file to create (default: %(default)s)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory '{results_dir}' does not exist.")

    rows, columns = collect_rows(results_dir)
    if not rows:
        raise SystemExit(f"No metrics.json files were found under '{results_dir}'.")

    ordered_columns = ["experiment"] + [
        col for col in sorted(columns) if col != "experiment"
    ]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ordered_columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()