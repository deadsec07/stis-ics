from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


def _flatten_metrics(name: str, payload: dict) -> dict[str, object]:
    metrics = payload.get("metrics", payload)
    return {
        "model": name,
        "num_samples": metrics.get("num_samples"),
        "num_labeled_samples": metrics.get("num_labeled_samples"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "pr_auc": metrics.get("pr_auc"),
        "roc_auc": metrics.get("roc_auc"),
        "detection_delay": metrics.get("detection_delay"),
    }


def _format_value(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _to_markdown_table(table: pd.DataFrame) -> str:
    headers = list(table.columns)
    rows = [[_format_value(row.get(header)) for header in headers] for row in table.to_dict(orient="records")]
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *data_rows])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a compact benchmark summary table for STIS and baselines.")
    parser.add_argument("--stis-report", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.stis_report, "r", encoding="utf-8") as handle:
        stis_report = json.load(handle)
    with open(args.baseline_report, "r", encoding="utf-8") as handle:
        baseline_report = json.load(handle)

    rows = [_flatten_metrics("stis", stis_report)]
    for name, payload in baseline_report.items():
        rows.append(_flatten_metrics(name, payload))

    table = pd.DataFrame(rows).sort_values(["f1", "recall", "pr_auc"], ascending=False, na_position="last")
    csv_path = output_dir / "benchmark_summary.csv"
    md_path = output_dir / "benchmark_summary.md"
    table.to_csv(csv_path, index=False)

    markdown_table = _to_markdown_table(table)
    md_path.write_text(
        "# Benchmark Summary\n\n"
        "This table is generated from the current STIS evaluation report and baseline benchmark output.\n\n"
        f"{markdown_table}\n",
        encoding="utf-8",
    )
    print({"csv": str(csv_path), "markdown": str(md_path)})


if __name__ == "__main__":
    main()
