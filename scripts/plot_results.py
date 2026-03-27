from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from stis.config import load_dataset_config
from stis.plotting import plot_score_timeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate core STIS plots from saved outputs.")
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dataset_config = load_dataset_config(args.dataset_config)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(model_dir / "stis_scores.csv")
    if dataset_config.timestamp_column in scores.columns:
        timestamps = pd.to_datetime(scores[dataset_config.timestamp_column], errors="coerce")
    else:
        timestamps = pd.Series(range(len(scores)))
    labels = scores["label"] if "label" in scores.columns else None
    threshold = None
    plot_score_timeline(
        timestamps=timestamps,
        scores=scores["stis_score"],
        labels=labels,
        output_path=output_dir / "score_timeline_replot.png",
        threshold=threshold,
    )
    print({"output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
