from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from stis.baselines import build_baselines
from stis.config import load_dataset_config
from stis.evaluation import classification_metrics
from stis.loader import load_train_test
from stis.preprocessing import Preprocessor
from stis.thresholding import ThresholdModel
from stis.utils import write_json


def _labels(frame: pd.DataFrame, label_column: str | None, positive_values: list[int], ignore_values: list[int]) -> pd.Series | None:
    if not label_column or label_column not in frame.columns:
        return None
    raw = pd.to_numeric(frame[label_column], errors="coerce")
    labels = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    valid_mask = ~raw.isna()
    if ignore_values:
        valid_mask &= ~raw.isin(ignore_values)
    labels.loc[valid_mask] = raw.loc[valid_mask].isin(positive_values).astype(int)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline anomaly detectors.")
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dataset_config = load_dataset_config(args.dataset_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw, features = load_train_test(dataset_config)
    preprocessor = Preprocessor(fill_method=dataset_config.fill_method).fit(train_raw, features)
    train = preprocessor.transform(train_raw)
    test = preprocessor.transform(test_raw)
    labels = _labels(
        test,
        dataset_config.label_column,
        dataset_config.positive_label_values,
        dataset_config.ignore_label_values,
    )

    reports = {}
    score_table = pd.DataFrame(index=test.index)
    for name, baseline in build_baselines().items():
        baseline.fit(train[features])
        train_scores = baseline.score(train[features])
        test_scores = baseline.score(test[features])
        threshold = ThresholdModel(
            method=dataset_config.threshold.method,
            percentile=dataset_config.threshold.percentile,
            fixed=dataset_config.threshold.fixed,
        ).fit(train_scores)
        predictions = threshold.predict(test_scores)
        reports[name] = classification_metrics(labels, predictions, test_scores)
        reports[name]["threshold"] = threshold.threshold_
        score_table[name] = test_scores

    score_table.to_csv(output_dir / "baseline_scores.csv", index=False)
    write_json(output_dir / "baseline_report.json", reports)
    print(reports)


if __name__ == "__main__":
    main()
