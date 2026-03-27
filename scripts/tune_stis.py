from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import pickle
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from stis.config import load_dataset_config
from stis.evaluation import classification_metrics
from stis.loader import load_train_test
from stis.pipeline import _labels
from stis.thresholding import ThresholdModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-search alpha/beta/gamma/threshold on a trained STIS model.")
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.8, 1.0, 1.2])
    parser.add_argument("--betas", nargs="+", type=float, default=[0.8, 1.0, 1.2, 1.5, 2.0])
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5])
    parser.add_argument("--percentiles", nargs="+", type=float, default=[90, 92, 94, 95, 96, 97])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = load_dataset_config(args.dataset_config)
    with open(Path(args.model_dir) / "model.pkl", "rb") as handle:
        bundle = pickle.load(handle)

    train_raw, test_raw, _ = load_train_test(dataset_config)
    preprocessor = bundle["preprocessor"]
    discretizer = bundle["discretizer"]
    scorer = bundle["scorer"]
    state_columns = bundle["state_columns"]

    train = preprocessor.transform(train_raw)
    test = preprocessor.transform(test_raw)
    train_states = discretizer.transform_to_state_labels(train[state_columns])
    test_states = discretizer.transform_to_state_labels(test[state_columns])
    train_artifacts = scorer.score(train, train_states)
    test_artifacts = scorer.score(test, test_states)
    labels = _labels(test, dataset_config)

    rows: list[dict[str, float | int | None]] = []
    for alpha, beta, gamma, percentile in product(args.alphas, args.betas, args.gammas, args.percentiles):
        train_scores = (
            alpha * train_artifacts.components["value_deviation"]
            + beta * train_artifacts.components["transition_rarity"]
            + gamma * train_artifacts.components["constraint_score"]
        )
        test_scores = (
            alpha * test_artifacts.components["value_deviation"]
            + beta * test_artifacts.components["transition_rarity"]
            + gamma * test_artifacts.components["constraint_score"]
        )
        threshold = ThresholdModel(method="percentile", percentile=percentile).fit(train_scores)
        metrics = classification_metrics(labels, threshold.predict(test_scores), test_scores)
        rows.append(
            {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "percentile": percentile,
                "threshold": threshold.threshold_,
                "f1": metrics.get("f1"),
                "recall": metrics.get("recall"),
                "precision": metrics.get("precision"),
                "pr_auc": metrics.get("pr_auc"),
                "detection_delay": metrics.get("detection_delay"),
            }
        )

    table = pd.DataFrame(rows).sort_values(
        ["f1", "recall", "precision", "pr_auc", "detection_delay"],
        ascending=[False, False, False, False, True],
        na_position="last",
    )
    table.to_csv(output_dir / "stis_tuning_results.csv", index=False)
    best = table.iloc[0].to_dict()
    with open(output_dir / "stis_tuning_best.json", "w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2)
    print(best)


if __name__ == "__main__":
    main()
