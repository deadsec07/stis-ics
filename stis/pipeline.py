from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DatasetConfig, RuleConfig
from .constraints import ConstraintEngine
from .discretizer import StateDiscretizer
from .evaluation import classification_metrics
from .loader import load_train_test
from .plotting import (
    plot_feature_contributions,
    plot_score_timeline,
    plot_top_constraints,
    plot_transition_heatmap,
)
from .preprocessing import Preprocessor
from .scorer import STISScorer
from .thresholding import ThresholdModel
from .transition_model import TransitionModel


def _labels(frame: pd.DataFrame, config: DatasetConfig) -> pd.Series | None:
    if config.label_column and config.label_column in frame.columns:
        raw = pd.to_numeric(frame[config.label_column], errors="coerce")
        labels = pd.Series(pd.NA, index=frame.index, dtype="Float64")
        valid_mask = ~raw.isna()
        if config.ignore_label_values:
            valid_mask &= ~raw.isin(config.ignore_label_values)
        labels.loc[valid_mask] = raw.loc[valid_mask].isin(config.positive_label_values).astype(int)
        return labels
    return None


def _state_columns(config: DatasetConfig, feature_columns: list[str]) -> list[str]:
    configured = config.discretizer.columns or config.actuator_columns + config.context_columns
    if not configured:
        configured = feature_columns[: min(8, len(feature_columns))]
    return [c for c in configured if c in feature_columns]


def train_stis_pipeline(
    dataset_config: DatasetConfig,
    rules_config: RuleConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw, feature_columns = load_train_test(dataset_config)
    preprocessor = Preprocessor(fill_method=dataset_config.fill_method).fit(train_raw, feature_columns)
    train = preprocessor.transform(train_raw)
    test = preprocessor.transform(test_raw)

    state_columns = _state_columns(dataset_config, feature_columns)
    discretizer = StateDiscretizer(
        strategy=dataset_config.discretizer.strategy,
        bins=dataset_config.discretizer.bins,
        columns=state_columns,
    ).fit(train[state_columns])
    train_states = discretizer.transform_to_state_labels(train[state_columns])
    test_states = discretizer.transform_to_state_labels(test[state_columns])

    transition_model = TransitionModel(epsilon=dataset_config.scoring.epsilon).fit(train_states)
    constraint_engine = ConstraintEngine(rules_config.rules)
    scorer = STISScorer(dataset_config.scoring, transition_model, constraint_engine, feature_columns).fit(train)

    train_artifacts = scorer.fit_component_normalizer(train, train_states)
    threshold_model = ThresholdModel(
        method=dataset_config.threshold.method,
        percentile=dataset_config.threshold.percentile,
        fixed=dataset_config.threshold.fixed,
    ).fit(train_artifacts.scores)
    ablation_thresholds = {
        name: ThresholdModel(
            method=dataset_config.threshold.method,
            percentile=dataset_config.threshold.percentile,
            fixed=dataset_config.threshold.fixed,
        ).fit(scores)
        for name, scores in scorer.ablation_scores(train_artifacts).items()
    }

    with open(outdir / "model.pkl", "wb") as handle:
        pickle.dump(
            {
                "preprocessor": preprocessor,
                "discretizer": discretizer,
                "transition_model": transition_model,
                "threshold_model": threshold_model,
                "scorer": scorer,
                "feature_columns": feature_columns,
                "state_columns": state_columns,
                "dataset_name": dataset_config.name,
                "ablation_thresholds": ablation_thresholds,
            },
            handle,
        )

    summary = {
        "dataset": dataset_config.name,
        "num_train_rows": int(len(train)),
        "num_test_rows": int(len(test)),
        "num_features": int(len(feature_columns)),
        "state_columns": state_columns,
        "threshold": threshold_model.threshold_,
    }
    with open(outdir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def evaluate_stis_pipeline(
    dataset_config: DatasetConfig,
    rules_config: RuleConfig,
    model_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    model_dir = Path(model_dir)
    outdir = Path(output_dir or model_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "model.pkl", "rb") as handle:
        model_bundle = pickle.load(handle)

    _, test_raw, _ = load_train_test(dataset_config)
    preprocessor = model_bundle["preprocessor"]
    discretizer = model_bundle["discretizer"]
    threshold_model = model_bundle["threshold_model"]
    scorer: STISScorer = model_bundle["scorer"]
    state_columns: list[str] = model_bundle["state_columns"]
    ablation_thresholds: dict[str, ThresholdModel] = model_bundle.get("ablation_thresholds", {})

    test = preprocessor.transform(test_raw)
    test_states = discretizer.transform_to_state_labels(test[state_columns])
    artifacts = scorer.score(test, test_states)
    predictions = threshold_model.predict(artifacts.scores)
    labels = _labels(test, dataset_config)

    metrics = classification_metrics(labels, predictions, artifacts.scores)
    ablations = {}
    for name, scores in scorer.ablation_scores(artifacts).items():
        variant_threshold = ablation_thresholds.get(name, threshold_model)
        ablations[name] = classification_metrics(labels, variant_threshold.predict(scores), scores)
        ablations[name]["threshold"] = variant_threshold.threshold_

    components = artifacts.components.copy()
    if dataset_config.timestamp_column in test.columns:
        components[dataset_config.timestamp_column] = test[dataset_config.timestamp_column]
    components["prediction"] = predictions
    if labels is not None:
        components["label"] = labels
    components.to_csv(outdir / "stis_scores.csv", index=False)
    artifacts.value_feature_scores.to_csv(outdir / "feature_contributions.csv", index=False)
    transition_matrix = scorer.transition_model.to_matrix()
    transition_matrix.to_csv(outdir / "transition_matrix.csv")

    report = {
        "dataset": dataset_config.name,
        "threshold": threshold_model.threshold_,
        "metrics": metrics,
        "ablations": ablations,
    }
    with open(outdir / "evaluation_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    timestamps = test[dataset_config.timestamp_column] if dataset_config.timestamp_column in test.columns else pd.Series(range(len(test)))
    plot_score_timeline(timestamps, artifacts.scores, labels, outdir / "score_timeline.png", threshold_model.threshold_)
    plot_transition_heatmap(transition_matrix, outdir / "transition_heatmap.png")
    plot_top_constraints(artifacts.constraint_events, outdir / "top_constraints.png")
    top_index = int(artifacts.scores.idxmax())
    plot_feature_contributions(artifacts.value_feature_scores, top_index, outdir / "feature_contributions.png")
    return report
