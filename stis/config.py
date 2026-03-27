from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ThresholdConfig:
    method: str = "percentile"
    percentile: float = 99.0
    fixed: float | None = None


@dataclass
class ScoringConfig:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    epsilon: float = 1e-6
    deviation_method: str = "mad"
    rolling_window: int = 50


@dataclass
class DiscretizerConfig:
    strategy: str = "quantile"
    bins: int = 4
    columns: list[str] = field(default_factory=list)
    include_context: bool = True


@dataclass
class DatasetConfig:
    name: str
    train_path: str
    test_path: str
    timestamp_column: str
    label_column: str | None = None
    positive_label_values: list[int] = field(default_factory=lambda: [1])
    ignore_label_values: list[int] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    sensor_columns: list[str] = field(default_factory=list)
    actuator_columns: list[str] = field(default_factory=list)
    context_columns: list[str] = field(default_factory=list)
    drop_columns: list[str] = field(default_factory=list)
    timestamp_format: str | None = None
    sort_by_time: bool = True
    fill_method: str = "ffill_bfill"
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    discretizer: DiscretizerConfig = field(default_factory=DiscretizerConfig)
    attack_intervals_path: str | None = None


@dataclass
class RuleConfig:
    rules: list[dict[str, Any]] = field(default_factory=list)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_dataset_config(path: str | Path) -> DatasetConfig:
    raw = _read_yaml(path)
    threshold = ThresholdConfig(**raw.get("threshold", {}))
    scoring = ScoringConfig(**raw.get("scoring", {}))
    discretizer = DiscretizerConfig(**raw.get("discretizer", {}))
    return DatasetConfig(
        name=raw["name"],
        train_path=raw["train_path"],
        test_path=raw["test_path"],
        timestamp_column=raw["timestamp_column"],
        label_column=raw.get("label_column"),
        positive_label_values=raw.get("positive_label_values", [1]),
        ignore_label_values=raw.get("ignore_label_values", []),
        feature_columns=raw.get("feature_columns", []),
        sensor_columns=raw.get("sensor_columns", []),
        actuator_columns=raw.get("actuator_columns", []),
        context_columns=raw.get("context_columns", []),
        drop_columns=raw.get("drop_columns", []),
        timestamp_format=raw.get("timestamp_format"),
        sort_by_time=raw.get("sort_by_time", True),
        fill_method=raw.get("fill_method", "ffill_bfill"),
        threshold=threshold,
        scoring=scoring,
        discretizer=discretizer,
        attack_intervals_path=raw.get("attack_intervals_path"),
    )


def load_rules_config(path: str | Path) -> RuleConfig:
    raw = _read_yaml(path)
    return RuleConfig(rules=raw.get("rules", []))
