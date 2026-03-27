from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DatasetConfig


def load_csv_dataset(path: str | Path, config: DatasetConfig) -> pd.DataFrame:
    frame = pd.read_csv(path, skipinitialspace=True)
    frame.columns = [str(column).strip() for column in frame.columns]
    if config.drop_columns:
        frame = frame.drop(columns=[c for c in config.drop_columns if c in frame.columns], errors="ignore")
    if config.timestamp_column in frame.columns:
        frame[config.timestamp_column] = pd.to_datetime(
            frame[config.timestamp_column],
            format=config.timestamp_format,
            errors="coerce",
        )
    if config.sort_by_time and config.timestamp_column in frame.columns:
        frame = frame.sort_values(config.timestamp_column).reset_index(drop=True)
    return frame


def resolve_feature_columns(frame: pd.DataFrame, config: DatasetConfig) -> list[str]:
    if config.feature_columns:
        return [c for c in config.feature_columns if c in frame.columns]
    excluded = {config.timestamp_column, config.label_column, *config.drop_columns}
    return [c for c in frame.columns if c not in excluded]


def load_train_test(config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = load_csv_dataset(config.train_path, config)
    test = load_csv_dataset(config.test_path, config)
    features = resolve_feature_columns(train, config)
    return train, test, features
