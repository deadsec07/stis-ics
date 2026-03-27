from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Preprocessor:
    fill_method: str = "ffill_bfill"

    def fit(self, frame: pd.DataFrame, feature_columns: list[str]) -> "Preprocessor":
        numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
        self.feature_columns_ = feature_columns
        self.medians_ = numeric.median(numeric_only=True).to_dict()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = frame.copy()
        transformed[self.feature_columns_] = transformed[self.feature_columns_].apply(
            pd.to_numeric, errors="coerce"
        )
        if self.fill_method == "ffill_bfill":
            transformed[self.feature_columns_] = transformed[self.feature_columns_].ffill().bfill()
        for column, value in self.medians_.items():
            transformed[column] = transformed[column].fillna(value)
        return transformed


def robust_feature_stats(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.Series, pd.Series]:
    median = frame[feature_columns].median()
    mad = (frame[feature_columns] - median).abs().median()
    mad = mad.replace(0.0, 1e-6)
    return median, mad


def zscore_feature_stats(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.Series, pd.Series]:
    mean = frame[feature_columns].mean()
    std = frame[feature_columns].std().replace(0.0, 1e-6)
    return mean, std


def normalized_deviation(
    frame: pd.DataFrame,
    reference_center: pd.Series,
    reference_scale: pd.Series,
    feature_columns: list[str],
) -> tuple[pd.Series, pd.DataFrame]:
    scores = (frame[feature_columns] - reference_center) / reference_scale
    scores = scores.replace([np.inf, -np.inf], 0.0).fillna(0.0).abs()
    aggregate = scores.mean(axis=1)
    return aggregate, scores
