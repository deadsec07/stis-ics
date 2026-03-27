from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ThresholdModel:
    method: str = "percentile"
    percentile: float = 99.0
    fixed: float | None = None
    threshold_: float | None = None

    def fit(self, scores: pd.Series) -> "ThresholdModel":
        if self.method == "fixed" and self.fixed is not None:
            self.threshold_ = float(self.fixed)
        else:
            self.threshold_ = float(np.percentile(scores, self.percentile))
        return self

    def predict(self, scores: pd.Series) -> pd.Series:
        if self.threshold_ is None:
            raise ValueError("ThresholdModel must be fit before predict")
        return (scores >= self.threshold_).astype(int)
