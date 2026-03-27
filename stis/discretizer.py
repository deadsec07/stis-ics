from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class StateDiscretizer:
    strategy: str = "quantile"
    bins: int = 4
    columns: list[str] | None = None

    def fit(self, frame: pd.DataFrame) -> "StateDiscretizer":
        columns = self.columns or list(frame.columns)
        self.columns_ = [c for c in columns if c in frame.columns]
        self.bin_edges_: dict[str, list[float]] = {}
        for column in self.columns_:
            series = frame[column]
            if self.strategy == "uniform":
                _, edges = pd.cut(series, bins=self.bins, labels=False, retbins=True, duplicates="drop")
            else:
                _, edges = pd.qcut(series, q=self.bins, labels=False, retbins=True, duplicates="drop")
            self.bin_edges_[column] = list(edges)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        discrete = pd.DataFrame(index=frame.index)
        for column in self.columns_:
            edges = self.bin_edges_[column]
            if len(edges) <= 1:
                discrete[column] = "bin_0"
                continue
            bins = pd.cut(frame[column], bins=edges, labels=False, include_lowest=True, duplicates="drop")
            discrete[column] = bins.fillna(0).astype(int).map(lambda idx: f"bin_{idx}")
        return discrete

    def transform_to_state_labels(self, frame: pd.DataFrame) -> pd.Series:
        discrete = self.transform(frame)
        return discrete.astype(str).agg("|".join, axis=1)
