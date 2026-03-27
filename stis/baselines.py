from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class BaselineModel(Protocol):
    def fit(self, train_features: pd.DataFrame) -> "BaselineModel":
        ...

    def score(self, test_features: pd.DataFrame) -> pd.Series:
        ...


@dataclass
class IsolationForestBaseline:
    random_state: int = 42

    def fit(self, train_features: pd.DataFrame) -> "IsolationForestBaseline":
        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(train_features)
        self.model_ = IsolationForest(random_state=self.random_state, contamination="auto")
        self.model_.fit(X)
        return self

    def score(self, test_features: pd.DataFrame) -> pd.Series:
        X = self.scaler_.transform(test_features)
        return pd.Series(-self.model_.score_samples(X), index=test_features.index, name="isolation_forest")


@dataclass
class OneClassSVMBaseline:
    nu: float = 0.05
    gamma: str = "scale"

    def fit(self, train_features: pd.DataFrame) -> "OneClassSVMBaseline":
        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(train_features)
        self.model_ = OneClassSVM(nu=self.nu, gamma=self.gamma)
        self.model_.fit(X)
        return self

    def score(self, test_features: pd.DataFrame) -> pd.Series:
        X = self.scaler_.transform(test_features)
        return pd.Series(-self.model_.score_samples(X), index=test_features.index, name="one_class_svm")


@dataclass
class LocalOutlierFactorBaseline:
    n_neighbors: int = 20

    def fit(self, train_features: pd.DataFrame) -> "LocalOutlierFactorBaseline":
        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(train_features)
        self.model_ = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True)
        self.model_.fit(X)
        return self

    def score(self, test_features: pd.DataFrame) -> pd.Series:
        X = self.scaler_.transform(test_features)
        return pd.Series(-self.model_.score_samples(X), index=test_features.index, name="lof")


@dataclass
class WindowedAutoencoderBaseline:
    window_size: int = 10
    hidden_layer_sizes: tuple[int, ...] = (64, 16, 64)
    random_state: int = 42

    def fit(self, train_features: pd.DataFrame) -> "WindowedAutoencoderBaseline":
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(train_features)
        windows = self._build_windows(scaled)
        self.model_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            random_state=self.random_state,
            max_iter=200,
        )
        self.model_.fit(windows, windows)
        self.feature_dim_ = scaled.shape[1]
        return self

    def score(self, test_features: pd.DataFrame) -> pd.Series:
        scaled = self.scaler_.transform(test_features)
        windows = self._build_windows(scaled)
        recon = self.model_.predict(windows)
        errors = ((windows - recon) ** 2).mean(axis=1)
        pad = np.repeat(errors[0], self.window_size - 1) if len(errors) else np.array([])
        full = np.concatenate([pad, errors]) if len(errors) else np.zeros(len(test_features))
        return pd.Series(full, index=test_features.index, name="windowed_autoencoder")

    def _build_windows(self, array: np.ndarray) -> np.ndarray:
        if len(array) < self.window_size:
            return np.repeat(array.reshape(1, -1), 1, axis=0)
        return np.stack(
            [array[idx : idx + self.window_size].reshape(-1) for idx in range(len(array) - self.window_size + 1)]
        )


def build_baselines() -> dict[str, BaselineModel]:
    return {
        "isolation_forest": IsolationForestBaseline(),
        "one_class_svm": OneClassSVMBaseline(),
        "lof": LocalOutlierFactorBaseline(),
        "windowed_autoencoder": WindowedAutoencoderBaseline(),
    }
