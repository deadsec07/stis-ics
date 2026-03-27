from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import ScoringConfig
from .constraints import ConstraintEngine
from .preprocessing import normalized_deviation, robust_feature_stats, zscore_feature_stats
from .transition_model import TransitionModel


@dataclass
class STISArtifacts:
    scores: pd.Series
    components: pd.DataFrame
    value_feature_scores: pd.DataFrame
    constraint_events: list[list[Any]]
    state_labels: pd.Series


class STISScorer:
    def __init__(
        self,
        scoring_config: ScoringConfig,
        transition_model: TransitionModel,
        constraint_engine: ConstraintEngine,
        feature_columns: list[str],
    ) -> None:
        self.config = scoring_config
        self.transition_model = transition_model
        self.constraint_engine = constraint_engine
        self.feature_columns = feature_columns

    def fit(self, normal_train_frame: pd.DataFrame) -> "STISScorer":
        if self.config.deviation_method == "zscore":
            self.reference_center_, self.reference_scale_ = zscore_feature_stats(normal_train_frame, self.feature_columns)
        else:
            self.reference_center_, self.reference_scale_ = robust_feature_stats(normal_train_frame, self.feature_columns)
        return self

    def fit_component_normalizer(self, frame: pd.DataFrame, state_labels: pd.Series) -> STISArtifacts:
        raw = self._raw_component_scores(frame, state_labels)
        components = pd.DataFrame(
            {
                "value_deviation": raw["value_scores"],
                "transition_rarity": raw["rarity"],
                "constraint_score": raw["constraint_score"],
            },
            index=frame.index,
        )
        self.component_center_ = components.median()
        self.component_scale_ = (components.quantile(0.95) - self.component_center_).replace(0.0, 1.0).fillna(1.0)
        self.component_scale_ = self.component_scale_.clip(lower=1.0)
        return self._build_artifacts(frame, state_labels, raw)

    def score(self, frame: pd.DataFrame, state_labels: pd.Series) -> STISArtifacts:
        raw = self._raw_component_scores(frame, state_labels)
        return self._build_artifacts(frame, state_labels, raw)

    def _raw_component_scores(self, frame: pd.DataFrame, state_labels: pd.Series) -> dict[str, Any]:
        value_scores, feature_scores = normalized_deviation(
            frame,
            self.reference_center_,
            self.reference_scale_,
            self.feature_columns,
        )
        rarity = self.transition_model.rarity_scores(state_labels)
        constraint_score, events = self.constraint_engine.score(frame)
        return {
            "value_scores": value_scores,
            "feature_scores": feature_scores,
            "rarity": rarity,
            "constraint_score": constraint_score,
            "events": events,
        }

    def _normalize_component(self, name: str, values: pd.Series) -> pd.Series:
        center = getattr(self, "component_center_", pd.Series(dtype=float)).get(name, 0.0)
        scale = getattr(self, "component_scale_", pd.Series(dtype=float)).get(name, 1.0)
        scale = scale if scale and scale > 0 else 1.0
        return ((values - center) / scale).clip(lower=0.0)

    def _build_artifacts(self, frame: pd.DataFrame, state_labels: pd.Series, raw: dict[str, Any]) -> STISArtifacts:
        value_component = self._normalize_component("value_deviation", raw["value_scores"])
        rarity_component = self._normalize_component("transition_rarity", raw["rarity"])
        constraint_component = self._normalize_component("constraint_score", raw["constraint_score"])
        total = (
            self.config.alpha * value_component
            + self.config.beta * rarity_component
            + self.config.gamma * constraint_component
        )
        components = pd.DataFrame(
            {
                "value_deviation": value_component,
                "transition_rarity": rarity_component,
                "constraint_score": constraint_component,
                "raw_value_deviation": raw["value_scores"],
                "raw_transition_rarity": raw["rarity"],
                "raw_constraint_score": raw["constraint_score"],
                "stis_score": total,
            },
            index=frame.index,
        )
        return STISArtifacts(
            scores=total.rename("stis_score"),
            components=components,
            value_feature_scores=raw["feature_scores"],
            constraint_events=raw["events"],
            state_labels=state_labels.rename("state_label"),
        )

    def ablation_scores(self, artifacts: STISArtifacts) -> dict[str, pd.Series]:
        components = artifacts.components
        return {
            "value_only": components["value_deviation"],
            "value_transition": components["value_deviation"] + components["transition_rarity"],
            "full": components["stis_score"],
        }
