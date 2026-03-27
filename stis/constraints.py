from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ConstraintEvent:
    rule_name: str
    score: float
    message: str


class ConstraintEngine:
    def __init__(self, rules: list[dict[str, Any]] | None = None) -> None:
        self.rules = rules or []

    def score(self, frame: pd.DataFrame) -> tuple[pd.Series, list[list[ConstraintEvent]]]:
        total_scores = []
        all_events: list[list[ConstraintEvent]] = []
        for idx in range(len(frame)):
            row_events: list[ConstraintEvent] = []
            for rule in self.rules:
                event = self._evaluate_rule(frame, idx, rule)
                if event is not None:
                    row_events.append(event)
            total_scores.append(sum(event.score for event in row_events))
            all_events.append(row_events)
        return pd.Series(total_scores, index=frame.index, name="constraint_score"), all_events

    def _evaluate_rule(
        self,
        frame: pd.DataFrame,
        index: int,
        rule: dict[str, Any],
    ) -> ConstraintEvent | None:
        rule_type = rule.get("type")
        weight = float(rule.get("weight", 1.0))
        name = rule.get("name", rule_type or "rule")

        if rule_type == "expected_response":
            return self._expected_response(frame, index, rule, name, weight)
        if rule_type == "unexpected_rise":
            return self._unexpected_rise(frame, index, rule, name, weight)
        if rule_type == "balance_consistency":
            return self._balance_consistency(frame, index, rule, name, weight)
        return None

    def _expected_response(self, frame: pd.DataFrame, index: int, rule: dict[str, Any], name: str, weight: float) -> ConstraintEvent | None:
        trigger = rule["trigger_column"]
        response = rule["response_column"]
        lag = int(rule.get("lag", 1))
        min_delta = float(rule.get("min_delta", 0.0))
        if index == 0 or index + lag >= len(frame):
            return None
        prev_trigger = frame.iloc[index - 1][trigger]
        curr_trigger = frame.iloc[index][trigger]
        if curr_trigger > prev_trigger:
            start = frame.iloc[index][response]
            end = frame.iloc[index + lag][response]
            if end - start < min_delta:
                return ConstraintEvent(
                    rule_name=name,
                    score=weight,
                    message=f"{trigger} increased but {response} failed to rise by {min_delta} within {lag} steps",
                )
        return None

    def _unexpected_rise(self, frame: pd.DataFrame, index: int, rule: dict[str, Any], name: str, weight: float) -> ConstraintEvent | None:
        control = rule["control_column"]
        observed = rule["observed_column"]
        control_off_value = rule.get("control_off_value", 0)
        max_delta = float(rule.get("max_delta", 0.0))
        lag = int(rule.get("lag", 1))
        if index == 0 or index + lag >= len(frame):
            return None
        if frame.iloc[index][control] == control_off_value:
            start = frame.iloc[index][observed]
            end = frame.iloc[index + lag][observed]
            if end - start > max_delta:
                return ConstraintEvent(
                    rule_name=name,
                    score=weight,
                    message=f"{observed} rose unexpectedly while {control} was off",
                )
        return None

    def _balance_consistency(self, frame: pd.DataFrame, index: int, rule: dict[str, Any], name: str, weight: float) -> ConstraintEvent | None:
        level = rule["level_column"]
        inlet = rule["inlet_column"]
        outlet = rule["outlet_column"]
        inlet_on_value = rule.get("inlet_on_value", 1)
        outlet_off_value = rule.get("outlet_off_value", 0)
        tolerance = float(rule.get("tolerance", 0.0))
        lag = int(rule.get("lag", 1))
        if index == 0 or index + lag >= len(frame):
            return None
        if frame.iloc[index][inlet] == inlet_on_value and frame.iloc[index][outlet] == outlet_off_value:
            start = frame.iloc[index][level]
            end = frame.iloc[index + lag][level]
            if end < start - tolerance:
                return ConstraintEvent(
                    rule_name=name,
                    score=weight,
                    message=f"{level} dropped despite active inlet and inactive outlet",
                )
        return None
