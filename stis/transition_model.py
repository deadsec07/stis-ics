from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import log

import pandas as pd


@dataclass
class TransitionModel:
    epsilon: float = 1e-6
    transition_counts_: dict[str, Counter] = field(default_factory=dict)
    state_counts_: Counter = field(default_factory=Counter)

    def fit(self, state_labels: pd.Series) -> "TransitionModel":
        transitions: dict[str, Counter] = defaultdict(Counter)
        for current, nxt in zip(state_labels.iloc[:-1], state_labels.iloc[1:]):
            transitions[str(current)][str(nxt)] += 1
            self.state_counts_[str(current)] += 1
        self.transition_counts_ = dict(transitions)
        return self

    def probability(self, current_state: str, next_state: str) -> float:
        current_state = str(current_state)
        next_state = str(next_state)
        numerator = self.transition_counts_.get(current_state, Counter()).get(next_state, 0)
        denominator = self.state_counts_.get(current_state, 0)
        if denominator == 0:
            return self.epsilon
        return max(numerator / denominator, self.epsilon)

    def rarity_scores(self, state_labels: pd.Series) -> pd.Series:
        values = [0.0]
        for current, nxt in zip(state_labels.iloc[:-1], state_labels.iloc[1:]):
            prob = self.probability(str(current), str(nxt))
            values.append(-log(prob + self.epsilon))
        return pd.Series(values, index=state_labels.index, name="transition_rarity")

    def to_matrix(self) -> pd.DataFrame:
        all_states = sorted(set(self.transition_counts_.keys()) | {state for nexts in self.transition_counts_.values() for state in nexts})
        matrix = pd.DataFrame(0.0, index=all_states, columns=all_states)
        for source, counts in self.transition_counts_.items():
            total = max(self.state_counts_.get(source, 0), 1)
            for target, count in counts.items():
                matrix.loc[source, target] = count / total
        return matrix
