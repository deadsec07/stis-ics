from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import tempfile

cache_dir = Path(tempfile.gettempdir()) / "stis-matplotlib-cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _attack_windows(labels: pd.Series | None) -> list[tuple[int, int]]:
    if labels is None:
        return []
    labels = labels.fillna(0).astype(int).reset_index(drop=True)
    windows: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(labels):
        if value == 1 and start is None:
            start = idx
        if value == 0 and start is not None:
            windows.append((start, idx - 1))
            start = None
    if start is not None:
        windows.append((start, len(labels) - 1))
    return windows


def plot_score_timeline(
    timestamps: pd.Series,
    scores: pd.Series,
    labels: pd.Series | None,
    output_path: str | Path,
    threshold: float | None = None,
) -> None:
    plt.figure(figsize=(14, 4))
    plt.plot(timestamps, scores, label="STIS score", linewidth=1.2)
    if threshold is not None:
        plt.axhline(threshold, color="red", linestyle="--", label="threshold")
    for start, end in _attack_windows(labels):
        plt.axvspan(timestamps.iloc[start], timestamps.iloc[end], color="orange", alpha=0.2)
    plt.title("Anomaly Score Over Time")
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_transition_heatmap(matrix: pd.DataFrame, output_path: str | Path, max_states: int = 25) -> None:
    matrix = matrix.iloc[:max_states, :max_states]
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix.values, aspect="auto", cmap="viridis")
    plt.colorbar(label="Transition probability")
    plt.title("Transition Rarity Model")
    plt.xlabel("Next state")
    plt.ylabel("Current state")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_top_constraints(constraint_events: list[list], output_path: str | Path, top_n: int = 10) -> None:
    counts = Counter()
    for row_events in constraint_events:
        for event in row_events:
            counts[event.rule_name] += 1
    names, values = zip(*counts.most_common(top_n)) if counts else ([], [])
    plt.figure(figsize=(10, 4))
    plt.bar(names, values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Top Violated Constraints")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_feature_contributions(feature_scores: pd.DataFrame, index: int, output_path: str | Path, top_n: int = 15) -> None:
    row = feature_scores.iloc[index].sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10, 4))
    plt.bar(row.index.astype(str), row.values)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Feature Contribution Breakdown @ index {index}")
    plt.ylabel("Absolute normalized deviation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
