from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from stis.config import load_dataset_config, load_rules_config
from stis.pipeline import evaluate_stis_pipeline, train_stis_pipeline


def build_synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    n_train = 400
    n_test = 300

    def _frame(n: int) -> pd.DataFrame:
        timestamp = pd.date_range("2024-01-01", periods=n, freq="min")
        pump = (np.sin(np.arange(n) / 20) > 0).astype(int)
        valve = (np.sin(np.arange(n) / 30) > 0.2).astype(int)
        flow = pump * 5 + valve * 2 + rng.normal(0, 0.2, n)
        pressure = pump * 3 + rng.normal(0, 0.15, n)
        tank_level = 40 + np.cumsum((valve * 0.1) - 0.05 + rng.normal(0, 0.02, n))
        return pd.DataFrame(
            {
                "timestamp": timestamp,
                "P101": pump,
                "MV101": valve,
                "FIT101": flow,
                "PIT101": pressure,
                "LIT101": tank_level,
                "mode": np.where(valve == 1, 1, 0),
                "label": np.zeros(n, dtype=int),
            }
        )

    train = _frame(n_train)
    test = _frame(n_test)
    anomaly_slice = slice(170, 210)
    test.loc[anomaly_slice, "P101"] = 0
    test.loc[anomaly_slice, "PIT101"] += np.linspace(0.0, 3.0, len(test.loc[anomaly_slice]))
    test.loc[anomaly_slice, "MV101"] = 1
    test.loc[anomaly_slice, "FIT101"] -= 2.5
    test.loc[anomaly_slice, "LIT101"] -= np.linspace(0.0, 4.0, len(test.loc[anomaly_slice]))
    test.loc[anomaly_slice, "label"] = 1
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic STIS demo with generated plots.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, test = build_synthetic_frames()
    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    dataset_config_payload = {
        "name": "synthetic_demo",
        "train_path": str(output_dir / "train.csv"),
        "test_path": str(output_dir / "test.csv"),
        "timestamp_column": "timestamp",
        "label_column": "label",
        "sensor_columns": ["FIT101", "PIT101", "LIT101"],
        "actuator_columns": ["P101", "MV101"],
        "context_columns": ["mode"],
        "feature_columns": ["P101", "MV101", "FIT101", "PIT101", "LIT101", "mode"],
        "threshold": {"method": "percentile", "percentile": 99.0},
        "scoring": {"alpha": 1.0, "beta": 1.0, "gamma": 1.5, "deviation_method": "mad"},
        "discretizer": {"strategy": "quantile", "bins": 3, "columns": ["P101", "MV101", "mode", "FIT101"]},
    }
    rules_config_payload = {
        "rules": [
            {
                "name": "valve_open_without_flow_response",
                "type": "expected_response",
                "trigger_column": "MV101",
                "response_column": "FIT101",
                "lag": 2,
                "min_delta": 0.5,
                "weight": 1.5,
            },
            {
                "name": "pressure_rises_with_pump_off",
                "type": "unexpected_rise",
                "control_column": "P101",
                "observed_column": "PIT101",
                "control_off_value": 0,
                "lag": 2,
                "max_delta": 0.3,
                "weight": 2.0,
            },
            {
                "name": "level_drops_despite_inlet",
                "type": "balance_consistency",
                "level_column": "LIT101",
                "inlet_column": "MV101",
                "outlet_column": "P101",
                "inlet_on_value": 1,
                "outlet_off_value": 0,
                "lag": 2,
                "tolerance": 0.2,
                "weight": 1.5,
            },
        ]
    }

    dataset_yaml = output_dir / "synthetic_dataset.yaml"
    rules_yaml = output_dir / "synthetic_rules.yaml"
    with open(dataset_yaml, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_config_payload, handle, sort_keys=False)
    with open(rules_yaml, "w", encoding="utf-8") as handle:
        yaml.safe_dump(rules_config_payload, handle, sort_keys=False)

    train_summary = train_stis_pipeline(load_dataset_config(dataset_yaml), load_rules_config(rules_yaml), output_dir)
    report = evaluate_stis_pipeline(load_dataset_config(dataset_yaml), load_rules_config(rules_yaml), output_dir, output_dir)
    print({"training": train_summary, "evaluation": report})


if __name__ == "__main__":
    main()
