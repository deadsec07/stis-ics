from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stis.config import load_dataset_config, load_rules_config
from stis.pipeline import evaluate_stis_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained STIS model.")
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--constraints-config", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    dataset_config = load_dataset_config(args.dataset_config)
    rules_config = load_rules_config(args.constraints_config)
    report = evaluate_stis_pipeline(dataset_config, rules_config, args.model_dir, args.output_dir)
    print(report)


if __name__ == "__main__":
    main()
