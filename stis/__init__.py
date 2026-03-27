"""STIS package."""

from .config import load_dataset_config, load_rules_config
from .pipeline import train_stis_pipeline, evaluate_stis_pipeline

__all__ = [
    "load_dataset_config",
    "load_rules_config",
    "train_stis_pipeline",
    "evaluate_stis_pipeline",
]
