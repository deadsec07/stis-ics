# STIS: State Transition Integrity Scoring for ICS Anomaly Detection

STIS is an interpretable anomaly-detection framework for industrial control systems (ICS) that scores **state transitions**, not just isolated point values. The method combines continuous sensor deviation, rarity of discrete state changes, and configurable physical/control consistency rules.

## Algorithm

For timestep `t`, define system state:

`S(t) = {sensor values, actuator states, context}`

Then score the transition `S(t) -> S(t+1)` as:

```text
STIS(t) = alpha * ValueDeviation(t)
        + beta  * TransitionRarity(t)
        + gamma * ConstraintViolationScore(t)
```

Where:

- `ValueDeviation`: normalized deviation from recent normal behavior using rolling z-score or robust MAD statistics.
- `TransitionRarity`: `-log(P(S(t+1) | S(t)) + epsilon)` under a first-order Markov transition model built from normal training data.
- `ConstraintViolationScore`: penalty from a configurable rule engine that encodes simple physical/control consistency checks.

## Why STIS

Standard anomaly detectors often flag unusual values but miss semantically wrong transitions, such as:

- a valve opening without the expected downstream flow increase
- a pump being off while pressure rises
- a tank level dropping despite active inflow and inactive outflow

STIS is designed to capture these integrity failures while remaining interpretable and dataset-portable through configuration files.

## Project Layout

```text
stis-ics/
├── configs/
│   ├── constraints/
│   └── datasets/
├── data/
│   ├── processed/
│   └── raw/
├── experiments/
├── notebooks/
├── results/
│   └── plots/
├── scripts/
└── stis/
```

## Dataset Format

The project expects CSV time-series files with:

- one timestamp column
- multivariate feature columns
- optional label column for point-wise attack/anomaly labels

Train data may be normal-only. Test data can contain attacks/anomalies.

Dataset-specific mapping is provided through YAML files in [`configs/datasets/sw_at.yaml`](/Users/alihasnat/Private/dev/stis-ics/configs/datasets/sw_at.yaml), [`configs/datasets/wadi.yaml`](/Users/alihasnat/Private/dev/stis-ics/configs/datasets/wadi.yaml), and [`configs/datasets/tep.yaml`](/Users/alihasnat/Private/dev/stis-ics/configs/datasets/tep.yaml).

A public BATADAL configuration is also included for users who cannot access SWaT/WaDi:

- [`configs/datasets/batadal.yaml`](/Users/alihasnat/Private/dev/stis-ics/configs/datasets/batadal.yaml)
- [`configs/datasets/batadal_unlabeled_test.yaml`](/Users/alihasnat/Private/dev/stis-ics/configs/datasets/batadal_unlabeled_test.yaml)
- [`configs/constraints/batadal_rules.yaml`](/Users/alihasnat/Private/dev/stis-ics/configs/constraints/batadal_rules.yaml)

## Assumptions

- timestamps are ordered or sortable
- actuator/context variables can be discretized
- train split can be treated as mostly or fully normal
- missing values are handled with simple forward/backward fill plus median fallback
- physical/control rules are approximate and configured externally

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How To Run

Train STIS:

```bash
python scripts/train_stis.py \
  --dataset-config configs/datasets/sw_at.yaml \
  --constraints-config configs/constraints/sw_at_rules.yaml \
  --output-dir results/sw_at
```

Evaluate STIS:

```bash
python scripts/eval_stis.py \
  --dataset-config configs/datasets/sw_at.yaml \
  --constraints-config configs/constraints/sw_at_rules.yaml \
  --model-dir results/sw_at
```

Run baselines:

```bash
python scripts/run_baselines.py \
  --dataset-config configs/datasets/sw_at.yaml \
  --output-dir results/sw_at
```

Plot outputs:

```bash
python scripts/plot_results.py \
  --dataset-config configs/datasets/sw_at.yaml \
  --model-dir results/sw_at \
  --output-dir results/plots/sw_at
```

Run a fully synthetic end-to-end demo:

```bash
python experiments/demo_synthetic.py --output-dir results/demo
```

Run BATADAL with public files:

```bash
python scripts/train_stis.py \
  --dataset-config configs/datasets/batadal.yaml \
  --constraints-config configs/constraints/batadal_rules.yaml \
  --output-dir results/batadal
```

```bash
python scripts/eval_stis.py \
  --dataset-config configs/datasets/batadal.yaml \
  --constraints-config configs/constraints/batadal_rules.yaml \
  --model-dir results/batadal
```

Run unlabeled BATADAL scoring on the public holdout file:

```bash
python scripts/eval_stis.py \
  --dataset-config configs/datasets/batadal_unlabeled_test.yaml \
  --constraints-config configs/constraints/batadal_rules.yaml \
  --model-dir results/batadal \
  --output-dir results/batadal_unlabeled
```

Generate a compact benchmark table:

```bash
python scripts/summarize_benchmarks.py \
  --stis-report results/batadal/evaluation_report.json \
  --baseline-report results/batadal/baselines/baseline_report.json \
  --output-dir results/batadal
```

## Evaluation

The framework computes:

- point-wise precision, recall, F1
- ROC-AUC when both classes exist
- PR-AUC
- confusion matrix
- mean detection delay over contiguous anomaly windows
- ablation variants:
  - `value_only`
  - `value_transition`
  - `full`

## Visualizations

Generated plots include:

- anomaly score over time with attack windows
- transition rarity heatmap
- top violated constraints
- feature contribution breakdown for a selected alert

## Benchmark Notes

- Isolation Forest, One-Class SVM, and Local Outlier Factor are included.
- An optional sequence autoencoder baseline is provided via `MLPRegressor` as a lightweight placeholder. Replace it with a true LSTM autoencoder if your environment includes TensorFlow or PyTorch.
- The code keeps feature handling, thresholds, and rules config-driven so SWaT, WADI, and TEP mappings can be swapped with minimal code changes.
- BATADAL `dataset04` contains attack labels for only a subset of rows. In this repo, `ATT_FLAG = 1` is treated as attack and `ATT_FLAG = -999` is treated as unlabeled/ignored during metric computation.

## Dataset-Specific TODOs

- confirm exact SWaT timestamp, label, and actuator column names
- confirm WADI flow/pressure tag names and attack label encoding
- map TEP benchmark files into the CSV schema or add a loader adapter
- refine physical lag tolerances and tolerable deltas per dataset
- choose a final subset of variables for state discretization

## Future Improvements

- higher-order or conditional transition models
- online adaptation under concept drift
- SHAP-like local explanations for value deviation
- richer temporal rule templates
- event-level alert grouping and root-cause ranking
