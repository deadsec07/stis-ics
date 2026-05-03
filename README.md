# STIS: State Transition Integrity Scoring for ICS Anomaly Detection

STIS is an interpretable ICS anomaly-detection framework by A A Hasnat that scores state transitions, not just isolated point values. The method combines continuous sensor deviation, rarity of discrete state changes, and configurable physical or control consistency rules.

Links:
- Live site: https://deadsec07.github.io/stis-ics/
- Repository: https://github.com/deadsec07/stis-ics
- Main site: https://hnetechnologies.com/
- Creator profile: https://deadsec07.github.io/

## Algorithm

For timestep `t`, define system state:

`S(t) = {sensor values, actuator states, context}`

Then score the transition `S(t) -> S(t+1)` as:

```text
STIS(t) = alpha * ValueDeviation(t)
        + beta  * TransitionRarity(t)
        + gamma * ConstraintViolationScore(t)
```

## Why STIS

Standard anomaly detectors often flag unusual values but miss semantically wrong transitions. STIS is designed to capture these integrity failures while remaining interpretable and portable across ICS datasets through configuration files.

## Project Layout

```text
stis-ics/
├── configs/
├── data/
├── experiments/
├── notebooks/
├── results/
├── scripts/
└── stis/
```

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

Run BATADAL with public files:

```bash
python scripts/train_stis.py \
  --dataset-config configs/datasets/batadal.yaml \
  --constraints-config configs/constraints/batadal_rules.yaml \
  --output-dir results/batadal
```

## Evaluation

The framework computes precision, recall, F1, ROC-AUC where applicable, PR-AUC, confusion matrix, detection delay, and ablation variants.

## Future Improvements

- Higher-order or conditional transition models
- Online adaptation under concept drift
- Richer temporal rule templates
- Event-level alert grouping and root-cause ranking
