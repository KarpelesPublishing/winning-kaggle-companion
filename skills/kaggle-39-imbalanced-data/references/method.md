# Chapter 39: Imbalanced Data

**How much log loss does negative downsampling cost, and at what keep rate does the analytic correction stop being as good as calibrating on development rows?**

Competitions with millions of negatives invite downsampling, and a probability metric then scores the sampled prevalence rather than the real one. The chapter gives a correction and a calibration route; this measures where each holds.

## The experiment

A constructed task with about 1.3% positives: 40,000 training rows, 30,000 development rows and 60,000 test rows, all at natural prevalence. Every positive and a fraction r (the control) of the negatives are kept for training. A gradient-boosted model and a logistic regression are fitted on the sample. Three outputs are scored by log loss and AUC on the test rows: raw sampled probabilities, the chapter's correction p r / (p r + 1 - p), and Platt scaling fitted on the development rows. The reference is the same model trained on every row. Results are means over 5 worlds.

Control: Share of negatives kept for training (100% (no downsampling), 2%, 0.5%, 0.2%; default 0.02).

## Measured results

| Measure | 100% (no downsampling) | 2% | 0.5% | 0.2% |
|---|---|---|---|---|
| Training rows | 40,000 | 1,302 | 684 | 563 |
| Raw log loss | 0.056 | 0.338 | 0.971 | 1.906 |
| Corrected log loss | 0.056 | 0.056 | 0.063 | 0.089 |
| Calibrated log loss | 0.056 | 0.055 | 0.058 | 0.059 |
| Mean corrected prediction | 0.012 vs prevalence 0.013 | 0.013 vs prevalence 0.013 | 0.024 vs prevalence 0.013 | 0.050 vs prevalence 0.013 |
| AUC | 0.851 | 0.853 | 0.834 | 0.814 |

## What the result says (default, share of negatives kept for training = 0.02)

Keeping 2% of negatives leaves 1,302 training rows. Raw boosted-model probabilities average 0.230 against a true prevalence of 0.013, so log loss is 0.338 against 0.056 when trained on all rows (0.338 / 0.056 = 6.0 times). The analytic correction brings it to 0.0556 (0.0556 - 0.0563 = -0.0007 against the all-rows model, mean prediction 0.013); calibrating on development rows gives 0.0554. The logistic model's corrected log loss is +0.0001 from its own all-rows reference. AUC moves +0.001 from 0.851.

- Raw against all rows: 0.338 / 0.056 = 6.0 (ratio of log losses).
- Correction, boosted: 0.0556 - 0.0563 = -0.0007.
- Calibration, boosted: 0.0554 - 0.0563 = -0.0009.
- Correction, logistic: 0.0580 - 0.0579 = +0.0001.

## Apply it to a competition

- Downsample negatives only inside training, keep development and assessment rows at natural prevalence, and score with the competition metric.
- Never submit raw sampled probabilities to a probability metric; apply the correction or a development-fitted calibrator.
- Check the corrected probabilities: their mean on natural-prevalence rows should match the prevalence; if not, calibrate instead.
- Keep enough positives and negatives for the model to be calibrated on the sample; very aggressive keep rates also cost AUC.

## Assumptions and limits

Constructed data; HistGradientBoostingClassifier with 60 trees stands in for LightGBM. The correction's own assumptions hold by construction (negatives kept independently with equal probability, features unchanged). The keep rate where the correction falls behind depends on the number of positives and the model.

Constructed data. The correction ties with calibration at a 2% keep rate here; it fails only at far more aggressive rates and only for the boosted model.

## Reproduce it

The chapter notebook `notebooks/39-imbalanced-data.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch39` (`run`, `explain`, `draw`).

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from kaggle_companion.activities._common import clean

SEED = 39
REPLICATES = 5          # independent worlds per keep rate; results are means over them
N_TRAIN, N_DEV, N_TEST = 40_000, 30_000, 60_000
EPS = 1e-6              # probabilities are kept inside [1e-6, 1 - 1e-6] so log loss stays finite


def generate(n, rng):
    """Eight features, a bent and interacting score, about 1.3% positives."""
    X = rng.normal(size=(n, 8))
    score = X[:, 0] + 0.8 * np.maximum(X[:, 1], 0) + 0.7 * X[:, 2] * X[:, 3] - 0.6 * np.abs(X[:, 4]) + 0.4 * X[:, 5]
    return X, (rng.random(n) < 1 / (1 + np.exp(-(1.2 * score - 5.6)))).astype(int)


def to_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def correct(p, keep):
    """The chapter's correction: natural odds equal sampled odds times the negative keep rate r."""
    return p * keep / (p * keep + 1 - p)


def one_world(keep, seed):
    rng = np.random.default_rng(seed)
    X_train, y_train = generate(N_TRAIN, rng)
    X_dev, y_dev = generate(N_DEV, rng)                       # natural prevalence: used only to fit the calibrator
    X_test, y_test = generate(N_TEST, rng)                    # natural prevalence: used only for scoring
    sampled = (y_train == 1) | (rng.random(N_TRAIN) < keep)   # keep every positive, each negative with probability `keep`
    out = {"training_rows": int(sampled.sum()), "prevalence": float(y_test.mean())}
    makers = {"gbm": lambda: HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=15, random_state=0),
              "logistic": lambda: LogisticRegression(max_iter=500)}
    for name, make in makers.items():
        model = make().fit(X_train[sampled], y_train[sampled])
        p_dev = np.clip(model.predict_proba(X_dev)[:, 1], EPS, 1 - EPS)
        p_test = np.clip(model.predict_proba(X_test)[:, 1], EPS, 1 - EPS)
        calibrator = LogisticRegression(C=1e6).fit(to_logit(p_dev)[:, None], y_dev)      # Platt scaling on development rows
        outputs = {"raw": p_test, "corrected": np.clip(correct(p_test, keep), EPS, 1 - EPS),
                   "calibrated": calibrator.predict_proba(to_logit(p_test)[:, None])[:, 1]}
        out[name] = {k: [log_loss(y_test, v), roc_auc_score(y_test, v), float(v.mean())] for k, v in outputs.items()}
    return out


def run(keep_rate):
    worlds = [one_world(keep_rate, SEED * 100 + r) for r in range(REPLICATES)]
    # Reference: the same models trained on every row (keep rate 1), same worlds.
    base = worlds if keep_rate == 1 else [one_world(1, SEED * 100 + r) for r in range(REPLICATES)]

    def mean(group, model, arm, j):
        return float(np.mean([w[model][arm][j] for w in group]))
    result = {"keep_rate": keep_rate, "training_rows": float(np.mean([w["training_rows"] for w in worlds])),
              "prevalence": float(np.mean([w["prevalence"] for w in worlds])), "replicates": REPLICATES}
    for model in ("gbm", "logistic"):
        result[model] = {arm: {"log_loss": mean(worlds, model, arm, 0), "auc": mean(worlds, model, arm, 1),
                               "mean_prediction": mean(worlds, model, arm, 2)} for arm in ("raw", "corrected", "calibrated")}
        result[model]["no_downsampling_log_loss"] = mean(base, model, "raw", 0)
        result[model]["no_downsampling_auc"] = mean(base, model, "raw", 1)
    result["calibrated_beats_corrected_gbm"] = float(np.mean([w["gbm"]["calibrated"][0] < w["gbm"]["corrected"][0] for w in worlds]))
    return clean(result)
```

Book location: Chapter 39, Negative Downsampling at Scale. Constructed example: seeded synthetic data, a boosted model and a logistic regression, measured by the chapter activity.
