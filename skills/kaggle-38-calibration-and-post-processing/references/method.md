# Chapter 38: Calibration and Post-Processing

**How large must the development set be before isotonic calibration beats the simpler Platt calibrator on new rows?**

Calibration is fitted, so it can overfit like any model. A calibrator that looks best on its own development rows can be the worst choice on the leaderboard, and the size of the development set decides which one to trust.

## The experiment

A constructed binary task. A Gaussian naive Bayes base model is trained once on 3,000 rows of six correlated copies of one signal, so it double counts the evidence and is overconfident; the true log odds also bend, so no two-parameter curve fits them exactly. For each development-set size (the control), 30 independent development sets are drawn. On each, Platt scaling (logistic regression on the base log odds) and isotonic regression are fitted, then scored by log loss on their own development rows and on 20,000 assessment rows.

Control: Development rows used to fit the calibrator (50, 200, 1,000, 5,000; default 200).

## Measured results

| Measure | 50 | 200 | 1,000 | 5,000 |
|---|---|---|---|---|
| No calibration, new rows | 0.959 | 0.959 | 0.959 | 0.959 |
| Platt, new rows | 0.587 | 0.566 | 0.561 | 0.560 |
| Isotonic, new rows | 0.694 | 0.589 | 0.557 | 0.554 |
| Isotonic, own rows | 0.480 | 0.509 | 0.542 | 0.550 |
| Isotonic beats Platt | 7% of draws | 20% of draws | 93% of draws | 100% of draws |

## What the result says (default, development rows used to fit the calibrator = 200)

With 200 development rows, calibration cuts log loss on new rows from 0.959 to 0.566 with Platt (0.959 - 0.566 = 0.393) and 0.589 with isotonic, so Platt is better on new rows (isotonic won 20% of 30 draws). On its own development rows isotonic scores 0.509, 0.080 better than it does on new rows. AUC moves from 0.768 to 0.768 (Platt) and 0.760 (isotonic).

- Gain from Platt on new rows: 0.959 - 0.566 = 0.393.
- Isotonic minus Platt on new rows: 0.589 - 0.566 = +0.023.
- Isotonic optimism: 0.589 - 0.509 = 0.080 (new rows minus own rows).
- Platt optimism: 0.566 - 0.552 = +0.015.

## Apply it to a competition

- Fit calibration on predictions whose labels the base model never saw, then score it on a different partition.
- Never choose a calibrator by its score on the rows it was fitted to; that comparison always favours the most flexible one.
- With a few hundred development rows, start with Platt or temperature scaling; consider isotonic only when thousands of rows show it winning on assessment.
- Calibration fixes log loss and calibration error, not AUC: if the metric is ranking, spend the effort elsewhere.

## Assumptions and limits

Constructed data, one overconfident base model and probabilities kept inside [0.001, 0.999]. The crossover size depends on how bent the true calibration curve is; with a curve Platt can represent exactly, isotonic needs even more rows to catch up.

Constructed data; the crossover size is a property of this generator, not a rule for every competition.

## Reproduce it

The chapter notebook `notebooks/38-calibration-and-post-processing.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch38` (`run`, `explain`, `draw`).

```python
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB

from kaggle_companion.activities._common import clean

SEED = 38
DRAWS = 30        # independent development sets per size; results are means over draws
EPS = 1e-3        # probabilities are kept inside [0.001, 0.999] so one confident miss cannot dominate log loss


def generate(n, rng):
    """Six noisy copies of one hidden signal plus two noise features. The true log odds bend upward for positive signal."""
    z = rng.normal(size=n)
    X = np.column_stack([z + rng.normal(0, 0.6, n) for _ in range(6)] + [rng.normal(size=n), rng.normal(size=n)])
    logit = 0.4 * z + 1.6 * np.maximum(z, 0) - 1.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def to_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expected_calibration_error(y, p, bins=10):
    """The chapter's population-weighted ECE with equal-width bins; empty bins contribute nothing."""
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((b == k).mean() * abs(y[b == k].mean() - p[b == k].mean()) for k in range(bins) if (b == k).any()))


def reliability(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return [[float(p[b == k].mean()), float(y[b == k].mean()), int((b == k).sum())] for k in range(bins) if (b == k).sum() >= 20]


def run(development_rows):
    rng = np.random.default_rng(SEED)
    X_base, y_base = generate(3000, rng)
    base = GaussianNB().fit(X_base, y_base)                    # frozen base model: never sees development or assessment labels
    X_a, y_a = generate(20000, rng)                            # assessment rows, used only for scoring
    p_a = np.clip(base.predict_proba(X_a)[:, 1], EPS, 1 - EPS)

    scores = {k: [] for k in ("none", "platt", "isotonic", "none_dev", "platt_dev", "isotonic_dev",
                              "auc_platt", "auc_isotonic")}
    for draw in range(DRAWS):
        X_d, y_d = generate(development_rows, rng)
        p_d = np.clip(base.predict_proba(X_d)[:, 1], EPS, 1 - EPS)
        platt = LogisticRegression(C=1e6).fit(to_logit(p_d)[:, None], y_d)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p_d, y_d)
        cal = {"platt": lambda p: platt.predict_proba(to_logit(p)[:, None])[:, 1], "isotonic": iso.predict}
        scores["none"].append(log_loss(y_a, p_a))
        scores["none_dev"].append(log_loss(y_d, p_d, labels=[0, 1]))
        for name, f in cal.items():
            scores[name].append(log_loss(y_a, f(p_a)))
            scores[name + "_dev"].append(log_loss(y_d, f(p_d), labels=[0, 1]))   # scored on the rows it was fitted to
            scores["auc_" + name].append(roc_auc_score(y_a, f(p_a)))
        if draw == 0:
            curves = {"none": reliability(y_a, p_a), "platt": reliability(y_a, cal["platt"](p_a)),
                      "isotonic": reliability(y_a, cal["isotonic"](p_a))}
            ece = {"none": expected_calibration_error(y_a, p_a),
                   "platt": expected_calibration_error(y_a, cal["platt"](p_a)),
                   "isotonic": expected_calibration_error(y_a, cal["isotonic"](p_a))}

    mean = {k: float(np.mean(v)) for k, v in scores.items()}
    return clean({
        "development_rows": development_rows,
        "assessment": {k: mean[k] for k in ("none", "platt", "isotonic")},
        "development": {k: mean[k + "_dev"] for k in ("none", "platt", "isotonic")},
        "isotonic_beats_platt": float(np.mean(np.array(scores["isotonic"]) < np.array(scores["platt"]))),
        "auc": {"none": roc_auc_score(y_a, p_a), "platt": mean["auc_platt"], "isotonic": mean["auc_isotonic"]},
        "ece_first_draw": ece, "reliability_first_draw": curves,
        "draws": DRAWS, "assessment_rows": len(y_a),
    })
```

Book location: Chapter 38, Fit Calibration on the Correct Predictions. Constructed example: seeded synthetic data, a naive Bayes base model and two calibrators, measured by the chapter activity.
