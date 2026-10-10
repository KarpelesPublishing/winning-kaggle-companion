# Chapter 20: Horse Health: Small Data, Big Discipline

**How much of the accuracy a tuned class factor adds on its own OOF rows survives on new rows?**

With 3 classes and micro-F1 (which equals accuracy for single-label prediction), a per-class factor looks like free accuracy on the rows that produced it. The chapter says an OOF base matrix does not make the fitted factors' same-row score independent; this measures by how much.

## The experiment

Fourteen constructed three-class datasets per development size (about 45%, 33% and 22% of rows), eight numeric features. A small boosted-tree classifier gives 5-fold OOF probabilities. A grid of 169 factor triples is searched for the highest OOF accuracy (divide each class probability by its factor, take argmax). Same-row gain is that accuracy minus plain argmax accuracy on the OOF rows. Cross-half gain tunes on one half of the OOF rows and scores the other half. New-row gain applies the tuned factors to the model refitted on all development rows and scores 4,000 fresh rows. Two kinds of probabilities are compared: the model's own, and the same probabilities divided by training class frequency and renormalized, which distorts them toward the rare classes as class-weighted training does.

Control: Development rows (250, 500, 988: the chapter's development size, 3,000; default 988).

## Measured results

| Measure | 250 | 500 | 988: the chapter's development size | 3,000 |
|---|---|---|---|---|
| Plain, same-row gain (points) | 2.97 | 1.89 | 1.03 | 0.31 |
| Plain, new-row gain (points) | -0.56 | -0.15 | -0.52 | -0.03 |
| Rebalanced, same-row gain (points) | 3.14 | 2.96 | 2.06 | 1.95 |
| Rebalanced, new-row gain (points) | +0.41 | +1.03 | +0.72 | +1.45 |
| Plain argmax accuracy on OOF rows | 0.573 | 0.616 | 0.629 | 0.654 |

## What the result says (default, development rows = 988)

With 988 development rows the model's own probabilities gain 1.03 accuracy points from tuned factors on the rows that tuned them and -0.52 on 4,000 new rows, so 1.03 - (-0.52) = 1.55 points is overstatement. For the rebalanced probabilities the same-row gain is 2.06 and the new-row gain +0.72 points: the factors pay there. The same-row gain is positive in both cases and cannot tell them apart. The cross-half gains are -0.94 and +0.79, which does separate them.

- Plain: same-row +1.03, cross-half -0.94, new rows -0.52 points.
- Rebalanced: same-row +2.06, cross-half +0.79, new rows +0.72 points.
- Plain overstatement: 1.03 - (-0.52) = 1.55 points.
- Rebalanced overstatement: 2.06 - 0.72 = 1.34 points.

## Apply it to a competition

- Score plain argmax first; treat any tuned factors as a candidate decision rule that must beat it on rows it was not tuned on.
- Tune factors inside an outer fold, or on half of the OOF rows and score the other half, or on an untouched holdout; never report the same-row gain.
- With few rows, expect the same-row gain to be largest and the least trustworthy; with many rows it shrinks toward the real gain.
- If your model's probabilities were rebalanced (class weights, resampling), factors are more likely to pay; re-check them whenever the model changes.

## Assumptions and limits

Constructed data and one boosted-tree stand-in for the chapter's gradient boosting libraries. The rebalanced probabilities are a constructed distortion, not a recipe. Fourteen datasets per size give new-row gains with standard errors of at most about 0.4 points, so smaller differences, including the rebalanced gain at 250 rows, are not resolved.

Constructed data. Whether factors pay depends on the probabilities, the metric and the number of rows; the contrast, not the size, is the lesson.

## Reproduce it

The chapter notebook `notebooks/20-horse-health-small-data.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch20` (`run`, `explain`, `draw`).

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from kaggle_companion.activities._common import clean

SEED = 20
REPLICATES = 14      # independent constructed datasets per development size
N_NEW = 4000         # fresh rows from the same generator, never used for fitting or tuning
GRID = np.exp(np.linspace(np.log(0.5), np.log(2.0), 13))
FACTORS = np.array([(1.0, a, b) for a in GRID for b in GRID])   # divide class 0, 1, 2 probabilities by these, then take argmax


def generate(n, rng, weights, bias):
    """Three-class task with moderate imbalance (about 45% / 33% / 22%) and a few nonlinear terms."""
    X = rng.normal(size=(n, 8))
    z = X @ weights + bias + 0.5 * np.column_stack([np.sin(2 * X[:, 0]), X[:, 1] * X[:, 2], np.zeros(n)])
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return X, np.minimum((rng.random(n)[:, None] > p.cumsum(1)).sum(1), 2)


def model():
    return HistGradientBoostingClassifier(learning_rate=0.1, max_leaf_nodes=6, max_iter=40, early_stopping=False, random_state=0)


def tune(proba, y):
    """Grid-search the factors that maximize accuracy on these rows; return them and that accuracy."""
    accuracy = (np.argmax(proba[:, None, :] / FACTORS[None], 2) == y[:, None]).mean(0)
    best = int(np.argmax(accuracy))
    return FACTORS[best], accuracy[best]


def gain(proba, y, factors):
    """Accuracy of factor-adjusted argmax minus accuracy of plain argmax, on these rows."""
    return float((np.argmax(proba / factors, 1) == y).mean() - (proba.argmax(1) == y).mean())


def one_dataset(n, seed):
    rng = np.random.default_rng(seed)
    weights, bias = rng.normal(0, 0.55, (8, 3)), np.array([0.5, 0.0, -0.5])
    X, y = generate(n, rng, weights, bias)
    X_new, y_new = generate(N_NEW, rng, weights, bias)
    oof = np.zeros((n, 3))
    for fit, out in StratifiedKFold(5, shuffle=True, random_state=1).split(X, y):
        oof[out] = model().fit(X[fit], y[fit]).predict_proba(X[out])
    new = model().fit(X, y).predict_proba(X_new)         # the deployed model, refitted on every development row
    prior = np.bincount(y, minlength=3) / n
    half = rng.permutation(n)
    halves = (half[: n // 2], half[n // 2:])
    out = {}
    # "plain": the model's own probabilities. "rebalanced": divided by training class frequency and renormalized,
    # which distorts them toward the rare classes in the way class-weighted training does.
    for name, (p_oof, p_new) in {"plain": (oof, new), "rebalanced": (oof / prior, new / prior)}.items():
        p_oof, p_new = p_oof / p_oof.sum(1, keepdims=True), p_new / p_new.sum(1, keepdims=True)
        factors, tuned_accuracy = tune(p_oof, y)
        # Same rows: tuned on the OOF rows and scored on those same rows. Cross-half: tuned on one half, scored on the other.
        same_rows = tuned_accuracy - (p_oof.argmax(1) == y).mean()
        cross = np.mean([gain(p_oof[b], y[b], tune(p_oof[a], y[a])[0]) for a, b in (halves, halves[::-1])])
        out[name] = (same_rows, cross, gain(p_new, y_new, factors), (p_oof.argmax(1) == y).mean())
    return out


def run(development_rows):
    sets = [one_dataset(development_rows, SEED * 1000 + i) for i in range(REPLICATES)]
    models = {}
    for name in ("plain", "rebalanced"):
        v = np.array([s[name] for s in sets])
        models[name] = {"same_rows": float(v[:, 0].mean()), "cross_half": float(v[:, 1].mean()), "new_rows": float(v[:, 2].mean()),
                        "new_rows_se": float(v[:, 2].std() / np.sqrt(REPLICATES)), "argmax_accuracy": float(v[:, 3].mean())}
    return clean({"development_rows": development_rows, "new_rows": N_NEW, "replicates": REPLICATES, "models": models})
```

Book location: Chapter 20, Per-Class Threshold Optimization on OOF. Constructed example: seeded synthetic three-class datasets and a boosted-tree classifier, measured by the chapter activity.
