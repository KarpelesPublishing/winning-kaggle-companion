# Chapter 16: The Fourth-Root Blend

**How alike can three models' predictions be before a blend stops beating the best single model, and do the weights matter?**

The chapter treats a blend as a candidate that must beat the strongest component and cost less than it gains. Prediction correlation is its screen, so the useful number is where the blend's gain disappears.

## The experiment

A constructed binary task. A hidden signal drives the label; three logistic regressions each see a noisy copy of it (noise 1.0, 1.15 and 1.3), and the share of that noise common to the three copies is the control. Models are fitted on 600 rows. On 300 development rows the strongest model is chosen, fourth-root weights are computed from each model's log loss against a measured constant-prediction baseline, and a hill-climbing search fits weights. Everything is scored by log loss on 20,000 assessment rows. Forty independent datasets are averaged.

Control: Share of each model's noise that is common to all three (0: independent errors, 0.3, 0.7, 0.95: nearly identical errors; default 0.95).

## Measured results

| Measure | 0: independent errors | 0.3 | 0.7 | 0.95: nearly identical errors |
|---|---|---|---|---|
| Prediction correlation | 0.43 | 0.60 | 0.83 | 0.97 |
| Best single model | 0.619 | 0.616 | 0.615 | 0.613 |
| Equal blend | 0.598 | 0.606 | 0.616 | 0.623 |
| Fourth-root blend | 0.598 | 0.606 | 0.616 | 0.623 |
| Fitted blend | 0.599 | 0.605 | 0.612 | 0.613 |

## What the result says (default, share of each model's noise that is common to all three = 0.95)

At prediction correlation 0.97, the equal blend does worse than the best single model: 0.613 - 0.623 = -0.010 log loss on 20,000 assessment rows, and it beats the best model in 0% of 40 datasets. Fourth-root weights score 0.623 (weights differ by 0.028 on average). Against equal weights, fitted weights change log loss by -0.0100 on the rows the weights were fitted to and by -0.0096 on assessment rows (negative means fitted weights do better).

- Equal blend: 0.613 - 0.623 = -0.010 (best model minus blend; positive means the blend wins).
- Fourth-root blend: 0.613 - 0.623 = -0.010.
- Fitted blend: 0.613 - 0.613 = 0.000.
- Fourth-root weights differ from each other by 0.028 on average (equal weights differ by 0).

## Apply it to a competition

- Screen the candidates by prediction correlation, but decide by the blend's score on assessment rows, not by a correlation cutoff.
- Compare equal weights, the fourth-root heuristic and any fitted weights against the strongest single model, which is the real baseline.
- Use a measured dummy predictor as the fourth-root baseline, and expect weights close to equal.
- Keep the single model when the blend does not beat it, and count the extra fits and inference as cost.

## Assumptions and limits

Constructed data and three logistic regressions standing in for three different model families; the models differ only in noise, whereas real models also differ in bias. The baseline for the fourth-root rule is the measured log loss of a constant-prior predictor, as the chapter requires, not a target mean passed off as a score. Correlation here is measured on the 300 development rows.

Constructed data with models that differ only in noise. The crossover correlation depends on how different the models' strengths are; with models of equal strength an equal blend would not fall below the best one.

## Reproduce it

The chapter notebook `notebooks/16-fourth-root-blend.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch16` (`run`, `explain`, `draw`).

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from kaggle_companion.activities._common import clean

SEED = 16
REPLICATES = 40                      # independent constructed datasets; every estimate is their mean
NOISE = np.array([1.0, 1.15, 1.3])   # how noisy each model's view of the hidden signal is: three similar-strength models
N_TRAIN, N_DEV, N_ASSESS = 600, 300, 20000
EPS = 1e-4


def generate(rng, n, shared):
    """A hidden signal drives the label. Each model sees a noisy copy; `shared` is the share of each copy's noise
    variance that is common to all three models, so it sets how alike their errors are."""
    signal = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-1.6 * signal))).astype(int)
    common = rng.normal(size=n)
    views = [signal + sd * (np.sqrt(shared) * common + np.sqrt(1 - shared) * rng.normal(size=n)) for sd in NOISE]
    return np.column_stack(views), y


def fourth_root_weights(scores, baseline, maximize=True):
    """The chapter's heuristic: compress each positive margin over a measured baseline with a fourth root."""
    scores = np.asarray(scores, dtype=float)
    margin = scores - baseline if maximize else baseline - scores
    raw = np.maximum(margin, 0) ** 0.25
    return raw / raw.sum()


def greedy_weights(y, P, step=0.1, max_iterations=30):
    """Hill-climbing on development log loss: start from the best model, move weight to whichever model helps most."""
    loss = lambda q: log_loss(y, np.clip(q, EPS, 1 - EPS))
    weights = np.eye(P.shape[1])[int(np.argmin([loss(P[:, j]) for j in range(P.shape[1])]))]
    current = P @ weights
    for _ in range(max_iterations):
        trial = [loss((1 - step) * current + step * P[:, j]) for j in range(P.shape[1])]
        j = int(np.argmin(trial))
        if trial[j] >= loss(current) - 1e-12:
            break
        weights = weights * (1 - step)
        weights[j] += step
        current = P @ weights
    return weights


def run(shared):
    loss = lambda y, q: log_loss(y, np.clip(q, EPS, 1 - EPS))
    rows = []
    for rep in range(REPLICATES):
        rng = np.random.default_rng(SEED * 100 + rep)
        (X, y), (Xd, yd), (Xa, ya) = [generate(rng, n, shared) for n in (N_TRAIN, N_DEV, N_ASSESS)]
        models = [LogisticRegression().fit(X[:, [m]], y) for m in range(3)]      # one model per view
        Pd = np.column_stack([models[m].predict_proba(Xd[:, [m]])[:, 1] for m in range(3)])
        Pa = np.column_stack([models[m].predict_proba(Xa[:, [m]])[:, 1] for m in range(3)])
        dev_loss = np.array([loss(yd, Pd[:, m]) for m in range(3)])
        dummy = loss(yd, np.full(N_DEV, y.mean()))                              # measured baseline: a constant-prior predictor
        w_root = fourth_root_weights(dev_loss, dummy, maximize=False)
        w_greedy = greedy_weights(yd, Pd)
        logits = np.log(Pd / (1 - Pd))
        corr = np.corrcoef(logits.T)
        rows.append({
            "best": loss(ya, Pa[:, int(np.argmin(dev_loss))]),                  # strongest model, chosen on development rows
            "equal": loss(ya, Pa.mean(axis=1)), "root": loss(ya, Pa @ w_root), "greedy": loss(ya, Pa @ w_greedy),
            "equal_dev": loss(yd, Pd.mean(axis=1)), "greedy_dev": loss(yd, Pd @ w_greedy),
            "correlation": (corr[0, 1] + corr[0, 2] + corr[1, 2]) / 3, "root_spread": w_root.max() - w_root.min(),
            "w_root": w_root, "w_greedy": w_greedy,
        })
    col = lambda k: np.array([r[k] for r in rows])
    gain = {k: col("best") - col(k) for k in ("equal", "root", "greedy")}      # positive: the blend beats the best single model
    return clean({
        "shared_noise": shared, "correlation": float(col("correlation").mean()), "root_weight_spread": float(col("root_spread").mean()),
        "assessment_loss": {k: float(col(k).mean()) for k in ("best", "equal", "root", "greedy")},
        "gain_over_best": {k: float(v.mean()) for k, v in gain.items()},
        "gain_standard_error": {k: float(v.std() / np.sqrt(REPLICATES)) for k, v in gain.items()},
        "share_equal_beats_best": float((gain["equal"] > 0).mean()),
        "mean_weights": {"root": np.mean([r["w_root"] for r in rows], axis=0), "greedy": np.mean([r["w_greedy"] for r in rows], axis=0)},
        "greedy_vs_equal": {"development": float((col("equal_dev") - col("greedy_dev")).mean()),
                            "assessment": float((col("equal") - col("greedy")).mean())},
        "replicates": REPLICATES, "assessment_rows": N_ASSESS, "development_rows": N_DEV,
    })
```

Book location: Chapter 16, The Fourth-Root Weight Formula. Constructed example: seeded synthetic data and three logistic regressions, measured by the chapter activity.
