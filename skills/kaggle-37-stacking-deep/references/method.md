# Chapter 37: Stacking Deep

**How far does a meta-model's score on its own OOF rows drift from an honest assessment as the library of base models grows, and when does a regularized meta-model overtake the unregularized one?**

Every extra base model is another column for the meta-model to fit. The score on the rows it was fitted to keeps improving, so it cannot say when to stop; the chapter asks for nested assessment and baselines to decide that.

## The experiment

A constructed regression task with 250 development rows, 10 features, a bent signal and noise of variance 1. The library is the control: that many ridge, nearest-neighbor and shallow-tree models on random feature subsets. Five outer folds; inside each, 3-fold inner OOF predictions build the meta training matrix, the base models refit on the outer training rows predict the outer fold, and the meta-model (ordinary least squares, or ridge with alpha 50) predicts from those. A global scheme instead builds one OOF matrix and cross-validates the meta-model on it. Fresh-row scores come from 1,000 new rows. Results are means over 5 independent worlds.

Control: Number of base models in the library (2, 10, 30, 50; default 30).

## Measured results

| Measure | 2 | 10 | 30 | 50 |
|---|---|---|---|---|
| Own OOF rows | 2.98 | 2.31 | 1.66 | 1.27 |
| Global-matrix meta-CV | 2.99 | 2.45 | 2.14 | 2.29 |
| Nested | 2.99 | 2.45 | 2.18 | 2.38 |
| Fresh rows | 3.01 | 2.43 | 2.23 | 2.32 |
| Equal average, nested | 3.00 | 2.80 | 2.74 | 2.75 |
| Ridge stack, nested | 3.05 | 2.47 | 2.13 | 2.01 |

## What the result says (default, number of base models in the library = 30)

With 30 base models the least squares stack scores 1.66 on its own OOF rows but 2.18 under nested assessment (2.18 - 1.66 = 0.52 of optimism); the global OOF matrix gives 2.14 and fresh rows 2.23. Against the strongest single model the stack gains 1.08 on its own rows but +0.54 nested (2.72 - 2.18 = 0.54). The equal average scores 2.74 and the ridge stack 2.13, which is -0.05 relative to least squares (a negative sign means ridge is better; it was better in 40% of 5 worlds).

- Optimism of the least squares stack: 2.18 - 1.66 = 0.52.
- Honest gain over the strongest single model: 2.72 - 2.18 = 0.54.
- Gain the own-row scores would claim: 2.74 - 1.66 = 1.08.
- Nested ridge against nested least squares: 2.13 - 2.18 = -0.05.

## Apply it to a competition

- Report a stack with nested assessment (base stage rebuilt inside each outer partition), never with its score on its own OOF rows.
- Compare against the strongest single model and the equal average under the same outer rows; a stack must beat both to be worth its extra stage.
- Limit the meta-model's freedom: ridge, nonnegative weights or a smaller library once the library is large relative to the rows it is fitted on.
- Archive the recipe and row IDs, and keep the negative result if the stack does not clear the baselines.

## Assumptions and limits

Constructed data with simple scikit-learn base models, not GBM libraries; one fixed ridge strength (alpha 50) and one data size. The library size where ridge overtakes least squares depends on the rows and on how correlated the base models are. The nested estimate is slightly pessimistic because its meta-model trains on 80% of the rows.

Constructed data; the library size where regularization wins is a property of this generator. At 50 models the unregularized stack still beats the strongest single model on average, but by far less than its own-row score suggests.

## Reproduce it

The chapter notebook `notebooks/37-stacking-deep.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch37` (`run`, `explain`, `draw`).

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from kaggle_companion.activities._common import clean

SEED = 37
REPLICATES = 5          # independent worlds per library size; results are means over them
N_DEV, N_FRESH, FEATURES = 250, 1000, 10
RIDGE_ALPHA = 50.0      # the "simple regularized meta model" the chapter prefers


def generate(n, rng):
    """A bent, interacting signal that no single base model captures, plus noise with variance 1."""
    X = rng.normal(size=(n, FEATURES))
    signal = 1.2 * np.sin(1.5 * X[:, 0]) + 0.8 * X[:, 1] * X[:, 2] + 0.6 * X[:, 3] ** 2 + 0.5 * X[:, 4] - 0.4 * X[:, 5]
    return X, signal + rng.normal(0, 1.0, n)


def make_library(size, rng):
    """`size` base models: ridge, kNN and shallow trees, each seeing a random subset of 3 to 6 features."""
    return [(i % 3, np.sort(rng.choice(FEATURES, rng.integers(3, 7), replace=False))) for i in range(size)]


def base_predictions(library, X_fit, y_fit, X_new):
    """Fit every base model on (X_fit, y_fit) and return their predictions for X_new as columns."""
    out = np.empty((len(X_new), len(library)))
    for j, (kind, cols) in enumerate(library):
        model = (Ridge(alpha=1.0) if kind == 0 else KNeighborsRegressor(15) if kind == 1
                 else DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=j))
        out[:, j] = model.fit(X_fit[:, cols], y_fit).predict(X_new[:, cols])
    return out


def oof_matrix(library, X, y, folds):
    matrix = np.zeros((len(y), len(library)))
    for fit, out in folds:
        matrix[out] = base_predictions(library, X[fit], y[fit], X[out])
    return matrix


def mse(pred, y):
    return float(np.mean((pred - y) ** 2))


def one_world(size, seed):
    rng = np.random.default_rng(seed)
    X, y = generate(N_DEV, rng)
    X_fresh, y_fresh = generate(N_FRESH, rng)               # new rows: the reference no estimate may touch
    library = make_library(size, rng)
    outer = list(KFold(5, shuffle=True, random_state=1).split(X))

    # Global scheme: one OOF matrix for the whole development set, then meta-CV on that matrix.
    G = oof_matrix(library, X, y, outer)
    global_cv = np.zeros(N_DEV)
    for fit, out in KFold(5, shuffle=True, random_state=2).split(G):
        global_cv[out] = LinearRegression().fit(G[fit], y[fit]).predict(G[out])

    # Nested scheme: rebuild the base stage inside each outer training partition.
    nested = {k: np.zeros(N_DEV) for k in ("ols", "ridge", "average", "best")}
    own = {k: [] for k in nested}                           # each method scored on the rows it was fitted or selected on
    fresh = {"ols": [], "ridge": []}
    for fit, out in outer:
        inner = list(KFold(3, shuffle=True, random_state=3).split(fit))
        G_in = oof_matrix(library, X[fit], y[fit], inner)    # inner OOF base predictions for outer training
        both = base_predictions(library, X[fit], y[fit], np.vstack([X[out], X_fresh]))
        G_out, G_new = both[:len(out)], both[len(out):]
        best = int(np.argmin(((G_in - y[fit][:, None]) ** 2).mean(axis=0)))
        metas = {"ols": LinearRegression().fit(G_in, y[fit]), "ridge": Ridge(alpha=RIDGE_ALPHA).fit(G_in, y[fit])}
        for k, meta in metas.items():
            nested[k][out] = meta.predict(G_out)
            own[k].append(mse(meta.predict(G_in), y[fit]))
            fresh[k].append(mse(meta.predict(G_new), y_fresh))
        nested["average"][out], own["average"] = G_out.mean(axis=1), own["average"] + [mse(G_in.mean(axis=1), y[fit])]
        nested["best"][out], own["best"] = G_out[:, best], own["best"] + [mse(G_in[:, best], y[fit])]
    return {"global_cv": mse(global_cv, y), "nested": {k: mse(v, y) for k, v in nested.items()},
            "own": {k: float(np.mean(v)) for k, v in own.items()}, "fresh": {k: float(np.mean(v)) for k, v in fresh.items()}}


def run(library_size):
    worlds = [one_world(library_size, SEED * 100 + r) for r in range(REPLICATES)]

    def avg(get):
        return float(np.mean([get(w) for w in worlds]))
    return clean({
        "library_size": library_size,
        "ols": {"own": avg(lambda w: w["own"]["ols"]), "global_cv": avg(lambda w: w["global_cv"]),
                "nested": avg(lambda w: w["nested"]["ols"]), "fresh": avg(lambda w: w["fresh"]["ols"])},
        "methods": {k: {"own": avg(lambda w, k=k: w["own"][k]), "nested": avg(lambda w, k=k: w["nested"][k])}
                    for k in ("best", "average", "ols", "ridge")},
        "ridge_fresh": avg(lambda w: w["fresh"]["ridge"]),
        "ridge_beats_ols_nested": float(np.mean([w["nested"]["ridge"] < w["nested"]["ols"] for w in worlds])),
        "noise_variance": 1.0, "replicates": REPLICATES, "development_rows": N_DEV, "fresh_rows": N_FRESH,
    })
```

Book location: Chapter 37, Distinguish Meta Fitting from Outer Assessment. Constructed example: seeded synthetic regression data and scikit-learn base models, measured by the chapter activity.
