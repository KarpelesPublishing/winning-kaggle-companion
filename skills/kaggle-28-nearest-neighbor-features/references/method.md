# Chapter 28: Nearest-Neighbor Features

**How many neighbours does it take before including the row itself stops distorting the estimate, and what does the legitimate feature gain?**

Neighbour target features look like a free gain, and a self-included version looks like a miracle. Measuring the three constructions against new rows shows where the shine is leakage and what remains once it is removed.

## The experiment

Six constructed datasets, each with 40 tight clusters in five dimensions with event rates of 0.12 or 0.88, 1,500 training rows and 4,000 new rows from the same generator. The feature is the mean label of the K nearest training rows after a training-fitted scaler. It is built three ways: with the row itself among its neighbours, with the row excluded once on the full training set, and nested inside each of five outer folds. A histogram gradient boosting classifier is scored by AUC, in 5-fold cross-validation and on the new rows, and compared with the same model without the feature.

Control: Neighbours K in the target-mean feature (1: the single nearest row, 5, 20, 50; default 5).

## Measured results

| Measure | 1: the single nearest row | 5 | 20 | 50 |
|---|---|---|---|---|
| Self-included CV | 1.000 | 0.921 | 0.880 | 0.865 |
| Self-included, new rows | 0.754 | 0.856 | 0.869 | 0.867 |
| Nested CV | 0.827 | 0.853 | 0.860 | 0.858 |
| Honest feature, new rows | 0.836 | 0.858 | 0.865 | 0.866 |
| No feature, new rows | 0.836 | 0.836 | 0.836 | 0.836 |

## What the result says (default, neighbours k in the target-mean feature = 5)

With K = 5: the self-included pipeline reports 0.921 but scores 0.856 on new rows, 0.921 - 0.856 = 0.065 of optimism. Excluding the row once reports 0.853; the nested estimate 0.853 differs from the new-row score 0.858 by -0.005. Against 0.836 without the feature, the honest feature adds 0.022 AUC on new rows. In a self-included feature the row's own label is 1/5 of the mean.

- Self included: 0.921 - 0.856 = 0.065 (estimate minus new rows).
- Self excluded, then CV: 0.853 - 0.858 = -0.005.
- Nested: 0.853 - 0.858 = -0.005.
- Value of the honest feature on new rows: 0.858 - 0.836 = +0.022.

## Apply it to a competition

- Never encode training rows in place with the database that contains them; drop each row's own entry, as a leave-one-out query does.
- Inside every outer fold, build the training rows' feature from the outer-training rows, then query the validation rows from the same database.
- Compare against the same model without the feature on matched rows, and try several K: a tiny K can be pure noise once the leak is gone.
- Treat a near-perfect estimate from a neighbour feature as a signal to check the exclusion, not as a result.

## Assumptions and limits

Constructed data with strong local structure, so the legitimate gain is real here and may be smaller on a competition table. In this generator excluding the row once on the full training set and nesting give estimates within about 0.01 AUC of each other and of the new-row score; with other data the second boundary can matter more, which is why the chapter nests by default. A histogram gradient boosting classifier stands in for LightGBM.

Constructed data with strong local structure; the sizes of these effects are properties of this generator, not a competition result.

## Reproduce it

The chapter notebook `notebooks/28-nearest-neighbor-features.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch28` (`run`, `explain`, `draw`).

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from kaggle_companion.activities._common import clean

SEED = 28
REPLICATES = 6          # independent constructed datasets; every estimate is their mean
DIMS, BLOBS = 5, 40
N_TRAIN, N_NEW = 1500, 4000


def generate(rng):
    """Forty tight clusters in five dimensions, each with its own event rate (0.12 or 0.88).
    The label is a property of the neighbourhood, so neighbours' labels carry real signal."""
    centers = rng.uniform(-3, 3, (BLOBS, DIMS))
    rate = np.where(rng.random(BLOBS) < 0.5, 0.12, 0.88)

    def rows(n):
        b = rng.integers(0, BLOBS, n)
        return centers[b] + rng.normal(0, 0.55, (n, DIMS)), (rng.random(n) < rate[b]).astype(int)

    return rows(N_TRAIN), rows(N_NEW)


def neighbour_mean(X_db, y_db, X_query, k, exclude_self):
    """Mean label of the k nearest database rows. The scaler and the database come from X_db only.
    With exclude_self the query rows ARE the database rows, so each one's first neighbour is itself and is dropped."""
    scaler = StandardScaler().fit(X_db)
    nn = NearestNeighbors(n_neighbors=k + int(exclude_self)).fit(scaler.transform(X_db))
    ids = nn.kneighbors(scaler.transform(X_query))[1]
    return y_db[ids[:, int(exclude_self):]].mean(axis=1)


def auc(X_fit, y_fit, X_eval, y_eval):
    model = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=8, random_state=0).fit(X_fit, y_fit)
    return roc_auc_score(y_eval, model.predict_proba(X_eval)[:, 1])


def with_feature(X, f):
    return np.column_stack([X, f])


def one_replicate(k, seed):
    (X, y), (X_new, y_new) = generate(np.random.default_rng(seed))
    outer = list(KFold(5, shuffle=True, random_state=1).split(X))
    cv = lambda col: np.mean([auc(with_feature(X[a], col[a]), y[a], with_feature(X[b], col[b]), y[b]) for a, b in outer])

    baseline = (np.mean([auc(X[a], y[a], X[b], y[b]) for a, b in outer]), auc(X, y, X_new, y_new))
    query_new = neighbour_mean(X, y, X_new, k, exclude_self=False)    # new rows are queried against the training rows

    # 1. Self included: the training column is built from the same rows it describes, each row's own label among them.
    col_self = neighbour_mean(X, y, X, k, exclude_self=False)
    # 2. Self excluded once on the full training set, then ordinary cross-validation on that column.
    col_loo = neighbour_mean(X, y, X, k, exclude_self=True)
    # 3. Nested: inside each outer fold, exclude self among the outer-training rows and query the validation rows from them.
    nested = np.mean([auc(with_feature(X[a], neighbour_mean(X[a], y[a], X[a], k, True)), y[a],
                          with_feature(X[b], neighbour_mean(X[a], y[a], X[b], k, False)), y[b]) for a, b in outer])
    deployed_loo = auc(with_feature(X, col_loo), y, with_feature(X_new, query_new), y_new)
    return {"baseline": baseline,
            "self_included": (cv(col_self), auc(with_feature(X, col_self), y, with_feature(X_new, query_new), y_new)),
            "self_excluded": (cv(col_loo), deployed_loo),
            "nested": (nested, deployed_loo)}      # pipelines 2 and 3 deploy the same model; they differ in the estimate


def run(k):
    reps = [one_replicate(k, SEED * 100 + i) for i in range(REPLICATES)]
    out = {"neighbors": k, "replicates": REPLICATES, "training_rows": N_TRAIN, "new_rows": N_NEW}
    for name in ("baseline", "self_included", "self_excluded", "nested"):
        out[name] = {"cv": float(np.mean([r[name][0] for r in reps])), "new_rows": float(np.mean([r[name][1] for r in reps]))}
    out["per_replicate_cv"] = {n: [r[n][0] for r in reps] for n in ("self_included", "self_excluded", "nested")}
    return clean(out)
```

Book location: Chapter 28, A Three-Row Query. Constructed example: six seeded synthetic datasets with clustered labels and a histogram gradient boosting model, measured by the chapter activity.
