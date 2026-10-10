# Chapter 12: The GBM Trio

**How much does it matter whether a category is fed as ordered integer codes, as native categorical splits or as one-hot columns?**

The chapter warns that integer codes impose an order the category does not have, and that native categorical support has its own requirements. Whether the choice is worth a deadline hour depends on how many levels the column has.

## The experiment

A constructed binary task with four numeric features and one category whose levels have random effects and arbitrary integer codes. One scikit-learn histogram gradient boosting configuration (100 rounds, learning rate 0.1, 15 leaves) is fitted on 3,000 rows and scored by AUC on 6,000 new rows, with the category as ordered integer codes, as native categorical splits, as one-hot columns, and left out. Five independent datasets per setting; the control is the number of levels.

Control: Levels in the category (8, 32, 64, 128 (about 23 rows per level); default 128).

## Measured results

| Measure | 8 | 32 | 64 | 128 (about 23 rows per level) |
|---|---|---|---|---|
| Native splits | 0.792 | 0.788 | 0.784 | 0.782 |
| Ordered codes | 0.791 | 0.779 | 0.765 | 0.751 |
| One-hot | 0.791 | 0.788 | 0.787 | 0.769 |
| No category | 0.715 | 0.715 | 0.710 | 0.707 |
| One-hot columns | 12 | 36 | 68 | 132 |

## What the result says (default, levels in the category = 128)

With 128 levels (about 23 training rows each), native categorical splits score 0.782 AUC. Ordered integer codes score 0.751, so 0.782 - 0.751 = 0.031. One-hot columns score 0.769 with 132 columns instead of 5. Leaving the category out scores 0.707. The choice of representation is worth a comparison at this size.

- Ordered codes behind native: 0.782 - 0.751 = 0.031 AUC.
- One-hot behind native: 0.782 - 0.769 = 0.013 AUC.
- Value of the category: 0.782 - 0.707 = 0.075 AUC.
- Columns fed to the trees: 5 (codes, native) against 132 (one-hot).

## Apply it to a competition

- Fix rounds, leaves and folds first, then change only the category representation, so a difference can be attributed to it.
- Compare ordered codes, native categorical splits and one-hot on the same rows; the gap grows with the number of levels.
- Keep the cheapest representation that is within noise of the best, and record columns and fit time next to the score.
- Native handling still needs consistent category meaning at inference and an unknown-level policy; it does not replace the outer assessment.

## Assumptions and limits

Constructed data and scikit-learn's HistGradientBoostingClassifier as the stand-in: LightGBM, XGBoost and CatBoost are not installed here. The three representations correspond to treating a label-encoded column as numeric, to the native categorical paths of LightGBM and CatBoost (and XGBoost with categorical support enabled), and to one-hot encoding. The numbers belong to this learner and generator. Fit time is not reported because it varies by machine; the column count is the deterministic cost proxy.

Constructed data with random level effects and a stand-in learner. The gap between native splits and one-hot columns is a few thousandths of AUC up to 64 levels and about 0.01 at 128 levels, and it changed size across seed bases, so no general ranking between them is claimed.

## Reproduce it

The chapter notebook `notebooks/12-the-gbm-trio.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch12` (`run`, `explain`, `draw`).

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from kaggle_companion.activities._common import clean

SEED = 12
REPLICATES = 5                 # independent constructed datasets; every estimate is their mean
N_TRAIN, N_NEW = 3000, 6000
NUMERIC = 4                    # numeric feature columns before the category column


def generate(levels, seed):
    """Four numeric features plus one category with `levels` levels. Each level has a random effect on the log odds,
    and the integer code given to a level is arbitrary, as it is for a label-encoded column."""
    rng = np.random.default_rng(seed)
    effect = rng.normal(0.0, 1.0, levels)
    code = rng.permutation(levels)

    def rows(n):
        level = rng.integers(0, levels, n)
        x = rng.normal(size=(n, NUMERIC))
        logit = 0.9 * x[:, 0] - 0.6 * x[:, 1] + 0.5 * x[:, 2] * x[:, 3] + effect[level]
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        return x, level, y

    return rows(N_TRAIN), rows(N_NEW), code


def learner(**extra):
    """One fixed configuration for every representation: the comparison changes the input, not the budget."""
    return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_leaf_nodes=15, random_state=0, **extra)


def run(levels):
    scores = {"none": [], "ordinal": [], "native": [], "onehot": []}
    for rep in range(REPLICATES):
        (x, lv, y), (x_new, lv_new, y_new), code = generate(levels, SEED * 100 + rep)
        ordinal, ordinal_new = np.column_stack([x, code[lv]]), np.column_stack([x_new, code[lv_new]])
        onehot, onehot_new = np.column_stack([x, np.eye(levels)[lv]]), np.column_stack([x_new, np.eye(levels)[lv_new]])
        fits = {"none": (learner(), x, x_new),                                   # no category column at all
                "ordinal": (learner(), ordinal, ordinal_new),                    # codes treated as ordered numbers
                "native": (learner(categorical_features=[NUMERIC]), ordinal, ordinal_new),   # native categorical splits
                "onehot": (learner(), onehot, onehot_new)}                       # one 0/1 column per level
        for name, (model, a, b) in fits.items():
            scores[name].append(roc_auc_score(y_new, model.fit(a, y).predict_proba(b)[:, 1]))
    return clean({
        "levels": levels, "columns": {"none": NUMERIC, "ordinal": NUMERIC + 1, "native": NUMERIC + 1, "onehot": NUMERIC + levels},
        "auc": {k: float(np.mean(v)) for k, v in scores.items()}, "per_replicate": scores,
        "rows_per_level": N_TRAIN / levels, "replicates": REPLICATES, "train_rows": N_TRAIN, "new_rows": N_NEW,
    })
```

Book location: Chapter 12, Feature Types and Framework Fit. Constructed example: seeded synthetic data and scikit-learn histogram gradient boosting, measured by the chapter activity.
