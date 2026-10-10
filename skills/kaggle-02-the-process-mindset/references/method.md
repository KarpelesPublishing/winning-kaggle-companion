# Chapter 2: The Process Mindset

**How much of the winning version's public gain is still there on rows the choice never used, as the number of logged versions grows?**

A log of many versions invites the question of which one to keep. If the public score picks it, the winner's public gain contains a share of luck that grows with the number of versions, and the log has created no new independent labels.

## The experiment

Sixteen constructed competitions. Each has 800 training rows, 2,000 test rows (400 public and 1,600 private, drawn at random ten times) and 20,000 fresh rows as the truth. A baseline logistic regression uses 6 columns; version k adds one of 100 candidate columns and refits, and only 2 of the 100 candidates truly help, by a small amount. The control is how many versions are logged (the baseline plus candidates). The winner is the version with the lowest public log loss. Gain is the baseline's loss minus the winner's loss, so a positive gain looks like improvement.

Control: Versions logged (including the baseline) (3, 10, 30, 100; default 30).

## Measured results

| Measure | 3 | 10 | 30 | 100 |
|---|---|---|---|---|
| Public gain | 0.0007 | 0.0030 | 0.0045 | 0.0083 |
| Private gain | -0.0003 | 0.0001 | 0.0008 | 0.0032 |
| Fresh-row gain | -0.0004 | 0.0001 | 0.0010 | 0.0036 |
| Public minus private | +0.0010 | +0.0029 | +0.0037 | +0.0051 |
| Picks truly better | 13% | 19% | 34% | 76% |

## What the result says (default, versions logged (including the baseline) = 30)

With 30 versions logged, the winner's public gain is 0.0045 and its private gain is 0.0008, so 0.0045 - 0.0008 = 0.0037 of the public gain is not there on private rows. On 20,000 fresh rows the winner gains +0.0010. Most of the public gain is the luck of picking the best of many comparisons. The winner truly beats the baseline in 34% of the picks.

- Overstatement: 0.0045 - 0.0008 = +0.0037 (public gain minus private gain).
- Private gain against the truth: 0.0008 - 0.0010 = -0.0002.
- Public gain against the truth: 0.0045 - 0.0010 = +0.0035.
- Picks that truly beat the baseline: 34% of 160 public-split picks.

## Apply it to a competition

- Record which feedback chose each version; once the public score picks a version it is development evidence.
- Do not read a public gain from the best of many versions as the gain you will see on hidden rows; expect it to shrink.
- Keep a comparison that played no part in the pick (a reserved block or a private score) and compare the winner with the baseline there.
- Adding runs adds no independent labels: before logging more versions, ask what new information each one carries.

## Assumptions and limits

Constructed competitions with a logistic regression, one added column per version, and 2 truly useful candidates among 100. With every candidate useless, the private gain is negative at every N (checked by the build with the two useful candidates removed). The gain sizes are properties of this generator.

Constructed data. Whether the pick helps on hidden rows depends on how many versions truly help; here 2 of 100 do, by design, and the sizes are properties of this generator.

## Reproduce it

The chapter notebook `notebooks/02-the-process-mindset.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch02` (`run`, `explain`, `draw`).

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

from kaggle_companion.activities._common import clean

SEED = 2
REPLICATES = 16                      # independent constructed competitions; every estimate is their mean
SPLITS = 10                          # random public/private splits of each competition's test rows
N_TRAIN, N_TEST, N_FRESH = 800, 2000, 20000
PUBLIC_SHARE = 0.20                  # public leaderboard rows: 400 of the 2,000 test rows
N_BASE, N_CANDIDATES, N_USEFUL, USEFUL_WEIGHT = 6, 100, 2, 0.25
LOGGED = [3, 10, 30, 100]            # numbers of logged versions the control can take


def generate(n, rng, w_base, w_cand):
    """Six base columns and 100 candidate columns. Only two candidates truly help, and only a little."""
    base = rng.normal(size=(n, N_BASE))
    cand = rng.normal(size=(n, N_CANDIDATES))
    p = 1 / (1 + np.exp(-(base @ w_base + cand @ w_cand)))
    return np.column_stack([base, cand]), (rng.random(n) < p).astype(int)


def row_loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def one_competition(seed):
    """Version 0 is the baseline. Version k adds candidate column k and refits. Returns every version's
    per-row test loss and its mean loss on a large fresh sample (the truth)."""
    rng = np.random.default_rng(seed)
    w_base = rng.normal(0, 0.6, N_BASE)
    w_cand = np.zeros(N_CANDIDATES)
    w_cand[rng.choice(N_CANDIDATES, N_USEFUL, replace=False)] = USEFUL_WEIGHT
    X, y = generate(N_TRAIN, rng, w_base, w_cand)
    X_test, y_test = generate(N_TEST, rng, w_base, w_cand)
    X_fresh, y_fresh = generate(N_FRESH, rng, w_base, w_cand)
    test_loss = np.zeros((max(LOGGED), N_TEST))
    fresh_loss = np.zeros(max(LOGGED))
    for k in range(max(LOGGED)):
        cols = list(range(N_BASE)) + ([] if k == 0 else [N_BASE + k - 1])
        model = LogisticRegression(max_iter=200).fit(X[:, cols], y)
        test_loss[k] = row_loss(model.predict_proba(X_test[:, cols])[:, 1], y_test)
        fresh_loss[k] = row_loss(model.predict_proba(X_fresh[:, cols])[:, 1], y_fresh).mean()
    # Each split draws a public sample; the winner is the version with the lowest public loss among the first N.
    gains = {n: [] for n in LOGGED}
    for _ in range(SPLITS):
        public = np.zeros(N_TEST, bool)
        public[rng.choice(N_TEST, int(PUBLIC_SHARE * N_TEST), replace=False)] = True
        pub_loss, priv_loss = test_loss[:, public].mean(1), test_loss[:, ~public].mean(1)
        for n in LOGGED:
            k = int(np.argmin(pub_loss[:n]))
            # gain = baseline loss minus the winner's loss, so positive means the winner looks better
            gains[n].append([pub_loss[0] - pub_loss[k], priv_loss[0] - priv_loss[k], fresh_loss[0] - fresh_loss[k]])
    return {n: np.mean(g, axis=0) for n, g in gains.items()}, {n: np.mean(np.array(g)[:, 2] > 0) for n, g in gains.items()}


def run(versions):
    studies = [one_competition(SEED * 1000 + i) for i in range(REPLICATES)]
    curve = {}
    for n in LOGGED:
        g = np.array([s[0][n] for s in studies])           # replicates x (public, private, fresh)
        curve[n] = {"public": g[:, 0].mean(), "private": g[:, 1].mean(), "fresh": g[:, 2].mean(),
                    "public_sd": g[:, 0].std(), "private_sd": g[:, 1].std(),
                    "truly_better": float(np.mean([s[1][n] for s in studies])),
                    "per_study_public": g[:, 0], "per_study_private": g[:, 1]}
    return clean({"versions": versions, "selected": curve[versions],
                  "curve": [{"versions": n, **curve[n]} for n in LOGGED],
                  "replicates": REPLICATES, "splits": SPLITS, "public_rows": int(PUBLIC_SHARE * N_TEST),
                  "private_rows": N_TEST - int(PUBLIC_SHARE * N_TEST), "fresh_rows": N_FRESH})
```

Book location: Chapter 2, Preserve the Comparison. Constructed example: sixteen seeded synthetic competitions and logistic regressions, measured by the chapter activity.
