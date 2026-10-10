# Chapter 3: Building a CV That Matches the Test Set

**When the relationship between features and outcomes changes from season to season, how far does each validation scheme's Brier score sit from the score on the next season?**

Leave-one-season-out is a respectable group assessment, but its training set contains seasons from after the held-out one. For a competition that forecasts a future season, the estimate you use to choose models then describes a different question from the one the leaderboard asks.

## The experiment

A constructed league of 8 seasons with 400 games each and 4 features. A logistic model gives win odds from season-specific weights; the weights take a random step from one season to the next, and the size of the step is the control (0 means no drift). Seasons 1 to 7 are development. Both schemes score seasons 5 to 7 (1,200 games, pooled into one Brier score): leave-one-season-out trains on the other six development seasons, past-only trains on earlier seasons only. The reference is the final recipe, fitted on all seven development seasons, scored on 4,000 fresh games of season 8. Every number is the mean of 100 independent histories.

Control: Season-to-season drift (standard deviation of the weight step) (0: no drift, 0.25: mild, 0.5: moderate, 1.0: strong; default 0.5).

## Measured results

| Measure | 0: no drift | 0.25: mild | 0.5: moderate | 1.0: strong |
|---|---|---|---|---|
| Leave-one-season-out Brier | 0.159 | 0.167 | 0.184 | 0.217 |
| Past-only Brier | 0.159 | 0.170 | 0.190 | 0.229 |
| Next-season Brier | 0.159 | 0.173 | 0.196 | 0.235 |
| Optimism, leave-one-season-out | 0.000 | 0.006 | 0.012 | 0.017 |
| Optimism, past-only | 0.000 | 0.003 | 0.006 | 0.006 |

## What the result says (default, season-to-season drift (standard deviation of the weight step) = 0.5)

At drift 0.5 the final recipe scores a Brier of 0.196 on the next season. Leave-one-season-out reports 0.184, so 0.196 - 0.184 = 0.012 of optimism. Past-only reports 0.190, 0.196 - 0.190 = 0.006. Leave-one-season-out is the more optimistic scheme. Leave-one-season-out was the more optimistic in 85% of 100 histories.

- Leave-one-season-out: 0.196 - 0.184 = 0.012 (next season minus estimate).
- Past-only: 0.196 - 0.190 = 0.006.
- Difference between the schemes: 0.190 - 0.184 = 0.006 of Brier.
- Share of histories where leave-one-season-out was more optimistic: 85%.

## Apply it to a competition

- Write down whether the test is a future season, a new group or independent rows before choosing a splitter.
- For a future season, train only on earlier seasons whose labels would have been revealed, and report which seasons receive predictions.
- Label leave-one-season-out as a historical grouped check, and compare it with the past-only estimate: a large difference points to drift.
- Expect even past-only validation to be somewhat optimistic when the relationship moves, and keep a margin before trusting a small gain.

## Assumptions and limits

Constructed league with a random-walk relationship, one logistic model and 100 histories. Real drift may be abrupt, seasonal or absent, and past-only validation trains on only four to six seasons where leave-one-season-out trains on six. At no drift the two schemes are tied and both match the next-season score.

Constructed data with a random-walk relationship; the sizes are properties of this generator, not a competition result. Past-only is closer to the next season, not exact.

## Reproduce it

The chapter notebook `notebooks/03-building-cv-that-matches.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch03` (`run`, `explain`, `draw`).

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

from kaggle_companion.activities._common import clean

SEED = 3
REPLICATES = 100      # independent constructed histories; every estimate is their mean
SEASONS, GAMES, FEATURES = 8, 400, 4
FIRST_SCORED = 4      # score seasons 5, 6 and 7 (0-based 4, 5, 6); season 8 is the future
FUTURE_GAMES = 4000


def generate(drift, seed):
    """Seven development seasons and a large season 8. The weights that turn features into win odds
    wander from season to season; `drift` is the size of the wander (their length stays constant)."""
    rng = np.random.default_rng(seed)
    weights = np.zeros((SEASONS, FEATURES))
    weights[0] = rng.normal(0, 1.0, FEATURES)
    for s in range(1, SEASONS):
        step = weights[s - 1] + rng.normal(0, drift, FEATURES)
        weights[s] = step * np.linalg.norm(weights[0]) / np.linalg.norm(step)

    def games(s, n):
        x = rng.normal(size=(n, FEATURES))
        win = rng.random(n) < 1 / (1 + np.exp(-x @ weights[s]))
        return x, win.astype(int)

    dev = [games(s, GAMES) for s in range(SEASONS - 1)]
    return dev, games(SEASONS - 1, FUTURE_GAMES)


def fit(seasons):
    X = np.vstack([s[0] for s in seasons])
    y = np.concatenate([s[1] for s in seasons])
    return LogisticRegression(max_iter=500).fit(X, y)


def brier(y, p):
    return float(np.mean((y - p) ** 2))


def one_history(drift, seed):
    dev, (X_next, y_next) = generate(drift, seed)
    held, loso, past = [], [], []
    for s in range(FIRST_SCORED, SEASONS - 1):
        X, y = dev[s]
        held.append(y)
        # Leave-one-season-out: train on every other development season, later ones included.
        loso.append(fit([d for k, d in enumerate(dev) if k != s]).predict_proba(X)[:, 1])
        # Past-only: train on the seasons before s, as a real forecast would.
        past.append(fit(dev[:s]).predict_proba(X)[:, 1])
    y = np.concatenate(held)
    # Truth: the final recipe (all seven development seasons) on the next season, 4,000 fresh games.
    truth = brier(y_next, fit(dev).predict_proba(X_next)[:, 1])
    return brier(y, np.concatenate(loso)), brier(y, np.concatenate(past)), truth


def run(drift):
    runs = np.array([one_history(drift, SEED * 1000 + i) for i in range(REPLICATES)])
    loso, past, truth = runs.T
    return clean({
        "drift": drift,
        "loso": float(loso.mean()), "past_only": float(past.mean()), "next_season": float(truth.mean()),
        # optimism = Brier on the next season minus the estimate (positive: the estimate looked better than reality)
        "optimism": {"loso": float((truth - loso).mean()), "past_only": float((truth - past).mean())},
        "per_history": {"loso": (truth - loso).tolist(), "past_only": (truth - past).tolist()},
        "loso_more_optimistic": float(np.mean((truth - loso) > (truth - past))),
        "replicates": REPLICATES, "scored_games": GAMES * (SEASONS - 1 - FIRST_SCORED), "future_games": FUTURE_GAMES,
    })
```

Book location: Chapter 3, March Mania as a Special Case. Constructed example: seeded synthetic seasons of games and a logistic model, measured by the chapter activity. The seasons are not NCAA results.
