# Chapter 19: March Mania: Prediction at Scale

**If the team history quietly includes later seasons, would the development score tell you, and would the label-flip test?**

A leave-one-season-out feature build is an easy mistake and rarely inflates the score by much. The chapter's test, reversing the later labels and checking that nothing earlier moves, detects it exactly. Measuring both shows why the chapter relies on the test.

## The experiment

Forty constructed fixtures from the chapter's generator (12 teams, 8 seasons of round-robin games, drifting strengths), plus the chapter's own seed 42. Each pipeline fits logistic regression and a small boosted tree on seasons 1 to 4, selects among them and a constant 0.5 by Brier loss on seasons 5 and 6, refits on seasons 1 to 6 and forecasts seasons 7 and 8 using only legal history. The control adds that many later seasons to every team's history when the features are built, as a leave-one-season-out build would. At 0 the pipeline is the chapter's. The label-flip test reverses every season 7 and 8 outcome and measures how much the selected recipe's development Brier moves.

Control: Later seasons included in each game's team history (0: legal, past-only history, 1: one later season, 2: two later seasons, 3: three later seasons; default 1).

## Measured results

| Measure | 0: legal, past-only history | 1: one later season | 2: two later seasons | 3: three later seasons |
|---|---|---|---|---|
| Development estimate | 0.2282 | 0.2257 | 0.2241 | 0.2248 |
| Forecast of seasons 7-8 | 0.2377 | 0.2359 | 0.2365 | 0.2377 |
| Label-flip change | 0.0000 | 0.0038 | 0.0110 | 0.0133 |
| Recipe changed by flip | 0.000 | 0.075 | 0.175 | 0.200 |
| Chapter fixture, logistic dev Brier | 0.217400 | 0.210300 | 0.212700 | 0.215200 |

## What the result says (default, later seasons included in each game's team history = 1)

With 1 later season(s) in the history, the development estimate averages 0.2257 and the forecast of seasons 7 and 8 scores 0.2359, so 0.2359 - 0.2257 = 0.0102. Against the legal pipeline the estimate moves -0.0026 (standard error 0.0004). The label-flip test fails: reversing the season 7 and 8 labels moves the development Brier by 0.0038 on average and changes the selected recipe in 0.075 of fixtures. The chapter's own fixture moves by 0.0034.

- Development estimate: 0.2257; forecast of seasons 7 and 8: 0.2359.
- Gap: 0.2359 - 0.2257 = 0.0102 (later seasons are harder here: best possible 0.216 then 0.220).
- Estimate against the legal pipeline: -0.0026.
- Label-flip change in the development Brier: 0.0038 - 0 = 0.0038 (a legal pipeline gives 0).

## Apply it to a competition

- Build features for every row from only the outcomes revealed before that row's forecast time; never build them once from the whole table.
- Run the label-flip test after each pipeline change: reverse the assessment labels and confirm that development scores and the selected recipe stay identical.
- Do not use the gap between development and later scores to detect leakage; seasons differ in difficulty and the gap is noisy.
- Report the development-to-assessment difference in the handover record, as the chapter does, rather than hiding it behind the development score.

## Assumptions and limits

Constructed fixtures in which team strengths drift and the spread of strengths shrinks, so later seasons are inherently harder; two fixed candidates and a single feature. In a real task the size of a leak depends on how much the later outcomes say about the earlier rows. The flip test measures a dependency, not its cost.

Constructed data. The leak here is small and the activity says so: the chapter's claim is that the boundary can be tested exactly, not that leaked history ruins a score.

## Reproduce it

The chapter notebook `notebooks/19-march-mania-full-walkthrough.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch19` (`run`, `explain`, `draw`).

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from kaggle_companion.activities._common import clean

SEED = 19
FIXTURES = 40       # independent constructed seasons-of-games; every estimate is their mean
CHAPTER_SEED = 42   # the chapter's own fixture, reported separately
TEAMS = 12


def fixture(seed):
    """Eight seasons of round-robin games between 12 teams whose strengths drift (the chapter's generator)."""
    rng = np.random.default_rng(seed)
    strength = rng.normal(size=TEAMS)
    season, a, b, y, p = [], [], [], [], []
    for s in range(1, 9):
        strength = 0.85 * strength + rng.normal(0, 0.25, size=TEAMS)
        for i in range(TEAMS):
            for j in range(i + 1, TEAMS):
                prob = 1 / (1 + np.exp(-(strength[i] - strength[j])))
                season.append(s); a.append(i); b.append(j); p.append(prob); y.append(int(rng.random() < prob))
    return {"season": np.array(season), "a": np.array(a), "b": np.array(b), "y": np.array(y), "p": np.array(p)}


def rate_difference(fx, leak):
    """Smoothed win rate of team A minus team B, from the history visible at the start of each game's season.

    Legal history is every earlier season. `leak` adds that many LATER seasons, as a leave-one-season-out
    feature build would. leak = 0 reproduces the chapter's pipeline."""
    season, a, b, y = fx["season"], fx["a"], fx["b"], fx["y"]
    out = np.zeros(len(y))
    for s in range(1, 9):
        use = (season < s) | ((season > s) & (season <= s + leak))
        wins, games = np.zeros(TEAMS), np.zeros(TEAMS)
        np.add.at(wins, a[use], y[use]); np.add.at(wins, b[use], 1 - y[use])
        np.add.at(games, a[use], 1); np.add.at(games, b[use], 1)
        rate = (wins + 1) / (games + 2)                 # Beta(1, 1) smoothing, as in the chapter
        here = season == s
        out[here] = rate[a[here]] - rate[b[here]]
    return out


def candidates():
    """The chapter's two fixed candidates; a constant 0.5 is the third."""
    return {"logistic": LogisticRegression(C=1.0, random_state=42, max_iter=1000),
            "small_tree": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, l2_regularization=1.0,
                                                         early_stopping=False, random_state=42)}


def pipeline(fx, leak):
    """Fit on seasons 1-4, select on 5-6 by Brier, refit on 1-6, then forecast 7-8 with legal history only."""
    x = rate_difference(fx, leak)[:, None]
    season, y = fx["season"], fx["y"]
    train, dev, fit, test = season <= 4, (season == 5) | (season == 6), season <= 6, season >= 7
    scores = {"constant": brier_score_loss(y[dev], np.full(dev.sum(), 0.5))}
    for name, model in candidates().items():
        scores[name] = brier_score_loss(y[dev], model.fit(x[train], y[train]).predict_proba(x[dev])[:, 1])
    order = ["constant", "logistic", "small_tree"]
    selected = min(order, key=lambda n: (scores[n], order.index(n)))
    if selected == "constant":
        p = np.full(test.sum(), 0.5)
    else:   # at forecast time only legal history exists, whatever history built the training rows
        p = candidates()[selected].fit(x[fit], y[fit]).predict_proba(rate_difference(fx, 0)[test][:, None])[:, 1]
    return scores, selected, brier_score_loss(y[test], p)


def flip_assessment_labels(fx):
    """The chapter's test: reverse every season 7 and 8 outcome and see whether anything earlier moves."""
    return {**fx, "y": np.where(fx["season"] >= 7, 1 - fx["y"], fx["y"])}


def run(leaked_seasons):
    fixtures = [fixture(SEED * 1000 + i) for i in range(FIXTURES)]
    estimate, delivered, change, reselected, shift = [], [], [], [], []
    for fx in fixtures:
        scores, selected, outcome = pipeline(fx, leaked_seasons)
        legal_scores, legal_selected, _ = pipeline(fx, 0)
        flipped_scores, flipped_selected, _ = pipeline(flip_assessment_labels(fx), leaked_seasons)
        estimate.append(scores[selected])
        delivered.append(outcome)
        change.append(abs(flipped_scores[selected] - scores[selected]))   # selected recipe's development Brier
        reselected.append(flipped_selected != selected)
        shift.append(scores[selected] - legal_scores[legal_selected])
    ref = fixture(CHAPTER_SEED)
    ref_scores, ref_selected, ref_outcome = pipeline(ref, leaked_seasons)
    ref_flip = pipeline(flip_assessment_labels(ref), leaked_seasons)[0]
    dev, test = np.isin(fixtures[0]["season"], [5, 6]), fixtures[0]["season"] >= 7
    best = lambda m: float(np.mean([(f["p"][m] * (1 - f["p"][m])).mean() for f in fixtures]))   # Brier of the true probabilities
    return clean({
        "leaked_seasons": leaked_seasons, "fixtures": FIXTURES,
        "estimate": float(np.mean(estimate)), "delivered": float(np.mean(delivered)),
        "best_possible_development": best(dev), "best_possible_assessment": best(test),
        "estimate_shift": float(np.mean(shift)), "estimate_shift_se": float(np.std(shift) / np.sqrt(FIXTURES)),
        "flip_change": float(np.mean(change)), "flip_change_per_fixture": change,
        "recipe_changed": float(np.mean(reselected)),
        "chapter_fixture": {"scores": ref_scores, "selected": ref_selected, "delivered": ref_outcome,
                            "flip_change": abs(ref_flip[ref_selected] - ref_scores[ref_selected])},
    })
```

Book location: Chapter 19, Run the Complete Constructed Forecast. Constructed example: forty seeded synthetic fixtures from the chapter's generator, logistic regression and a boosted tree, measured by the chapter activity.
