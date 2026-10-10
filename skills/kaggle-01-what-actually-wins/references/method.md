# Chapter 1: What Actually Wins on Kaggle

**When the model gets more capacity on the same columns, which assessment rewards the upgrade, and which one tracks the score on subjects the model never saw?**

A competitor who suspects the algorithm is the problem will try a bigger model. If the split does not match the test, the experiment cannot say whether the bigger model helped, and the choice made from it can be the wrong one.

## The experiment

Eight constructed studies of a sensor task. Each has 36 subjects with 40 windows each. Three noisy activity channels carry a signal shared by everyone; four more channels are a fixed fingerprint of the subject, and every subject has its own label bias. Boosted trees of the capacity chosen by the control (leaves per tree) are scored by AUC three ways: random 5-fold on the windows, 5-fold with whole subjects held out, and a fresh set of 150 new subjects. Scores pool all held-out predictions per study and average over the studies; the model with 2 leaves per tree is the comparison.

Control: Model capacity (leaves per tree) (2: one split per tree, 8, 31: deep trees; default 31).

## Measured results

| Measure | 2: one split per tree | 8 | 31: deep trees |
|---|---|---|---|
| Random-fold AUC | 0.707 | 0.746 | 0.735 |
| Subject-fold AUC | 0.585 | 0.581 | 0.576 |
| New-subject AUC | 0.625 | 0.598 | 0.585 |
| Random-fold optimism | +0.082 | +0.147 | +0.149 |
| Upgrade effect, new subjects | +0.000 | -0.027 | -0.040 |

## What the result says (default, model capacity (leaves per tree) = 31)

With 31 leaves per tree, random folds report 0.735 and the score on new subjects is 0.585, so 0.735 - 0.585 = 0.149 of optimism. Subject-held-out folds report 0.576, -0.009 from the new-subject score. Against the 2-leaf model the upgrade moves random folds by +0.027, subject folds by -0.008 and new subjects by -0.040. The upgrade looks like progress in random folds and is a loss on new subjects.

- Random folds: 0.735 - 0.585 = +0.149 (estimate minus new subjects).
- Subject-held-out folds: 0.576 - 0.585 = -0.009.
- Upgrade effect on random folds: 0.735 - 0.707 = +0.027.
- Upgrade effect on new subjects: 0.585 - 0.625 = -0.040.

## Apply it to a competition

- Write the contract before the next run: what will a new row be, what inputs exist at prediction time, and how is it scored.
- Split development data by that unit (subject, entity or date block) before comparing any two models.
- If a random-fold gain and a held-out-unit result disagree, trust the held-out-unit result and look for columns that identify the unit.
- Treat a suspiciously large random-fold gain from added capacity as a question about the split, not as proof the model fits the task better.

## Assumptions and limits

Constructed data and a boosted-tree model from scikit-learn standing in for a gradient-boosting library. The fingerprint channels are built to be useless for new subjects, which is the chapter's familiar-subject case. In this generator the simplest model scores highest on new subjects; with different data a larger model can help there too, and the lesson is that only a matching split can tell. The subject-fold estimate is noisy and sits below the new-subject score (by 0.041 at 2 leaves): each fold model trains on fewer subjects, and the pooled predictions mix five fold models whose overall level differs.

Constructed data. The fingerprint is built to carry no information about new subjects, and the sizes of these effects are properties of this generator, not a competition result.

## Reproduce it

The chapter notebook `notebooks/01-what-actually-wins.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch01` (`run`, `explain`, `draw`).

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

from kaggle_companion.activities._common import clean

SEED = 1
REPLICATES = 8                  # independent constructed studies; every estimate is their mean
N_SUBJECTS, ROWS_EACH = 36, 40  # development data: 36 subjects with 40 sensor windows each
NEW_SUBJECTS, NEW_ROWS = 150, 8 # the real test: 150 subjects never seen in development
ACTIVITY_WEIGHTS = np.array([0.9, -0.7, 0.5])
BASELINE_LEAVES = 2             # the model before the "upgrade": one split per tree


def generate(n_subjects, rows_each, rng):
    """Sensor windows. Three noisy activity channels carry a shared signal. Four more channels are a
    subject fingerprint (constant within a subject), and each subject also has its own label bias."""
    fingerprint = rng.normal(0, 1, (n_subjects, 4))
    bias = rng.normal(0, 1.5, n_subjects)
    subject = np.repeat(np.arange(n_subjects), rows_each)
    activity = rng.normal(size=(len(subject), 3))
    logit = activity @ ACTIVITY_WEIGHTS + bias[subject]
    y = (rng.random(len(subject)) < 1 / (1 + np.exp(-logit))).astype(int)
    X = np.column_stack([activity + rng.normal(0, 0.8, activity.shape),
                         fingerprint[subject] + rng.normal(0, 0.15, (len(subject), 4))])
    return X, y, subject


def make_model(leaves):
    return HistGradientBoostingClassifier(max_leaf_nodes=leaves, max_iter=40, learning_rate=0.2,
                                          min_samples_leaf=5, random_state=0)


def pooled_auc(leaves, X, y, splits):
    """Predict every row from a model that did not see it, then compute one AUC over all rows."""
    pred = np.zeros(len(y))
    for fit, out in splits:
        pred[out] = make_model(leaves).fit(X[fit], y[fit]).predict_proba(X[out])[:, 1]
    return roc_auc_score(y, pred)


def one_study(leaves, seed):
    rng = np.random.default_rng(seed)
    X, y, subject = generate(N_SUBJECTS, ROWS_EACH, rng)
    X_new, y_new, _ = generate(NEW_SUBJECTS, NEW_ROWS, rng)
    out = {}
    for name, size in (("model", leaves), ("baseline", BASELINE_LEAVES)):
        out[name] = {
            "random_folds": pooled_auc(size, X, y, KFold(5, shuffle=True, random_state=0).split(X)),
            "subject_folds": pooled_auc(size, X, y, GroupKFold(5).split(X, y, subject)),
            "new_subjects": roc_auc_score(y_new, make_model(size).fit(X, y).predict_proba(X_new)[:, 1]),
        }
    return out


def run(leaves):
    studies = [one_study(leaves, SEED * 1000 + i) for i in range(REPLICATES)]
    keys = ["random_folds", "subject_folds", "new_subjects"]
    mean = lambda arm, k: float(np.mean([s[arm][k] for s in studies]))
    return clean({
        "leaves": leaves,
        "model": {k: mean("model", k) for k in keys},
        "baseline": {k: mean("baseline", k) for k in keys},
        "per_study": {k: [s["model"][k] for s in studies] for k in keys},
        "replicates": REPLICATES, "subjects": N_SUBJECTS, "rows_each": ROWS_EACH,
        "new_subjects_count": NEW_SUBJECTS, "baseline_leaves": BASELINE_LEAVES,
    })
```

Book location: Chapter 1, Start with the Prediction Contract. Constructed example: eight seeded synthetic subject studies and boosted trees, measured by the chapter activity.
