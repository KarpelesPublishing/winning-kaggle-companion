# Chapter 26: Advanced Target Encoding

**How much smoothing does a target encoding need, and does the answer change with how strongly categories really differ?**

The chapter says to start from the smoothed mean and reach for a richer encoder only when it supplies a missing function. Measuring the cost of too little and too much smoothing shows whether the simple default is enough and what the richer rules would have to beat.

## The experiment

Thirty constructed binary tasks per setting with 300 categories of Zipf-like frequency (many categories have fewer than 5 of the 6,000 training rows), two numeric features and true category effects with the control's standard deviation. Each pipeline cross-fits the encoded training column in 5 folds, fits a logistic regression and encodes 20,000 new rows with a map fitted on all training rows. Smoothed means with pseudocount m from 0 to 300 are compared with the chapter's Weight of Evidence at smoothing 0.5, by log loss on all new rows and on those whose category has fewer than 5 training rows.

Control: True spread of category effects (standard deviation, log odds) (0.2: categories barely differ, 0.5, 1.0, 1.5: categories differ strongly; default 1.0).

## Measured results

| Measure | 0.2: categories barely differ | 0.5 | 1.0 | 1.5: categories differ strongly |
|---|---|---|---|---|
| Best pseudocount m | 100 | 30 | 10 | 3 |
| Log loss at best m | 0.630 | 0.619 | 0.580 | 0.527 |
| Log loss with no smoothing | 0.631 | 0.632 | 0.619 | 0.580 |
| Log loss at m = 10 | 0.630 | 0.620 | 0.580 | 0.531 |
| Weight of evidence | 0.631 | 0.624 | 0.584 | 0.530 |

## What the result says (default, true spread of category effects (standard deviation, log odds) = 1.0)

With category spread 1.0, the lowest new-row log loss is 0.580 at m = 10. No smoothing scores 0.619, so 0.619 - 0.580 = 0.039 is what skipping smoothing costs; the Chapter 9 default m = 10 is the best setting here. On rare categories (6.7% of new rows) no smoothing scores 0.695 against 0.621 at m = 10. Weight of evidence at smoothing 0.5 scores 0.584.

- Best pseudocount: m = 10 with log loss 0.580.
- No smoothing: 0.619 - 0.580 = 0.039 worse than the best.
- Default m = 10: 0.580 - 0.580 = 0.000 worse than the best.
- Rare categories: m = 0 gives 0.695, m = 10 gives 0.621; weight of evidence 0.640.

## Apply it to a competition

- Start from the smoothed mean and pick m by inner validation, never m = 0 for a category with few rows.
- Cross-fit the training column and map new rows with a table fitted on all training rows, as Chapter 9 requires; return and store the map.
- Report log loss for rare and common categories separately: the gain from shrinkage sits in the rare ones.
- Replace the baseline with Weight of Evidence, a mixed model or a native encoder only if it beats the best smoothed mean on untouched rows.

## Assumptions and limits

Constructed data with normally distributed category effects, a logistic model and cross-fitted columns. Weight of Evidence is the chapter's fit_woe formula with smoothing 0.5. No James-Stein, mixed-model or native-categorical encoder is run, so this does not rank those. When categories barely differ the differences between m values are within about 0.002 log loss and should not be over-read.

Constructed data. The best m depends on the generator's frequencies and spread; the pattern (best m falls as the spread rises, no smoothing hurts rare categories) is the claim, not the particular values.

## Reproduce it

The chapter notebook `notebooks/26-advanced-target-encoding.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch26` (`run`, `explain`, `draw`).

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

from kaggle_companion.activities._common import clean

SEED = 26
REPLICATES = 30       # independent constructed datasets per setting; every number is their mean
K, N_TRAIN, N_NEW = 300, 6000, 20000
PSEUDOCOUNTS = [0, 1, 3, 10, 30, 100, 300]   # smoothing strength m; 0 means no smoothing
WOE_SMOOTH = 0.5      # the chapter's Weight of Evidence default
RARE = 5              # a category with fewer than 5 development rows is rare


def generate(spread, seed):
    """300 categories with Zipf-like frequencies, so a few are common and many are rare. True category effects ~ Normal(0, spread)."""
    rng = np.random.default_rng(seed)
    effect = rng.normal(0, spread, K)
    freq = 1 / np.arange(1, K + 1)
    freq /= freq.sum()
    cat, cat_new = rng.choice(K, N_TRAIN, p=freq), rng.choice(K, N_NEW, p=freq)

    def rows(c):
        x = rng.normal(size=(len(c), 2))
        logit = 0.6 * x[:, 0] - 0.4 * x[:, 1] + effect[c] - 0.3
        return x, (rng.random(len(c)) < 1 / (1 + np.exp(-logit))).astype(int)

    (x, y), (x_new, y_new) = rows(cat), rows(cat_new)
    return cat, x, y, cat_new, x_new, y_new


def smoothed_mean(cat, y, m):
    """The chapter 9 encoding e_c = (s_c + m*mu) / (n_c + m); with m = 0 it is each category's own mean (the global mean if unseen)."""
    mu = y.mean()
    s, n = np.bincount(cat, weights=y, minlength=K), np.bincount(cat, minlength=K)
    return (s + m * mu) / (n + m) if m > 0 else np.where(n > 0, s / np.maximum(n, 1), mu)


def weight_of_evidence(cat, y, smooth=WOE_SMOOTH):
    """The chapter's fit_woe and transform_woe: log of (smoothed share among positives) over (smoothed share among negatives).

    As in the chapter, the pseudocount goes to every category seen in fitting (k_seen*smooth in each denominator), and a
    category unseen in fitting gets the neutral value 0."""
    seen = np.bincount(cat, minlength=K) > 0
    k_seen = seen.sum()
    pos = np.bincount(cat, weights=y, minlength=K) + smooth
    neg = np.bincount(cat, weights=1 - y, minlength=K) + smooth
    woe = np.log((pos / (y.sum() + k_seen * smooth)) / (neg / ((1 - y).sum() + k_seen * smooth)))
    return np.where(seen, woe, 0.0)


def to_logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def evaluate(data, encoder, transform=lambda v: v):
    """The deployable pipeline: cross-fit the training column, fit the model, encode new rows with the map from all training rows."""
    cat, x, y, cat_new, x_new, y_new = data
    column = np.zeros(len(cat))
    for fit, out in KFold(5, shuffle=True, random_state=2).split(cat):
        column[out] = encoder(cat[fit], y[fit])[cat[out]]
    full_map = encoder(cat, y)
    model = LogisticRegression(C=10, max_iter=500).fit(np.column_stack([x, transform(column)]), y)
    p = model.predict_proba(np.column_stack([x_new, transform(full_map[cat_new])]))[:, 1]
    rare = np.bincount(cat, minlength=K)[cat_new] < RARE
    return [log_loss(y_new, p), log_loss(y_new[rare], p[rare], labels=[0, 1]), log_loss(y_new[~rare], p[~rare], labels=[0, 1]), rare.mean()]


def run(category_spread):
    sets = [generate(category_spread, SEED * 1000 + i) for i in range(REPLICATES)]
    by_m = {str(m): np.mean([evaluate(d, lambda c, y, m=m: smoothed_mean(c, y, m), to_logit) for d in sets], axis=0) for m in PSEUDOCOUNTS}
    woe = np.mean([evaluate(d, weight_of_evidence) for d in sets], axis=0)
    best = min(PSEUDOCOUNTS, key=lambda m: by_m[str(m)][0])
    return clean({
        "category_spread": category_spread, "replicates": REPLICATES, "categories": K, "training_rows": N_TRAIN, "new_rows": N_NEW,
        "rare_share_of_new_rows": float(woe[3]), "best_m": best,
        "overall": {str(m): by_m[str(m)][0] for m in PSEUDOCOUNTS}, "rare": {str(m): by_m[str(m)][1] for m in PSEUDOCOUNTS},
        "common": {str(m): by_m[str(m)][2] for m in PSEUDOCOUNTS},
        "woe": {"overall": woe[0], "rare": woe[1], "common": woe[2]},
    })
```

Book location: Chapter 26, Which One to Use When. Constructed example: thirty seeded synthetic tasks per setting with Zipf-like category frequencies and a logistic regression, measured by the chapter activity.
