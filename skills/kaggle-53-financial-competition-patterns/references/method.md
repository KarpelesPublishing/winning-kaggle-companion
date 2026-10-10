# Chapter 53: Financial Competition Patterns

**When does removing factor exposure from the features help the score, and when does it cost correlation?**

A feature that carries market exposure is legal, but whether it helps depends on the scored target and on whether the factor premium seen in training continues. Neutralizing is a candidate representation to compare under the same metric, not a rule.

## The experiment

Eight constructed markets of 120 assets. Five features each mix a hidden alpha signal with the asset's market exposure (beta). Returns are a small alpha term, beta times a daily factor return, and noise. Ridge regression is trained on 200 dates while the factor premium averages 0.5, then scored on 200 later dates in which the premium is the control. Features are neutralized by a per-date cross-sectional regression on beta, removing 0%, 50% or 100% of the exposure. Each model is scored by the mean per-date Spearman correlation against the raw returns and against the returns left after removing the day's beta effect.

Control: Mean factor return in the scored period (training period: 0.5) (0.5: premium persists, 0.25: half as strong, 0: premium gone, -0.5: premium flips sign; default -0.5).

## Measured results

| Measure | 0.5: premium persists | 0.25: half as strong | 0: premium gone | -0.5: premium flips sign |
|---|---|---|---|---|
| Raw features, raw returns | 0.274 | 0.165 | 0.045 | -0.186 |
| Neutralized, raw returns | 0.063 | 0.065 | 0.066 | 0.061 |
| Raw features, neutral returns | 0.055 | 0.055 | 0.055 | 0.055 |
| Neutralized, neutral returns | 0.072 | 0.072 | 0.072 | 0.072 |
| Cost of neutralizing (raw returns) | +0.211 | +0.100 | -0.021 | -0.247 |

## What the result says (default, mean factor return in the scored period (training period: 0.5) = -0.5)

With a premium of -0.50 in the scored period (0.5 in training), raw features score -0.186 against raw returns and fully neutralized features 0.061, so 0.061 - (-0.186) = 0.247 is what neutralizing gains there. Against factor-neutral returns the scores are 0.055 and 0.072. The daily correlation on raw returns has a standard deviation of 0.230 with raw features and 0.082 with neutralized ones.

- Cost of full neutralization on raw returns: -0.186 - 0.061 = -0.247.
- Gain on factor-neutral returns: 0.072 - 0.055 = +0.017.
- Half neutralization on raw returns: -0.087, between the two ends.
- Daily swing on raw returns, raw features minus neutralized: 0.230 - 0.082 = +0.148.

## Apply it to a competition

- Find out whether the official score is on raw returns or on a factor-neutral or residual target before neutralizing anything.
- Compare the raw and neutralized versions of the same model on the exact metric, on later periods, and keep the unneutralized comparator.
- Look at the per-date scores as well as the mean: a model whose daily correlation swings with the factor is betting on the premium.
- Never alter the requested target to follow a recipe; neutralize inputs or predictions only when the comparison supports it.

## Assumptions and limits

Constructed markets with one factor, a static beta per asset and an alpha signal that does not change; Ridge regression stands in for any model and eight markets give the means. Exposure here is observed without error, which is generous: with a noisy exposure estimate, neutralization would remove less. The premium in the scored period is set by hand to show the regimes. The result concerns the scoring target, and is not a claim about any real competition or about trading profit.

Constructed data with a strong factor; the sizes are properties of this generator. Neutralization gains on the factor-neutral target are small here (about 0.02); the large effects are the regime-dependent ones on raw returns.

## Reproduce it

The chapter notebook `notebooks/53-financial-competition-patterns.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch53` (`run`, `explain`, `draw`).

```python
import numpy as np
from sklearn.linear_model import Ridge

from kaggle_companion.activities._common import clean

SEED = 53
REPLICATES = 8                  # independent constructed markets; every estimate is their mean
ASSETS, TRAIN_DATES, TEST_DATES = 120, 200, 200
LOAD = np.array([0.8, 0.6, 0.5, 0.4, 0.3])      # how strongly each feature carries the market exposure (beta)
SIGNAL = np.array([0.5, 0.4, 0.4, 0.3, 0.3])    # how strongly each feature carries the true alpha signal
TRAIN_PREMIUM = 0.5             # mean daily factor return while the model is trained
FRACTIONS = [0.0, 0.5, 1.0]     # share of the exposure removed from each feature


def make_market(rng, beta, dates, premium):
    """Features, exposures and next-day returns for `dates` dates. A common factor return moves every asset by its beta."""
    alpha = rng.normal(size=(dates, ASSETS))                       # hidden idiosyncratic signal
    exposure = np.tile(beta, (dates, 1))
    X = np.stack([SIGNAL[k] * alpha + LOAD[k] * exposure + rng.normal(size=(dates, ASSETS)) for k in range(len(LOAD))], axis=-1)
    factor = rng.normal(premium, 0.5, size=(dates, 1))             # the factor premium: mean `premium`, daily noise 0.5
    ret = 0.06 * alpha + 0.5 * exposure * factor + 0.5 * rng.normal(size=(dates, ASSETS))
    return X, exposure, ret


def neutralize(X, exposure, fraction):
    """Cross-sectional regression per date: subtract `fraction` of each feature's projection on the exposure."""
    b = exposure - exposure.mean(axis=1, keepdims=True)
    slope = ((X - X.mean(axis=1, keepdims=True)) * b[..., None]).sum(axis=1) / (b ** 2).sum(axis=1, keepdims=True)
    return X - fraction * b[..., None] * slope[:, None, :]


def factor_neutral(ret, exposure):
    """The return left after the day's regression on exposure: what a factor-neutral metric scores."""
    b = exposure - exposure.mean(axis=1, keepdims=True)
    centered = ret - ret.mean(axis=1, keepdims=True)
    return centered - b * ((b * centered).sum(axis=1, keepdims=True) / (b ** 2).sum(axis=1, keepdims=True))


def ranks(a):
    """Per-date ranks scaled to [-0.5, 0.5]."""
    return (a.argsort(axis=1).argsort(axis=1) + 0.5) / a.shape[1] - 0.5


def date_correlations(pred, target):
    """Spearman correlation between prediction and target on each date."""
    p, t = ranks(pred), ranks(target)
    return (p * t).sum(axis=1) / np.sqrt((p ** 2).sum(axis=1) * (t ** 2).sum(axis=1))


def one_market(test_premium, replicate):
    rng = np.random.default_rng(SEED * 100 + replicate)
    beta = rng.normal(size=ASSETS)
    X_train, e_train, r_train = make_market(rng, beta, TRAIN_DATES, TRAIN_PREMIUM)
    X_test, e_test, r_test = make_market(rng, beta, TEST_DATES, test_premium)
    targets = {"raw": r_test, "neutral": factor_neutral(r_test, e_test)}
    out = {}
    for fraction in FRACTIONS:
        # Features are neutralized with the same rule in training and in the later period; the target is never altered.
        model = Ridge(alpha=100).fit(neutralize(X_train, e_train, fraction).reshape(-1, len(LOAD)), ranks(r_train).reshape(-1))
        pred = model.predict(neutralize(X_test, e_test, fraction).reshape(-1, len(LOAD))).reshape(TEST_DATES, ASSETS)
        for name, target in targets.items():
            c = date_correlations(pred, target)
            out[(fraction, name)] = (float(c.mean()), float(c.std()))
    return out


def run(test_premium):
    markets = [one_market(test_premium, i) for i in range(REPLICATES)]
    score = {}
    for name in ("raw", "neutral"):
        score[name] = {str(f): {"mean": float(np.mean([m[(f, name)][0] for m in markets])),
                               "daily_sd": float(np.mean([m[(f, name)][1] for m in markets])),
                               "per_market": [m[(f, name)][0] for m in markets]} for f in FRACTIONS}
    return clean({"test_premium": test_premium, "score": score, "train_premium": TRAIN_PREMIUM,
                  "replicates": REPLICATES, "assets": ASSETS, "test_dates": TEST_DATES})
```

Book location: Chapter 53, Feature Neutralization: Removing Market Exposure. Constructed example: eight seeded synthetic markets of 120 assets and a Ridge model, measured by the chapter activity.
