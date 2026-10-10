# Chapter 41: Online Learning

**How much of an adaptive model's advantage over a frozen model survives as labels arrive later, and how wrong is a replay that ignores the delay?**

In a live stream the newest labels are not yet known. A replay that fits on the most recent rows by event time reports a gain the system cannot deliver; the chapter asks for release time to decide eligibility.

## The experiment

A constructed stream of 2,000 rows with three features and a linear signal whose coefficients are a fixed stable part plus a drifting part that is pulled back toward zero; label noise has variance 1. The control is the delay between a row and the release of its label. At each origin from step 300 the model predicts, scored by squared error. A frozen ridge is fitted once on the labels released before step 300; the rolling policy refits ridge on the latest released rows; the incremental policy updates a discounted ridge one released row at a time. The window length and discount are chosen on the first 500 origins and every score is reported on the later 1,200. A fourth policy fits the rolling window on the latest rows by event time, ignoring the delay; it is invalid. Results are means over 20 streams.

Control: Label delay (steps between a row and its label's release) (0, 10, 40, 100; default 40).

## Measured results

| Measure | 0 | 10 | 40 | 100 |
|---|---|---|---|---|
| Frozen | 2.565 | 2.571 | 2.623 | 2.704 |
| Rolling refit | 1.518 | 1.823 | 2.320 | 2.481 |
| Incremental | 1.455 | 1.730 | 2.174 | 2.328 |
| Invalid replay | 1.518 | 1.518 | 1.518 | 1.518 |
| Selected window | 30 rows | 34 rows | 126 rows | 213 rows |
| Rolling gain over frozen | 1.047 | 0.748 | 0.303 | 0.223 |

## What the result says (default, label delay (steps between a row and its label's release) = 40)

With labels released 40 steps after their rows, the frozen model scores 2.623, the rolling refit 2.320 (window 126 rows on average) and the incremental policy 2.174. The rolling gain over frozen is 2.623 - 2.320 = 0.303. The invalid replay scores 1.518, suggesting a gain of 1.105 over frozen; its optimism against the eligible rolling refit is 2.320 - 1.518 = 0.802. The noise floor is 1.000.

- Rolling gain over frozen: 2.623 - 2.320 = 0.303.
- Incremental gain over frozen: 2.623 - 2.174 = 0.449.
- Optimism of the invalid replay: 2.320 - 1.518 = 0.802.
- Rolling error above the noise floor: 2.320 - 1.000 = 1.320.

## Apply it to a competition

- Write the timeline first: observation time, prediction origin and label-release time, with eligibility defined as release before origin.
- Replay frozen, incremental and rolling policies on the same stream; if a policy only wins when it can see unreleased labels, it has not won.
- Select window length and update schedule on earlier origins and report on later ones; expect longer windows as delay grows.
- Report the gain over a frozen model at the delay of the real system, not at zero delay.

## Assumptions and limits

Constructed linear stream with stable coefficients plus mean-reverting drift, ridge models and fixed settings grids. The frozen model is fitted once on the first 300 steps' released labels. Delays are in steps; the shape of the curves depends on the drift speed relative to the delay.

Constructed stream; the incremental policy beat the rolling refit at every delay here, a property of this drift and these settings grids. At a delay of 100 the rolling refit's edge over the frozen model is about as large as the variation between seed bases (under one other seed base it reversed); the incremental policy kept its edge.

## Reproduce it

The chapter notebook `notebooks/41-online-learning.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch41` (`run`, `explain`, `draw`).

```python
import numpy as np

from kaggle_companion.activities._common import clean

SEED = 41
REPLICATES = 20         # independent streams per delay; results are means over them
STEPS, WARM, SELECT = 2000, 300, 500   # prediction origins run from WARM; the first SELECT origins choose each policy's setting
FEATURES, RIDGE = 3, 1.0
DRIFT_NOISE, DRIFT_PULL = 0.12, 0.02   # the drifting part wanders and is pulled back toward zero
WINDOWS = (30, 60, 120, 240)           # candidate rolling-window lengths, in released rows
DISCOUNTS = (0.95, 0.98, 0.99, 0.995)  # candidate forgetting factors for the incremental policy
POINTS = 40                            # resolution of the cumulative error curves


def make_stream(rng):
    """Row t has features x_t and label y_t = w_t . x_t + noise, where w_t = a stable part + a drifting part."""
    X = rng.normal(size=(STEPS, FEATURES))
    stable = rng.normal(0, 1, FEATURES)                    # the part of the signal a frozen model can keep
    spread = DRIFT_NOISE / np.sqrt(1 - (1 - DRIFT_PULL) ** 2)   # long-run spread of the drifting part
    drift = np.zeros((STEPS, FEATURES))
    drift[0] = rng.normal(0, spread, FEATURES)
    for t in range(1, STEPS):
        drift[t] = drift[t - 1] * (1 - DRIFT_PULL) + DRIFT_NOISE * rng.normal(size=FEATURES)
    w = stable + drift
    return X, (w * X).sum(axis=1) + rng.normal(0, 1, STEPS)


def ridge(X, y):
    return np.linalg.solve(X.T @ X + RIDGE * np.eye(FEATURES), X.T @ y)


def replay(X, y, delay):
    """Squared error at every origin t >= WARM for each policy. Row i is released at i + delay and usable only at origins t > i + delay."""
    origins = range(WARM, STEPS)
    errors = {"frozen": None}
    frozen = ridge(X[:WARM - delay], y[:WARM - delay])           # fitted once, on rows released before the first origin
    errors["frozen"] = (y[WARM:] - X[WARM:] @ frozen) ** 2
    for window in WINDOWS:
        eligible, ignoring_delay = [], []
        for t in origins:
            hi = t - delay                                         # rows 0 .. hi-1 are released before origin t
            w = ridge(X[max(0, hi - window):hi], y[max(0, hi - window):hi])
            eligible.append((y[t] - X[t] @ w) ** 2)
            w = ridge(X[max(0, t - window):t], y[max(0, t - window):t])   # INVALID: also uses labels not yet released
            ignoring_delay.append((y[t] - X[t] @ w) ** 2)
        errors[("rolling", window)], errors[("invalid", window)] = np.array(eligible), np.array(ignoring_delay)
    for discount in DISCOUNTS:                                     # incremental: discounted ridge, one released row at a time
        A, b = np.zeros((FEATURES, FEATURES)), np.zeros(FEATURES)
        for i in range(WARM - delay):
            A, b = discount * A + np.outer(X[i], X[i]), discount * b + X[i] * y[i]
        errs = []
        for t in origins:
            i = t - delay - 1                                      # the row released since the previous origin
            if i >= WARM - delay:
                A, b = discount * A + np.outer(X[i], X[i]), discount * b + X[i] * y[i]
            errs.append((y[t] - X[t] @ np.linalg.solve(A + RIDGE * np.eye(FEATURES), b)) ** 2)
        errors[("incremental", discount)] = np.array(errs)
    return errors


def pick(errors, kind, options):
    """Choose the setting with the lowest error on the first SELECT origins; report it on the later origins only."""
    best = min(options, key=lambda o: errors[(kind, o)][:SELECT].mean())
    return best, errors[(kind, best)][SELECT:]


def run(delay):
    scores = {k: [] for k in ("frozen", "rolling", "incremental", "invalid")}
    settings = {"rolling": [], "incremental": []}
    curves = {k: [] for k in scores}
    for r in range(REPLICATES):
        X, y = make_stream(np.random.default_rng(SEED * 100 + r))
        errors = replay(X, y, delay)
        later = {"frozen": errors["frozen"][SELECT:]}
        for kind, options in (("rolling", WINDOWS), ("incremental", DISCOUNTS), ("invalid", WINDOWS)):
            chosen, later[kind] = pick(errors, kind, options)
            if kind in settings:
                settings[kind].append(chosen)
        for k, e in later.items():
            scores[k].append(e.mean())
            running = np.cumsum(e) / np.arange(1, len(e) + 1)
            curves[k].append(running[np.linspace(0, len(e) - 1, POINTS).astype(int)])
    mean = {k: float(np.mean(v)) for k, v in scores.items()}
    return clean({
        "delay": delay, "mse": mean,
        "gain_rolling": mean["frozen"] - mean["rolling"], "gain_incremental": mean["frozen"] - mean["incremental"],
        "gain_invalid": mean["frozen"] - mean["invalid"],
        "selected_window": float(np.mean(settings["rolling"])), "selected_discount": float(np.mean(settings["incremental"])),
        "curve_steps": np.linspace(1, STEPS - WARM - SELECT, POINTS), "curves": {k: np.mean(v, axis=0) for k, v in curves.items()},
        "noise_floor": 1.0, "replicates": REPLICATES, "scored_origins": STEPS - WARM - SELECT,
    })
```

Book location: Chapter 41, A Delayed-Label Contrast. Constructed example: a seeded synthetic drifting stream and ridge models, measured by the chapter activity.
