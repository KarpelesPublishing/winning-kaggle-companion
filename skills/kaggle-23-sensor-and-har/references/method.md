# Chapter 23: Sensors and HAR: The Cross-Subject Problem

**How far above the score on new people does a random stratified split sit, and does leave-one-subject-out close the gap?**

The chapter's central mistake is validating a cross-subject task with folds that mix subjects. Measuring the size of the error as people become more distinctive shows when random folds mislead and what LOSO recovers.

## The experiment

Four constructed cohorts per setting. Each person produces continuous 6-channel recordings (3 accelerometer and 3 gyroscope axes at 50 Hz) of six activities, cut into 2-second windows with 50% overlap, 24 windows per person and activity. A person's cadence, amplitude, wrist orientation and sensor bias differ from the average by an amount the control sets. Trees are fitted on windowed features (mean, spread, energy, zero crossings, dominant frequency, axis correlations). Macro-F1 is pooled over 8 development subjects with random stratified 5-fold and with leave-one-subject-out, and compared with the score on 12 new people.

Control: How much each person differs from the average (0 means identical people) (0: identical people, 0.5, 1.0, 1.5: very distinct people; default 1.0).

## Measured results

| Measure | 0: identical people | 0.5 | 1.0 | 1.5: very distinct people |
|---|---|---|---|---|
| Random stratified macro-F1 | 0.955 | 0.840 | 0.799 | 0.788 |
| Leave-one-subject-out macro-F1 | 0.953 | 0.746 | 0.588 | 0.485 |
| Macro-F1 on new people | 0.961 | 0.774 | 0.604 | 0.506 |
| Random minus new people | -0.005 | +0.066 | +0.195 | +0.282 |
| Sitting vs standing, new people | 0.971 | 0.698 | 0.528 | 0.448 |

## What the result says (default, how much each person differs from the average (0 means identical people) = 1.0)

At distinctiveness 1.0, random stratified folds report macro-F1 0.799 and leave-one-subject-out 0.588, while the model scores 0.604 on 12 new people. The random estimate is off by 0.799 - 0.604 = 0.195; leave-one-subject-out is off by 0.017. For sitting against standing alone the three scores are 0.724, 0.473 and 0.528. The random split overstates the transfer.

- Random stratified: 0.799 - 0.604 = +0.195 (estimate minus new people).
- Leave-one-subject-out: 0.588 - 0.604 = -0.017.
- Sitting against standing, random minus new people: 0.724 - 0.528 = +0.196.
- Cohorts averaged: 4, each with 8 development subjects and 12 new people.

## Apply it to a competition

- Put every window of a person (and every overlapping window of a recording) on one side of each split; use leave-one-subject-out or grouped folds.
- Report the subject-held-out score as the estimate, and keep the random-split score only as a diagnostic of how much people differ.
- Check confusion between near-identical classes (sitting and standing) on held-out subjects, where orientation differences hurt most.
- Fit any normalization from permitted rows only, and treat whole-subject normalization as transductive if the test subject's batch is not available.

## Assumptions and limits

Constructed signals with a simple generator, six activities, and extremely randomized trees standing in for the chapter's gradient boosting. Subject differences are a built-in mechanism of cadence, amplitude and orientation; real subjects differ in other ways. With identical people the overlapping windows cost nothing measurable here because the activities are separable; with weaker class signal they might, and this activity does not test that.

Constructed data. How large the subject effect is depends entirely on the generator's person-to-person variation; the claim is the direction and the agreement of leave-one-subject-out with new people, not the sizes.

## Reproduce it

The chapter notebook `notebooks/23-sensor-and-har.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch23` (`run`, `explain`, `draw`).

```python
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from kaggle_companion.activities._common import clean

SEED = 23
DATASETS = 4                 # independent constructed cohorts; every estimate is their mean
DEV_SUBJECTS, NEW_SUBJECTS, WINDOWS = 8, 12, 24   # windows per subject and activity
HZ, T, STEP = 50, 100, 50    # 50 Hz sampling, 2-second windows, 50% overlap
# activity: (cadence in Hz, amplitude, vertical share, rhythm irregularity, gravity tilt in radians)
ACTIVITIES = {"walking": (1.8, 1.0, 0.8, 0.1, 0.0), "running": (2.8, 1.7, 0.8, 0.1, 0.0), "cycling": (1.4, 0.8, 0.3, 0.05, 0.0),
              "stairs": (1.6, 1.1, 0.8, 0.7, 0.0), "sitting": (0.0, 0.0, 0.0, 0.0, 0.0), "standing": (0.0, 0.0, 0.0, 0.0, 0.35)}
NAMES = list(ACTIVITIES)


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def make_subject(distinctiveness, rng):
    """A person: a cadence multiplier, an amplitude multiplier, a wrist orientation and a sensor bias."""
    d = distinctiveness
    return {"cadence": 1 + 0.15 * d * rng.normal(), "amplitude": np.exp(0.35 * d * rng.normal()),
            "rotation": rotation(0.45 * d * rng.normal()), "bias": 0.3 * d * rng.normal(size=3)}


def recording(activity, person, rng):
    """One continuous 6-channel recording (3 accelerometer + 3 gyroscope axes), cut into 50%-overlapping windows."""
    cadence, amp, vertical, irregular, tilt = ACTIVITIES[activity]
    length = (WINDOWS - 1) * STEP + T
    t = np.arange(length) / HZ
    f, a = cadence * person["cadence"], amp * person["amplitude"]
    phase = rng.uniform(0, 2 * np.pi) + np.cumsum(rng.normal(0, irregular * 0.15, length))   # irregular rhythm drifts the phase
    wave = np.sin(2 * np.pi * f * t + phase)
    acc = np.column_stack([a * (1 - vertical) * np.sin(2 * np.pi * f * t + phase + 1.0), 0.4 * a * wave ** 2, a * vertical * wave])
    acc = (acc + rotation(tilt) @ np.array([0, 0, 1.0])) @ person["rotation"].T + person["bias"]   # gravity, then the person's orientation
    gyro = 0.5 * a * np.column_stack([np.cos(2 * np.pi * f * t + phase), np.sin(2 * np.pi * f * t + phase), wave]) @ person["rotation"].T
    signal = np.column_stack([acc, gyro]) + rng.normal(0, 0.8, (length, 6))
    return np.stack([signal[i * STEP:i * STEP + T] for i in range(WINDOWS)])


def window_features(w):
    """Per channel: mean, standard deviation, energy, zero-crossing rate, dominant frequency and its magnitude; plus axis correlations."""
    mean, sd, energy = w.mean(1), w.std(1), (w ** 2).mean(1)
    zero_cross = ((w[:, 1:] - mean[:, None]) * (w[:, :-1] - mean[:, None]) < 0).mean(1)
    spectrum = np.abs(np.fft.rfft(w - mean[:, None], axis=1))
    peak, magnitude = spectrum[:, 1:].argmax(1) + 1, spectrum[:, 1:].max(1)
    corr = [np.einsum("nt,nt->n", w[:, :, i] - mean[:, None, i], w[:, :, j] - mean[:, None, j]) / (T * sd[:, i] * sd[:, j] + 1e-6)
            for i, j in ((0, 1), (0, 2), (1, 2))]
    return np.column_stack([mean, sd, energy, zero_cross, peak * HZ / T, magnitude / T] + [c[:, None] for c in corr])


def cohort(n_subjects, distinctiveness, rng):
    X, y, who = [], [], []
    for s in range(n_subjects):
        person = make_subject(distinctiveness, rng)
        for k, name in enumerate(NAMES):
            X.append(window_features(recording(name, person, rng)))
            y += [k] * WINDOWS
            who += [s] * WINDOWS
    return np.vstack(X), np.array(y), np.array(who)


def model():
    return ExtraTreesClassifier(n_estimators=60, random_state=0, n_jobs=1)


def macro_f1(y, pred, labels=None):
    return f1_score(y, pred, labels=labels, average="macro")


def one_cohort(distinctiveness, seed):
    rng = np.random.default_rng(seed)
    X, y, who = cohort(DEV_SUBJECTS, distinctiveness, rng)
    X_new, y_new, _ = cohort(NEW_SUBJECTS, distinctiveness, rng)      # different people, never seen in development
    random_pred, subject_pred = np.zeros(len(y), int), np.zeros(len(y), int)
    for fit, val in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):      # the chapter's mistake
        random_pred[val] = model().fit(X[fit], y[fit]).predict(X[val])
    for fit, val in LeaveOneGroupOut().split(X, y, who):                               # the chapter's fix
        subject_pred[val] = model().fit(X[fit], y[fit]).predict(X[val])
    new_pred = model().fit(X, y).predict(X_new)
    seated = [NAMES.index("sitting"), NAMES.index("standing")]
    return [macro_f1(y, random_pred), macro_f1(y, subject_pred), macro_f1(y_new, new_pred),
            macro_f1(y, random_pred, seated), macro_f1(y, subject_pred, seated), macro_f1(y_new, new_pred, seated)]


def run(distinctiveness):
    scores = np.array([one_cohort(distinctiveness, SEED * 1000 + i) for i in range(DATASETS)])
    mean = scores.mean(0)
    return clean({
        "distinctiveness": distinctiveness, "datasets": DATASETS, "dev_subjects": DEV_SUBJECTS, "new_subjects": NEW_SUBJECTS,
        "random": mean[0], "loso": mean[1], "new": mean[2],
        "sit_stand": {"random": mean[3], "loso": mean[4], "new": mean[5]},
        "per_dataset": {"random": scores[:, 0], "loso": scores[:, 1], "new": scores[:, 2]},
    })
```

Book location: Chapter 23, The Fix: Leave-One-Subject-Out CV. Constructed example: four seeded synthetic cohorts of simulated wearable recordings and extremely randomized trees, measured by the chapter activity.
