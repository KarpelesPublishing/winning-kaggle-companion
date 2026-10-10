"""Chapter 23: Sensors and HAR. Random-window, leave-one-subject-out and new-subject scores as people differ more."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
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
# notebook-end


SPEC = {
    "chapter": 23,
    "chapter_title": "Sensors and HAR: The Cross-Subject Problem",
    "subtitle": "Match the validation boundary to the test boundary: new subjects, not new windows.",
    "summary": ("Random stratified folds put the same person's windows on both sides of the split. One demonstration measures how far that estimate "
                "sits above leave-one-subject-out and above the score on genuinely new people, as people differ more from one another."),
    "title": "Random folds against leave-one-subject-out, scored on new people",
    "question": "How far above the score on new people does a random stratified split sit, and does leave-one-subject-out close the gap?",
    "why": ("The chapter's central mistake is validating a cross-subject task with folds that mix subjects. Measuring the size of the error as "
            "people become more distinctive shows when random folds mislead and what LOSO recovers."),
    "method": ("Four constructed cohorts per setting. Each person produces continuous 6-channel recordings (3 accelerometer and 3 gyroscope axes at 50 Hz) "
               "of six activities, cut into 2-second windows with 50% overlap, 24 windows per person and activity. A person's cadence, amplitude, "
               "wrist orientation and sensor bias differ from the average by an amount the control sets. Trees are fitted on windowed features (mean, spread, "
               "energy, zero crossings, dominant frequency, axis correlations). Macro-F1 is pooled over 8 development subjects with random stratified 5-fold "
               "and with leave-one-subject-out, and compared with the score on 12 new people."),
    "control": {"key": "distinctiveness", "label": "How much each person differs from the average (0 means identical people)",
                "values": [0, 0.5, 1.0, 1.5], "default": 1.0,
                "value_labels": ["0: identical people", "0.5", "1.0", "1.5: very distinct people"]},
    "source_section": "The Fix: Leave-One-Subject-Out CV",
    "symbols": ("F1_c is the F1 score of activity class c, C the number of classes, y the true activity and the prediction is the model's "
                "class for a window. Macro-F1 averages F1_c over classes."),
    "explanation": ("With identical people a model that has seen a window's neighbours cannot cheat, because nothing about the person helps. "
                    "As people differ, the random split lets the model learn each person's own cadence, amplitude and orientation and recognize their "
                    "other windows, so it scores far above what it can do on someone new. Leave-one-subject-out withholds the whole person and lands "
                    "near the new-people score."),
    "application": ("Group the validation splits by subject (and by recording or session where those also repeat), report the subject-held-out "
                    "estimate, and check class-pair confusions such as sitting against standing separately."),
    "assumptions": ("Constructed signals with a simple generator, six activities, and extremely randomized trees standing in for the chapter's gradient "
                    "boosting. Subject differences are a built-in mechanism of cadence, amplitude and orientation; real subjects differ in other ways. "
                    "With identical people the overlapping windows cost nothing measurable here because the activities are separable; with weaker class "
                    "signal they might, and this activity does not test that."),
    "prediction": "With moderately distinct people (1.0), how does the macro-F1 of random stratified 5-fold compare with the score on new people?",
    "prediction_options": ["About the same", "About 0.2 higher", "Lower"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "Random folds report 0.799 against 0.604 on new people, 0.195 higher, while leave-one-subject-out reports 0.588.",
        "incorrect": "Random folds report 0.799 against 0.604 on new people, 0.195 higher. Leave-one-subject-out reports 0.588, much closer.",
    },
    "check": "With identical people (0), why do all three estimates agree even though the random folds mix overlapping windows?",
    "answer": ("Windows from one person look alike only because the person is distinctive, and with identical people there is no person-specific "
               "pattern to memorize. The estimates then agree (0.955, 0.953 and 0.961), so in this generator the overlap alone costs almost nothing. "
               "The error appears as soon as people differ and grows with how much they do."),
    "provenance": "Constructed example: four seeded synthetic cohorts of simulated wearable recordings and extremely randomized trees, measured by the chapter activity.",
    "apply": [
        "Put every window of a person (and every overlapping window of a recording) on one side of each split; use leave-one-subject-out or grouped folds.",
        "Report the subject-held-out score as the estimate, and keep the random-split score only as a diagnostic of how much people differ.",
        "Check confusion between near-identical classes (sitting and standing) on held-out subjects, where orientation differences hurt most.",
        "Fit any normalization from permitted rows only, and treat whole-subject normalization as transductive if the test subject's batch is not available.",
    ],
    "honesty": ("Constructed data. How large the subject effect is depends entirely on the generator's person-to-person variation; the "
                "claim is the direction and the agreement of leave-one-subject-out with new people, not the sizes."),
}

EQUATIONS = [{"tex": r"\mathrm{macro\text{-}F1} = \frac{1}{C}\sum_{c=1}^{C} F1_c",
              "alt": "macro F1 is the mean over C classes of the per-class F1 score",
              "basis": "The metric the chapter's leave-one-subject-out code reports (average='macro'); not a display equation in the manuscript."}]
NCOLS = 2
HEIGHT = 4.4
SCHEMES = [("random", "Random\nstratified"), ("loso", "Leave one\nsubject out")]


def _panel(ax, values, dots, new_value, ylabel, xlabel, show_dots=True):
    colors = [COLORS["terracotta"], COLORS["teal"]]
    for x, key in enumerate(("random", "loso")):
        ax.bar(x, values[key], width=0.55, color=colors[x], edgecolor=COLORS["ink"], lw=0.6)
        ax.text(x, values[key] + 0.045, fmt(values[key]), ha="center", va="bottom", fontsize=10, bbox={"fc": "white", "ec": "none", "pad": 1.5})
        if show_dots:
            ax.scatter([x + (i - (len(dots[key]) - 1) / 2) * 0.07 for i in range(len(dots[key]))], dots[key], s=16, color=COLORS["ink"], zorder=3,
                       label="One cohort" if x == 0 else None)
    ax.axhline(new_value, color=COLORS["gold"], ls=(0, (4, 3)), lw=1.6, label=f"Score on new people: {fmt(new_value)}")
    ax.set_xticks(range(2), [label for _, label in SCHEMES], fontsize=10)
    ax.set_ylim(0, 1.42)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)


def draw(axes, result, parameter):
    left, right = axes
    _panel(left, result, result["per_dataset"], result["new"], "Macro-F1, all six activities", f"Validation scheme, distinctiveness {parameter}")
    left.legend(loc="upper left", frameon=False, fontsize=10)
    _panel(right, result["sit_stand"], None, result["sit_stand"]["new"], "Macro-F1, sitting and standing only", "Validation scheme", show_dots=False)
    right.legend(loc="upper left", frameon=False, fontsize=10)


def explain(result, parameter):
    r, l, n = result["random"], result["loso"], result["new"]
    gap = r - n
    ss = result["sit_stand"]
    interpretation = (
        f"At distinctiveness {parameter}, random stratified folds report macro-F1 {fmt(r)} and leave-one-subject-out {fmt(l)}, while the model scores "
        f"{fmt(n)} on {result['new_subjects']} new people. The random estimate is off by {fmt(r)} - {fmt(n)} = {fmt(gap)}; leave-one-subject-out is "
        f"off by {fmt(abs(l - n))}. For sitting against standing alone the three scores are {fmt(ss['random'])}, {fmt(ss['loso'])} and {fmt(ss['new'])}. "
        + ("The random split overstates the transfer." if gap > 0.03 else "At this setting both splits estimate the new-people score about equally well."))
    steps = [
        f"Random stratified: {fmt(r)} - {fmt(n)} = {signed(gap)} (estimate minus new people).",
        f"Leave-one-subject-out: {fmt(l)} - {fmt(n)} = {signed(l - n)}.",
        f"Sitting against standing, random minus new people: {fmt(ss['random'])} - {fmt(ss['new'])} = {signed(ss['random'] - ss['new'])}.",
        f"Cohorts averaged: {result['datasets']}, each with {result['dev_subjects']} development subjects and {result['new_subjects']} new people.",
    ]
    metrics = {"Random stratified macro-F1": fmt(r), "Leave-one-subject-out macro-F1": fmt(l), "Macro-F1 on new people": fmt(n),
               "Random minus new people": signed(gap), "Sitting vs standing, new people": fmt(ss["new"])}
    alt = (f"Two panels of bars for random stratified and leave-one-subject-out macro-F1 at distinctiveness {parameter}, with a dashed line at the "
           f"score on new people, {fmt(n)}; left is all six activities with dots per cohort, right is sitting and standing only.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    zero = results[0]
    assert max(abs(zero["random"] - zero["new"]), abs(zero["loso"] - zero["new"]), abs(zero["random"] - zero["loso"])) < 0.02, "identical people should agree"
    values = sorted(results)
    gaps = [results[v]["random"] - results[v]["new"] for v in values]
    assert all(b > a for a, b in zip(gaps, gaps[1:])), f"random-split error should grow with distinctiveness: {gaps}"
    for v in values[1:]:
        res = results[v]
        assert res["random"] - res["new"] > 0.04, f"random split should overstate at {v}"
        assert abs(res["loso"] - res["new"]) < 0.7 * (res["random"] - res["new"]), f"LOSO should be closer than random at {v}"
        assert res["sit_stand"]["random"] > res["sit_stand"]["new"] + 0.05, f"sitting vs standing should be overstated at {v}"
    assert fmt(zero["random"]) == "0.955" and fmt(zero["loso"]) == "0.953" and fmt(zero["new"]) == "0.961", "check answer numbers"
    mid = results[1.0]
    assert fmt(mid["random"]) == "0.799" and fmt(mid["new"]) == "0.604" and fmt(mid["random"] - mid["new"]) == "0.195", "prediction feedback numbers"
    assert fmt(mid["loso"]) == "0.588", "prediction feedback numbers"
    assert 0.12 < mid["random"] - mid["new"] < 0.3, "prediction option says about 0.2"
