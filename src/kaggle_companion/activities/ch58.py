"""Chapter 58: Sensor Embeddings and Augmentation. A left-right sensor swap with and without relabelling, scored on held-out subjects."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, KFold

from kaggle_companion.activities._common import clean

SEED = 58
REPLICATES = 6                     # independent constructed cohorts; every estimate is their mean
TRAIN_SUBJECTS, TEST_SUBJECTS = 12, 20
RATE, WINDOW, STRIDE, RECORDING = 32, 64, 32, 320   # 32 Hz, 2 s windows with 50% overlap, 10 s recordings
NOISE = 0.6
CLASSES = ["rest", "walk", "wave_right", "wave_left"]
MIRROR = {0: 0, 1: 1, 2: 3, 3: 2}  # what a left-right swap does to each label: waving with one hand becomes waving with the other


def new_subject(rng):
    """Per-person differences: sensor gains and offsets, walking and waving style."""
    return {"gain": np.exp(rng.normal(0, 0.2, 2)), "bias": rng.normal(0, 0.4, (2, 3)), "swing": rng.uniform(1.0, 2.0),
            "step_hz": rng.uniform(1.6, 2.4), "phase": rng.uniform(0, 6.28), "wave": rng.uniform(0.5, 1.6), "wave_hz": rng.uniform(1.5, 3.0)}


def recording(activity, subject, rng):
    """(time, sensor, axis) readings. Sensor 0 is the left wrist, 1 the right; axes are lateral x, forward y, vertical z.
    The two wrist units are mounted as mirror images, so a left-hand wave has the opposite lateral sign to a right-hand wave."""
    t = np.arange(RECORDING) / RATE
    x = np.zeros((RECORDING, 2, 3))
    if activity == 1:                                    # walk: arms swing forward and back in antiphase
        a, f, ph = subject["swing"], subject["step_hz"], subject["phase"]
        x[:, 0, 1] = a * np.sin(2 * np.pi * f * t + ph)
        x[:, 1, 1] = a * np.sin(2 * np.pi * f * t + ph + np.pi)
        x[:, :, 2] = (0.3 * a * np.sin(4 * np.pi * f * t + 0.5))[:, None]
    if activity in (2, 3):                               # wave: sideways oscillation with the arm raised to the side
        sensor, sign = (1, 1.0) if activity == 2 else (0, -1.0)
        x[:, sensor, 0] = sign * (subject["wave"] * np.sin(2 * np.pi * subject["wave_hz"] * t) + 0.8)
    return x * subject["gain"][None, :, None] + subject["bias"][None] + rng.normal(0, NOISE, x.shape)


def cohort(subjects, left_wave_subjects, rng):
    """Windows, labels and subject ids. Only the first `left_wave_subjects` people recorded the left-hand wave."""
    windows, labels, owner = [], [], []
    for i in range(subjects):
        person = new_subject(rng)
        for activity in range(4):
            if activity == 3 and i >= left_wave_subjects:
                continue
            x = recording(activity, person, rng)
            for start in range(0, RECORDING - WINDOW + 1, STRIDE):
                windows.append(x[start:start + WINDOW])
                labels.append(activity)
                owner.append(i)
    return np.array(windows), np.array(labels), np.array(owner)


def swap_wrists(windows):
    """The augmentation: exchange the two sensors and flip the lateral axis, because the units are mirrored."""
    swapped = windows[:, :, ::-1, :].copy()
    swapped[..., 0] *= -1
    return swapped


def features(windows):
    """Per sensor and axis: mean, std, min, max; per sensor: strongest spectral peak; plus left-right forward correlation."""
    centered = windows - windows.mean(axis=1, keepdims=True)
    stats = np.stack([windows.mean(1), windows.std(1), windows.min(1), windows.max(1)], axis=-1)       # (n, sensor, axis, 4)
    peak = np.abs(np.fft.rfft(centered, axis=1))[:, 1:].max(axis=(1, 3))                                # (n, sensor)
    a, b = centered[:, :, 0, 1], centered[:, :, 1, 1]
    corr = (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    return np.column_stack([stats.reshape(len(windows), -1), peak, corr])


def forest():
    return RandomForestClassifier(60, random_state=0, n_jobs=1)


def macro_f1(y_true, y_pred, labels):
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def one_cohort(left_wave_subjects, replicate):
    rng = np.random.default_rng(SEED * 100 + replicate)
    W, y, owner = cohort(TRAIN_SUBJECTS, left_wave_subjects, rng)
    W_test, y_test, _ = cohort(TEST_SUBJECTS, TEST_SUBJECTS, rng)         # unseen people, every class recorded
    F, F_test = features(W), features(W_test)
    out = {}
    for policy in ("none", "swap_relabel", "swap_keep_labels"):
        if policy == "none":
            F_fit, y_fit = F, y
        else:
            relabel = np.array([MIRROR[k] for k in y]) if policy == "swap_relabel" else y   # the wrong version keeps the old label
            F_fit, y_fit = np.vstack([F, features(swap_wrists(W))]), np.concatenate([y, relabel])
        pred = forest().fit(F_fit, y_fit).predict(F_test)
        out[policy] = {"macro_f1": macro_f1(y_test, pred, [0, 1, 2, 3]), "left_wave_f1": macro_f1(y_test, pred, [3])}
    # Two ways to estimate the unaugmented model from the training people alone, over the classes present in training.
    seen = sorted(set(y))
    window_cv, subject_cv = np.zeros(len(y), dtype=int), np.zeros(len(y), dtype=int)
    for fit, held in KFold(3, shuffle=True, random_state=0).split(F):
        window_cv[held] = forest().fit(F[fit], y[fit]).predict(F[held])
    for fit, held in GroupKFold(3).split(F, y, owner):
        subject_cv[held] = forest().fit(F[fit], y[fit]).predict(F[held])
    out["window_cv"], out["subject_cv"] = macro_f1(y, window_cv, seen), macro_f1(y, subject_cv, seen)
    return out


def run(left_wave_subjects):
    sets = [one_cohort(left_wave_subjects, i) for i in range(REPLICATES)]
    mean = lambda f: float(np.mean([f(s) for s in sets]))
    policies = {p: {"macro_f1": mean(lambda s: s[p]["macro_f1"]), "left_wave_f1": mean(lambda s: s[p]["left_wave_f1"]),
                    "per_cohort": [s[p]["macro_f1"] for s in sets]} for p in ("none", "swap_relabel", "swap_keep_labels")}
    return clean({"left_wave_subjects": left_wave_subjects, "policies": policies,
                  "window_cv": mean(lambda s: s["window_cv"]), "subject_cv": mean(lambda s: s["subject_cv"]),
                  "replicates": REPLICATES, "train_subjects": TRAIN_SUBJECTS, "test_subjects": TEST_SUBJECTS})
# notebook-end


SPEC = {
    "chapter": 58,
    "chapter_title": "Sensor Embeddings and Augmentation",
    "subtitle": "A left-right swap is valid only when the label means the same thing after the swap, and assessment subjects stay unseen.",
    "summary": ("A mirror transform doubles the training windows, but it also changes what a handed label means. One demonstration "
                "measures a wrist-sensor classifier with no augmentation, a swap that relabels waving hands, and a swap that keeps the labels, on "
                "people never seen in training."),
    "title": "Left-right swap augmentation on unseen subjects, by how many people recorded a left-hand wave",
    "question": "When does a left-right swap help a wearable activity classifier, and what does it cost when the label is not updated with the swap?",
    "why": ("A swap is a hypothesis about symmetry. It is a free source of examples for rare handed classes when the label is updated with "
            "the geometry, and a source of wrong labels when it is not. Scores must come from people the model has not seen."),
    "method": ("Constructed wrist-sensor recordings: two three-axis units (left and right, mounted as mirror images), 32 Hz, 2 s windows "
               "with 50% overlap. Four activities: rest, walk (arms swing in antiphase), wave with the right hand and wave with the left hand. "
               "People differ in sensor gain and offset and in walking and waving style. A random forest on window summaries (mean, std, "
               "min, max, spectral peak per sensor, left-right correlation) is trained on 12 people, of whom only the number set by the "
               "control recorded the left-hand wave, and scored by macro F1 on 20 unseen people who recorded all four activities. "
               "The swap exchanges the two units and flips the lateral axis. It is applied three ways: not at all, with waving "
               "relabelled to the other hand, and with the original label kept."),
    "control": {"key": "left_wave_subjects", "label": "Training people who recorded the left-hand wave (of 12)",
                "values": [0, 1, 3, 12], "default": 1,
                "value_labels": ["0: none", "1", "3", "12: all"]},
    "source_section": "Keep Assessment Subjects Independent",
    "symbols": ("A window has shape (time, sensors, 3). The swap S exchanges the sensor axis and negates the lateral axis, and "
                "the label map m sends wave_right to wave_left, wave_left to wave_right and leaves rest and walk fixed. Macro F1 is "
                "the mean of the per-class F1 scores."),
    "explanation": ("Right-hand waves look like left-hand waves after the swap, so the relabelled swap turns the abundant class "
                    "into examples of the rare one: it matters most when few people recorded the left wave, and does little when many did. "
                    "Keeping the original label teaches the model that left-handed waving is a right-hand wave, which "
                    "makes it far worse than no augmentation whenever any left-hand waves were recorded (with none, there is nothing to spoil). A window-level cross-validation estimate leaks each person's style and a split among the training people "
                    "cannot see a class that none of them recorded; with a single left-wave recorder, a person-level split can also land far from the unseen-people score."),
    "application": ("Apply a mirror augmentation only where the label is invariant or is updated with the geometry, test it by "
                    "subject-held-out score with and without it, and split by person before cutting windows."),
    "assumptions": ("Constructed signals in which the left wave is the exact mirror of the right wave, so the relabelled swap is valid by "
                    "design; a real non-dominant hand can differ and the gain would then be smaller. Gradient-boosted trees or a "
                    "neural model on raw windows would give different absolute scores. The classes separate by amplitude "
                    "and spectrum, so the axis flip itself matters little in this generator; it is kept because it is the correct "
                    "geometry for mirrored units. Twelve training people and six cohorts give the means."),
    "prediction": "If nobody in the training set recorded the left-hand wave, what macro F1 does the relabelled swap reach on unseen people, against no augmentation?",
    "prediction_options": ["About the same as no augmentation", "Far higher: about 0.95 against 0.65", "Slightly lower"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "The relabelled swap reaches a macro F1 of 0.954 against 0.650 for no augmentation, because it creates the missing class from right-hand waves.",
        "incorrect": "The relabelled swap reaches a macro F1 of 0.954 against 0.650 for no augmentation. It creates the missing left-wave class from right-hand waves, which is why it is not the same.",
    },
    "check": "With all 12 people recording the left wave the relabelled swap gains almost nothing, and the label-keeping swap loses a lot. Why do the two differ so much?",
    "answer": ("When both hands were recorded, the relabelled swap only adds near-copies of what the model already has, so it gains little (0.956 to 0.959). "
               "The swap that keeps the label adds a mirrored copy of every window, and the copies of right-hand waves look like left-hand waves but carry the right-hand label. "
               "That harm barely shrinks as real left-hand data grows (0.249 below no augmentation with 1 person, 0.208 with all 12, "
               "where the macro F1 is 0.748), so a transform must be checked for what it does to the labels, "
               "not only for how much data it adds."),
    "provenance": "Constructed example: six seeded cohorts of synthetic wrist-sensor recordings and a random forest, measured by the chapter activity.",
    "apply": [
        "Write down the sensor order, axis meaning and which labels depend on side before choosing any symmetry transform.",
        "Apply the swap together with the matching label change, or disable it for classes whose label does not survive the swap.",
        "Score each augmentation alone on people the model has never seen, and keep the no-augmentation model as the comparison.",
        "Split by person or recording before cutting overlapping windows, and expect a window-level or in-pool estimate to be higher than the unseen-person score.",
    ],
    "honesty": ("Constructed data in which the mirror assumption holds exactly. The large gain at 0 and 1 people comes from a class that is nearly "
                "absent from the training set; with plenty of left-hand data the swap is close to neutral."),
}

EQUATIONS = [{"tex": r"\tilde{x}[t, s, a] = \sigma_a\, x[t, \pi(s), a], \qquad \tilde{y} = m(y)",
              "alt": "the swapped window x tilde at time t, sensor s and axis a equals sigma a times x at time t, the partner sensor pi of s, axis a; the new label y tilde equals m of y",
              "basis": "The activity's own swap and label map: pi exchanges the two wrists, sigma_a is -1 on the lateral axis and 1 otherwise. The chapter's swap_sensor_pairs code is a pairing operation without a display equation."}]
NCOLS = 2
HEIGHT = 4.4
POLICIES = [("none", "No\naugmentation", COLORS["light"]), ("swap_relabel", "Swap,\nrelabelled", COLORS["teal"]),
            ("swap_keep_labels", "Swap,\nlabel kept", COLORS["terracotta"])]


def draw(axes, result, parameter):
    left, right = axes
    xs = list(range(len(POLICIES)))
    for x, (key, _, color) in zip(xs, POLICIES):
        p = result["policies"][key]
        left.bar(x, p["macro_f1"], width=0.6, color=color, edgecolor=COLORS["ink"], lw=0.6)
        dots = p["per_cohort"]
        left.scatter([x + (i - (len(dots) - 1) / 2) * 0.05 for i in range(len(dots))], dots, s=12, color=COLORS["ink"], zorder=3,
                     label="One cohort" if x == 0 else None)
        left.text(x, max(dots + [p["macro_f1"]]) + 0.02, fmt(p["macro_f1"]), ha="center", va="bottom", fontsize=10)
    left.set_xticks(xs, [name for _, name, _ in POLICIES])
    left.set_xlim(-0.6, len(POLICIES) - 0.4)
    left.set_ylim(0, 1.2)
    left.set_ylabel("Macro F1 on 20 unseen people")
    left.set_xlabel(f"Augmentation, {parameter} of 12 people recorded the left wave")
    left.legend(loc="upper left", frameon=False, fontsize=10)

    estimates = [("Window-level\nCV", result["window_cv"], COLORS["gold"]), ("Person-level\nCV", result["subject_cv"], COLORS["navy"]),
                 ("Unseen\npeople", result["policies"]["none"]["macro_f1"], COLORS["teal"])]
    for x, (name, value, color) in enumerate(estimates):
        right.bar(x, value, width=0.6, color=color, edgecolor=COLORS["ink"], lw=0.6)
        right.text(x, value + 0.02, fmt(value), ha="center", va="bottom", fontsize=10)
    right.set_xticks(range(len(estimates)), [name for name, _, _ in estimates])
    right.set_xlim(-0.6, len(estimates) - 0.4)
    right.set_ylim(0, 1.2)
    right.set_ylabel("Macro F1, no augmentation")
    right.set_xlabel("How the model is assessed")


def diff(a, b, digits=3):
    """Difference of the two numbers as displayed, so the hand calculation on the page adds up."""
    return float(fmt(a, digits)) - float(fmt(b, digits))


def explain(result, parameter):
    n, s, k = (result["policies"][key] for key in ("none", "swap_relabel", "swap_keep_labels"))
    gain = diff(s["macro_f1"], n["macro_f1"])
    loss = diff(n["macro_f1"], k["macro_f1"])
    inflation = diff(result["window_cv"], n["macro_f1"])
    interpretation = (
        f"When {parameter} of 12 training people recorded the left-hand wave, the model with no augmentation scores a macro F1 of {fmt(n['macro_f1'])} "
        f"on unseen people and the relabelled swap {fmt(s['macro_f1'])}, so {fmt(s['macro_f1'])} - {fmt(n['macro_f1'])} = {fmt(gain)} is the gain "
        f"(left-wave F1 {fmt(n['left_wave_f1'])} to {fmt(s['left_wave_f1'])}). The swap that keeps the label scores {fmt(k['macro_f1'])}, "
        + (f"{fmt(n['macro_f1'])} - {fmt(k['macro_f1'])} = {fmt(loss)} below no augmentation. " if loss >= 0 else
           f"{fmt(k['macro_f1'])} - {fmt(n['macro_f1'])} = {fmt(-loss)} above no augmentation. ") +
        f"Window-level cross-validation on the training people reports {fmt(result['window_cv'])} and person-level {fmt(result['subject_cv'])}, "
        f"against {fmt(n['macro_f1'])} on unseen people.")
    steps = [
        f"Gain from the relabelled swap: {fmt(s['macro_f1'])} - {fmt(n['macro_f1'])} = {signed(gain)}.",
        f"Cost of keeping the label: {fmt(k['macro_f1'])} - {fmt(n['macro_f1'])} = {signed(diff(k['macro_f1'], n['macro_f1']))}.",
        f"Left-wave F1 with the relabelled swap: {fmt(s['left_wave_f1'])}, without augmentation {fmt(n['left_wave_f1'])}.",
        f"Window-level CV minus unseen people: {fmt(result['window_cv'])} - {fmt(n['macro_f1'])} = {signed(inflation)}.",
    ]
    metrics = {"No augmentation": fmt(n["macro_f1"]), "Swap, relabelled": fmt(s["macro_f1"]), "Swap, label kept": fmt(k["macro_f1"]),
               "Left-wave F1, relabelled swap": fmt(s["left_wave_f1"]), "Window-level CV (no augmentation)": fmt(result["window_cv"])}
    alt = (f"Left: macro F1 on unseen people with {parameter} of 12 training people recording the left wave: no augmentation "
           f"{fmt(n['macro_f1'])}, relabelled swap {fmt(s['macro_f1'])}, label-keeping swap {fmt(k['macro_f1'])}. Right: window-level CV "
           f"{fmt(result['window_cv'])}, person-level CV {fmt(result['subject_cv'])} and unseen-people score {fmt(n['macro_f1'])} for the unaugmented model.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def _qualitative(results):
    """Directional claims; checked again under a different seed base."""
    for c, res in results.items():
        n, s, k = (res["policies"][key]["macro_f1"] for key in ("none", "swap_relabel", "swap_keep_labels"))
        assert s >= n - 0.02, f"relabelled swap should not hurt at {c}"
        assert (k < n - 0.1) if c > 0 else (k < n + 0.02), f"keeping the label should hurt a lot once left-wave data exists, at {c}"
        assert res["window_cv"] > res["subject_cv"] - 0.01, f"window-level CV should not be below person-level at {c}"
    assert results[0]["policies"]["swap_relabel"]["macro_f1"] - results[0]["policies"]["none"]["macro_f1"] > 0.25, "swap creates the missing class"
    assert results[0]["policies"]["none"]["left_wave_f1"] < 0.05, "no left-wave data, no left-wave F1"
    assert results[1]["policies"]["swap_relabel"]["macro_f1"] - results[1]["policies"]["none"]["macro_f1"] > 0.05, "one person: swap helps clearly"
    assert results[12]["policies"]["swap_relabel"]["macro_f1"] - results[12]["policies"]["none"]["macro_f1"] < 0.03, "all people: swap near neutral"
    gains = [results[c]["policies"]["swap_relabel"]["macro_f1"] - results[c]["policies"]["none"]["macro_f1"] for c in (0, 1, 3, 12)]
    assert gains == sorted(gains, reverse=True), "gain shrinks as left-wave data grows"
    assert results[0]["window_cv"] - results[0]["policies"]["none"]["macro_f1"] > 0.25, "in-pool CV cannot see the absent class"
    assert abs(results[1]["subject_cv"] - results[1]["policies"]["none"]["macro_f1"]) > 0.1, "person-level CV is far off with one left-wave recorder"
    assert results[12]["window_cv"] > results[12]["policies"]["none"]["macro_f1"], "window-level CV overstates"


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    _qualitative(results)
    r0, r12 = results[0]["policies"], results[12]["policies"]
    assert fmt(r0["swap_relabel"]["macro_f1"]) == "0.954" and fmt(r0["none"]["macro_f1"]) == "0.650", "feedback numbers"
    assert fmt(r12["none"]["macro_f1"]) == "0.956" and fmt(r12["swap_relabel"]["macro_f1"]) == "0.959", "answer numbers"
    assert fmt(r12["swap_keep_labels"]["macro_f1"]) == "0.748", "answer: label kept at 12 people"
    r1 = results[1]["policies"]
    assert fmt(diff(r1["none"]["macro_f1"], r1["swap_keep_labels"]["macro_f1"])) == "0.249", "answer: harm with 1 person"
    assert fmt(diff(r12["none"]["macro_f1"], r12["swap_keep_labels"]["macro_f1"])) == "0.208", "answer: harm with 12 people"
