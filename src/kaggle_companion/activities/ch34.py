"""Chapter 34: Pseudo-Labeling. The eligible teacher against the "all models except k" teacher, by number of recipients."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from kaggle_companion.activities._common import clean

SEED = 34
REPLICATES = 8          # independent constructed datasets; every estimate is their mean
FEATURES, N_LABELED, N_FRESH, FOLDS = 8, 600, 5000, 3
CONFIDENCE = 0.3        # keep a pseudo-label only when the teacher's probability is more than 0.3 from 0.5


def make_signal(rng):
    """A fixed nonlinear log-odds function: a linear part, a sine of a random projection and one interaction."""
    w, v = rng.normal(size=FEATURES), rng.normal(size=FEATURES)
    return lambda X: 1.2 * (X @ w) / np.sqrt(FEATURES) + 0.9 * np.sin(X @ v) + 0.6 * X[:, 0] * X[:, 1]


def sample(signal, n, rng):
    X = rng.normal(size=(n, FEATURES))
    return X, (rng.random(n) < 1 / (1 + np.exp(-signal(X)))).astype(int)


def model():
    return HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=8, max_bins=32, random_state=0)


def student(X_train, y_train, X_recipients, teacher_probability):
    """Fit on the training rows plus the recipients whose pseudo-label the teacher is confident about."""
    keep = np.abs(teacher_probability - 0.5) > CONFIDENCE
    X = np.vstack([X_train, X_recipients[keep]])
    y = np.concatenate([y_train, (teacher_probability[keep] > 0.5).astype(int)])
    return model().fit(X, y), keep


def one_replicate(recipients, seed):
    rng = np.random.default_rng(seed)
    signal = make_signal(rng)
    X, y = sample(signal, N_LABELED, rng)                     # genuinely labeled rows: the only rows ever scored by CV
    X_new, y_new = sample(signal, N_FRESH, rng)               # fresh rows from the same generator: the "truth"
    X_rec, y_rec = sample(signal, recipients, rng)            # recipients; y_rec is used only to audit pseudo-label accuracy
    folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=1).split(X, y))
    fold_models = [model().fit(X[a], y[a]) for a, _ in folds]   # ordinary K-fold models: model k never saw fold k

    pooled = {name: np.zeros(N_LABELED) for name in ("supervised", "eligible", "shortcut")}
    fresh = {name: [] for name in pooled}
    audit = {"eligible": [], "shortcut": []}
    for k, (train, valid) in enumerate(folds):
        teachers = {
            "eligible": fold_models[k].predict_proba(X_rec)[:, 1],       # model k: trained on outer-training rows only
            # The shortcut: average every model except k. Each of them was trained on fold k's labels.
            "shortcut": np.mean([fold_models[j].predict_proba(X_rec)[:, 1] for j in range(FOLDS) if j != k], axis=0)}
        fitted = {"supervised": fold_models[k]}
        for name, probability in teachers.items():
            fitted[name], keep = student(X[train], y[train], X_rec, probability)
            audit[name].append((keep.sum(), float(np.mean((probability[keep] > 0.5) == y_rec[keep])) if keep.any() else 0.0))
        for name, fit in fitted.items():
            pooled[name][valid] = fit.predict_proba(X[valid])[:, 1]       # score the untouched genuine fold
            fresh[name].append(roc_auc_score(y_new, fit.predict_proba(X_new)[:, 1]))
    out = {name: (roc_auc_score(y, pooled[name]), float(np.mean(fresh[name]))) for name in pooled}
    out["kept"] = {n: float(np.mean([a[0] for a in audit[n]])) for n in audit}
    out["all_labels_fresh"] = roc_auc_score(y_new, model().fit(X, y).predict_proba(X_new)[:, 1])   # reference: all 600 labels, no pseudo-labels
    out["accuracy"] = {n: float(np.mean([a[1] for a in audit[n]])) for n in audit}
    return out


def run(recipients):
    reps = [one_replicate(recipients, SEED * 100 + i) for i in range(REPLICATES)]
    result = {"recipients": recipients, "replicates": REPLICATES, "labeled_rows": N_LABELED, "fresh_rows": N_FRESH}
    for name in ("supervised", "eligible", "shortcut"):
        result[name] = {"cv": float(np.mean([r[name][0] for r in reps])), "fresh": float(np.mean([r[name][1] for r in reps]))}
    result["kept"] = {n: float(np.mean([r["kept"][n] for r in reps])) for n in ("eligible", "shortcut")}
    result["all_labels_fresh"] = float(np.mean([r["all_labels_fresh"] for r in reps]))
    result["eligible_gain_positive"] = int(sum(r["eligible"][1] > r["supervised"][1] for r in reps))   # datasets where it helped
    result["accuracy"] = {n: float(np.mean([r["accuracy"][n] for r in reps])) for n in ("eligible", "shortcut")}
    return clean(result)
# notebook-end


SPEC = {
    "chapter": 34,
    "chapter_title": "Pseudo-Labeling",
    "subtitle": "A teacher is eligible by the labels it trained on, not by its fold number.",
    "summary": ("Pseudo-labels are only as clean as the labels their teacher saw. One demonstration compares the eligible teacher with the "
                "common shortcut, averaging every fold model except k, and measures how far each pipeline's cross-validation score "
                "sits from its score on fresh rows as the number of recipients grows."),
    "title": "The eligible teacher against the \"all models except k\" teacher, by number of recipients",
    "question": "How much does the shortcut teacher inflate the student's cross-validation score, and what does honest pseudo-labeling add?",
    "why": ("Pseudo-labeling is often reported as a large gain, and the shortcut that produces a large gain is easy to write. Scoring both "
            "pipelines against fresh rows shows how much of the gain is information from the validation labels."),
    "method": ("Eight constructed binary tasks with 600 labeled rows, eight features and a nonlinear log-odds function, plus a pool of unlabeled "
               "recipients whose size is the control. Three fold models are trained as in ordinary 3-fold training: model k excludes fold k. "
               "For fold k the eligible teacher is model k; the shortcut teacher averages the other two models, which were trained on "
               "fold k. Recipients whose teacher probability is above 0.8 or below 0.2 are added to the training rows with the teacher's hard "
               "label, a histogram gradient boosting student is fitted, and it is scored on the untouched labeled fold (pooled AUC) and on "
               "5,000 fresh rows. The supervised model with no pseudo-labels is the baseline."),
    "control": {"key": "recipients", "label": "Unlabeled recipients available for pseudo-labeling",
                "values": [250, 750, 1500, 4500], "default": 1500,
                "value_labels": ["250", "750", "1,500", "4,500"]},
    "source_section": "Trace the Teacher's Training Labels",
    "symbols": ("m_j is the model that excludes fold j, x a recipient, p_k(x) the teacher probability used for outer fold k, and K the number of "
                "folds. The eligible teacher is m_k; the shortcut averages m_j over j not equal to k, and each of those models trained on fold k's labels."),
    "explanation": ("Model k never saw fold k, but the other models did. Averaging them gives pseudo-labels that carry what fold k's labels "
                    "taught those models, so the student arrives at fold k partly informed about it and cross-validation rewards the leak. "
                    "The more recipients, the more of that information reaches the student. The eligible teacher's pseudo-labels carry only "
                    "outer-training information, so the estimate stays honest and the gain is small."),
    "application": ("For every outer fold build teachers from outer-training labels only (a fold model, an inner-fold ensemble), add recipients "
                    "to training only, and score untouched genuine labels. Report the supervised baseline on the same folds, and audit "
                    "the pseudo-labels' accuracy on any rows where the truth is known."),
    "assumptions": ("Constructed data with a stable relationship and a histogram gradient boosting model for teacher and student. The shortcut "
                    "student is also better on fresh rows than the baseline here, because its teachers together saw all 600 labels; that real "
                    "gain is small and roughly what a fit on all 600 labels delivers. What the shortcut inflates is the cross-validation "
                    "estimate. A single run of 3 folds is noisy, so eight datasets are averaged."),
    "prediction": "With 1,500 recipients, the shortcut teacher's cross-validation AUC is how far above the same student's score on fresh rows?",
    "prediction_options": ["Within 0.01", "About 0.05", "About 0.15"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "The shortcut reports 0.775 against 0.721 on fresh rows, 0.054 higher; the eligible teacher reports 0.707 against 0.709.",
        "incorrect": "The shortcut reports 0.775 against 0.721 on fresh rows, 0.054 higher; the eligible teacher reports 0.707 against 0.709. The shortcut's teachers had seen the fold they were scoring.",
    },
    "check": "What does honest pseudo-labeling add over the supervised baseline at 1,500 recipients, and what did the shortcut claim?",
    "answer": ("The eligible teacher improves the fresh-row score from 0.705 to 0.709: a gain of 0.004, small but genuine. The shortcut "
               "claims 0.775 against the baseline's 0.704, a gain of 0.071, while its fresh-row score is 0.721, 0.016 above the baseline."),
    "provenance": "Constructed example: eight seeded synthetic binary tasks, fold models as teachers and a histogram gradient boosting student, measured by the chapter activity.",
    "apply": [
        "Write down, for each pseudo-label, which labels its teacher trained on; a teacher is eligible only if its training labels exclude the fold being scored.",
        "Score each outer fold's untouched genuine labels, never the pseudo-labeled recipients and never rows that duplicate a validation row.",
        "Compare against the supervised baseline on the same folds, and treat a large gain as a reason to re-check the teacher.",
        "Tune confidence, weights and recipient counts inside the outer training data, not on the assessment fold.",
    ],
    "honesty": "Constructed data; honest pseudo-labeling added about 0.01 AUC at most here, and the sizes of every effect are properties of this generator, not a competition result.",
}

EQUATIONS = [{"tex": r"p_k^{\text{eligible}}(x) = m_k(x), \qquad p_k^{\text{shortcut}}(x) = \frac{1}{K-1}\sum_{j \ne k} m_j(x)",
              "alt": "the eligible teacher probability for fold k is the output of model k; the shortcut teacher probability is the average of the outputs of all other models j different from k",
              "basis": "The teacher rule the chapter states in words (Trace the Teacher's Training Labels); written as an equation by the activity."}]
NCOLS = 1
HEIGHT = 4.5
PIPELINES = [("supervised", "Supervised\nbaseline"), ("eligible", "Eligible teacher\n(model k)"), ("shortcut", "All models\nexcept k")]


def draw(ax, result, parameter):
    xs = np.arange(len(PIPELINES))
    cv = [result[k]["cv"] for k, _ in PIPELINES]
    fresh = [result[k]["fresh"] for k, _ in PIPELINES]
    ax.bar(xs - 0.2, cv, width=0.4, color=COLORS["light"], edgecolor=COLORS["ink"], lw=0.6, label="Cross-validation estimate")
    ax.bar(xs + 0.2, fresh, width=0.4, color=COLORS["teal"], edgecolor=COLORS["ink"], lw=0.6, label="AUC on 5,000 fresh rows")
    for x, a, b in zip(xs, cv, fresh):
        ax.text(x - 0.2, a + 0.004, fmt(a), ha="center", va="bottom", fontsize=10)
        ax.text(x + 0.2, b + 0.004, fmt(b), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(xs, [n for _, n in PIPELINES])
    ax.set_ylim(0.62, 0.88)
    ax.set_ylabel("AUC")
    ax.set_xlabel(f"Teacher used for the pseudo-labels, {parameter:,} recipients")
    ax.legend(loc="upper left", frameon=False, fontsize=10)


def explain(result, parameter):
    s, e, w = result["supervised"], result["eligible"], result["shortcut"]
    claimed = float(fmt(w["cv"])) - float(fmt(s["cv"]))
    inflation = float(fmt(w["cv"])) - float(fmt(w["fresh"]))
    honest_gap = float(fmt(e["cv"])) - float(fmt(e["fresh"]))
    honest_gain = float(fmt(e["fresh"])) - float(fmt(s["fresh"]))
    interpretation = (
        f"With {parameter:,} recipients the shortcut teacher reports {fmt(w['cv'])} but its student scores {fmt(w['fresh'])} on fresh rows, "
        f"{fmt(w['cv'])} - {fmt(w['fresh'])} = {fmt(inflation)} of optimism, a claimed gain of {fmt(claimed)} over the baseline's {fmt(s['cv'])}. "
        f"The eligible teacher reports {fmt(e['cv'])} and scores {fmt(e['fresh'])} on fresh rows, {signed(honest_gap)} apart, a real change of "
        f"{signed(honest_gain)} against the baseline's {fmt(s['fresh'])}. The teachers kept {fmt(result['kept']['eligible'], 0)} (eligible) "
        f"and {fmt(result['kept']['shortcut'], 0)} (shortcut) pseudo-labels per fold, with accuracy {fmt(result['accuracy']['eligible'])} and "
        f"{fmt(result['accuracy']['shortcut'])}.")
    steps = [
        f"Shortcut: {fmt(w['cv'])} - {fmt(w['fresh'])} = {fmt(inflation)} (estimate minus fresh rows).",
        f"Eligible: {fmt(e['cv'])} - {fmt(e['fresh'])} = {signed(honest_gap)}.",
        f"Claimed gain of the shortcut: {fmt(w['cv'])} - {fmt(s['cv'])} = {fmt(claimed)}.",
        f"Real gain of the eligible teacher on fresh rows: {fmt(e['fresh'])} - {fmt(s['fresh'])} = {signed(honest_gain)}.",
    ]
    metrics = {"Shortcut CV": fmt(w["cv"]), "Shortcut, fresh rows": fmt(w["fresh"]), "Eligible CV": fmt(e["cv"]),
               "Eligible, fresh rows": fmt(e["fresh"]), "Baseline CV": fmt(s["cv"]), "Baseline, fresh rows": fmt(s["fresh"])}
    alt = (f"Paired bars of cross-validation AUC and fresh-row AUC for the supervised baseline, the eligible teacher and the all-models-except-k "
           f"teacher at {parameter:,} recipients; the shortcut reports {fmt(inflation)} above its fresh-row score.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    sizes = (250, 750, 1500, 4500)
    gaps = [results[n]["shortcut"]["cv"] - results[n]["shortcut"]["fresh"] for n in sizes]
    assert gaps[0] > 0 and gaps[0] < gaps[1] < gaps[2] < gaps[3], "shortcut optimism should grow with recipients"
    for n, res in results.items():
        assert abs(res["eligible"]["cv"] - res["eligible"]["fresh"]) < 0.025, f"eligible estimate should describe fresh rows at {n}"
        assert abs(res["supervised"]["cv"] - res["supervised"]["fresh"]) < 0.025, f"baseline estimate should describe fresh rows at {n}"
        assert abs(res["eligible"]["fresh"] - res["supervised"]["fresh"]) < 0.015, f"honest pseudo-labeling gain should be small at {n}"
        assert res["shortcut"]["cv"] - res["supervised"]["cv"] > 0.02, f"shortcut should claim a gain at {n}"
        assert res["shortcut"]["fresh"] > res["supervised"]["fresh"] - 0.003, f"shortcut student should not be worse on fresh rows at {n}"
        assert res["shortcut"]["fresh"] - res["supervised"]["fresh"] < 0.03, f"real gain of the shortcut student should be small at {n}"
    r = results[1500]
    diff = lambda a, b: fmt(float(fmt(a)) - float(fmt(b)))      # the difference of the two numbers as displayed
    assert fmt(r["shortcut"]["cv"]) == "0.775" and fmt(r["shortcut"]["fresh"]) == "0.721", "prediction feedback numbers"
    assert diff(r["shortcut"]["cv"], r["shortcut"]["fresh"]) == "0.054", "prediction feedback gap"
    assert fmt(r["eligible"]["cv"]) == "0.707" and fmt(r["eligible"]["fresh"]) == "0.709", "prediction feedback numbers"
    assert fmt(r["supervised"]["fresh"]) == "0.705" and fmt(r["supervised"]["cv"]) == "0.704", "check answer baseline"
    assert diff(r["eligible"]["fresh"], r["supervised"]["fresh"]) == "0.004", "check answer honest gain"
    assert diff(r["shortcut"]["cv"], r["supervised"]["cv"]) == "0.071", "check answer claimed gain"
    assert diff(r["shortcut"]["fresh"], r["supervised"]["fresh"]) == "0.016", "check answer real gain of the shortcut student"
    assert r["eligible_gain_positive"] >= 6, "check answer: the small honest gain should be positive in most datasets"
    for n, res in results.items():
        assert abs(res["shortcut"]["fresh"] - res["all_labels_fresh"]) < 0.01, f"shortcut student should be close to an all-label fit at {n}"
