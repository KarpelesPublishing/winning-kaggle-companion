"""Chapter 1: What Actually Wins on Kaggle. A model upgrade scored by random folds, subject-held-out folds and new subjects."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
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
# notebook-end


SPEC = {
    "chapter": 1,
    "chapter_title": "What Actually Wins on Kaggle",
    "subtitle": "Match the representation to the task, then test the complete pipeline.",
    "summary": ("Familiar-subject rows and new-subject rows share columns but need different assessment. One demonstration measures "
                "what a model upgrade does to a random-fold estimate, a subject-held-out estimate and the score on subjects the model never saw."),
    "title": "A model upgrade, scored three ways",
    "question": ("When the model gets more capacity on the same columns, which assessment rewards the upgrade, and which one "
                 "tracks the score on subjects the model never saw?"),
    "why": ("A competitor who suspects the algorithm is the problem will try a bigger model. If the split does not match the test, "
            "the experiment cannot say whether the bigger model helped, and the choice made from it can be the wrong one."),
    "method": ("Eight constructed studies of a sensor task. Each has 36 subjects with 40 windows each. Three noisy activity channels carry a signal "
               "shared by everyone; four more channels are a fixed fingerprint of the subject, and every subject has its own label bias. "
               "Boosted trees of the capacity chosen by the control (leaves per tree) are scored by AUC three ways: random 5-fold on "
               "the windows, 5-fold with whole subjects held out, and a fresh set of 150 new subjects. Scores pool all held-out "
               "predictions per study and average over the studies; the model with 2 leaves per tree is the comparison."),
    "control": {"key": "leaves", "label": "Model capacity (leaves per tree)",
                "values": [2, 8, 31], "default": 31,
                "value_labels": ["2: one split per tree", "8", "31: deep trees"]},
    "source_section": "Start with the Prediction Contract",
    "symbols": ("AUC_random, AUC_subject and AUC_new are the pooled AUCs from random folds, subject-held-out folds and new subjects; "
                "the estimate error is a fold estimate minus the new-subject score, so a positive value is optimism."),
    "explanation": ("Because rows from the same subject sit on both sides of a random split, a flexible model can use the fingerprint "
                    "channels to look up each subject's bias, and the random-fold score rewards that. New subjects have fingerprints "
                    "and biases it has never seen, so the same lookup is noise there. Subject-held-out folds remove the lookup from the "
                    "estimate. Move the control to see where the random-fold ranking and the new-subject ranking of the models disagree."),
    "application": ("Before choosing between models, write down what a new row will be (a new subject, a later date or a familiar entity) "
                    "and split development data to match. Judge each upgrade by the matching estimate, not by the one that is cheapest to run."),
    "assumptions": ("Constructed data and a boosted-tree model from scikit-learn standing in for a gradient-boosting library. The fingerprint "
                    "channels are built to be useless for new subjects, which is the chapter's familiar-subject case. In this generator "
                    "the simplest model scores highest on new subjects; with different data a larger model can help there too, and the "
                    "lesson is that only a matching split can tell. The subject-fold estimate is noisy and sits below the "
                    "new-subject score (by 0.041 at 2 leaves): each fold model trains on fewer subjects, and the pooled "
                    "predictions mix five fold models whose overall level differs."),
    "prediction": ("Upgrade from trees with 2 leaves to trees with 31 leaves on the same columns. What happens to the random-fold "
                   "estimate and to the score on new subjects?"),
    "prediction_options": ["Both rise by about 0.03",
                           "The random-fold estimate rises by about 0.03 while the new-subject score falls by about 0.04",
                           "Neither moves by more than 0.01"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "Random folds rise from 0.707 to 0.735 while the score on new subjects falls from 0.625 to 0.585.",
        "incorrect": "Random folds rise from 0.707 to 0.735, which looks like progress, while the score on new subjects falls from 0.625 to 0.585.",
    },
    "check": "Which of the three assessments would have told you not to adopt the bigger model, and what does that say about where to spend effort first?",
    "answer": ("The subject-held-out folds did not reward it: they changed by -0.008 at 31 leaves, within the noise of eight "
               "studies, while random folds moved by +0.027 and the new-subject score by -0.040. Random folds sit 0.149 above the new-subject score and subject folds 0.009 below it. "
               "The model change could not resolve a split that does not match the test; fixing the split came first."),
    "provenance": "Constructed example: eight seeded synthetic subject studies and boosted trees, measured by the chapter activity.",
    "apply": [
        "Write the contract before the next run: what will a new row be, what inputs exist at prediction time, and how is it scored.",
        "Split development data by that unit (subject, entity or date block) before comparing any two models.",
        "If a random-fold gain and a held-out-unit result disagree, trust the held-out-unit result and look for columns that identify the unit.",
        "Treat a suspiciously large random-fold gain from added capacity as a question about the split, not as proof the model fits the task better.",
    ],
    "honesty": ("Constructed data. The fingerprint is built to carry no information about new subjects, and the sizes of these effects "
                "are properties of this generator, not a competition result."),
}

EQUATIONS = [{"tex": r"\text{error} = \mathrm{AUC}_{\text{fold estimate}} - \mathrm{AUC}_{\text{new subjects}}",
              "alt": "error equals the AUC from a fold estimate minus the AUC on new subjects",
              "basis": "The activity's own measure of how far an assessment drifts from the test population; not a display equation in the manuscript."}]
NCOLS = 1
HEIGHT = 4.4
SCHEMES = [("random_folds", "Random folds"), ("subject_folds", "Subject-held-out folds"), ("new_subjects", "New subjects (the test)")]


def draw(ax, result, parameter):
    xs = list(range(len(SCHEMES)))
    means = [result["model"][k] for k, _ in SCHEMES]
    colors = [COLORS["terracotta"], COLORS["light"], COLORS["teal"]]
    ax.bar(xs, means, width=0.55, color=colors, edgecolor=COLORS["ink"], lw=0.6)
    for x, (k, _) in zip(xs, SCHEMES):
        dots = result["per_study"][k]
        ax.scatter([x + (i - (len(dots) - 1) / 2) * 0.04 for i in range(len(dots))], dots, s=14, color=COLORS["ink"], zorder=3,
                   label="One study" if x == 0 else None)
        ax.text(x, max(max(dots), result["baseline"][k]) + 0.014, fmt(result["model"][k]), ha="center", va="bottom", fontsize=10)
    if result["leaves"] != result["baseline_leaves"]:
        ax.scatter(xs, [result["baseline"][k] for k, _ in SCHEMES], marker="D", s=46, facecolor="white", edgecolor=COLORS["gold"],
                   lw=1.6, zorder=4, label=f"Same assessment, {result['baseline_leaves']} leaves per tree")
    ax.axhline(result["model"]["new_subjects"], color=COLORS["gold"], ls=(0, (4, 3)), lw=1.4,
               label=f"New-subject score: {fmt(result['model']['new_subjects'])}")
    ax.set_xticks(xs, [name for _, name in SCHEMES])
    ax.set_ylim(0.45, 0.95)
    ax.set_ylabel("AUC (pooled held-out predictions)")
    ax.set_xlabel(f"Assessment, trees with {parameter} leaves")
    ax.legend(loc="upper right", frameon=False, fontsize=10, ncol=1)


def explain(result, parameter):
    m, b = result["model"], result["baseline"]
    gain_r = m["random_folds"] - b["random_folds"]
    gain_s = m["subject_folds"] - b["subject_folds"]
    gain_n = m["new_subjects"] - b["new_subjects"]
    err_r = m["random_folds"] - m["new_subjects"]
    err_s = m["subject_folds"] - m["new_subjects"]
    if parameter == result["baseline_leaves"]:
        upgrade = "This is the comparison model itself, so no upgrade is measured; raise the capacity to see one."
    elif gain_r > 0.01 and gain_n < -0.01:
        upgrade = "The upgrade looks like progress in random folds and is a loss on new subjects."
    elif gain_n > 0.01:
        upgrade = "Here the upgrade also helps on new subjects."
    else:
        upgrade = "Here the upgrade changes the new-subject score very little."
    interpretation = (
        f"With {parameter} leaves per tree, random folds report {fmt(m['random_folds'])} and the score on new subjects is "
        f"{fmt(m['new_subjects'])}, so {fmt(m['random_folds'])} - {fmt(m['new_subjects'])} = {fmt(err_r)} of optimism. "
        f"Subject-held-out folds report {fmt(m['subject_folds'])}, {signed(err_s)} from the new-subject score. "
        f"Against the {result['baseline_leaves']}-leaf model the upgrade moves random folds by {signed(gain_r)}, "
        f"subject folds by {signed(gain_s)} and new subjects by {signed(gain_n)}. {upgrade}")
    steps = [
        f"Random folds: {fmt(m['random_folds'])} - {fmt(m['new_subjects'])} = {signed(err_r)} (estimate minus new subjects).",
        f"Subject-held-out folds: {fmt(m['subject_folds'])} - {fmt(m['new_subjects'])} = {signed(err_s)}.",
        f"Upgrade effect on random folds: {fmt(m['random_folds'])} - {fmt(b['random_folds'])} = {signed(gain_r)}.",
        f"Upgrade effect on new subjects: {fmt(m['new_subjects'])} - {fmt(b['new_subjects'])} = {signed(gain_n)}.",
    ]
    metrics = {"Random-fold AUC": fmt(m["random_folds"]), "Subject-fold AUC": fmt(m["subject_folds"]),
               "New-subject AUC": fmt(m["new_subjects"]), "Random-fold optimism": signed(err_r),
               "Upgrade effect, new subjects": signed(gain_n)}
    alt = (f"Bars of AUC from random folds, subject-held-out folds and new subjects for trees with {parameter} leaves, with a dot for each of "
           f"{result['replicates']} studies and a dashed line at the new-subject score of {fmt(m['new_subjects'])}; random folds are "
           f"{signed(err_r)} from it.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    small, mid, big = results[2], results[8], results[31]
    for leaves, res in results.items():
        m = res["model"]
        assert m["random_folds"] - m["new_subjects"] > 0.05, f"random folds not optimistic at {leaves}"
        assert abs(m["subject_folds"] - m["new_subjects"]) < abs(m["random_folds"] - m["new_subjects"]) - 0.03, \
            f"subject folds not closer to new subjects at {leaves}"
        assert abs(m["subject_folds"] - m["new_subjects"]) < 0.06, f"subject folds far from new subjects at {leaves}"
    for res in (mid, big):
        b, m = res["baseline"], res["model"]
        assert m["random_folds"] - b["random_folds"] > 0.01, "upgrade should raise the random-fold estimate"
        assert m["new_subjects"] - b["new_subjects"] < -0.01, "upgrade should lower the new-subject score"
        assert m["subject_folds"] - b["subject_folds"] < 0.01, "subject folds should not reward the upgrade"
    assert small["model"] == small["baseline"], "2 leaves is the comparison model"
    # Ranking disagreement: random folds rank the 2-leaf model last, new subjects rank it first.
    assert small["model"]["random_folds"] < min(mid["model"]["random_folds"], big["model"]["random_folds"])
    assert small["model"]["new_subjects"] > max(mid["model"]["new_subjects"], big["model"]["new_subjects"])
    m, b = big["model"], big["baseline"]
    assert (fmt(b["random_folds"]), fmt(m["random_folds"])) == ("0.707", "0.735"), "prediction feedback numbers"
    assert (fmt(b["new_subjects"]), fmt(m["new_subjects"])) == ("0.625", "0.585"), "prediction feedback numbers"
    assert fmt(m["random_folds"] - b["random_folds"]) == "0.027" and fmt(m["new_subjects"] - b["new_subjects"]) == "-0.040"
    assert fmt(m["subject_folds"] - b["subject_folds"]) == "-0.008", "answer number"
    assert fmt(m["random_folds"] - m["new_subjects"]) == "0.149" and fmt(m["subject_folds"] - m["new_subjects"]) == "-0.009"
    assert fmt(small["model"]["subject_folds"] - small["model"]["new_subjects"]) == "-0.041", "assumptions number"
    # "Within the noise of eight studies": the paired subject-fold change is under two standard errors.
    diff = np.array(big["per_study"]["subject_folds"]) - np.array(small["per_study"]["subject_folds"])
    assert abs(diff.mean()) < 2 * diff.std(ddof=1) / np.sqrt(len(diff)), "subject-fold change should be within noise"
