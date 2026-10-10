"""Chapter 40: Knowledge Distillation. Hard-label, mixed and pure soft-target students against a forest teacher, by training budget."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

from kaggle_companion.activities._common import clean

warnings.filterwarnings("ignore")   # one-epoch warm-start fits warn about convergence by design
SEED = 40
REPLICATES = 5          # independent worlds per training budget; results are means over them
N_TRAIN, N_DEV, N_TEST, FEATURES = 375, 125, 6000, 8
FLIP = 0.2              # share of training and development labels flipped
EPS = 1e-4
ALPHAS = (0.0, 0.5, 1.0)  # weight on the teacher's soft target; 0 is hard labels only


def generate(n, rng):
    """Eight features and a bent, interacting true probability."""
    X = rng.normal(size=(n, FEATURES))
    z = 1.4 * np.sin(1.3 * X[:, 0]) + 0.9 * X[:, 1] * X[:, 2] + 0.8 * X[:, 3] - 0.7 * np.abs(X[:, 4]) + 0.3
    return X, 1 / (1 + np.exp(-1.5 * z))


def forest():
    return RandomForestClassifier(60, min_samples_leaf=5, random_state=0, n_jobs=1)


def train_student(X, target, X_dev, y_dev, X_test, epochs, seed):
    """Soft-target cross-entropy via duplicated rows: each row counts as a 1 with weight q and as a 0 with weight 1 - q.

    Returns test probabilities after the last epoch and at the epoch with the best development log loss (early stopping)."""
    X2, y2, w2 = np.vstack([X, X]), np.r_[np.ones(len(X)), np.zeros(len(X))], np.r_[target, 1 - target]
    net = MLPClassifier((32, 16), alpha=1e-3, max_iter=1, warm_start=True, learning_rate_init=0.01, batch_size=64, random_state=seed)
    best_loss, best_pred, best_epoch = np.inf, None, 0
    for epoch in range(1, epochs + 1):
        net.fit(X2, y2, sample_weight=w2)                      # one more epoch from the current weights
        if epoch % 3 == 0 or epoch == epochs:
            loss = log_loss(y_dev, np.clip(net.predict_proba(X_dev)[:, 1], EPS, 1 - EPS), labels=[0, 1])
            if loss < best_loss:
                best_loss, best_pred, best_epoch = loss, np.clip(net.predict_proba(X_test)[:, 1], EPS, 1 - EPS), epoch
    parameters = sum(w.size for w in net.coefs_) + sum(b.size for b in net.intercepts_)
    return np.clip(net.predict_proba(X_test)[:, 1], EPS, 1 - EPS), best_pred, best_epoch, parameters


def one_world(epochs, seed):
    rng = np.random.default_rng(seed)
    X, p = generate(N_TRAIN + N_DEV, rng)
    clean_y = (rng.random(len(p)) < p).astype(int)
    y = np.where(rng.random(len(p)) < FLIP, 1 - clean_y, clean_y)      # noisy labels for training and development
    X_test, p_test = generate(N_TEST, rng)
    y_test = (rng.random(N_TEST) < p_test).astype(int)                 # genuine, unflipped test labels
    X_tr, y_tr, X_dev, y_dev = X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]

    soft = np.zeros(N_TRAIN)                                            # cross-fitted teacher: no row is predicted by a model that saw its label
    for fit, out in KFold(5, shuffle=True, random_state=0).split(X_tr):
        soft[out] = forest().fit(X_tr[fit], y_tr[fit]).predict_proba(X_tr[out])[:, 1]
    teacher = forest().fit(X_tr, y_tr)
    pred = np.clip(teacher.predict_proba(X_test)[:, 1], EPS, 1 - EPS)
    out = {"teacher": [log_loss(y_test, pred), roc_auc_score(y_test, pred)],
           "teacher_nodes": sum(t.tree_.node_count for t in teacher.estimators_)}
    for alpha in ALPHAS:
        target = np.clip(alpha * soft + (1 - alpha) * y_tr, 0, 1)
        last, tuned, epoch, parameters = train_student(X_tr, target, X_dev, y_dev, X_test, epochs, seed)
        out[alpha] = [log_loss(y_test, last), roc_auc_score(y_test, last), log_loss(y_test, tuned), roc_auc_score(y_test, tuned), epoch]
        out["student_parameters"] = parameters
    return out


def run(epochs):
    worlds = [one_world(epochs, SEED * 100 + r) for r in range(REPLICATES)]

    def avg(get):
        return float(np.mean([get(w) for w in worlds]))
    arms = {}
    for alpha in ALPHAS:
        arms[str(alpha)] = {"fixed_log_loss": avg(lambda w: w[alpha][0]), "fixed_auc": avg(lambda w: w[alpha][1]),
                            "stopped_log_loss": avg(lambda w: w[alpha][2]), "stopped_auc": avg(lambda w: w[alpha][3]),
                            "stopped_epoch": avg(lambda w: w[alpha][4])}
    return clean({
        "epochs": epochs, "arms": arms,
        "teacher": {"log_loss": avg(lambda w: w["teacher"][0]), "auc": avg(lambda w: w["teacher"][1])},
        "mixed_beats_hard_stopped": avg(lambda w: w[0.5][2] < w[0.0][2]),
        "pure_worse_than_hard_stopped": avg(lambda w: w[1.0][2] > w[0.0][2]),
        "teacher_nodes": avg(lambda w: w["teacher_nodes"]), "student_parameters": worlds[0]["student_parameters"],
        "flip": FLIP, "replicates": REPLICATES,
    })
# notebook-end


SPEC = {
    "chapter": 40,
    "chapter_title": "Knowledge Distillation",
    "subtitle": "Soft teacher targets are one candidate supervision signal: compare them with a tuned hard-label student.",
    "summary": ("A forest teacher's cross-fitted probabilities train a small neural student. One demonstration compares students fed hard "
                "labels, a half-and-half mix and pure soft targets, with and without early stopping on development rows, against the "
                "teacher, on genuine test labels, as the training budget grows."),
    "title": "Hard, mixed and pure soft-target students, by training budget",
    "question": "Do soft teacher targets beat hard labels for a small neural student, or does that depend on how well the hard-label baseline is tuned?",
    "why": ("A distilled student can look like a large win against an untuned hard-label baseline. The chapter asks for the hard-label "
            "endpoint of alpha to stay in the comparison and for the student to be judged on genuine held-out labels."),
    "method": ("A constructed binary task with 375 training and 125 development rows whose labels are flipped at 20%, and 6,000 "
               "test rows with unflipped labels. The teacher is a 60-tree random forest; its cross-fitted probabilities (5 folds) "
               "are the soft targets. The student is a small neural network (833 parameters) trained on alpha times the soft "
               "target plus (1 - alpha) times the hard label with soft cross-entropy, for alpha 0 (hard only), 0.5 and 1 (pure). "
               "The control is the number of epochs allowed. Each student is scored after its last epoch and at the epoch "
               "with the best development log loss. Results are means over 5 worlds."),
    "control": {"key": "epochs", "label": "Training epochs allowed",
                "values": [6, 15, 30, 45], "default": 30,
                "value_labels": ["6", "15", "30", "45"]},
    "source_section": "GBM-to-NN Soft Label Distillation",
    "symbols": ("t is the teacher's cross-fitted probability for a training row, y its (noisy) hard label, alpha the weight on the "
                "soft target, q the mixed target the student is trained on and p the student's predicted probability."),
    "explanation": ("Soft targets are smoother than 0 and 1, so a student trained on them overfits more slowly, which makes it look "
                    "much better than a hard-label student that is left to train to the end. Early stopping on development rows gives the hard-label "
                    "student the same protection for free, and the gap disappears. Pure distillation copies cross-fitted teacher "
                    "predictions, each from a forest that saw only 80% of the rows and their noisy labels, so the student inherits "
                    "that teacher's errors and adds its own approximation error."),
    "application": ("Keep alpha = 0 and early stopping in the comparison, choose alpha and the stopping epoch on development rows, and "
                    "report the student against the teacher and a tuned hard-label model on genuine labels, with artifact size."),
    "assumptions": ("Constructed data with 20% flipped labels, a random forest standing in for the chapter's GBM teacher and a small "
                    "scikit-learn network as the "
                    "student, trained with soft cross-entropy. This is the chapter's soft-target variant, not matched-temperature "
                    "distillation. Early stopping uses development labels that are as noisy as the training labels, so the stopping epoch is itself noisy. "
                    "Other noise levels, student sizes and teachers can change the ordering."),
    "prediction": ("After 30 epochs with no early stopping, a half-soft student (alpha 0.5) has a log loss of 0.644 against 0.991 for the "
                   "hard-label student. If both are early-stopped on development rows instead, how do they compare?"),
    "prediction_options": ["The half-soft student is still clearly better", "About the same, within 0.01", "The hard-label student is clearly better"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "Early-stopped, the hard-label student scores 0.598 and the half-soft student 0.602: within 0.005. The big gap was the untuned baseline overfitting.",
        "incorrect": "Early-stopped, the hard-label student scores 0.598 and the half-soft student 0.602: within 0.005, not a clear win for either. The big gap was the untuned baseline overfitting.",
    },
    "check": "Pure distillation (alpha = 1) stays worse than hard labels after early stopping and behind the teacher. Why, and what does that say about when to distill?",
    "answer": ("Alpha = 1 trains the student on the teacher's cross-fitted predictions, which were fitted to the noisy labels on 80% of the "
               "rows, so the student inherits those errors: 0.627 against 0.598 for hard labels and 0.594 for the teacher. Here the "
               "hard-label student (833 parameters) is already within 0.005 of the teacher (about 4,100 tree nodes), so distillation "
               "was not needed to compress it. Distill when a tuned hard-label student falls short of the teacher, not by default."),
    "provenance": "Constructed example: seeded synthetic data, a random forest teacher and a small neural student, measured by the chapter activity.",
    "apply": [
        "Always include alpha = 0 (hard labels only) and early stopping on development rows as the baseline before crediting soft targets.",
        "Generate teacher targets with cross-fitting inside each outer training partition; a teacher that saw a row's label leaks it into the student.",
        "Judge student, teacher and hard-label baseline on the same genuine labels, and report parameters or nodes and latency beside the loss.",
        "Prefer a mixed alpha chosen on development rows over pure distillation, and treat any gain as a hypothesis until the blend is assessed.",
    ],
    "honesty": ("Constructed data, one noise level and one student. Soft targets slow overfitting but did not beat an early-stopped hard-label "
                "student here; a result in the other direction would need its own evidence."),
}

EQUATIONS = [{"tex": r"q_i = \alpha\, t_i + (1-\alpha)\, y_i, \qquad \mathcal{L} = -\tfrac{1}{n}\sum_i \left[q_i \log p_i + (1-q_i)\log(1-p_i)\right]",
              "alt": "The student target q i equals alpha times the teacher probability t i plus one minus alpha times the hard label y i, and the loss is the average soft cross-entropy between q i and the student probability p i",
              "basis": "The chapter's alpha-mixed soft-target objective (code and prose in Chapter 40, GBM-to-NN Soft Label Distillation); no display equation."}]
NCOLS = 2
HEIGHT = 4.4
ARM_LABELS = [("0.0", "Hard labels\n(alpha 0)"), ("0.5", "Mixed\n(alpha 0.5)"), ("1.0", "Pure soft\n(alpha 1)")]


def draw(axes, result, parameter):
    xs = list(range(len(ARM_LABELS)))
    panels = [(axes[0], "log_loss", "Test log loss (lower is better)", (0.0, 1.5), 0.02, 3),
              (axes[1], "auc", "Test AUC (higher is better)", (0.5, 0.92), 0.01, 3)]
    for ax, metric, label, ylim, pad, digits in panels:
        fixed = [result["arms"][k]["fixed_" + metric] for k, _ in ARM_LABELS]
        stopped = [result["arms"][k]["stopped_" + metric] for k, _ in ARM_LABELS]
        ax.bar([x - 0.2 for x in xs], fixed, width=0.4, color=COLORS["light"], edgecolor=COLORS["ink"], lw=0.6,
               label=f"After epoch {parameter}")
        ax.bar([x + 0.2 for x in xs], stopped, width=0.4, color=COLORS["teal"], edgecolor=COLORS["ink"], lw=0.6,
               label="Early-stopped on dev rows")
        for x, a, b in zip(xs, fixed, stopped):
            floor = ylim[0] + 0.02 * (ylim[1] - ylim[0])           # labels sit at the foot of each bar, clear of the teacher line
            ax.text(x - 0.2, floor, fmt(a, digits), ha="center", va="bottom", fontsize=10, color=COLORS["ink"])
            ax.text(x + 0.2, floor, fmt(b, digits), ha="center", va="bottom", fontsize=10, color="white")
        ax.axhline(result["teacher"][metric], color=COLORS["terracotta"], ls=(0, (4, 3)), lw=1.4,
                   label=f"Forest teacher: {fmt(result['teacher'][metric], 3)}")
        ax.set_xticks(xs, [name for _, name in ARM_LABELS], fontsize=10)
        ax.set_ylim(*ylim)
        ax.set_ylabel(label)
        ax.set_xlabel("Student training target")
    axes[0].legend(loc="upper right", frameon=False, fontsize=10)
    axes[1].legend(loc="upper right", frameon=False, fontsize=10)


def explain(result, parameter):
    a, t = result["arms"], result["teacher"]
    hard, mixed, pure = a["0.0"], a["0.5"], a["1.0"]
    naive_gain = hard["fixed_log_loss"] - mixed["fixed_log_loss"]
    tuned_gain = hard["stopped_log_loss"] - mixed["stopped_log_loss"]
    interpretation = (
        f"With {parameter} epochs allowed and no early stopping, the hard-label student scores {fmt(hard['fixed_log_loss'])} and the half-soft "
        f"student {fmt(mixed['fixed_log_loss'])}, a gain of {fmt(hard['fixed_log_loss'])} - {fmt(mixed['fixed_log_loss'])} = {signed(naive_gain)}. "
        f"Early-stopped on development rows they score {fmt(hard['stopped_log_loss'])} and {fmt(mixed['stopped_log_loss'])}, a gain of "
        f"{fmt(hard['stopped_log_loss'])} - {fmt(mixed['stopped_log_loss'])} = {signed(tuned_gain)}. Pure soft targets score "
        f"{fmt(pure['stopped_log_loss'])} early-stopped, and the forest teacher {fmt(t['log_loss'])}. The student has "
        f"{result['student_parameters']:,} parameters against about {round(result['teacher_nodes']):,} tree nodes. "
        f"The mixed student beat the hard-label student in {round(100 * result['mixed_beats_hard_stopped'])}% of {result['replicates']} worlds once both were early-stopped.")
    steps = [
        f"Apparent gain from soft targets, no early stopping: {fmt(hard['fixed_log_loss'])} - {fmt(mixed['fixed_log_loss'])} = {signed(naive_gain)}.",
        f"Gain once both are early-stopped: {fmt(hard['stopped_log_loss'])} - {fmt(mixed['stopped_log_loss'])} = {signed(tuned_gain)}.",
        f"Pure soft against hard labels, early-stopped: {fmt(pure['stopped_log_loss'])} - {fmt(hard['stopped_log_loss'])} = {signed(pure['stopped_log_loss'] - hard['stopped_log_loss'])}.",
        f"Tuned hard-label student against the teacher: {fmt(hard['stopped_log_loss'])} - {fmt(t['log_loss'])} = {signed(hard['stopped_log_loss'] - t['log_loss'])}.",
    ]
    metrics = {"Hard labels, last epoch": fmt(hard["fixed_log_loss"]), "Mixed, last epoch": fmt(mixed["fixed_log_loss"]),
               "Hard labels, early-stopped": fmt(hard["stopped_log_loss"]), "Mixed, early-stopped": fmt(mixed["stopped_log_loss"]),
               "Pure soft, early-stopped": fmt(pure["stopped_log_loss"]), "Forest teacher": fmt(t["log_loss"])}
    alt = (f"Left: test log loss for students trained on hard labels, a half-soft mix and pure soft targets after {parameter} epochs "
           f"and when early-stopped on development rows, with the forest teacher at {fmt(t['log_loss'])}. Right: the same comparison by AUC.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for e, res in results.items():
        a, t = res["arms"], res["teacher"]
        assert a["0.0"]["stopped_log_loss"] < t["log_loss"] + 0.015, f"tuned hard student close to the teacher at {e}"
        assert abs(a["0.5"]["stopped_log_loss"] - a["0.0"]["stopped_log_loss"]) < 0.01, f"mixed ties hard once early-stopped at {e}"
        assert a["1.0"]["stopped_log_loss"] > a["0.0"]["stopped_log_loss"] + 0.02, f"pure soft is worse at {e}"
        assert res["pure_worse_than_hard_stopped"] == 1.0
        assert a["1.0"]["stopped_log_loss"] > t["log_loss"] + 0.02, f"pure student stays behind the teacher at {e}"
    for e in (15, 30, 45):
        a = results[e]["arms"]
        assert a["0.0"]["fixed_log_loss"] > a["0.5"]["fixed_log_loss"] + 0.05, f"soft targets look better without early stopping at {e}"
    assert results[45]["arms"]["0.0"]["fixed_log_loss"] > 2 * results[45]["arms"]["0.0"]["stopped_log_loss"], "untuned hard student overfits badly"
    assert results[6]["arms"]["0.0"]["fixed_log_loss"] <= results[6]["arms"]["0.5"]["fixed_log_loss"] + 0.01, "short budget: no advantage for soft"
    s = results[30]
    assert fmt(s["arms"]["0.5"]["fixed_log_loss"]) == "0.644" and fmt(s["arms"]["0.0"]["fixed_log_loss"]) == "0.991", "prediction numbers"
    assert fmt(s["arms"]["0.0"]["stopped_log_loss"]) == "0.598" and fmt(s["arms"]["0.5"]["stopped_log_loss"]) == "0.602"
    assert s["arms"]["0.5"]["stopped_log_loss"] - s["arms"]["0.0"]["stopped_log_loss"] < 0.005, "within 0.005"
    assert fmt(s["arms"]["1.0"]["stopped_log_loss"]) == "0.627" and fmt(s["teacher"]["log_loss"]) == "0.594"
    assert s["student_parameters"] == 833 and 4000 < s["teacher_nodes"] < 4200, "artifact sizes in the answer"
    assert s["arms"]["0.0"]["stopped_log_loss"] - s["teacher"]["log_loss"] < 0.005, "hard student within 0.005 of the teacher"
