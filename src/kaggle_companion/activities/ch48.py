"""Chapter 48: Pseudo-Labeling for CV. A confidence threshold, a teacher rebuilt inside each outer fold, and the gain on fresh rows."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

from kaggle_companion.activities._common import clean

SEED = 48
DRAWS = 16                  # independent worlds; every estimate is a mean over draws
N_LABELED, N_POOL, N_FRESH = 100, 1000, 2000
DIM, NOISE = 6, 1.6        # features per row and within-cluster noise


def make_world(rng):
    """Three clusters per class in six dimensions: the same world supplies labeled rows, the unlabeled pool and fresh rows."""
    return rng.normal(size=(3, DIM)) * 2.0, rng.normal(size=(3, DIM)) * 2.0


def sample(world, n, rng):
    y = (rng.random(n) < 0.5).astype(int)
    cluster = rng.integers(0, 3, n)
    X = np.where(y[:, None] == 1, world[1][cluster], world[0][cluster]) + NOISE * rng.normal(size=(n, DIM))
    return X, y


def teacher(seed):
    """High-capacity teacher: it memorises the labeled rows it is trained on (which is what makes leakage matter)."""
    return ExtraTreesClassifier(n_estimators=30, random_state=seed, n_jobs=1)


def student(seed):
    return ExtraTreesClassifier(n_estimators=30, min_samples_leaf=3, random_state=seed, n_jobs=1)


def pseudo_set(teacher_X, teacher_y, pool, threshold, seed):
    """Fit a teacher, keep the pool rows whose top class probability reaches the threshold, and return rows and pseudo-labels."""
    prob = teacher(seed).fit(teacher_X, teacher_y).predict_proba(pool)
    keep = prob.max(1) >= threshold
    return pool[keep], prob.argmax(1)[keep], keep, prob.argmax(1)


def run(threshold):
    rng = np.random.default_rng(SEED)
    keys = ("coverage", "pseudo_accuracy", "supervised", "teacher_alone", "student", "leaky_estimate", "leaky_true", "nested_estimate", "nested_true",
            "supervised_estimate")
    rows = {k: [] for k in keys}
    for draw in range(DRAWS):
        world = make_world(rng)
        X, y = sample(world, N_LABELED, rng)
        pool, pool_y = sample(world, N_POOL, rng)
        X_fresh, y_fresh = sample(world, N_FRESH, rng)
        # Final pipeline: teacher on all labeled rows, student on labeled rows plus confident pseudo-labels.
        px, py, keep, guess = pseudo_set(X, y, pool, threshold, draw)
        rows["coverage"].append(keep.mean())
        rows["pseudo_accuracy"].append((py == pool_y[keep]).mean() if keep.any() else np.nan)
        rows["supervised"].append(student(draw).fit(X, y).score(X_fresh, y_fresh))
        rows["teacher_alone"].append(teacher(draw).fit(X, y).score(X_fresh, y_fresh))   # the other supervised candidate
        rows["student"].append(student(draw).fit(np.vstack([X, px]), np.r_[y, py]).score(X_fresh, y_fresh))
        # Three-fold cross-validation of the whole pipeline, with the teacher either global (leaky) or rebuilt per fold.
        cells = {k: [] for k in ("leaky_estimate", "leaky_true", "nested_estimate", "nested_true", "supervised_estimate")}
        for fit, hold in StratifiedKFold(3, shuffle=True, random_state=draw).split(X, y):
            for tag, (tx, ty) in (("leaky", (X, y)), ("nested", (X[fit], y[fit]))):    # global teacher sees the hold-out labels
                px_f, py_f, _, _ = pseudo_set(tx, ty, pool, threshold, draw)
                model = student(draw).fit(np.vstack([X[fit], px_f]), np.r_[y[fit], py_f])
                cells[tag + "_estimate"].append(model.score(X[hold], y[hold]))
                cells[tag + "_true"].append(model.score(X_fresh, y_fresh))
            cells["supervised_estimate"].append(student(draw).fit(X[fit], y[fit]).score(X[hold], y[hold]))
        for k, v in cells.items():
            rows[k].append(np.mean(v))
    mean = {k: float(np.nanmean(v)) for k, v in rows.items()}
    gain = np.array(rows["student"]) - np.array(rows["supervised"])
    gain_vs_teacher = np.array(rows["student"]) - np.array(rows["teacher_alone"])
    leak = np.array(rows["leaky_estimate"]) - np.array(rows["leaky_true"])
    honest = np.array(rows["nested_estimate"]) - np.array(rows["nested_true"])
    return clean({
        "threshold": threshold, **mean,
        "gain": gain.mean(), "gain_se": gain.std(ddof=1) / np.sqrt(DRAWS), "gain_win_rate": float((gain > 0).mean()),
        "gain_vs_teacher": gain_vs_teacher.mean(), "gain_vs_teacher_se": gain_vs_teacher.std(ddof=1) / np.sqrt(DRAWS),
        "leaky_error": leak.mean(), "leaky_error_se": leak.std(ddof=1) / np.sqrt(DRAWS),
        "nested_error": honest.mean(), "nested_error_se": honest.std(ddof=1) / np.sqrt(DRAWS),
        "per_draw": {"supervised": rows["supervised"], "student": rows["student"]},
        "labeled": N_LABELED, "pool": N_POOL, "fresh": N_FRESH, "draws": DRAWS,
    })
# notebook-end


SPEC = {
    "chapter": 48,
    "chapter_title": "Pseudo-Labeling for CV",
    "subtitle": "Rebuild the teacher inside each outer fold, treat confidence as a selection rule rather than proof, and score the student on genuine labels.",
    "summary": ("Pseudo-labels add targets the teacher chose. One demonstration varies the confidence threshold and measures the "
                "pseudo-label accuracy, the student's gain on fresh rows, and how far cross-validation overstates the score when the "
                "teacher has already seen the validation labels."),
    "title": "Pseudo-label student against a supervised model, with and without teacher leakage, by confidence threshold",
    "question": "Does a stricter confidence threshold make pseudo-labeling pay, and does it remove the leak from a teacher that saw the validation labels?",
    "why": ("The chapter says a confidence threshold controls selection but does not prove correctness, and cannot repair teacher "
            "leakage. Sixteen constructed worlds put numbers on both statements: which threshold helps the student, and how large the "
            "cross-validation error from a leaky teacher is at each."),
    "method": ("Constructed binary task in six dimensions with three noisy clusters per class: 100 labeled rows, 1,000 unlabeled "
               "pool rows with known labels used only for scoring, and 2,000 fresh rows. The teacher is an extremely randomized "
               "trees classifier (it memorises its training rows); the student is a second trees model trained on the labeled rows plus "
               "the pool rows whose top teacher probability reaches the control's threshold, using the teacher's labels. The student is "
               "compared with the same model trained on labeled rows only. Three-fold cross-validation of the whole pipeline is run "
               "twice: with one teacher trained on all 100 labeled rows (so it has seen every validation label) and with a teacher "
               "rebuilt from each fold's training rows. Each estimate is compared with the score on fresh rows of the same student models, "
               "averaged over 16 worlds."),
    "control": {"key": "threshold", "label": "Confidence threshold for keeping a pseudo-label",
                "values": [0.6, 0.8, 0.9, 0.99], "default": 0.9,
                "value_labels": ["0.6: keep most of the pool", "0.8", "0.9", "0.99: keep only the surest rows"]},
    "source_section": "What Can Go Wrong",
    "symbols": ("The threshold tau keeps a pool row when the teacher's highest class probability is at least tau. Coverage is the share "
                "of the pool kept, pseudo-label accuracy the share of kept rows whose teacher label is correct, and the gain is the "
                "student's accuracy on fresh rows minus the supervised model's."),
    "explanation": ("Raising the threshold makes the kept pseudo-labels more accurate and far fewer, so the student has less new "
                    "information to learn from. Here the gain is largest at the loosest threshold, where the labels are least accurate, "
                    "and gone at the strictest. Separately, a teacher trained on the validation fold's labels passes that knowledge to "
                    "the student through its pseudo-labels, so cross-validation overstates the score at every threshold that keeps "
                    "a meaningful share of the pool, however accurate the kept labels are."),
    "application": ("Rebuild the teacher from outer-training labels inside every fold, choose the threshold by the student's score on "
                    "genuine held-out labels rather than by pseudo-label accuracy, and keep the best supervised-only model, "
                    "including the teacher itself, as the comparator."),
    "assumptions": ("Constructed clusters, one teacher family and a small labeled set; the pool has the same distribution as the labeled "
                    "rows, which is the favourable case (no confirmation of a shifted region). The leak matters here because the "
                    "teacher memorises; a smoother teacher leaks less. The nested pipeline is unbiased by construction (its "
                    "teacher never sees the scored labels); its remaining error, up to about 0.015 here, is noise from scoring on "
                    "100 labeled rows per world, shared across thresholds because the same rows are reused. Gains of 0.01 or less "
                    "are comparable to their standard errors. The supervised comparator has the student's settings; the memorising "
                    "teacher on its own scores 0.875 on fresh rows, so against it the student gains only 0.010 at threshold 0.6 "
                    "and nothing at 0.9."),
    "prediction": "At threshold 0.9, the kept pseudo-labels are about 98% correct. How much does the student improve on the supervised model on fresh rows?",
    "prediction_options": ["Clearly, by more than 0.02 accuracy", "Barely, by less than 0.01 accuracy", "It is worse than the supervised model"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "The gain is 0.006 (standard error 0.003): 98.1% accurate labels on 35.3% of the pool add little new information.",
        "incorrect": "The gain is 0.006 (standard error 0.003): the kept labels are 98.1% accurate but cover only 35.3% of the pool, so they add little the labeled rows did not already teach.",
    },
    "check": "At threshold 0.9 the labels are 98% correct, yet the leaky cross-validation still overstates the fresh score. Why, and what fixes it?",
    "answer": ("Leakage is about which labels the teacher saw, not whether its pseudo-labels are right. A teacher trained on all labeled "
               "rows has seen the validation fold, and its pseudo-labels carry that back to the student: the leaky estimate is 0.033 above the "
               "fresh score (standard error 0.009), against 0.003 for a teacher rebuilt inside each fold. Rebuild the teacher per outer fold; "
               "a stricter threshold only shrinks the leak by shrinking coverage."),
    "provenance": "Constructed example: seeded synthetic clusters and extremely randomized trees teacher and student, measured by the chapter activity.",
    "apply": [
        "Rebuild the teacher inside every outer fold from that fold's training labels, and generate pseudo-labels only for permitted recipients.",
        "Compare the student with the best supervised-only model, the teacher included, on genuine held-out labels; do not judge a threshold by the accuracy of its pseudo-labels.",
        "Report coverage with accuracy: a threshold that keeps few rows can be accurate and useless.",
        "Audit the kept rows by class, source or subject where genuine labels exist, to catch concentrated pseudo-labels.",
    ],
    "honesty": "Constructed data and a teacher that memorises; the sizes of these effects are properties of this generator, not a competition result.",
}

EQUATIONS = [{"tex": r"\text{keep pool row } i \iff \max_k \hat p_{ik} \ge \tau",
              "alt": "keep pool row i if and only if the largest teacher class probability for row i is at least tau",
              "basis": "The confidence-filtering rule of Confidence Filtering and The Core Idea; the manuscript states it in prose, so this is the activity's own label."}]
NCOLS = 2
HEIGHT = 4.4


def draw(axes, result, parameter):
    ax, bx = axes
    groups = [("Teacher saw the\nvalidation labels", result["leaky_estimate"], result["leaky_true"], COLORS["terracotta"]),
              ("Teacher rebuilt\nper fold", result["nested_estimate"], result["nested_true"], COLORS["teal"])]
    for x, (_, est, true, color) in enumerate(groups):
        ax.plot([x, x], [true, est], color=color, lw=3, zorder=1)
        ax.scatter([x], [est], s=110, marker="D", color=color, zorder=3, edgecolor="white", linewidth=1,
                   label="Cross-validation estimate" if x == 0 else None)
        ax.scatter([x], [true], s=90, color=COLORS["ink"], zorder=3, label="Score on fresh rows" if x == 0 else None)
        up = 1 if est >= true else -1             # the higher marker's label goes above it, the lower one's below
        ax.annotate(fmt(est), (x, est), xytext=(14, 7 * up), textcoords="offset points", va="center", fontsize=10)
        ax.annotate(fmt(true), (x, true), xytext=(14, -7 * up), textcoords="offset points", va="center", fontsize=10)
    ax.set_xticks([0, 1], [g[0] for g in groups])
    ax.set_xlim(-0.5, 1.7)
    ax.set_ylim(0.78, 0.96)
    ax.set_ylabel("Accuracy of the student pipeline")
    ax.set_xlabel("Cross-validation pipeline")
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    sup, stu = result["per_draw"]["supervised"], result["per_draw"]["student"]
    for a, b in zip(sup, stu):
        bx.plot([0, 1], [a, b], color=COLORS["light"], lw=1, zorder=1)
    bx.scatter([0] * len(sup), sup, s=20, color=COLORS["grey"], zorder=2, label="One world")
    bx.scatter([1] * len(stu), stu, s=20, color=COLORS["grey"], zorder=2)
    for x, v, c in ((0, result["supervised"], COLORS["ink"]), (1, result["student"], COLORS["teal"])):
        bx.scatter([x], [v], s=130, marker="D", color=c, zorder=3, edgecolor="white", linewidth=1)
        bx.text(x + (-0.1 if x == 0 else 0.1), v, fmt(v), ha="right" if x == 0 else "left", va="center", fontsize=10)
    bx.axhline(result["teacher_alone"], color=COLORS["gold"], ls=(0, (4, 3)), lw=1.4,
               label=f"Teacher alone: {fmt(result['teacher_alone'])}")
    bx.set_xticks([0, 1], ["Supervised\nonly", f"Student, threshold\n{parameter}"])
    bx.set_xlim(-0.6, 1.6)
    bx.set_ylim(0.70, 0.98)
    bx.set_ylabel("Accuracy on 2,000 fresh rows")
    bx.set_xlabel(f"Kept {100 * result['coverage']:.0f}% of pool, {100 * result['pseudo_accuracy']:.1f}% correct")
    bx.legend(loc="lower right", frameon=False, fontsize=10)


def explain(result, parameter):
    gain, se = result["gain"], result["gain_se"]
    leak, nested = result["leaky_error"], result["nested_error"]
    if gain > 3 * se:
        verdict = "The student is clearly better than the supervised model."
    elif gain > 2 * se:
        verdict = "The gain is small, between two and three standard errors, and may not survive another seed."
    elif gain < -2 * se:
        verdict = "The student is clearly worse than the supervised model."
    else:
        verdict = "The gain is within two standard errors of zero."
    shown = round(result['student'], 3) - round(result['supervised'], 3)   # difference of the displayed numbers
    gain_text = (f"{fmt(result['student'])} - {fmt(result['supervised'])} = {fmt(shown)}" if shown >= 0
                 else f"{fmt(result['supervised'])} - {fmt(result['student'])} = {fmt(-shown)} lost")
    interpretation = (
        f"At threshold {parameter} the teacher's labels are kept for {100 * result['coverage']:.1f}% of the pool and are "
        f"{100 * result['pseudo_accuracy']:.1f}% correct. The student scores {fmt(result['student'])} on fresh rows against "
        f"{fmt(result['supervised'])} for the supervised model: {gain_text} (standard error {fmt(se)}). {verdict} "
        f"Cross-validation with a teacher that saw all labeled rows reports {fmt(result['leaky_estimate'])} against a fresh score of "
        f"{fmt(result['leaky_true'])}, an error of {signed(leak)}; rebuilding the teacher inside each fold gives an error of {signed(nested)}. "
        f"The teacher used alone scores {fmt(result['teacher_alone'])} on fresh rows, so the student is "
        f"{signed(result['gain_vs_teacher'])} against the better of the two supervised models.")
    steps = [f"Student gain on fresh rows: {fmt(result['student'])} - {fmt(result['supervised'])} = {signed(gain)}.",
             f"Leaky estimate error: {fmt(result['leaky_estimate'])} - {fmt(result['leaky_true'])} = {signed(leak)} (standard error {fmt(result['leaky_error_se'])}).",
             f"Nested estimate error: {fmt(result['nested_estimate'])} - {fmt(result['nested_true'])} = {signed(nested)} (standard error {fmt(result['nested_error_se'])}).",
             f"Pseudo-labels kept: {100 * result['coverage']:.1f}% of {result['pool']} pool rows, {100 * result['pseudo_accuracy']:.1f}% correct."]
    metrics = {"Pseudo-labels kept": f"{100 * result['coverage']:.1f}%", "Pseudo-label accuracy": f"{100 * result['pseudo_accuracy']:.1f}%",
               "Student gain (standard error)": f"{signed(gain)} ({fmt(se)})", "Leaky CV error": signed(leak),
               "Nested CV error": signed(nested), "Supervised accuracy, fresh": fmt(result["supervised"])}
    alt = (f"Two panels at threshold {parameter}. Left, the cross-validation estimate and the fresh-row score for a teacher that saw the "
           f"validation labels ({fmt(result['leaky_estimate'])} against {fmt(result['leaky_true'])}) and for a teacher rebuilt per fold "
           f"({fmt(result['nested_estimate'])} against {fmt(result['nested_true'])}). Right, paired fresh-row accuracy of the supervised "
           f"model and the student for each of {result['draws']} worlds.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    ts = [0.6, 0.8, 0.9, 0.99]
    cov = [results[t]["coverage"] for t in ts]
    acc = [results[t]["pseudo_accuracy"] for t in ts]
    gain = [results[t]["gain"] for t in ts]
    assert cov == sorted(cov, reverse=True) and acc == sorted(acc), "stricter threshold: fewer, more accurate pseudo-labels"
    assert gain[0] > gain[1] > gain[2] > gain[3], "the gain falls as the threshold rises"
    assert gain[0] > 0.015 and gain[0] > 4 * results[0.6]["gain_se"], "the loosest threshold gives a clear gain"
    assert gain[2] < 0.01 and gain[3] < 2 * results[0.99]["gain_se"], "strict thresholds add little or nothing"
    for t in (0.6, 0.8, 0.9):
        r = results[t]
        assert r["leaky_error"] > 0.02, f"the leaky teacher overstates at {t}"
        assert abs(r["nested_error"]) < 0.02, f"the nested estimate stays close at {t}"
        assert r["leaky_error"] - r["nested_error"] > 0.02, f"leak exceeds nested error at {t}"
    assert max(abs(results[t]["nested_error"]) for t in ts) < 0.025
    assert fmt(results[0.6]["teacher_alone"]) == "0.875" and fmt(results[0.6]["gain_vs_teacher"]) == "0.010"
    assert results[0.6]["gain_vs_teacher"] > 2 * results[0.6]["gain_vs_teacher_se"], "student beats the teacher alone at 0.6"
    assert results[0.9]["gain_vs_teacher"] < results[0.9]["gain_vs_teacher_se"], "no gain over the teacher alone at 0.9"
    assert max(abs(results[t]["nested_error"]) for t in ts) < 0.02, "assumptions: nested error up to about 0.015"
    r9 = results[0.9]
    assert fmt(r9["gain"]) == "0.006" and fmt(r9["gain_se"]) == "0.003"
    assert f"{100 * r9['pseudo_accuracy']:.1f}" == "98.1" and f"{100 * r9['coverage']:.1f}" == "35.3"
    assert fmt(r9["leaky_error"]) == "0.033" and fmt(r9["leaky_error_se"]) == "0.009" and fmt(r9["nested_error"]) == "0.003"
