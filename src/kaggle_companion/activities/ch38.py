"""Chapter 38: Calibration and Post-Processing. Platt versus isotonic calibration, by development-set size."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB

from kaggle_companion.activities._common import clean

SEED = 38
DRAWS = 30        # independent development sets per size; results are means over draws
EPS = 1e-3        # probabilities are kept inside [0.001, 0.999] so one confident miss cannot dominate log loss


def generate(n, rng):
    """Six noisy copies of one hidden signal plus two noise features. The true log odds bend upward for positive signal."""
    z = rng.normal(size=n)
    X = np.column_stack([z + rng.normal(0, 0.6, n) for _ in range(6)] + [rng.normal(size=n), rng.normal(size=n)])
    logit = 0.4 * z + 1.6 * np.maximum(z, 0) - 1.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def to_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expected_calibration_error(y, p, bins=10):
    """The chapter's population-weighted ECE with equal-width bins; empty bins contribute nothing."""
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((b == k).mean() * abs(y[b == k].mean() - p[b == k].mean()) for k in range(bins) if (b == k).any()))


def reliability(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return [[float(p[b == k].mean()), float(y[b == k].mean()), int((b == k).sum())] for k in range(bins) if (b == k).sum() >= 20]


def run(development_rows):
    rng = np.random.default_rng(SEED)
    X_base, y_base = generate(3000, rng)
    base = GaussianNB().fit(X_base, y_base)                    # frozen base model: never sees development or assessment labels
    X_a, y_a = generate(20000, rng)                            # assessment rows, used only for scoring
    p_a = np.clip(base.predict_proba(X_a)[:, 1], EPS, 1 - EPS)

    scores = {k: [] for k in ("none", "platt", "isotonic", "none_dev", "platt_dev", "isotonic_dev",
                              "auc_platt", "auc_isotonic")}
    for draw in range(DRAWS):
        X_d, y_d = generate(development_rows, rng)
        p_d = np.clip(base.predict_proba(X_d)[:, 1], EPS, 1 - EPS)
        platt = LogisticRegression(C=1e6).fit(to_logit(p_d)[:, None], y_d)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p_d, y_d)
        cal = {"platt": lambda p: platt.predict_proba(to_logit(p)[:, None])[:, 1], "isotonic": iso.predict}
        scores["none"].append(log_loss(y_a, p_a))
        scores["none_dev"].append(log_loss(y_d, p_d, labels=[0, 1]))
        for name, f in cal.items():
            scores[name].append(log_loss(y_a, f(p_a)))
            scores[name + "_dev"].append(log_loss(y_d, f(p_d), labels=[0, 1]))   # scored on the rows it was fitted to
            scores["auc_" + name].append(roc_auc_score(y_a, f(p_a)))
        if draw == 0:
            curves = {"none": reliability(y_a, p_a), "platt": reliability(y_a, cal["platt"](p_a)),
                      "isotonic": reliability(y_a, cal["isotonic"](p_a))}
            ece = {"none": expected_calibration_error(y_a, p_a),
                   "platt": expected_calibration_error(y_a, cal["platt"](p_a)),
                   "isotonic": expected_calibration_error(y_a, cal["isotonic"](p_a))}

    mean = {k: float(np.mean(v)) for k, v in scores.items()}
    return clean({
        "development_rows": development_rows,
        "assessment": {k: mean[k] for k in ("none", "platt", "isotonic")},
        "development": {k: mean[k + "_dev"] for k in ("none", "platt", "isotonic")},
        "isotonic_beats_platt": float(np.mean(np.array(scores["isotonic"]) < np.array(scores["platt"]))),
        "auc": {"none": roc_auc_score(y_a, p_a), "platt": mean["auc_platt"], "isotonic": mean["auc_isotonic"]},
        "ece_first_draw": ece, "reliability_first_draw": curves,
        "draws": DRAWS, "assessment_rows": len(y_a),
    })
# notebook-end


SPEC = {
    "chapter": 38,
    "chapter_title": "Calibration and Post-Processing",
    "subtitle": "Postprocessing is another fitted model: give it a development set and a separate assessment.",
    "summary": ("A calibrator is fitted on development predictions and judged elsewhere. One demonstration measures how a "
                "two-parameter Platt calibrator and a flexible isotonic calibrator behave as the development set grows, on the "
                "rows they were fitted to and on new rows."),
    "title": "Platt or isotonic calibration, by the size of the development set",
    "question": "How large must the development set be before isotonic calibration beats the simpler Platt calibrator on new rows?",
    "why": ("Calibration is fitted, so it can overfit like any model. A calibrator that looks best on its own development rows "
            "can be the worst choice on the leaderboard, and the size of the development set decides which one to trust."),
    "method": ("A constructed binary task. A Gaussian naive Bayes base model is trained once on 3,000 rows of six correlated "
               "copies of one signal, so it double counts the evidence and is overconfident; the true log odds also bend, so no "
               "two-parameter curve fits them exactly. For each development-set size (the control), 30 independent development "
               "sets are drawn. On each, Platt scaling (logistic regression on the base log odds) and isotonic regression are "
               "fitted, then scored by log loss on their own development rows and on 20,000 assessment rows."),
    "control": {"key": "development_rows", "label": "Development rows used to fit the calibrator",
                "values": [50, 200, 1000, 5000], "default": 200,
                "value_labels": ["50", "200", "1,000", "5,000"]},
    "source_section": "Fit Calibration on the Correct Predictions",
    "symbols": ("n_b is the number of rows in probability bin b, n the total, y-bar_b the observed event rate and p-bar_b the "
                "mean predicted probability in that bin."),
    "explanation": ("Platt scaling fits two numbers, so it cannot chase noise, but it cannot follow a bent calibration curve either. "
                    "Isotonic regression fits a free monotone step function: with few rows it memorizes them and looks excellent "
                    "on its own development set; with many rows it follows the true curve and wins. Both are monotone, so neither "
                    "improves ranking."),
    "application": ("Fit every calibrator on development predictions, score it on a separate assessment partition, and compare it "
                    "with a simpler calibrator and with no calibration before keeping it."),
    "assumptions": ("Constructed data, one overconfident base model and probabilities kept inside [0.001, 0.999]. The crossover "
                    "size depends on how bent the true calibration curve is; with a curve Platt can represent exactly, isotonic "
                    "needs even more rows to catch up."),
    "prediction": "With 50 development rows, which calibrator has the lowest log loss on its own development rows, and which on new rows?",
    "prediction_options": ["Isotonic on both", "Isotonic on development rows, Platt on new rows", "Platt on both"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "Isotonic scores 0.480 on its own 50 rows but 0.694 on new rows, where Platt scores 0.587.",
        "incorrect": "Isotonic scores 0.480 on its own 50 rows, the best there, but 0.694 on new rows, where Platt scores 0.587. It memorized the development set.",
    },
    "check": "Why does the AUC barely change after calibration, even when log loss improves by a third?",
    "answer": ("Both calibrators are monotone, so they keep the base model's ordering of rows. AUC measures only ordering. Platt "
               "leaves it exactly unchanged; isotonic can merge nearby probabilities into ties, which slightly lowers AUC. The "
               "log-loss gain comes from fixing overconfidence, not from ranking better."),
    "provenance": "Constructed example: seeded synthetic data, a naive Bayes base model and two calibrators, measured by the chapter activity.",
    "apply": [
        "Fit calibration on predictions whose labels the base model never saw, then score it on a different partition.",
        "Never choose a calibrator by its score on the rows it was fitted to; that comparison always favours the most flexible one.",
        "With a few hundred development rows, start with Platt or temperature scaling; consider isotonic only when thousands of rows show it winning on assessment.",
        "Calibration fixes log loss and calibration error, not AUC: if the metric is ranking, spend the effort elsewhere.",
    ],
    "honesty": "Constructed data; the crossover size is a property of this generator, not a rule for every competition.",
}

EQUATIONS = [{"tex": r"\mathrm{ECE}=\sum_b \frac{n_b}{n}|\bar{y}_b-\bar{p}_b|",
              "alt": "E C E equals the sum over bins b of n b over n times the absolute value of y bar b minus p bar b",
              "basis": "The chapter's expected calibration error (display equation in Chapter 38, Define the Metric Contract)."}]
NCOLS = 2
HEIGHT = 4.4
METHODS = [("none", "No calibration", COLORS["terracotta"]), ("platt", "Platt", COLORS["teal"]),
           ("isotonic", "Isotonic", COLORS["gold"])]


def draw(axes, result, parameter):
    left, right = axes
    left.plot([0, 1], [0, 1], color=COLORS["grey"], lw=0.8, ls=":", label="Perfect calibration")
    for key, name, color in METHODS:
        pts = result["reliability_first_draw"][key]
        left.plot([q[0] for q in pts], [q[1] for q in pts], marker="o", ms=4, lw=1.4, color=color,
                  label=f"{name} (ECE {fmt(result['ece_first_draw'][key])})")
    left.set_xlim(0, 1)
    left.set_ylim(0, 1)
    left.set_xlabel("Predicted probability")
    left.set_ylabel("Observed event rate (assessment rows)")
    left.legend(loc="upper left", frameon=False, fontsize=10)

    xs = list(range(len(METHODS)))
    dev = [result["development"][k] for k, _, _ in METHODS]
    new = [result["assessment"][k] for k, _, _ in METHODS]
    right.bar([x - 0.2 for x in xs], dev, width=0.4, color=COLORS["light"], edgecolor=COLORS["ink"], lw=0.6,
              label="On its own development rows")
    right.bar([x + 0.2 for x in xs], new, width=0.4, color=COLORS["navy"], edgecolor=COLORS["ink"], lw=0.6,
              label="On 20,000 new rows")
    for x, a, b in zip(xs, dev, new):
        right.text(x - 0.2, a + 0.01, fmt(a), ha="center", va="bottom", fontsize=10)
        right.text(x + 0.2, b + 0.01, fmt(b), ha="center", va="bottom", fontsize=10)
    right.set_xticks(xs, [name for _, name, _ in METHODS])
    right.set_ylim(0, max(dev + new) * 1.25)
    right.set_ylabel("Log loss (lower is better)")
    right.set_xlabel(f"Calibrator, {parameter} development rows")
    right.legend(loc="upper right", frameon=False, fontsize=10)


def explain(result, parameter):
    a, d = result["assessment"], result["development"]
    winner = "isotonic" if a["isotonic"] < a["platt"] else "Platt"
    optimism = a["isotonic"] - d["isotonic"]
    interpretation = (
        f"With {parameter} development rows, calibration cuts log loss on new rows from {fmt(a['none'])} to {fmt(a['platt'])} "
        f"with Platt ({fmt(a['none'])} - {fmt(a['platt'])} = {fmt(a['none'] - a['platt'])}) and {fmt(a['isotonic'])} with "
        f"isotonic, so {winner} is better on new rows "
        f"(isotonic won {round(100 * result['isotonic_beats_platt'])}% of {result['draws']} draws). On its own development rows "
        f"isotonic scores {fmt(d['isotonic'])}, {fmt(optimism)} better than it does on new rows. "
        f"AUC moves from {fmt(result['auc']['none'])} to {fmt(result['auc']['platt'])} (Platt) and {fmt(result['auc']['isotonic'])} (isotonic).")
    steps = [
        f"Gain from Platt on new rows: {fmt(a['none'])} - {fmt(a['platt'])} = {fmt(a['none'] - a['platt'])}.",
        f"Isotonic minus Platt on new rows: {fmt(a['isotonic'])} - {fmt(a['platt'])} = {signed(a['isotonic'] - a['platt'])}.",
        f"Isotonic optimism: {fmt(a['isotonic'])} - {fmt(d['isotonic'])} = {fmt(optimism)} (new rows minus own rows).",
        f"Platt optimism: {fmt(a['platt'])} - {fmt(d['platt'])} = {signed(a['platt'] - d['platt'])}.",
    ]
    metrics = {"No calibration, new rows": fmt(a["none"]), "Platt, new rows": fmt(a["platt"]),
               "Isotonic, new rows": fmt(a["isotonic"]), "Isotonic, own rows": fmt(d["isotonic"]),
               "Isotonic beats Platt": f"{round(100 * result['isotonic_beats_platt'])}% of draws"}
    alt = (f"Left: reliability curves on assessment rows for no calibration, Platt and isotonic with {parameter} development rows. "
           f"Right: log loss on development rows and on new rows; isotonic scores {fmt(d['isotonic'])} on its own rows and "
           f"{fmt(a['isotonic'])} on new rows, Platt {fmt(a['platt'])}.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for n, res in results.items():
        a, d = res["assessment"], res["development"]
        assert a["platt"] < a["none"] - 0.3 and a["isotonic"] < a["none"] - 0.2, f"calibration should help a lot at {n}"
        assert d["isotonic"] < d["platt"], f"isotonic should look best on its own rows at {n}"
        assert abs(res["auc"]["platt"] - res["auc"]["none"]) < 1e-6, "Platt is monotone: AUC unchanged"
        assert res["auc"]["isotonic"] <= res["auc"]["none"] + 1e-6, "isotonic ties can only lower AUC"
        assert a["none"] - a["platt"] > a["none"] / 3, "check says log loss improves by a third"
    small, large = results[50], results[5000]
    assert fmt(small["development"]["isotonic"]) == "0.480" and fmt(small["assessment"]["isotonic"]) == "0.694"
    assert fmt(small["assessment"]["platt"]) == "0.587", "prediction feedback numbers"
    assert small["assessment"]["platt"] < small["assessment"]["isotonic"], "Platt should win with 50 rows"
    assert results[200]["assessment"]["platt"] < results[200]["assessment"]["isotonic"], "Platt should win with 200 rows"
    assert large["assessment"]["isotonic"] < large["assessment"]["platt"] and large["isotonic_beats_platt"] > 0.9, \
        "isotonic should win with 5,000 rows"
    assert results[200]["isotonic_beats_platt"] <= 0.25 and results[1000]["isotonic_beats_platt"] >= 0.9, \
        "apply text: a few hundred rows favour Platt, thousands favour isotonic"
