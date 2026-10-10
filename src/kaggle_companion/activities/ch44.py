"""Chapter 44: Loss Functions for CV. Cross-entropy against focal loss on a rare class, scored by the metric and by probability quality."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.metrics import average_precision_score, log_loss

from kaggle_companion.activities._common import clean

SEED = 44
DRAWS = 16        # independent training sets; every estimate is a mean over draws
N_TRAIN, N_DEV, N_ASSESS = 2000, 2000, 12000
PREVALENCE = 0.05


def generate(n, rng):
    """Rare positives (5%). Two features separate them, and 30% of positives form a hard subgroup that looks different."""
    y = (rng.random(n) < PREVALENCE).astype(int)
    X = rng.normal(size=(n, 6))
    X[:, 0] += 1.4 * y
    X[:, 1] += 0.9 * y
    hard = (y == 1) & (rng.random(n) < 0.3)
    X[hard, 0] -= 1.4          # the hard positives hide on feature 0 ...
    X[hard, 2] += 2.0          # ... and show up on feature 2 instead
    return X, y


def focal_loss(z, y, gamma):
    """Mean binary focal loss on logits z: -(1 - p_t)^gamma * log(p_t). gamma = 0 is ordinary cross-entropy."""
    p = np.clip(1 / (1 + np.exp(-z)), 1e-7, 1 - 1e-7)
    p_t = np.where(y == 1, p, 1 - p)
    return float(np.mean(-((1 - p_t) ** gamma) * np.log(p_t)))


def focal_grad(z, y, gamma):
    """Derivative of the focal loss with respect to each logit (checked against finite differences in verify)."""
    p = np.clip(1 / (1 + np.exp(-z)), 1e-7, 1 - 1e-7)
    q = 1 - p
    pos = gamma * p * q ** gamma * np.log(p) - q ** (gamma + 1)
    neg = -(gamma * q * p ** gamma * np.log(q) - p ** (gamma + 1))
    return np.where(y == 1, pos, neg)


def fit_mlp(X, y, gamma, seed, hidden=16, epochs=200, lr=0.03, decay=1e-3):
    """One-hidden-layer network trained full-batch with Adam on the focal loss. Returns a predict-probability function."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    params = [rng.normal(0, 1 / np.sqrt(d), (d, hidden)), np.zeros(hidden), rng.normal(0, 1 / np.sqrt(hidden), hidden), np.array([-2.0])]
    m1 = [np.zeros_like(p) for p in params]
    m2 = [np.zeros_like(p) for p in params]
    for t in range(1, epochs + 1):
        h = np.tanh(X @ params[0] + params[1])
        dz = focal_grad(h @ params[2] + params[3][0], y, gamma) / n
        dh = np.outer(dz, params[2]) * (1 - h ** 2)
        grads = [X.T @ dh + decay * params[0], dh.sum(0), h.T @ dz + decay * params[2], np.array([dz.sum()])]
        for i, g in enumerate(grads):                       # Adam update
            m1[i] = 0.9 * m1[i] + 0.1 * g
            m2[i] = 0.999 * m2[i] + 0.001 * g * g
            params[i] -= lr * (m1[i] / (1 - 0.9 ** t)) / (np.sqrt(m2[i] / (1 - 0.999 ** t)) + 1e-8)

    def logit(X_new):
        return np.tanh(X_new @ params[0] + params[1]) @ params[2] + params[3][0]
    return logit


def f1(y, flagged):
    """F1 of the positive class: 2TP / (2TP + FP + FN)."""
    tp = np.sum(flagged & (y == 1))
    return 2 * tp / max(2 * tp + np.sum(flagged & (y == 0)) + np.sum(~flagged & (y == 1)), 1)


def tuned_threshold(y, p):
    """The probability cutoff that maximises F1 on development rows."""
    cuts = np.linspace(0.02, 0.9, 60)
    return cuts[np.argmax([f1(y, p >= c) for c in cuts])]


def score(logit, X_dev, y_dev, X_a, y_a):
    z = logit(X_a)
    p = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)
    cut = tuned_threshold(y_dev, 1 / (1 + np.exp(-logit(X_dev))))
    return {"ap": average_precision_score(y_a, p), "f1_default": f1(y_a, p >= 0.5), "f1_tuned": f1(y_a, p >= cut),
            "log_loss": log_loss(y_a, p), "mean_probability": p.mean(), "flagged_default": (p >= 0.5).mean()}


def run(gamma):
    rng = np.random.default_rng(SEED)
    X_a, y_a = generate(N_ASSESS, rng)                         # assessment rows, used only for scoring
    rows = {"ce": [], "focal": []}
    for draw in range(DRAWS):
        X, y = generate(N_TRAIN, rng)
        X_d, y_d = generate(N_DEV, rng)
        for name, g in (("ce", 0.0), ("focal", float(gamma))):
            if name == "focal" and gamma == 0:             # gamma 0 is the comparator itself
                rows["focal"].append(rows["ce"][-1])
                continue
            rows[name].append(score(fit_mlp(X, y, g, seed=draw), X_d, y_d, X_a, y_a))
    keys = list(rows["ce"][0])
    mean = {name: {k: float(np.mean([r[k] for r in rs])) for k in keys} for name, rs in rows.items()}
    paired = {k: np.array([f[k] - c[k] for f, c in zip(rows["focal"], rows["ce"])]) for k in keys}
    # Sanity test from the chapter: gamma = 0 must reduce to ordinary cross-entropy.
    z_test = np.random.default_rng(0).normal(size=500)
    y_test = (np.random.default_rng(1).random(500) < 0.3).astype(int)
    p_test = 1 / (1 + np.exp(-z_test))
    return clean({
        "gamma": gamma,
        "ce": mean["ce"], "focal": mean["focal"],
        "ap_gain": paired["ap"].mean(), "ap_gain_se": paired["ap"].std(ddof=1) / np.sqrt(DRAWS) if gamma else 0.0,
        "ap_win_rate": float((paired["ap"] > 0).mean()) if gamma else 0.0,
        "f1_tuned_gain": paired["f1_tuned"].mean(), "f1_default_gain": paired["f1_default"].mean(),
        "gamma0_vs_log_loss": abs(focal_loss(z_test, y_test, 0.0) - log_loss(y_test, p_test)),
        "prevalence": PREVALENCE, "draws": DRAWS, "assessment_rows": N_ASSESS,
    })
# notebook-end


SPEC = {
    "chapter": 44,
    "chapter_title": "Loss Functions for CV",
    "subtitle": "Name the failure first, keep cross-entropy as the comparator, and judge a specialised loss on the metric that is scored.",
    "summary": ("A focal loss changes what the model is trained to fix. One demonstration trains the same small network with "
                "cross-entropy and with focal loss on a rare class, and measures the scored ranking metric, the F1 at a default "
                "cutoff and the quality of the probabilities."),
    "title": "Focal loss against its cross-entropy comparator on a rare class, by gamma",
    "question": "Does focal loss beat cross-entropy on the metric that is scored, and what does it cost elsewhere?",
    "why": ("Focal loss is a popular answer to class imbalance, and the chapter asks that it win against an unchanged "
            "cross-entropy reference on the scored metric before it is kept. Sixteen paired trainings show where the "
            "gain stops and what changes silently, such as the scale of the predicted probabilities."),
    "method": ("A constructed binary task with 5% positives, six numeric features and a hard subgroup (30% of positives look "
               "like negatives on the first feature). A one-hidden-layer network, 16 tanh units, is trained full-batch with "
               "Adam on the focal loss with the control's gamma and, from the same start and the same rows, on cross-entropy "
               "(gamma = 0). The 2,000 training rows hold about 100 positives. Both models are scored on 12,000 fresh rows by "
               "average precision (the ranking metric), by F1 at the default cutoff of 0.5, by F1 at a cutoff tuned on a "
               "separate 2,000 development rows, by log loss and by the mean predicted probability, averaged over 16 draws."),
    "control": {"key": "gamma", "label": "Focusing parameter gamma (0 is ordinary cross-entropy)",
                "values": [0, 1, 2, 5], "default": 2,
                "value_labels": ["0: cross-entropy", "1", "2: the common starting value", "5: strong focusing"]},
    "source_section": "Focal Loss for Imbalanced Classes",
    "symbols": ("p_t is the probability the model gives the true class, gamma the focusing parameter (0 gives cross-entropy), "
                "FL the focal loss for one row and AP the average precision of the ranking."),
    "explanation": ("The factor (1 - p_t)^gamma shrinks the loss of rows the model already classifies well, so training spends its "
                    "effort on hard rows, including the hidden subgroup of positives. The same factor also shrinks easy positives, and "
                    "it stops the probabilities from matching frequencies: they drift upward on a rare class. Raise gamma and "
                    "watch the ranking gain peak and turn into a loss while the log loss only gets worse."),
    "application": ("Keep an unweighted cross-entropy run as the reference, compare focal values on the scored metric over "
                    "several seeds, and check the probabilities of a focal model on development rows before trusting them: "
                    "if the score needs probabilities, fit a calibrator on separate development rows."),
    "assumptions": ("Constructed data, a small numpy network standing in for a CNN and a binary focal loss without the optional "
                    "alpha weights. The ranking gain at gamma = 2 is a property of this generator. A side check without the hard "
                    "subgroup still showed a smaller gain, so the subgroup explains only part of it; the rest is likely focal loss "
                    "damping overconfident fits on about 100 positives. Both losses share one optimiser setting; neither was retuned. On other data the gain can be zero or negative, which is why the "
                    "chapter asks for the comparison rather than prescribing gamma. Cutoffs are tuned for F1 only."),
    "prediction": "Compared with cross-entropy, what does focal loss with gamma = 2 do to the log loss on fresh rows?",
    "prediction_options": ["Lowers it, because the model focuses on hard rows", "Leaves it about the same",
                           "Raises it, because the probabilities are no longer calibrated"],
    "prediction_answer": 2,
    "prediction_feedback": {
        "correct": "The log loss rises from 0.155 to 0.256 even though the ranking improves: the mean predicted probability is 0.191 on a 5% task.",
        "incorrect": "The log loss rises from 0.155 to 0.256 even though the ranking improves: the focal model predicts a mean probability of 0.191 on a task whose positive rate is 5%.",
    },
    "check": "At gamma = 5 the F1 at the default cutoff collapses while the tuned-cutoff F1 barely moves. What does that say about comparing losses?",
    "answer": ("Strong focusing squeezes the probabilities into a narrow middle band (mean 0.332), so almost nothing reaches 0.5 "
               "(0.003 of rows are flagged), "
               "so F1 at the default cutoff falls from 0.218 to 0.080, yet with a cutoff tuned on development rows F1 is 0.340 "
               "against 0.341. A comparison is only fair when each loss gets its own decision rule: tuning the cutoff alone lifts the "
               "cross-entropy F1 by 0.123, while gamma = 2 adds only about 0.01 on top of a tuned cutoff and 0.021 average precision."),
    "provenance": "Constructed example: seeded synthetic rare-class data and a numpy network, measured by the chapter activity.",
    "apply": [
        "Train the plain cross-entropy (or BCE) reference first and keep it in every comparison, under the same seeds and budget.",
        "Check the implementation at gamma = 0: it must reproduce ordinary cross-entropy before any focal run is trusted.",
        "Compare gamma values on the scored metric, averaged over seeds, and give each loss its own cutoff chosen on development rows.",
        "If probabilities are scored or blended, check their quality after a focal run and fit a calibrator on separate development rows when it is poor.",
    ],
    "honesty": "Constructed data with a hard minority subgroup; the sizes of these effects are properties of this generator, not a competition result.",
}

EQUATIONS = [{"tex": r"\mathrm{FL}(p_t) = -(1 - p_t)^{\gamma}\,\log(p_t)",
              "alt": "Focal loss equals minus one minus p t, raised to the power gamma, times the natural log of p t",
              "basis": "The modulating factor of Focal Loss for Imbalanced Classes; the manuscript gives it in prose and code, not as a display equation."}]
NCOLS = 2
HEIGHT = 4.4


def draw(axes, result, parameter):
    ax, bx = axes
    ce, fo = result["ce"], result["focal"]
    metrics = [("ap", "Average\nprecision"), ("f1_tuned", "F1, tuned\ncutoff"), ("f1_default", "F1, cutoff\n0.5")]
    xs = np.arange(len(metrics))
    ax.bar(xs - 0.2, [ce[k] for k, _ in metrics], width=0.4, color=COLORS["light"], edgecolor=COLORS["ink"], lw=0.6,
           label="Cross-entropy")
    ax.bar(xs + 0.2, [fo[k] for k, _ in metrics], width=0.4, color=COLORS["teal"], edgecolor=COLORS["ink"], lw=0.6,
           label=f"Focal, gamma = {parameter}")
    for x, (k, _) in zip(xs, metrics):
        ax.text(x - 0.2, ce[k] + 0.006, fmt(ce[k]), ha="center", va="bottom", fontsize=10)
        ax.text(x + 0.2, fo[k] + 0.006, fmt(fo[k]), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(xs, [n for _, n in metrics])
    ax.set_ylim(0, 0.52)
    ax.set_ylabel("Score on 12,000 fresh rows")
    ax.set_xlabel("Metric")
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    bx.bar([0, 1], [ce["mean_probability"], fo["mean_probability"]], width=0.5, color=[COLORS["light"], COLORS["teal"]],
           edgecolor=COLORS["ink"], lw=0.6)
    for x, v in enumerate([ce["mean_probability"], fo["mean_probability"]]):
        bx.text(x, v + 0.008, fmt(v), ha="center", va="bottom", fontsize=10)
    bx.axhline(result["prevalence"], color=COLORS["gold"], ls=(0, (4, 3)), lw=1.6, label="True positive rate: 0.050")
    bx.set_xticks([0, 1], ["Cross-entropy", f"Focal, gamma = {parameter}"])
    bx.set_ylim(0, 0.4)
    bx.set_ylabel("Mean predicted probability")
    bx.set_xlabel("Training loss")
    bx.legend(loc="upper left", frameon=False, fontsize=10)


def shown_diff(a, b):
    """Difference of two numbers as displayed (three decimals), so each written subtraction checks by hand."""
    return round(a, 3) - round(b, 3)


def explain(result, parameter):
    ce, fo = result["ce"], result["focal"]
    gain, se = result["ap_gain"], result["ap_gain_se"]
    if parameter == 0:
        interpretation = ("Gamma 0 is ordinary cross-entropy, so the two models are the same run and every difference is zero. "
                          f"It is the reference: average precision {fmt(ce['ap'])}, F1 {fmt(ce['f1_default'])} at the default cutoff and "
                          f"log loss {fmt(ce['log_loss'])}. Its probabilities are honest: the mean predicted probability is "
                          f"{fmt(ce['mean_probability'])} against a {fmt(result['prevalence'])} positive rate, "
                          f"{fmt(max(result['prevalence'], ce['mean_probability']))} - {fmt(min(result['prevalence'], ce['mean_probability']))} = "
                          f"{fmt(abs(ce['mean_probability'] - result['prevalence']))} apart. The check in the notebook shows the focal formula and the library log loss "
                          f"agree to {result['gamma0_vs_log_loss']:.1e}.")
        steps = [f"Reference average precision: {fmt(ce['ap'])}.",
                 f"Reference log loss: {fmt(ce['log_loss'])}.",
                 f"Mean predicted probability against the true rate: {fmt(ce['mean_probability'])} against {fmt(result['prevalence'])}.",
                 "Every focal run below is compared with this one."]
    else:
        verdict = ("The ranking improves" if gain > 2 * se else "The ranking does not improve clearly" if gain > -2 * se else
                   "The ranking gets worse")
        if gain >= 0:
            ap_line = f"{fmt(fo['ap'])} - {fmt(ce['ap'])} = {fmt(shown_diff(fo['ap'], ce['ap']))}"
        else:
            ap_line = f"{fmt(ce['ap'])} - {fmt(fo['ap'])} = {fmt(shown_diff(ce['ap'], fo['ap']))} lost"
        interpretation = (
            f"At gamma {parameter}, average precision is {fmt(fo['ap'])} against {fmt(ce['ap'])} for cross-entropy: {ap_line} "
            f"(standard error {fmt(se)}, better in {round(100 * result['ap_win_rate'])}% of the draws). {verdict}. "
            f"At the default cutoff F1 moves from {fmt(ce['f1_default'])} to {fmt(fo['f1_default'])}; with a cutoff tuned on development rows "
            f"it moves from {fmt(ce['f1_tuned'])} to {fmt(fo['f1_tuned'])}. Log loss goes from {fmt(ce['log_loss'])} to {fmt(fo['log_loss'])} "
            f"because the mean predicted probability is {fmt(fo['mean_probability'])} on a {fmt(result['prevalence'])} positive rate.")
        steps = [f"Average precision: {fmt(fo['ap'])} - {fmt(ce['ap'])} = {signed(shown_diff(fo['ap'], ce['ap']))} (focal minus cross-entropy).",
                 f"F1 at the default cutoff: {fmt(fo['f1_default'])} - {fmt(ce['f1_default'])} = {signed(shown_diff(fo['f1_default'], ce['f1_default']))}.",
                 f"F1 at the tuned cutoff: {fmt(fo['f1_tuned'])} - {fmt(ce['f1_tuned'])} = {signed(shown_diff(fo['f1_tuned'], ce['f1_tuned']))}.",
                 f"Log loss: {fmt(fo['log_loss'])} - {fmt(ce['log_loss'])} = {signed(shown_diff(fo['log_loss'], ce['log_loss']))}."]
    metrics = {"Average precision, cross-entropy": fmt(ce["ap"]), "Average precision, focal": fmt(fo["ap"]),
               "Paired gain (standard error)": f"{signed(gain)} ({fmt(se)})", "F1 at cutoff 0.5, focal": fmt(fo["f1_default"]),
               "Log loss, cross-entropy": fmt(ce["log_loss"]), "Log loss, focal": fmt(fo["log_loss"])}
    alt = (f"Two panels. Left, paired bars of average precision, tuned-cutoff F1 and default-cutoff F1 for cross-entropy and for "
           f"focal loss with gamma {parameter}. Right, mean predicted probability for each loss against the true rate of 0.050; "
           f"the focal model's mean is {fmt(fo['mean_probability'])}.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for g, res in results.items():
        assert res["gamma0_vs_log_loss"] < 1e-9, "gamma 0 must reproduce cross-entropy"
    z = np.random.default_rng(5).normal(size=30)
    y = (np.random.default_rng(6).random(30) < 0.4).astype(int)
    for gamma in (0.0, 2.0):
        eps = 1e-6
        numeric = np.array([(focal_loss(z + eps * (np.arange(30) == i), y, gamma) - focal_loss(z - eps * (np.arange(30) == i), y, gamma))
                            / (2 * eps) * 30 for i in range(30)])
        assert np.allclose(numeric, focal_grad(z, y, gamma), atol=1e-5), "analytic gradient must match finite differences"
    r0, r1, r2, r5 = results[0], results[1], results[2], results[5]
    assert r0["ap_gain"] == 0 and r0["ce"] == r0["focal"], "gamma 0 is the comparator"
    # The ranking gain peaks near gamma 2 and is gone at gamma 5.
    assert r2["ap_gain"] > 0.015 and r2["ap_gain"] > 4 * r2["ap_gain_se"], "gamma 2 gain is clear"
    assert r2["ap_gain"] > r1["ap_gain"] > 0 and r5["ap_gain"] < 0.005, "gain peaks at gamma 2 and is gone at gamma 5"
    # Probability quality only gets worse as gamma grows.
    assert r0["ce"]["log_loss"] < r1["focal"]["log_loss"] < r2["focal"]["log_loss"] < r5["focal"]["log_loss"]
    assert r2["focal"]["mean_probability"] > 3 * results[2]["prevalence"], "focal probabilities drift well above the 5% rate"
    # Prediction feedback and answer numbers.
    assert fmt(r2["ce"]["log_loss"]) == "0.155" and fmt(r2["focal"]["log_loss"]) == "0.256"
    assert fmt(r2["focal"]["mean_probability"]) == "0.191"
    assert fmt(r5["focal"]["mean_probability"]) == "0.332"
    assert fmt(r5["focal"]["flagged_default"]) == "0.003" and fmt(r5["ce"]["f1_default"]) == "0.218"
    assert fmt(r5["focal"]["f1_default"]) == "0.080" and fmt(r5["focal"]["f1_tuned"]) == "0.340"
    assert fmt(r5["ce"]["f1_tuned"]) == "0.341" and abs(r5["f1_tuned_gain"]) < 0.01
    assert fmt(r2["ap_gain"]) == "0.021"
    assert fmt(r2["ce"]["f1_tuned"] - r2["ce"]["f1_default"]) == "0.123" and fmt(r2["f1_tuned_gain"]) == "0.010"
