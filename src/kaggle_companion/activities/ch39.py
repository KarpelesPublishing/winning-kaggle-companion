"""Chapter 39: Imbalanced Data. Negative downsampling, the analytic probability correction and development calibration, by keep rate."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from kaggle_companion.activities._common import clean

SEED = 39
REPLICATES = 5          # independent worlds per keep rate; results are means over them
N_TRAIN, N_DEV, N_TEST = 40_000, 30_000, 60_000
EPS = 1e-6              # probabilities are kept inside [1e-6, 1 - 1e-6] so log loss stays finite


def generate(n, rng):
    """Eight features, a bent and interacting score, about 1.3% positives."""
    X = rng.normal(size=(n, 8))
    score = X[:, 0] + 0.8 * np.maximum(X[:, 1], 0) + 0.7 * X[:, 2] * X[:, 3] - 0.6 * np.abs(X[:, 4]) + 0.4 * X[:, 5]
    return X, (rng.random(n) < 1 / (1 + np.exp(-(1.2 * score - 5.6)))).astype(int)


def to_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def correct(p, keep):
    """The chapter's correction: natural odds equal sampled odds times the negative keep rate r."""
    return p * keep / (p * keep + 1 - p)


def one_world(keep, seed):
    rng = np.random.default_rng(seed)
    X_train, y_train = generate(N_TRAIN, rng)
    X_dev, y_dev = generate(N_DEV, rng)                       # natural prevalence: used only to fit the calibrator
    X_test, y_test = generate(N_TEST, rng)                    # natural prevalence: used only for scoring
    sampled = (y_train == 1) | (rng.random(N_TRAIN) < keep)   # keep every positive, each negative with probability `keep`
    out = {"training_rows": int(sampled.sum()), "prevalence": float(y_test.mean())}
    makers = {"gbm": lambda: HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=15, random_state=0),
              "logistic": lambda: LogisticRegression(max_iter=500)}
    for name, make in makers.items():
        model = make().fit(X_train[sampled], y_train[sampled])
        p_dev = np.clip(model.predict_proba(X_dev)[:, 1], EPS, 1 - EPS)
        p_test = np.clip(model.predict_proba(X_test)[:, 1], EPS, 1 - EPS)
        calibrator = LogisticRegression(C=1e6).fit(to_logit(p_dev)[:, None], y_dev)      # Platt scaling on development rows
        outputs = {"raw": p_test, "corrected": np.clip(correct(p_test, keep), EPS, 1 - EPS),
                   "calibrated": calibrator.predict_proba(to_logit(p_test)[:, None])[:, 1]}
        out[name] = {k: [log_loss(y_test, v), roc_auc_score(y_test, v), float(v.mean())] for k, v in outputs.items()}
    return out


def run(keep_rate):
    worlds = [one_world(keep_rate, SEED * 100 + r) for r in range(REPLICATES)]
    # Reference: the same models trained on every row (keep rate 1), same worlds.
    base = worlds if keep_rate == 1 else [one_world(1, SEED * 100 + r) for r in range(REPLICATES)]

    def mean(group, model, arm, j):
        return float(np.mean([w[model][arm][j] for w in group]))
    result = {"keep_rate": keep_rate, "training_rows": float(np.mean([w["training_rows"] for w in worlds])),
              "prevalence": float(np.mean([w["prevalence"] for w in worlds])), "replicates": REPLICATES}
    for model in ("gbm", "logistic"):
        result[model] = {arm: {"log_loss": mean(worlds, model, arm, 0), "auc": mean(worlds, model, arm, 1),
                               "mean_prediction": mean(worlds, model, arm, 2)} for arm in ("raw", "corrected", "calibrated")}
        result[model]["no_downsampling_log_loss"] = mean(base, model, "raw", 0)
        result[model]["no_downsampling_auc"] = mean(base, model, "raw", 1)
    result["calibrated_beats_corrected_gbm"] = float(np.mean([w["gbm"]["calibrated"][0] < w["gbm"]["corrected"][0] for w in worlds]))
    return clean(result)
# notebook-end


SPEC = {
    "chapter": 39,
    "chapter_title": "Imbalanced Data",
    "subtitle": "Downsampling negatives is cheap, but it changes the probabilities; fix them and check the fix.",
    "summary": ("Keeping every positive and a fraction of the negatives cuts training cost and leaves ranking almost intact, but the "
                "raw probabilities describe the sampled data. One demonstration measures the damage to log loss and compares the "
                "chapter's analytic correction with a calibrator fitted on development rows, for a gradient-boosted model and a logistic model."),
    "title": "Downsampling negatives: raw, corrected and calibrated probabilities",
    "question": "How much log loss does negative downsampling cost, and at what keep rate does the analytic correction stop being as good as calibrating on development rows?",
    "why": ("Competitions with millions of negatives invite downsampling, and a probability metric then scores the sampled "
            "prevalence rather than the real one. The chapter gives a correction and a calibration route; this measures where each holds."),
    "method": ("A constructed task with about 1.3% positives: 40,000 training rows, 30,000 development rows and 60,000 test rows, all "
               "at natural prevalence. Every positive and a fraction r (the control) of the negatives are kept for training. A "
               "gradient-boosted model and a logistic regression are fitted on the sample. Three outputs are scored by log loss "
               "and AUC on the test rows: raw sampled probabilities, the chapter's correction p r / (p r + 1 - p), and Platt "
               "scaling fitted on the development rows. The reference is the same model trained on every row. Results are means over 5 worlds."),
    "control": {"key": "keep_rate", "label": "Share of negatives kept for training",
                "values": [1, 0.02, 0.005, 0.002], "default": 0.02,
                "value_labels": ["100% (no downsampling)", "2%", "0.5%", "0.2%"]},
    "source_section": "Negative Downsampling at Scale",
    "symbols": ("p is the probability from the model trained on the sample, r the fraction of negatives kept (all positives are "
                "kept) and p_true the corrected probability for the natural population."),
    "explanation": ("The correction follows from odds: keeping a fraction r of negatives multiplies the sampled odds by 1/r, so the "
                    "natural odds are the sampled odds times r. It assumes the model is calibrated on the sampled data. A logistic "
                    "regression stays calibrated there almost regardless of the sample size; a boosted model fitted on a few hundred "
                    "rows does not, and the correction then inherits its overconfidence. A calibrator fitted on natural-prevalence "
                    "development rows does not depend on that assumption."),
    "application": ("Downsample negatives to cut cost, then restore probabilities with the formula only when the model is calibrated "
                    "on the sample; otherwise calibrate on development rows with the natural prevalence, and assess on a separate partition."),
    "assumptions": ("Constructed data; HistGradientBoostingClassifier with 60 trees stands in for LightGBM. The correction's own "
                    "assumptions hold by construction (negatives kept independently with equal probability, features unchanged). "
                    "The keep rate where the correction falls behind depends on the number of positives and the model."),
    "prediction": ("Every positive and 2% of the negatives are kept for training. Scored on natural-prevalence rows, how does the raw "
                   "gradient-boosted model's log loss compare with the same model trained on all rows (0.056)?"),
    "prediction_options": ["About the same, because ranking is unchanged", "About twice as large", "About six times as large"],
    "prediction_answer": 2,
    "prediction_feedback": {
        "correct": "Raw log loss is 0.338, six times 0.056, while AUC is 0.853 against 0.851: ranking survives, probabilities do not.",
        "incorrect": "Raw log loss is 0.338, six times 0.056, while AUC is 0.853 against 0.851. Ranking survives; the probabilities describe the sampled prevalence, not the real one.",
    },
    "check": "At a 0.2% keep rate the analytic correction is clearly worse than development calibration for the boosted model but not for the logistic regression. Why?",
    "answer": ("The formula assumes the model is calibrated on the sample. With 563 training rows the boosted model is not, so the corrected "
               "probabilities average 0.050 against a true prevalence of 0.013 and log loss is 0.089 against 0.059 after calibration. "
               "The logistic regression has only nine parameters and stays calibrated on the sample, so its corrected "
               "log loss (0.059) is within 0.002 of the model trained on every row (0.058)."),
    "provenance": "Constructed example: seeded synthetic data, a boosted model and a logistic regression, measured by the chapter activity.",
    "apply": [
        "Downsample negatives only inside training, keep development and assessment rows at natural prevalence, and score with the competition metric.",
        "Never submit raw sampled probabilities to a probability metric; apply the correction or a development-fitted calibrator.",
        "Check the corrected probabilities: their mean on natural-prevalence rows should match the prevalence; if not, calibrate instead.",
        "Keep enough positives and negatives for the model to be calibrated on the sample; very aggressive keep rates also cost AUC.",
    ],
    "honesty": ("Constructed data. The correction ties with calibration at a 2% keep rate here; it fails only at far more "
                "aggressive rates and only for the boosted model."),
}

EQUATIONS = [{"tex": r"p_{\mathrm{true}}=\frac{p\,r}{p\,r+(1-p)}",
              "alt": "p true equals p times r divided by p times r plus one minus p",
              "basis": "The chapter's downsampling correction (display formula in Chapter 39, Negative Downsampling at Scale)."}]
NCOLS = 2
HEIGHT = 4.4
OUTPUTS = [("raw", "Raw", COLORS["terracotta"]), ("corrected", "Corrected", COLORS["teal"]), ("calibrated", "Calibrated\non dev rows", COLORS["gold"])]


def draw(axes, result, parameter):
    left, right = axes
    xs = list(range(len(OUTPUTS)))
    g = result["gbm"]
    values = [g[k]["log_loss"] for k, _, _ in OUTPUTS]
    left.bar(xs, values, color=[c for _, _, c in OUTPUTS], edgecolor=COLORS["ink"], lw=0.6, width=0.6)
    for x, v in zip(xs, values):
        left.text(x, v * 1.08, fmt(v), ha="center", va="bottom", fontsize=10)
    left.axhline(g["no_downsampling_log_loss"], color=COLORS["ink"], ls=(0, (4, 3)), lw=1.4,
                 label=f"Trained on all rows: {fmt(g['no_downsampling_log_loss'])}")
    left.set_yscale("log")
    left.set_ylim(0.03, 4)
    left.set_xticks(xs, [name for _, name, _ in OUTPUTS], fontsize=10)
    left.set_ylabel("Test log loss, gradient-boosted model (log scale)")
    left.set_xlabel(f"Probability output, {round(100 * parameter, 1):g}% of negatives kept")
    left.legend(loc="upper right", frameon=False, fontsize=10)

    bars = [("Boosted,\ncorrected", g["corrected"]["log_loss"] - g["no_downsampling_log_loss"], COLORS["teal"]),
            ("Boosted,\ncalibrated", g["calibrated"]["log_loss"] - g["no_downsampling_log_loss"], COLORS["gold"]),
            ("Logistic,\ncorrected", result["logistic"]["corrected"]["log_loss"] - result["logistic"]["no_downsampling_log_loss"], COLORS["navy"])]
    rx = list(range(len(bars)))
    right.bar(rx, [v for _, v, _ in bars], color=[c for _, _, c in bars], edgecolor=COLORS["ink"], lw=0.6, width=0.6)
    top = max(max(v for _, v, _ in bars), 0.001)
    for x, (_, v, _) in zip(rx, bars):
        right.text(x, v + (0.02 * top if v >= 0 else -0.02 * top), signed(v, 4), ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
    right.axhline(0, color=COLORS["ink"], lw=1.0)
    right.set_xticks(rx, [name for name, _, _ in bars], fontsize=10)
    low = min(min(v for _, v, _ in bars), 0)
    right.set_ylim(low - 0.2 * top, top * 1.3)
    right.set_ylabel("Test log loss minus training on all rows")
    right.set_xlabel(f"Model and output, {round(result['training_rows']):,} training rows")


def explain(result, parameter):
    g, lg = result["gbm"], result["logistic"]
    base = g["no_downsampling_log_loss"]
    ratio = g["raw"]["log_loss"] / base
    excess = g["corrected"]["log_loss"] - base
    lr_excess = lg["corrected"]["log_loss"] - lg["no_downsampling_log_loss"]
    auc_move = g["raw"]["auc"] - g["no_downsampling_auc"]
    interpretation = (
        f"Keeping {round(100 * parameter, 1):g}% of negatives leaves {round(result['training_rows']):,} training rows. Raw boosted-model probabilities average "
        f"{fmt(g['raw']['mean_prediction'])} against a true prevalence of {fmt(result['prevalence'])}, so log loss is "
        f"{fmt(g['raw']['log_loss'])} against {fmt(base)} when trained on all rows ({fmt(g['raw']['log_loss'])} / {fmt(base)} = "
        f"{fmt(ratio, 1)} times). The analytic correction brings it to {fmt(g['corrected']['log_loss'], 4)} "
        f"({fmt(g['corrected']['log_loss'], 4)} - {fmt(base, 4)} = {fmt(excess, 4)} against the all-rows model, mean prediction "
        f"{fmt(g['corrected']['mean_prediction'])}); calibrating on development rows gives {fmt(g['calibrated']['log_loss'], 4)}. "
        f"The logistic model's corrected log loss is {signed(lr_excess, 4)} from its own all-rows reference. AUC moves {signed(auc_move)} from {fmt(g['no_downsampling_auc'])}.")
    steps = [
        f"Raw against all rows: {fmt(g['raw']['log_loss'])} / {fmt(base)} = {fmt(ratio, 1)} (ratio of log losses).",
        f"Correction, boosted: {fmt(g['corrected']['log_loss'], 4)} - {fmt(base, 4)} = {signed(excess, 4)}.",
        f"Calibration, boosted: {fmt(g['calibrated']['log_loss'], 4)} - {fmt(base, 4)} = {signed(g['calibrated']['log_loss'] - base, 4)}.",
        f"Correction, logistic: {fmt(lg['corrected']['log_loss'], 4)} - {fmt(lg['no_downsampling_log_loss'], 4)} = {signed(lr_excess, 4)}.",
    ]
    metrics = {"Training rows": f"{round(result['training_rows']):,}", "Raw log loss": fmt(g["raw"]["log_loss"]),
               "Corrected log loss": fmt(g["corrected"]["log_loss"]), "Calibrated log loss": fmt(g["calibrated"]["log_loss"]),
               "Mean corrected prediction": f"{fmt(g['corrected']['mean_prediction'])} vs prevalence {fmt(result['prevalence'])}",
               "AUC": fmt(g["raw"]["auc"])}
    alt = (f"Left: test log loss of the boosted model's raw, corrected and development-calibrated probabilities when "
           f"{round(100 * parameter, 1):g}% of negatives are kept, on a log scale; raw is {fmt(g['raw']['log_loss'])} and the all-rows reference "
           f"is {fmt(base)}. Right: log loss above the all-rows reference for corrected and calibrated boosted output and corrected logistic output.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for r, res in results.items():
        g = res["gbm"]
        if r < 1:
            assert g["raw"]["log_loss"] > 1.8 * g["no_downsampling_log_loss"], f"raw probabilities should be badly off at {r}"
            assert abs(g["raw"]["auc"] - g["corrected"]["auc"]) < 1e-9, "the correction is monotone"
            assert res["logistic"]["corrected"]["log_loss"] - res["logistic"]["no_downsampling_log_loss"] < 0.002, f"logistic correction holds at {r}"
    g = results[0.02]["gbm"]
    assert abs(g["corrected"]["log_loss"] - g["calibrated"]["log_loss"]) < 0.0015, "correction ties calibration at 2%"
    for r in (0.005, 0.002):
        g = results[r]["gbm"]
        assert g["corrected"]["log_loss"] > g["calibrated"]["log_loss"] + 0.004, f"correction falls behind at {r}"
        assert results[r]["calibrated_beats_corrected_gbm"] >= 0.8
    assert results[0.002]["gbm"]["corrected"]["log_loss"] > 1.4 * results[0.002]["gbm"]["calibrated"]["log_loss"]
    assert results[0.002]["gbm"]["corrected"]["mean_prediction"] > 3 * results[0.002]["prevalence"], "overconfident at 0.2%"
    mid = results[0.02]["gbm"]
    assert fmt(mid["raw"]["log_loss"]) == "0.338" and fmt(mid["no_downsampling_log_loss"]) == "0.056", "prediction feedback"
    assert fmt(mid["raw"]["auc"]) == "0.853" and fmt(mid["no_downsampling_auc"]) == "0.851"
    assert 5.5 < mid["raw"]["log_loss"] / mid["no_downsampling_log_loss"] < 6.5, "about six times"
    tiny = results[0.002]
    assert round(tiny["training_rows"]) == 563, "check answer row count"
    assert fmt(tiny["gbm"]["corrected"]["mean_prediction"]) == "0.050" and fmt(tiny["prevalence"]) == "0.013"
    assert fmt(tiny["gbm"]["corrected"]["log_loss"]) == "0.089" and fmt(tiny["gbm"]["calibrated"]["log_loss"]) == "0.059"
    assert fmt(tiny["logistic"]["corrected"]["log_loss"]) == "0.059" and fmt(tiny["logistic"]["no_downsampling_log_loss"]) == "0.058"
    assert results[1]["gbm"]["no_downsampling_auc"] - results[0.002]["gbm"]["raw"]["auc"] > 0.03, "very aggressive rates cost AUC"
