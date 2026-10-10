"""Chapter 16: The Fourth-Root Blend. Equal, fourth-root and development-fitted weights, as the models' errors become more alike."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from kaggle_companion.activities._common import clean

SEED = 16
REPLICATES = 40                      # independent constructed datasets; every estimate is their mean
NOISE = np.array([1.0, 1.15, 1.3])   # how noisy each model's view of the hidden signal is: three similar-strength models
N_TRAIN, N_DEV, N_ASSESS = 600, 300, 20000
EPS = 1e-4


def generate(rng, n, shared):
    """A hidden signal drives the label. Each model sees a noisy copy; `shared` is the share of each copy's noise
    variance that is common to all three models, so it sets how alike their errors are."""
    signal = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-1.6 * signal))).astype(int)
    common = rng.normal(size=n)
    views = [signal + sd * (np.sqrt(shared) * common + np.sqrt(1 - shared) * rng.normal(size=n)) for sd in NOISE]
    return np.column_stack(views), y


def fourth_root_weights(scores, baseline, maximize=True):
    """The chapter's heuristic: compress each positive margin over a measured baseline with a fourth root."""
    scores = np.asarray(scores, dtype=float)
    margin = scores - baseline if maximize else baseline - scores
    raw = np.maximum(margin, 0) ** 0.25
    return raw / raw.sum()


def greedy_weights(y, P, step=0.1, max_iterations=30):
    """Hill-climbing on development log loss: start from the best model, move weight to whichever model helps most."""
    loss = lambda q: log_loss(y, np.clip(q, EPS, 1 - EPS))
    weights = np.eye(P.shape[1])[int(np.argmin([loss(P[:, j]) for j in range(P.shape[1])]))]
    current = P @ weights
    for _ in range(max_iterations):
        trial = [loss((1 - step) * current + step * P[:, j]) for j in range(P.shape[1])]
        j = int(np.argmin(trial))
        if trial[j] >= loss(current) - 1e-12:
            break
        weights = weights * (1 - step)
        weights[j] += step
        current = P @ weights
    return weights


def run(shared):
    loss = lambda y, q: log_loss(y, np.clip(q, EPS, 1 - EPS))
    rows = []
    for rep in range(REPLICATES):
        rng = np.random.default_rng(SEED * 100 + rep)
        (X, y), (Xd, yd), (Xa, ya) = [generate(rng, n, shared) for n in (N_TRAIN, N_DEV, N_ASSESS)]
        models = [LogisticRegression().fit(X[:, [m]], y) for m in range(3)]      # one model per view
        Pd = np.column_stack([models[m].predict_proba(Xd[:, [m]])[:, 1] for m in range(3)])
        Pa = np.column_stack([models[m].predict_proba(Xa[:, [m]])[:, 1] for m in range(3)])
        dev_loss = np.array([loss(yd, Pd[:, m]) for m in range(3)])
        dummy = loss(yd, np.full(N_DEV, y.mean()))                              # measured baseline: a constant-prior predictor
        w_root = fourth_root_weights(dev_loss, dummy, maximize=False)
        w_greedy = greedy_weights(yd, Pd)
        logits = np.log(Pd / (1 - Pd))
        corr = np.corrcoef(logits.T)
        rows.append({
            "best": loss(ya, Pa[:, int(np.argmin(dev_loss))]),                  # strongest model, chosen on development rows
            "equal": loss(ya, Pa.mean(axis=1)), "root": loss(ya, Pa @ w_root), "greedy": loss(ya, Pa @ w_greedy),
            "equal_dev": loss(yd, Pd.mean(axis=1)), "greedy_dev": loss(yd, Pd @ w_greedy),
            "correlation": (corr[0, 1] + corr[0, 2] + corr[1, 2]) / 3, "root_spread": w_root.max() - w_root.min(),
            "w_root": w_root, "w_greedy": w_greedy,
        })
    col = lambda k: np.array([r[k] for r in rows])
    gain = {k: col("best") - col(k) for k in ("equal", "root", "greedy")}      # positive: the blend beats the best single model
    return clean({
        "shared_noise": shared, "correlation": float(col("correlation").mean()), "root_weight_spread": float(col("root_spread").mean()),
        "assessment_loss": {k: float(col(k).mean()) for k in ("best", "equal", "root", "greedy")},
        "gain_over_best": {k: float(v.mean()) for k, v in gain.items()},
        "gain_standard_error": {k: float(v.std() / np.sqrt(REPLICATES)) for k, v in gain.items()},
        "share_equal_beats_best": float((gain["equal"] > 0).mean()),
        "mean_weights": {"root": np.mean([r["w_root"] for r in rows], axis=0), "greedy": np.mean([r["w_greedy"] for r in rows], axis=0)},
        "greedy_vs_equal": {"development": float((col("equal_dev") - col("greedy_dev")).mean()),
                            "assessment": float((col("equal") - col("greedy")).mean())},
        "replicates": REPLICATES, "assessment_rows": N_ASSESS, "development_rows": N_DEV,
    })
# notebook-end


SPEC = {
    "chapter": 16,
    "chapter_title": "The Fourth-Root Blend",
    "subtitle": "Equal averaging is the comparator; a blend must beat the strongest model on rows that chose none of its weights.",
    "summary": ("Blending pays only when the models' errors differ. One demonstration measures equal, fourth-root and development-fitted "
                "blends against the best single model as the models' prediction correlation rises."),
    "title": "Blending three similar models as their errors become alike",
    "question": "How alike can three models' predictions be before a blend stops beating the best single model, and do the weights matter?",
    "why": ("The chapter treats a blend as a candidate that must beat the strongest component and cost less than it gains. "
            "Prediction correlation is its screen, so the useful number is where the blend's gain disappears."),
    "method": ("A constructed binary task. A hidden signal drives the label; three logistic regressions each see a noisy copy of it "
               "(noise 1.0, 1.15 and 1.3), and the share of that noise common to the three copies is the control. Models are fitted "
               "on 600 rows. On 300 development rows the strongest model is chosen, fourth-root weights are computed from "
               "each model's log loss against a measured constant-prediction baseline, and a hill-climbing search fits weights. "
               "Everything is scored by log loss on 20,000 assessment rows. Forty independent datasets are averaged."),
    "control": {"key": "shared_noise", "label": "Share of each model's noise that is common to all three",
                "values": [0.0, 0.3, 0.7, 0.95], "default": 0.95,
                "value_labels": ["0: independent errors", "0.3", "0.7", "0.95: nearly identical errors"]},
    "source_section": "The Fourth-Root Weight Formula",
    "symbols": ("m_i is model i's margin over the baseline (baseline loss minus its development loss), b the loss of a defined dummy "
                "predictor and w_i the weight of model i. Gain over the best model is its assessment log loss minus the blend's; "
                "positive means the blend wins."),
    "explanation": ("Averaging cancels each model's own noise but not the noise the models share. With independent errors the "
                    "blend removes a lot; as the shared part grows the blend's gain shrinks, and because the three models are not "
                    "equally good the average eventually does worse than simply keeping the best one. Fourth-root weights compress "
                    "the margins so far that they land close to equal weights."),
    "application": ("Check prediction correlation first as a screen, then compare equal weights, the fourth-root heuristic and any fitted "
                    "weights with the strongest single model on assessment rows. Keep the single model when the blend does not beat it."),
    "assumptions": ("Constructed data and three logistic regressions standing in for three different model families; the models differ only "
                    "in noise, whereas real models also differ in bias. The baseline for the fourth-root rule is the measured log loss of a "
                    "constant-prior predictor, as the chapter requires, not a target mean passed off as a score. Correlation here is "
                    "measured on the 300 development rows."),
    "prediction": "When the three models' errors are almost identical (shared noise 0.95), how does the equal-weight blend compare with the best single model?",
    "prediction_options": ["Better, because averaging always helps", "About the same", "Worse"],
    "prediction_answer": 2,
    "prediction_feedback": {
        "correct": "The equal blend has log loss 0.623 against 0.613 for the best model: 0.010 worse, not better.",
        "incorrect": "The equal blend has log loss 0.623 against 0.613 for the best model: 0.010 worse. Shared noise does not average away, and the weaker models dilute the best one.",
    },
    "check": "Fourth-root weights and equal weights score almost the same here. Why, and does that make the heuristic a good idea?",
    "answer": ("The fourth root compresses margins so far that the three weights differ by only 0.028 on average, so the blend is nearly an equal "
               "average (log loss 0.623 for both). That is the chapter's point: it is a mild heuristic, not an optimality claim. "
               "It earns its place only if it beats equal weights on assessment rows, and here the difference is within noise."),
    "provenance": "Constructed example: seeded synthetic data and three logistic regressions, measured by the chapter activity.",
    "apply": [
        "Screen the candidates by prediction correlation, but decide by the blend's score on assessment rows, not by a correlation cutoff.",
        "Compare equal weights, the fourth-root heuristic and any fitted weights against the strongest single model, which is the real baseline.",
        "Use a measured dummy predictor as the fourth-root baseline, and expect weights close to equal.",
        "Keep the single model when the blend does not beat it, and count the extra fits and inference as cost.",
    ],
    "honesty": ("Constructed data with models that differ only in noise. The crossover correlation depends on how different the models' "
                "strengths are; with models of equal strength an equal blend would not fall below the best one."),
}

EQUATIONS = [{"tex": r"w_i = \frac{\max(m_i, 0)^{1/4}}{\sum_j \max(m_j, 0)^{1/4}}, \qquad m_i = b - \ell_i",
              "alt": "w i equals the fourth root of the larger of m i and zero, divided by the sum over j of the same quantity, where m i is the baseline loss b minus model i's loss",
              "basis": "The chapter's fourth_root_weights function for a loss to minimize (The Fourth-Root Weight Formula); the chapter shows it as code."}]
NCOLS = 2
HEIGHT = 4.4
BLENDS = [("equal", "Equal weights", COLORS["navy"]), ("root", "Fourth-root weights", COLORS["teal"]),
          ("greedy", "Fitted weights", COLORS["gold"])]


def draw(axes, result, parameter):
    left, right = axes
    xs = range(len(BLENDS))
    values = [result["gain_over_best"][k] for k, _, _ in BLENDS]
    errors = [result["gain_standard_error"][k] for k, _, _ in BLENDS]
    left.bar(xs, values, width=0.55, color=[c for _, _, c in BLENDS], edgecolor=COLORS["ink"], lw=0.6)
    left.errorbar(list(xs), values, yerr=errors, fmt="none", ecolor=COLORS["ink"], capsize=4, lw=1.2)
    for x, v, e in zip(xs, values, errors):
        left.text(x, v + e + 0.001 if v >= 0 else v - e - 0.001, signed(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
    left.axhline(0, color=COLORS["grey"], lw=0.9, label="Best single model")
    left.set_xticks(list(xs), [name for _, name, _ in BLENDS])
    bound = max(abs(v) + e for v, e in zip(values, errors)) + 0.008
    left.set_ylim(-bound, bound)
    left.set_ylabel("Log loss gain over the best single model")
    left.set_xlabel(f"Blend, correlation {fmt(result['correlation'], 2)} (bars: mean, lines: one standard error)")
    left.legend(loc="upper right", frameon=False, fontsize=10)

    w = result["mean_weights"]
    width = 0.26
    for offset, key, name, color in ((-width, "root", "Fourth-root", COLORS["teal"]), (0, None, "Equal", COLORS["light"]),
                                     (width, "greedy", "Fitted", COLORS["gold"])):
        vals = [1 / 3] * 3 if key is None else list(w[key])
        right.bar([i + offset for i in range(3)], vals, width=width, color=color, edgecolor=COLORS["ink"], lw=0.6, label=name)
        for i, v in enumerate(vals):
            right.text(i + offset, v + 0.01, fmt(v, 2), ha="center", va="bottom", fontsize=10, rotation=90)
    right.set_xticks(range(3), ["Model 1\n(least noisy)", "Model 2", "Model 3\n(noisiest)"])
    right.set_ylim(0, 1.3)
    right.set_ylabel("Mean weight in the blend")
    right.set_xlabel("Model (weights averaged over datasets)")
    right.legend(loc="upper right", frameon=False, fontsize=10, ncols=3)


def diff(a, b):
    """Difference of the displayed (3 decimal) values, so the hand calculation in the text adds up."""
    return fmt(round(a, 3) - round(b, 3))


def explain(result, parameter):
    L, g = result["assessment_loss"], result["gain_over_best"]
    gd = result["greedy_vs_equal"]
    verdict = "the equal blend beats the best single model" if g["equal"] > 0.002 else (
        "the equal blend does worse than the best single model" if g["equal"] < -0.002 else "the equal blend ties the best single model")
    interpretation = (
        f"At prediction correlation {fmt(result['correlation'], 2)}, {verdict}: {fmt(L['best'])} - {fmt(L['equal'])} = {diff(L['best'], L['equal'])} log loss "
        f"on {result['assessment_rows']:,} assessment rows, and it beats the best model in {round(100 * result['share_equal_beats_best'])}% of "
        f"{result['replicates']} datasets. Fourth-root weights score {fmt(L['root'])} (weights differ by {fmt(result['root_weight_spread'])} on average). "
        f"Against equal weights, fitted weights change log loss by {signed(-gd['development'], 4)} on the rows the weights were fitted to "
        f"and by {signed(-gd['assessment'], 4)} on assessment rows (negative means fitted weights do better).")
    steps = [
        f"Equal blend: {fmt(L['best'])} - {fmt(L['equal'])} = {diff(L['best'], L['equal'])} (best model minus blend; positive means the blend wins).",
        f"Fourth-root blend: {fmt(L['best'])} - {fmt(L['root'])} = {diff(L['best'], L['root'])}.",
        f"Fitted blend: {fmt(L['best'])} - {fmt(L['greedy'])} = {diff(L['best'], L['greedy'])}.",
        f"Fourth-root weights differ from each other by {fmt(result['root_weight_spread'])} on average (equal weights differ by 0).",
    ]
    metrics = {"Prediction correlation": fmt(result["correlation"], 2), "Best single model": fmt(L["best"]), "Equal blend": fmt(L["equal"]),
               "Fourth-root blend": fmt(L["root"]), "Fitted blend": fmt(L["greedy"])}
    alt = (f"Left: log loss gain over the best single model for equal ({signed(g['equal'])}), fourth-root ({signed(g['root'])}) and fitted "
           f"({signed(g['greedy'])}) weights at prediction correlation {fmt(result['correlation'], 2)}. Right: mean weight of each of the three models "
           f"under fourth-root, equal and fitted weights; fitted weights give the least noisy model {fmt(result['mean_weights']['greedy'][0], 2)}.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    order = sorted(results)
    corr = [results[s]["correlation"] for s in order]
    assert all(b > a for a, b in zip(corr, corr[1:])), "correlation should rise with shared noise"
    gains = [results[s]["gain_over_best"]["equal"] for s in order]
    assert all(b < a for a, b in zip(gains, gains[1:])), "equal-blend gain should fall as correlation rises"
    assert gains[0] > 0.01, "independent errors: the blend should clearly win"
    assert gains[-1] < -0.005, "nearly identical errors: the equal blend should lose to the best model"
    for s in order:
        res = results[s]
        assert abs(res["assessment_loss"]["root"] - res["assessment_loss"]["equal"]) < 0.002, f"fourth root should stay near equal at {s}"
        assert res["root_weight_spread"] < 0.1, f"fourth-root weights should be close to equal at {s}"
    hi = results[0.95]
    assert hi["share_equal_beats_best"] < 0.3, "equal should rarely beat the best at 0.95"
    assert hi["greedy_vs_equal"]["development"] > 0 and hi["greedy_vs_equal"]["assessment"] > 0, "fitted weights help versus equal when equal is poor"
    assert hi["mean_weights"]["greedy"][0] > 0.9, "at 0.95 the fitted weights should collapse onto the best model"
    for s in (0.0, 0.3, 0.7):
        gd = results[s]["greedy_vs_equal"]
        assert gd["assessment"] < gd["development"] - 0.0005, f"fitted weights should lose some gain out of sample at {s}"
    assert fmt(hi["assessment_loss"]["equal"]) == "0.623" and fmt(hi["assessment_loss"]["best"]) == "0.613", "prediction feedback numbers"
    assert fmt(hi["assessment_loss"]["equal"] - hi["assessment_loss"]["best"]) == "0.010", "prediction feedback gap"
    assert fmt(hi["root_weight_spread"]) == "0.028" and fmt(hi["assessment_loss"]["root"]) == "0.623", "check answer numbers"
