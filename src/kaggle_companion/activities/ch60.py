"""Chapter 60: Multi-Target Learning. A shared-trunk network with and without a dense auxiliary task, by how related the task is."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np

from kaggle_companion.activities._common import clean

SEED = 60
PROBLEMS, DRAWS = 12, 1     # 12 problems, each with its own latent structure and data draw: independent replicates
REPLICATES = PROBLEMS * DRAWS
D, HIDDEN = 10, 12          # input features and shared hidden units
N_MAIN, N_AUX, N_TEST = 120, 1200, 4000
N_FIT = 90                  # of the 120 main labels: 90 fit the network, 30 choose the stopping step
WEIGHTS = [0.0, 0.3, 1.0, 3.0]   # auxiliary loss weight; 0 is the single-task model
STEPS, LR, DECAY, CHECK_EVERY = 400, 0.01, 0.01, 25


def generate(relatedness, problem_seed, data_seed):
    """One problem: its latent structure comes from `problem_seed`, its rows from `data_seed`. Both targets are
    nonlinear functions of the same inputs. The main target reads the latent features tanh(X A). The auxiliary
    target is `relatedness` parts a different readout of the same features and the rest an unrelated signal v. Main labels are scarce (120 rows); auxiliary
    labels are dense (1,200 rows)."""
    prob = np.random.default_rng(problem_seed)
    A, B = prob.normal(size=(D, 3)), prob.normal(size=(D, 3))
    c_main, c_other, c_aux = prob.normal(size=(3, 3))
    probe = prob.normal(size=(5000, D))
    scale = [(np.tanh(probe @ A) @ c_main).std(), (np.tanh(probe @ B) @ c_other).std(), (np.tanh(probe @ A) @ c_aux).std()]
    rng = np.random.default_rng(data_seed)

    def rows(n):
        X = rng.normal(size=(n, D))
        u_main = np.tanh(X @ A) @ c_main / scale[0]
        v = np.tanh(X @ B) @ c_other / scale[1]
        u_aux = np.tanh(X @ A) @ c_aux / scale[2]
        y_main = u_main + rng.normal(0, 0.5, n)
        y_aux = relatedness * u_aux + np.sqrt(1 - relatedness ** 2) * v + rng.normal(0, 0.3, n)
        return X, y_main, y_aux
    return rows(N_MAIN), rows(N_AUX), rows(N_TEST)


def r2(y, pred):
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def fit_multitask(X_fit, y_fit, X_val, y_val, X_aux, y_aux, weight, seed, X_test):
    """Shared tanh layer, one linear head per target. Loss = main MSE + weight * auxiliary MSE, trained with Adam.
    The step is chosen by the main-label validation rows, exactly as for a single-task model; returns the
    test predictions at that step."""
    rng = np.random.default_rng(seed)
    P = {"W": rng.normal(0, 0.5, (D, HIDDEN)), "b": np.zeros(HIDDEN), "head_main": rng.normal(0, 0.3, HIDDEN),
         "head_aux": rng.normal(0, 0.3, HIDDEN), "c_main": np.zeros(1), "c_aux": np.zeros(1)}
    m1 = {k: np.zeros_like(v) for k, v in P.items()}
    m2 = {k: np.zeros_like(v) for k, v in P.items()}
    best_score, best_pred = -np.inf, None
    for t in range(1, STEPS + 1):
        H = np.tanh(X_fit @ P["W"] + P["b"])
        e = (H @ P["head_main"] + P["c_main"] - y_fit) / len(y_fit)          # main error signal
        g = {"head_main": H.T @ e, "c_main": e.sum(keepdims=True)}
        dH = np.outer(e, P["head_main"]) * (1 - H ** 2)
        g["W"], g["b"] = X_fit.T @ dH, dH.sum(0)
        if weight > 0:                                                         # auxiliary gradient joins the shared layer
            Ha = np.tanh(X_aux @ P["W"] + P["b"])
            ea = weight * (Ha @ P["head_aux"] + P["c_aux"] - y_aux) / len(y_aux)
            g["head_aux"], g["c_aux"] = Ha.T @ ea, ea.sum(keepdims=True)
            dHa = np.outer(ea, P["head_aux"]) * (1 - Ha ** 2)
            g["W"], g["b"] = g["W"] + X_aux.T @ dHa, g["b"] + dHa.sum(0)
        else:
            g["head_aux"], g["c_aux"] = np.zeros(HIDDEN), np.zeros(1)
        g["W"] = g["W"] + DECAY * P["W"]
        for k in P:                                                            # Adam update
            m1[k] = 0.9 * m1[k] + 0.1 * g[k]
            m2[k] = 0.999 * m2[k] + 0.001 * g[k] ** 2
            P[k] = P[k] - LR * (m1[k] / (1 - 0.9 ** t)) / (np.sqrt(m2[k] / (1 - 0.999 ** t)) + 1e-8)
        if t % CHECK_EVERY == 0:
            score = r2(y_val, np.tanh(X_val @ P["W"] + P["b"]) @ P["head_main"] + P["c_main"])
            if score > best_score:
                best_score = score
                best_pred = np.tanh(X_test @ P["W"] + P["b"]) @ P["head_main"] + P["c_main"]
    return best_pred


def run(relatedness):
    scores = {w: [] for w in WEIGHTS}
    for i in range(REPLICATES):
        problem, draw = divmod(i, DRAWS)
        (Xm, ym, _), (Xa, _, ya), (Xt, yt, _) = generate(relatedness, SEED * 100 + problem, SEED * 1000 + i)
        for w in WEIGHTS:
            pred = fit_multitask(Xm[:N_FIT], ym[:N_FIT], Xm[N_FIT:], ym[N_FIT:], Xa, ya, w, i, Xt)
            scores[w].append(r2(yt, pred))                                      # R2 on 4,000 fresh rows
    base = np.array(scores[0.0])
    gain = {w: np.array(scores[w]) - base for w in WEIGHTS}
    return clean({
        "relatedness": relatedness, "weights": WEIGHTS,
        "r2": {str(w): float(np.mean(scores[w])) for w in WEIGHTS},
        "per_replicate": {str(w): scores[w] for w in WEIGHTS},
        "gain": {str(w): float(gain[w].mean()) for w in WEIGHTS},
        "gain_se": {str(w): float(gain[w].std(ddof=1) / np.sqrt(REPLICATES)) for w in WEIGHTS},
        "gain_per_replicate": {str(w): gain[w] for w in WEIGHTS},
        "share_better": {str(w): float(np.mean(gain[w] > 0)) for w in WEIGHTS},
        "replicates": REPLICATES, "main_labels": N_MAIN, "auxiliary_labels": N_AUX, "test_rows": N_TEST,
    })
# notebook-end


SPEC = {
    "chapter": 60,
    "chapter_title": "Multi-Target Learning",
    "subtitle": "Compare the primary metric with and without the auxiliary task, and let relatedness and weight decide.",
    "summary": ("An auxiliary target feeds gradients into a shared layer. One demonstration measures held-out accuracy on a scarce "
                "main target, with and without a dense auxiliary target, as the auxiliary target's relatedness and loss weight change."),
    "title": "Held-out main-target accuracy with and without an auxiliary task",
    "question": "When does a dense auxiliary target improve a scarce main target on new rows, and how much does the loss weight matter?",
    "why": ("Adding targets is cheap and tempting, but the chapter's rule is to compare the primary metric with and without the "
            "auxiliary task under a fixed budget. The size and sign of the change depend on how related the tasks are and on the "
            "weight, and a single run can mislead either way."),
    "method": ("Twelve constructed problems, each with its own latent structure. The main target has only 120 "
               "labelled rows; the auxiliary target has 1,200. Both are nonlinear functions of the same ten inputs. The auxiliary target "
               "mixes a signal built on the main target's latent features (the same hidden directions, read out differently) with an "
               "unrelated signal; the control is the weight on the shared part (1.0 means entirely built on the shared features, 0 means "
               "an unrelated signal). A shared tanh layer of 12 units with one linear head per "
               "target is trained on the loss main MSE + w * auxiliary MSE, for w in 0 (single-task), 0.3, 1 and 3. Every model stops at "
               "the step that scores best on 30 held-back main-label rows, then is scored by R-squared on 4,000 new rows."),
    "control": {"key": "relatedness", "label": "Relatedness of the auxiliary target (share of its signal built on the main target's latent features)",
                "values": [1.0, 0.7, 0.3, 0.0], "default": 0.7,
                "value_labels": ["1.0: same latent features", "0.7", "0.3", "0: unrelated signal"]},
    "source_section": "Auxiliary Tasks: What Qualifies",
    "symbols": ("L is the training loss, MSE_main the mean squared error of the main head on its labelled rows, MSE_aux the mean "
                "squared error of the auxiliary head on its rows and w the auxiliary loss weight. w = 0 is the single-task model."),
    "explanation": ("The auxiliary gradient reaches the shared layer, so it shapes the features the main head reads. When the auxiliary "
                    "target is built on the same latent features, its 1,200 dense rows teach the shared layer features that 120 main labels "
                    "cannot, and a larger weight helps more. As its signal drifts away from the main target's, those extra rows teach "
                    "features the main head does not need, and the gain shrinks toward noise. The weight multiplies whatever the "
                    "auxiliary task teaches, useful or not."),
    "application": ("Train the multi-target model and a single-task reference under the same stopping rule, compare the primary "
                    "metric on rows neither used, and keep the auxiliary task only if the gain is clear, then tune its weight."),
    "assumptions": ("Constructed data and a small numpy network standing in for the chapter's PyTorch trunk. The main-target stopping step "
                    "comes from only 30 rows, which is noisy, and the size of the gain differs between problems, "
                    "so the chart shows each replicate. No gradient alignment is measured: a small or negative change is not proof "
                    "of gradient conflict, which is the chapter's own caution. Under two other seeds the weight-3 gain at relatedness 1.0 stayed "
                    "clear (about 0.07) and the related settings kept beating the unrelated ones, but the weight-3 gain at 0.7 fell to about "
                    "0.03, so the exact sizes and the order of 1.0 and 0.7 are not stable."),
    "prediction": ("With an auxiliary target built entirely on the main target's latent features (relatedness 1.0) and loss weight 3, what happens to the "
                   "main target's R-squared on new rows compared with the single-task model?"),
    "prediction_options": ["It falls: the auxiliary gradient interferes", "It barely changes (within 0.02)", "It rises by about 0.08"],
    "prediction_answer": 2,
    "prediction_feedback": {
        "correct": "R-squared rises from 0.574 to 0.653, a gain of 0.079: the dense auxiliary rows teach the shared layer features that 120 main labels cannot.",
        "incorrect": "R-squared rises from 0.574 to 0.653, a gain of 0.079. With a fully related auxiliary target the dense rows teach the shared layer useful features.",
    },
    "check": "At relatedness 0, is the weight-3 model better than the single-task model, and what does that say about adding an auxiliary task by default?",
    "answer": ("At relatedness 0 the weight-3 model changes R-squared by -0.004, against a standard error of 0.025, so there is no evidence "
               "of any gain, and the weight-1 model is 0.002 below the single-task model. At relatedness 1.0 the weight-3 gain is 0.079, "
               "about 5.8 standard errors. The auxiliary task has to earn its place against the single-task reference on the primary metric."),
    "provenance": "Constructed example: seeded synthetic regression problems and a small numpy shared-layer network, measured by the chapter activity.",
    "apply": [
        "Train a single-task reference with the same stopping rule and budget before adding any auxiliary target.",
        "Compare the primary metric alone on rows neither model used, and keep per-target results visible rather than only an average.",
        "Prefer auxiliary targets that share latent structure with the primary target and have denser labels than the primary target.",
        "Treat the auxiliary loss weight as a hyperparameter: raise it only while the primary metric keeps improving, and expect a flat or negative change when the tasks are weakly related.",
    ],
    "honesty": ("Constructed data. The gains are properties of this generator and of 12 replicates; the unrelated-task result is small and "
                "within noise, so the activity reports no reliable harm from an unrelated auxiliary task."),
}

EQUATIONS = [{"tex": r"L = \mathrm{MSE}_{\mathrm{main}} + w\,\mathrm{MSE}_{\mathrm{aux}}",
              "alt": "L equals the main task mean squared error plus w times the auxiliary task mean squared error",
              "basis": "The activity's weighted multi-target loss; the chapter's weighted-loss discussion (Loss Weighting and Gradient Balancing) has no display equation."}]
NCOLS = 2
HEIGHT = 4.4


def shown_gain(result, w):
    """Gain computed from the three-decimal R-squared values the page displays, so every hand calculation adds up."""
    return float(fmt(result["r2"][str(w)])) - float(fmt(result["r2"]["0.0"]))


def draw(axes, result, parameter):
    left, right = axes
    ws = result["weights"]
    labels = ["Single\ntask" if w == 0 else f"w = {w:g}" for w in ws]
    xs = list(range(len(ws)))
    means = [result["r2"][str(w)] for w in ws]
    top = max(max(result["per_replicate"][str(w)]) for w in ws)
    bottom = min(min(result["per_replicate"][str(w)]) for w in ws)
    for x, w, m in zip(xs, ws, means):
        dots = result["per_replicate"][str(w)]
        left.scatter([x + (i - (len(dots) - 1) / 2) * 0.035 for i in range(len(dots))], dots, s=10, color=COLORS["grey"], zorder=3)
        left.scatter([x], [m], s=70, marker="D", color=COLORS["light"] if w == 0 else COLORS["teal"], edgecolor=COLORS["ink"], zorder=4)
        left.text(x + 0.14, m, fmt(m), ha="left", va="center", fontsize=10, bbox={"fc": "white", "ec": "none", "pad": 1})
    left.set_ylim(bottom - 0.05, top + 0.05)
    left.axhline(means[0], color=COLORS["gold"], ls=(0, (4, 3)), lw=1.4)
    left.set_xticks(xs, labels)
    left.set_xlim(-0.5, len(ws) - 0.35)
    left.set_ylabel("R-squared on 4,000 new rows (diamond: mean)")
    left.set_xlabel(f"Auxiliary loss weight, relatedness {parameter:g}")

    right.axhline(0, color=COLORS["grey"], lw=0.8)
    for x, w in zip(xs[1:], ws[1:]):
        g = result["gain_per_replicate"][str(w)]
        right.scatter([x + (i - (len(g) - 1) / 2) * 0.035 for i in range(len(g))], g, s=10, color=COLORS["navy"], zorder=3)
        mean, se = result["gain"][str(w)], result["gain_se"][str(w)]
        right.errorbar([x], [mean], yerr=[2 * se], color=COLORS["terracotta"], marker="D", ms=6, lw=2, capsize=5, zorder=4)
        right.text(x + 0.17, mean, signed(shown_gain(result, w)), ha="left", va="center", fontsize=10, color=COLORS["terracotta"],
                   bbox={"fc": "white", "ec": "none", "pad": 1})
    right.set_xticks(xs[1:], labels[1:])
    right.set_xlim(0.4, len(ws) - 0.4)
    right.set_ylabel("Gain over single task (R-squared)")
    right.set_xlabel("Auxiliary loss weight (diamond: mean, bar: 2 standard errors)")


def explain(result, parameter):
    r2s, se = result["r2"], result["gain_se"]
    big = shown_gain(result, 3.0)
    gain = {k: shown_gain(result, float(k)) for k in ("0.3", "1.0", "3.0")}
    clear = result["gain"]["3.0"] > 2 * se["3.0"]
    interpretation = (
        f"At relatedness {parameter:g} the single-task model scores R-squared {fmt(r2s['0.0'])} on new rows. With the auxiliary task at "
        f"weight 3 it scores {fmt(r2s['3.0'])}, and {fmt(r2s['3.0'])} - {fmt(r2s['0.0'])} = {fmt(big)} (standard error {fmt(se['3.0'])}, "
        f"better in {round(100 * result['share_better']['3.0'])}% of {result['replicates']} replicates). "
        + ("The gain is more than two standard errors, so the auxiliary task earns its place. "
           if clear else "The gain is within two standard errors, so this run gives no clear reason to keep the auxiliary task. ")
        + f"Weight 1 changes the score by {signed(gain['1.0'])} and weight 0.3 by {signed(gain['0.3'])}.")
    steps = [
        f"Single-task model: R-squared {fmt(r2s['0.0'])} on {result['test_rows']:,} new rows.",
        f"Weight 3: {fmt(r2s['3.0'])} - {fmt(r2s['0.0'])} = {fmt(big)}, with standard error {fmt(se['3.0'])}.",
        f"Weight 1: {fmt(r2s['1.0'])} - {fmt(r2s['0.0'])} = {fmt(gain['1.0'])}.",
        f"Weight 0.3: {fmt(r2s['0.3'])} - {fmt(r2s['0.0'])} = {fmt(gain['0.3'])}.",
    ]
    metrics = {"Single-task R-squared": fmt(r2s["0.0"]), "R-squared at weight 0.3": fmt(r2s["0.3"]),
               "R-squared at weight 1": fmt(r2s["1.0"]), "R-squared at weight 3": fmt(r2s["3.0"]),
               "Gain at weight 3": signed(big), "Gain at weight 3, in standard errors": fmt(result["gain"]["3.0"] / se["3.0"], 1)}
    alt = (f"Left: bars of held-out R-squared for the single-task model ({fmt(r2s['0.0'])}) and for weights 0.3, 1 and 3 "
           f"({fmt(r2s['0.3'])}, {fmt(r2s['1.0'])}, {fmt(r2s['3.0'])}) at relatedness {parameter:g}, with a dot per replicate. "
           f"Right: the paired gain over the single-task model for each weight, with mean and two standard errors; the weight-3 gain is {signed(big)}.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    g3 = {rel: res["gain"]["3.0"] for rel, res in results.items()}
    assert min(g3[1.0], g3[0.7]) > g3[0.3] > g3[0.0], f"gain at weight 3 should fall from related to unrelated: {g3}"
    assert g3[1.0] > 2 * results[1.0]["gain_se"]["3.0"] and g3[0.7] > 2 * results[0.7]["gain_se"]["3.0"], "related settings: clear gain"
    full = results[1.0]
    assert full["gain"]["3.0"] > full["gain"]["1.0"] > full["gain"]["0.3"] > 0, "related task: larger weight helps more"
    assert fmt(results[0.7]["r2"]["0.0"]) == fmt(full["r2"]["0.0"]), "single-task model is the same at every relatedness"
    assert fmt(full["r2"]["0.0"]) == "0.574" and fmt(full["r2"]["3.0"]) == "0.653", "prediction feedback numbers"
    assert fmt(shown_gain(full, 3.0)) == "0.079", "prediction feedback gain"
    assert 0.07 < full["gain"]["3.0"] < 0.09, "prediction option says about 0.08"
    none = results[0.0]
    assert all(abs(none["gain"][k]) < 0.03 for k in ("0.3", "1.0", "3.0")), "unrelated task: gains small in either direction"
    assert abs(none["gain"]["3.0"]) < none["gain_se"]["3.0"] * 2, "unrelated task at weight 3 within two standard errors"
    assert fmt(shown_gain(none, 3.0)) == "-0.004" and fmt(none["gain_se"]["3.0"]) == "0.025", "check answer numbers"
    assert fmt(shown_gain(none, 1.0)) == "-0.002", "check answer weight-1 number"
    assert fmt(full["gain"]["3.0"] / full["gain_se"]["3.0"], 1) == "5.8", "check answer: about 5.8 standard errors"
