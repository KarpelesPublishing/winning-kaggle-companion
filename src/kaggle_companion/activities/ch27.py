"""Chapter 27: SHAP as a Feature Engineering Signal. Exact Shapley attribution for a pair of near-copies, then paired removal tests."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import itertools
import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from kaggle_companion.activities._common import clean

SEED = 27
REPLICATES = 8             # independent constructed datasets; every estimate is their mean
P = 6                      # columns: copy A, copy B and four pure-noise columns
N_TRAIN, N_VALID, N_FRESH = 1200, 1200, 6000
EXPLAIN_ROWS, BACKGROUND_ROWS = 100, 30


def generate(n, copy_corr, rng):
    """Columns 0 and 1 are two noisy readings of one hidden signal z; they correlate at `copy_corr` (1 = identical).
    Only z drives the label, so either reading alone carries what the model needs."""
    z = rng.normal(size=n)
    s = math.sqrt(1.0 / copy_corr - 1.0)                       # reading noise that gives the requested correlation
    scale = math.sqrt(1.0 + s * s)                             # keep each reading at unit variance
    readings = [(z + s * rng.normal(size=n)) / scale for _ in range(2)]
    X = np.column_stack(readings + [rng.normal(size=n) for _ in range(P - 2)])
    y = (rng.random(n) < 1 / (1 + np.exp(-1.8 * z))).astype(int)
    return X, y


def fit(X, y, seed):
    return HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=8, min_samples_leaf=20,
                                          random_state=seed).fit(X, y)


SUBSETS = [S for k in range(P + 1) for S in itertools.combinations(range(P), k)]


def mean_abs_shap(model, rows, background):
    """Exact interventional Shapley values for P columns, in predicted-probability units.
    v(S) = mean prediction when columns in S come from the explained row and the others from background rows."""
    n_rows, n_bg = len(rows), len(background)
    value = {}
    for S in SUBSETS:
        hybrid = np.tile(background, (n_rows, 1))
        if S:
            hybrid[:, list(S)] = np.repeat(rows, n_bg, axis=0)[:, list(S)]
        value[S] = model.predict_proba(hybrid)[:, 1].reshape(n_rows, n_bg).mean(axis=1)
    phi = np.zeros((n_rows, P))
    for j in range(P):
        for S in SUBSETS:
            if j not in S:
                weight = math.factorial(len(S)) * math.factorial(P - len(S) - 1) / math.factorial(P)
                phi[:, j] += weight * (value[tuple(sorted(S + (j,)))] - value[S])
    return np.abs(phi).mean(axis=0)


def one_replicate(copy_corr, seed):
    rng = np.random.default_rng(seed)
    X, y = generate(N_TRAIN, copy_corr, rng)
    Xv, yv = generate(N_VALID, copy_corr, rng)
    Xf, yf = generate(N_FRESH, copy_corr, rng)                  # fresh rows from the same generator: the "truth"
    model = fit(X, y, seed)
    attribution = mean_abs_shap(model, Xv[:EXPLAIN_ROWS], X[:BACKGROUND_ROWS])
    low = int(np.argmin(attribution[:2]))                       # the copy the attribution ranks lower
    high = 1 - low

    def loss(cols):                                             # refit on the kept columns, score on both row sets
        m = fit(X[:, cols], y, seed)
        return (log_loss(yv, m.predict_proba(Xv[:, cols])[:, 1]), log_loss(yf, m.predict_proba(Xf[:, cols])[:, 1]))

    keep_all = list(range(P))
    full = loss(keep_all)
    removals = {"drop_noise": [c for c in keep_all if c != 2],  # control: one column that carries nothing
                "drop_low_copy": [c for c in keep_all if c != low],
                "drop_both": [c for c in keep_all if c > 1]}
    out = {name: [a - b for a, b in zip(loss(cols), full)] for name, cols in removals.items()}   # [validation, fresh]
    return {"attr_low": attribution[low], "attr_high": attribution[high], "attr_noise": float(attribution[2:].mean()),
            "low_is_column_a": low == 0, "full_fresh_loss": full[1], **out}


def run(copy_corr):
    reps = [one_replicate(copy_corr, SEED * 100 + i) for i in range(REPLICATES)]
    mean = lambda f: float(np.mean([f(x) for x in reps]))
    return clean({
        "copy_corr": copy_corr,
        "attr_low": mean(lambda x: x["attr_low"]), "attr_high": mean(lambda x: x["attr_high"]),
        "attr_noise": mean(lambda x: x["attr_noise"]),
        "low_is_column_a": int(sum(x["low_is_column_a"] for x in reps)),
        "full_fresh_loss": mean(lambda x: x["full_fresh_loss"]),
        **{f"{k}_fresh": mean(lambda x, k=k: x[k][1]) for k in ("drop_noise", "drop_low_copy", "drop_both")},
        **{f"{k}_valid": mean(lambda x, k=k: x[k][0]) for k in ("drop_noise", "drop_low_copy", "drop_both")},
        "per_replicate": {"attr_low": [x["attr_low"] for x in reps], "attr_high": [x["attr_high"] for x in reps],
                          **{f"{k}_fresh": [x[k][1] for x in reps] for k in ("drop_noise", "drop_low_copy", "drop_both")}},
        "replicates": REPLICATES, "train_rows": N_TRAIN, "fresh_rows": N_FRESH,
    })
# notebook-end


SPEC = {
    "chapter": 27,
    "chapter_title": "SHAP as a Feature Engineering Signal",
    "subtitle": "Use attribution to form a hypothesis, then test the removal on the same development rows.",
    "summary": ("A near-zero attribution does not make a column unneeded. One demonstration attributes a model's predictions to two "
                "near-copies of one signal, then removes the low-attribution copy, then both, and measures the loss each time."),
    "title": "Low attribution for a copy, and what removing it costs",
    "question": "When two columns carry the same information, does the one with low attribution deserve to be dropped, and what does dropping both cost?",
    "why": ("Mean absolute SHAP is a common feature-selection screen, and copies of a signal are common in real tables. Measuring the "
            "attribution next to the removal tells you when a low value reflects redundancy rather than noise."),
    "method": ("Eight constructed datasets of 1,200 training rows with six columns. Columns A and B are two noisy readings of one "
               "hidden signal and correlate at the control value (1.0 is an identical copy); four more columns are pure noise. "
               "Only the signal drives the label. A histogram gradient boosting classifier is fitted, and exact Shapley values "
               "(by enumerating all 64 column subsets against 30 background rows) give mean absolute attribution on 100 "
               "validation rows. The copy ranked lower is then removed, then both copies, with a refit and the change in log "
               "loss measured on 6,000 fresh rows. Dropping one noise column is the control."),
    "control": {"key": "copy_corr", "label": "Correlation between the two copies (1.0 = identical columns)",
                "values": [0.5, 0.9, 0.99, 1.0], "default": 1.0,
                "value_labels": ["0.5: loosely related readings", "0.9", "0.99: near-copies", "1.0: identical copy"]},
    "source_section": "A Removal Contrast",
    "symbols": ("phi_j is the attribution of column j to one prediction, v(S) the mean model output when only the columns in S "
                "are taken from the explained row, and mean |phi_j| the quantity a SHAP screen ranks. The weight is the share of "
                "column orderings in which the columns of S come before j."),
    "explanation": ("Shapley attribution divides credit among the columns the fitted model used. With identical columns a tree "
                    "picks one of them at every split (here the first column), so the other receives exactly zero credit and ranks "
                    "below pure noise. With near-copies the model uses both and the split between them is arbitrary, changing "
                    "from dataset to dataset. In every case the pair is what the model needs: removing only the low-attribution "
                    "copy is nearly free when the copies are tight, removing both is expensive."),
    "application": ("Treat a low mean |SHAP| as a candidate for a removal experiment, not a verdict. Check correlated substitutes "
                    "first, remove the candidate and its substitutes as a group on the same development rows, and keep a "
                    "pure-noise column as the control for what removal costs when a column carries nothing."),
    "assumptions": ("Constructed data. The shap library is not installed, so attribution is computed by exact interventional "
                    "Shapley enumeration on predicted probability (the chapter's tree explainer is the efficient version of the "
                    "same idea, and its output is a margin by default), with a histogram gradient boosting classifier standing in "
                    "for LightGBM. Which of two identical columns receives the credit is decided by column order in this model; "
                    "other libraries break ties differently."),
    "prediction": "With two identical columns, what mean absolute attribution does the low-attribution copy receive, and what does dropping both cost?",
    "prediction_options": ["Half the other copy's share; dropping both costs little",
                           "Zero attribution; dropping both costs about 0.21 log loss",
                           "The same as a noise column; dropping both costs about 0.02 log loss"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "The low copy receives 0.000 attribution, below the 0.020 of a noise column, and dropping both costs 0.206 log loss on fresh rows.",
        "incorrect": "The low copy receives 0.000 attribution, below the 0.020 of a noise column, yet dropping both columns costs 0.206 log loss on fresh rows while dropping the low copy costs 0.000.",
    },
    "check": "Why does the low-attribution copy cost nothing to drop at correlation 1.0, but cost something at correlation 0.5?",
    "answer": ("At 1.0 the copy repeats what the other column already says, so the refitted model loses nothing (0.000). At 0.5 the "
               "two readings carry independent noise, so each one corrects the other, and dropping the low copy costs 0.037 on fresh rows. "
               "In both cases dropping both is the expensive step, so a removal test has to be run on the pair."),
    "provenance": "Constructed example: eight seeded synthetic datasets, a histogram gradient boosting model and exact Shapley enumeration, measured by the chapter activity.",
    "apply": [
        "Rank by mean |SHAP| across out-of-fold models, then list the columns that correlate with each low-ranked column before removing anything.",
        "Run the removal as paired experiments on the same development rows: remove the candidate alone, then the candidate with its substitutes.",
        "Keep a pure-noise column in the experiment so the cost of removing a useless column is measured, not assumed.",
        "Judge the selected subset on a separate assessment, and keep a failed removal in the experiment log.",
    ],
    "honesty": "Constructed data; the sizes of these effects are properties of this generator, not a competition result.",
}

EQUATIONS = [{"tex": r"\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!}\,\bigl[v(S\cup\{j\}) - v(S)\bigr]",
              "alt": "phi j equals the sum over subsets S of the other features of the weight |S| factorial times |F| minus |S| minus 1 factorial over |F| factorial, times v of S with j minus v of S",
              "basis": "The Shapley attribution the activity enumerates exactly; the chapter describes SHAP without a display equation (What SHAP Is Actually Telling You)."}]
NCOLS = 2
HEIGHT = 4.4
REMOVALS = [("drop_noise", "One noise\ncolumn"), ("drop_low_copy", "Low-attribution\ncopy"), ("drop_both", "Both\ncopies")]


def draw(axes, result, parameter):
    ax, ax2 = axes
    lo, hi = result["per_replicate"]["attr_low"], result["per_replicate"]["attr_high"]
    n = len(lo)
    xs = np.arange(n)
    ax.bar(xs - 0.2, hi, width=0.38, color=COLORS["teal"], edgecolor=COLORS["ink"], lw=0.6, label="Higher-ranked copy")
    ax.bar(xs + 0.2, lo, width=0.38, color=COLORS["light"], edgecolor=COLORS["ink"], lw=0.6, label="Lower-ranked copy")
    ax.axhline(result["attr_noise"], color=COLORS["gold"], ls=(0, (4, 3)), lw=1.6,
               label=f"A noise column: {fmt(result['attr_noise'])}")
    ax.set_xticks(xs, [str(i + 1) for i in xs])
    ax.set_ylim(0, 0.4)
    ax.set_xlabel("Constructed dataset (replicate)")
    ax.set_ylabel("Mean |attribution| (probability units)")
    ax.legend(loc="upper right", frameon=False, fontsize=10, ncol=1)

    means = [result[k + "_fresh"] for k, _ in REMOVALS]
    colors = [COLORS["light"], COLORS["gold"], COLORS["terracotta"]]
    pos = np.arange(len(REMOVALS))
    ax2.bar(pos, means, width=0.55, color=colors, edgecolor=COLORS["ink"], lw=0.6)
    for p, (k, _) in zip(pos, REMOVALS):
        dots = result["per_replicate"][k + "_fresh"]
        ax2.scatter([p + (i - (len(dots) - 1) / 2) * 0.05 for i in range(len(dots))], dots, s=14, color=COLORS["ink"], zorder=3,
                    label="One dataset" if p == 0 else None)
    for p, m in zip(pos, means):
        top = max(result["per_replicate"][REMOVALS[p][0] + "_fresh"])
        ax2.text(p, max(m, top) + 0.008, fmt(m), ha="center", va="bottom", fontsize=10)
    ax2.axhline(0, color=COLORS["grey"], lw=0.6)
    ax2.set_xticks(pos, [name for _, name in REMOVALS])
    ax2.set_ylim(-0.03, 0.31)
    ax2.set_ylabel("Log loss added by removal (fresh rows)")
    ax2.set_xlabel("What is removed, then refitted")
    ax2.legend(loc="upper left", frameon=False, fontsize=10)


def delta(x):
    """Signed change, but a value that rounds to zero is shown without a sign."""
    return fmt(0.0) if abs(x) < 0.0005 else signed(x)


def operand(x):
    """A number written as the right-hand side of a subtraction: a negative value goes in parentheses."""
    return fmt(x) if float(fmt(x)) >= 0 else f"({fmt(x)})"


def explain(result, parameter):
    lo, hi, nz = result["attr_low"], result["attr_high"], result["attr_noise"]
    one, both, noise = result["drop_low_copy_fresh"], result["drop_both_fresh"], result["drop_noise_fresh"]
    share = float(fmt(lo)) / (float(fmt(lo)) + float(fmt(hi)))
    column_a = result["low_is_column_a"]
    n = result["replicates"]
    if lo < nz:
        verdict = "The low copy ranks below a pure-noise column, so a screen would call it noise."
    else:
        verdict = "The low copy still ranks above a pure-noise column."
    interpretation = (
        f"At copy correlation {parameter}: the higher-ranked copy has mean |attribution| {fmt(hi)} and the lower-ranked copy {fmt(lo)}, "
        f"against {fmt(nz)} for a noise column. {verdict} Removing the low copy changes fresh-row log loss by {delta(one)}; "
        f"removing both copies changes it by {delta(both)}, and {fmt(both)} - {operand(one)} = {fmt(float(fmt(both)) - float(fmt(one)))} "
        f"is the extra cost of dropping the pair. The lower-ranked copy was column A in {column_a} of {n} datasets.")
    steps = [
        f"Share of the pair's attribution held by the low copy: {fmt(lo)} / ({fmt(lo)} + {fmt(hi)}) = {fmt(share)}.",
        f"Cost of dropping the low copy: {delta(one)} log loss; a noise column: {delta(noise)}.",
        f"Cost of dropping both: {fmt(both)} - {operand(one)} = {fmt(float(fmt(both)) - float(fmt(one)))} beyond the single removal.",
        f"Low copy was column A in {column_a} of {n} datasets" + (" (column order, not information, breaks a tie)." if column_a in (0, n) else "."),
    ]
    metrics = {"Higher copy |attribution|": fmt(hi), "Lower copy |attribution|": fmt(lo), "Noise column |attribution|": fmt(nz),
               "Drop low copy": delta(one), "Drop both copies": delta(both), "Drop a noise column": delta(noise)}
    alt = (f"Left: paired bars of the two copies' mean absolute attribution in eight constructed datasets at copy correlation {parameter}, "
           f"with a dashed line at the noise column level {fmt(nz)}. Right: log loss added on fresh rows by dropping a noise column "
           f"({delta(noise)}), the low-attribution copy ({delta(one)}) and both copies ({delta(both)}).")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for c, res in results.items():
        assert res["drop_both_fresh"] - res["drop_low_copy_fresh"] > 0.08, f"dropping both should be costly at {c}"
        assert abs(res["drop_noise_fresh"]) < 0.01, f"noise-column control should cost about nothing at {c}"
        assert res["attr_noise"] < res["attr_low"] + 0.02 or c == 1.0, f"noise reference off at {c}"
    ordered = [results[c]["drop_low_copy_fresh"] for c in (0.5, 0.9, 0.99)]
    assert ordered[0] > ordered[1] > 0.005 and ordered[1] > ordered[2] - 0.003, "cost of the low copy should fall as the copies tighten"
    exact = results[1.0]
    assert exact["attr_low"] == 0 and exact["attr_noise"] > 0.01, "identical copy should receive zero credit, below noise"
    assert abs(exact["drop_low_copy_fresh"]) < 0.005, "dropping an identical copy should cost nothing"
    assert fmt(exact["attr_low"]) == "0.000" and fmt(exact["attr_noise"]) == "0.020", "prediction feedback attribution numbers"
    assert fmt(exact["drop_both_fresh"]) == "0.206", "prediction feedback numbers"
    assert fmt(exact["drop_low_copy_fresh"]) == "0.000", "prediction feedback numbers"
    assert results[0.5]["drop_low_copy_fresh"] > 0.02 and fmt(results[0.5]["drop_low_copy_fresh"]) == "0.037", "check answer numbers"
    assert 0 < results[0.99]["low_is_column_a"] < results[0.99]["replicates"], "near-copies: the lower copy should vary by dataset"
