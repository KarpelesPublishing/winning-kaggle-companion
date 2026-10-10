"""Chapter 13: Hyperparameter Tuning with Optuna. Random search on boosted trees: the search's best score against a frozen assessment."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from kaggle_companion.activities._common import clean

SEED = 13
REPLICATES = 12         # independent constructed datasets; every estimate is their mean
N_SEARCH, N_ASSESS = 300, 4000   # rows the search may use, and rows reserved before any search
BASELINE = dict(learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0, max_iter=100)


def generate(n, rng):
    """Ten numeric features: two main effects, an interaction, a bend and a threshold, plus noise."""
    X = rng.normal(size=(n, 10))
    logit = 0.9 * X[:, 0] - 0.7 * X[:, 1] + 0.6 * X[:, 2] * X[:, 3] + 0.5 * np.sin(2 * X[:, 4]) + 0.4 * (X[:, 5] > 0.5)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def sample_config(rng):
    """One random-search trial: learning rate, capacity, leaf support, regularization and rounds."""
    return dict(learning_rate=float(10 ** rng.uniform(-2, -0.3)), max_leaf_nodes=int(rng.integers(4, 32)),
                min_samples_leaf=int(rng.integers(3, 50)), l2_regularization=float(10 ** rng.uniform(-3, 1.5)),
                max_iter=int(rng.integers(10, 40)))


def roc_auc_score(y, score):
    """AUC as the rank (Mann-Whitney) statistic: the chance a random positive outscores a random negative. Same value as
    scikit-learn's roc_auc_score, computed directly because the search calls it hundreds of times."""
    ranks = rankdata(score)
    pos = y == 1
    n_pos, n_neg = pos.sum(), len(y) - pos.sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def model(config):
    return HistGradientBoostingClassifier(max_bins=32, random_state=0, **config)


def cv_auc(config, X, y, folds):
    """Development score of one configuration: mean AUC over the search folds."""
    return float(np.mean([roc_auc_score(y[v], model(config).fit(X[t], y[t]).predict_proba(X[v])[:, 1]) for t, v in folds]))


def assess(config, X, y, X_assess, y_assess):
    """Refit on all search rows and score once on the reserved rows."""
    return roc_auc_score(y_assess, model(config).fit(X, y).predict_proba(X_assess)[:, 1])


def run(trials):
    search_path, assess_path = [], []
    for rep in range(REPLICATES):
        rng = np.random.default_rng(SEED * 100 + rep)
        X, y = generate(N_SEARCH + N_ASSESS, rng)
        Xs, ys, Xa, ya = X[:N_SEARCH], y[:N_SEARCH], X[N_SEARCH:], y[N_SEARCH:]
        folds = list(StratifiedKFold(3, shuffle=True, random_state=0).split(Xs, ys))
        base_cv, base_assess = cv_auc(BASELINE, Xs, ys, folds), assess(BASELINE, Xs, ys, Xa, ya)
        best, gain_search, gain_assess = -1.0, [], []
        for _ in range(trials):
            config = sample_config(rng)
            score = cv_auc(config, Xs, ys, folds)
            if score > best:                       # a new best: freeze it and score it once on the reserved rows
                best, frozen = score, assess(config, Xs, ys, Xa, ya)
            gain_search.append(best - base_cv)      # what the search reports: best score so far minus the baseline's
            gain_assess.append(frozen - base_assess)  # what the frozen winner earns on reserved rows over the baseline
        search_path.append(gain_search)
        assess_path.append(gain_assess)
    s, a = np.array(search_path), np.array(assess_path)
    return clean({
        "trials": trials,
        "search_gain": float(s[:, -1].mean()), "assessment_gain": float(a[:, -1].mean()),
        "share_helped": float((a[:, -1] > 0).mean()),
        "mean_search_path": s.mean(axis=0), "mean_assessment_path": a.mean(axis=0),
        "assessment_path_per_replicate": a, "per_replicate_final": {"search": s[:, -1], "assessment": a[:, -1]},
        "replicates": REPLICATES, "search_rows": N_SEARCH, "assessment_rows": N_ASSESS,
    })
# notebook-end


SPEC = {
    "chapter": 13,
    "chapter_title": "Hyperparameter Tuning with Optuna",
    "subtitle": "A tuned configuration's winning score is a development result until a frozen recipe is assessed elsewhere.",
    "summary": ("A hyperparameter search reports the best score it found. One demonstration runs a random search on boosted trees "
                "and measures that score against what the frozen winner earns on rows reserved before the search."),
    "title": "A random search's best score, against the frozen winner on reserved rows",
    "question": "As the trial budget grows, how much of the search's reported improvement shows up on rows the search never used?",
    "why": ("The chapter says an HPO winner's score is not an independently assessed improvement. The size of the difference, "
            "and how it grows with the budget, tells you how much of a tuning report to believe."),
    "method": ("A constructed binary task: 300 rows for the search and 4,000 rows reserved before it begins. A scikit-learn histogram gradient "
               "boosting model is tuned by random search (learning rate, leaves, leaf support, L2 and rounds) with 3-fold cross-validation, "
               "and the best configuration so far is refitted on the 300 rows and scored once on the reserved rows. Both are reported as "
               "gains over a fixed default configuration. The control is the number of trials; twelve independent datasets are averaged."),
    "control": {"key": "trials", "label": "Search trials",
                "values": [1, 4, 8, 16], "default": 16,
                "value_labels": ["1", "4", "8", "16"]},
    "source_section": "The Optuna Pattern",
    "symbols": ("G_search(T) is the best development score found in T trials minus the default configuration's development score, "
                "G_assess(T) is the frozen winner's AUC on the reserved rows minus the default's, and the optimism is their difference."),
    "explanation": ("Each trial is another noisy draw of a configuration's development score, so the best-so-far score can only rise. "
                    "Part of every rise is the winner's luck on these folds, which does not carry to new rows. The gain on reserved "
                    "rows grows more slowly and flattens, so the optimism tends to widen with the budget. With a single trial nothing is "
                    "selected, and the two gains differ only by noise."),
    "application": ("Report a tuning result as the frozen recipe's gain on reserved rows over the baseline, with the budget and cost. "
                    "Stop searching when the reserved-row gain stops moving, not when the development score stops climbing."),
    "assumptions": ("Constructed data, random search in place of Optuna's TPE sampler, scikit-learn HistGradientBoostingClassifier in place of "
                    "LightGBM, and 3 folds rather than 5; Optuna and LightGBM are not installed here. The chapter notes that random and TPE "
                    "searches both remain valid comparators. A 300-row search set is deliberately small, which makes the development score "
                    "noisy; with more rows the optimism shrinks."),
    "prediction": "After 16 random-search trials, how does the gain the search reports over the default compare with the gain on reserved rows?",
    "prediction_options": ["About the same", "Search gain is much larger, at least 1.5 times as big", "Reserved-row gain is larger"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "The search reports +0.032 AUC over the default; the frozen winner earns +0.008 on reserved rows.",
        "incorrect": "The search reports +0.032 AUC over the default; the frozen winner earns +0.008 on reserved rows. Most of the reported gain is the winner's luck on the search folds.",
    },
    "check": "The development score can only rise with more trials. Why is that not evidence that the tuned model keeps getting better?",
    "answer": ("The score is a running maximum of noisy estimates, so it is non-decreasing by construction. Going from 4 to 16 trials raises it by "
               "0.014 on the search folds but changes the reserved-row gain by less than 0.001; the extra development gain is mostly selection luck, "
               "which is why the chapter asks for a frozen recipe and an untouched comparison."),
    "provenance": "Constructed example: seeded synthetic data, a random search over boosted trees, measured by the chapter activity.",
    "apply": [
        "Reserve the assessment rows before the search, and refit the frozen best configuration on them exactly once.",
        "Report the gain over the untuned baseline on the reserved rows, next to the search's own best score and the elapsed cost.",
        "Treat a flat reserved-row gain as the signal to stop spending trials, even while the development score still rises.",
        "With little data, expect the development score to overstate the gain, and prefer coarse changes over many fine ones.",
    ],
    "honesty": ("Constructed data and a stand-in learner and sampler. The reserved-row gain is small and noisy (it is below zero in some "
                "datasets, see the dots), and the optimism is a property of this 300-row generator, not a constant."),
}

EQUATIONS = [{"tex": r"\mathrm{optimism}(T) = G_{\mathrm{search}}(T) - G_{\mathrm{assess}}(T)",
              "alt": "optimism at T trials equals the search gain at T minus the assessment gain at T",
              "basis": "The activity's own label for the gap between a search's best score and a frozen assessment (The Optuna Pattern); the chapter has no display equation."}]
NCOLS = 2
HEIGHT = 4.4


def draw(axes, result, parameter):
    left, right = axes
    n = result["trials"]
    xs = list(range(1, n + 1))
    for path in result["assessment_path_per_replicate"]:
        left.plot(xs, path, color=COLORS["teal"], lw=0.7, alpha=0.3)
    left.plot(xs, result["mean_search_path"], color=COLORS["terracotta"], lw=2.0, marker="o" if n <= 12 else None, ms=4,
              label="Search: best score so far")
    left.plot(xs, result["mean_assessment_path"], color=COLORS["teal"], lw=2.0, marker="o" if n <= 12 else None, ms=4,
              label="Frozen winner on reserved rows")
    left.axhline(0, color=COLORS["grey"], lw=0.8)
    left.set_xlabel("Trials run (thin lines: reserved-row gain, one dataset each)")
    left.set_ylabel("AUC gain over the default configuration")
    left.set_xlim(0.5, max(n, 2) + 0.5)
    top = max(max(result["mean_search_path"]), max(result["per_replicate_final"]["search"]), 0.03)
    low = min(min(min(p) for p in result["assessment_path_per_replicate"]), -0.01)
    left.set_ylim(low - 0.005, top + 0.045)
    left.legend(loc="upper left", frameon=False, fontsize=10)

    values = [result["search_gain"], result["assessment_gain"]]
    right.bar([0, 1], values, width=0.5, color=[COLORS["terracotta"], COLORS["teal"]], edgecolor=COLORS["ink"], lw=0.6)
    for x, key in enumerate(["search", "assessment"]):
        dots = result["per_replicate_final"][key]
        right.scatter([x + (i - (len(dots) - 1) / 2) * 0.05 for i in range(len(dots))], dots, s=12, color=COLORS["ink"], zorder=3)
        right.text(x, max([values[x]] + dots) + 0.004, fmt(values[x]), ha="center", va="bottom", fontsize=10)
    right.axhline(0, color=COLORS["grey"], lw=0.8)
    right.set_xticks([0, 1], ["Search\nreports", "Reserved\nrows"])
    right.set_ylim(low - 0.005, top + 0.045)
    right.set_ylabel("AUC gain over the default configuration")
    right.set_xlabel(f"After {n} trials (dots: one dataset each)")


def diff(a, b):
    """Difference of the displayed (3 decimal) values, so the hand calculation in the text adds up."""
    return fmt(round(a, 3) - round(b, 3))


def explain(result, parameter):
    s, a = result["search_gain"], result["assessment_gain"]
    a_txt = fmt(a) if round(a, 3) >= 0 else f"({fmt(a)})"     # a negative subtrahend is written in parentheses
    calc = f"{fmt(s)} - {a_txt} = {diff(s, a)}"
    if parameter == 1:
        verdict = (f"so {calc}. With one trial nothing is selected, so this difference is estimation noise, "
                   f"not the winner's luck")
    elif round(s, 3) - round(a, 3) >= 0.003:
        verdict = f"so {calc} of the reported gain is optimism"
    else:
        verdict = f"a difference of {calc}, within noise at this budget"
    interpretation = (
        f"After {parameter} trial{'s' if parameter != 1 else ''} the search reports a gain of {signed(s)} AUC over the default; the frozen "
        f"winner earns {signed(a)} on {result['assessment_rows']:,} reserved rows, {verdict}. "
        f"The winner beat the default on reserved rows in {round(100 * result['share_helped'])}% of the {result['replicates']} datasets.")
    steps = [
        f"Optimism: {calc} AUC (search gain minus reserved-row gain).",
        f"Share of the reported gain that carried over: {a_txt} / {fmt(s)} = {fmt(a / s)}."
        if s > 0.003 else "The search gain is too small to express a share.",
        f"Datasets where the tuned winner beat the default on reserved rows: {round(100 * result['share_helped'])}%.",
    ]
    metrics = {"Trials": str(parameter), "Search-reported gain": signed(s), "Gain on reserved rows": signed(a),
               "Optimism": diff(s, a), "Datasets helped": f"{round(100 * result['share_helped'])}%"}
    alt = (f"Left: mean AUC gain over the default against trials run, for the search's best score ({fmt(s)} at the end) and the frozen "
           f"winner on reserved rows ({fmt(a)}), with thin lines per dataset. Right: the two final gains with a dot per dataset.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    order = sorted(results)
    for a, b in zip(order, order[1:]):
        assert results[b]["search_gain"] >= results[a]["search_gain"] - 1e-9, "running best must not fall"
    big = results[16]
    assert big["search_gain"] - big["assessment_gain"] > 0.005, "optimism should be visible at 16 trials"
    assert big["search_gain"] > 1.5 * big["assessment_gain"], "prediction option says at least 1.5 times"
    opt = {t: results[t]["search_gain"] - results[t]["assessment_gain"] for t in order}
    assert opt[16] > opt[4] > opt[1], "optimism should widen with the budget"
    gain_search = results[16]["search_gain"] - results[4]["search_gain"]
    gain_assess = results[16]["assessment_gain"] - results[4]["assessment_gain"]
    assert gain_search > 0.002 and gain_search > gain_assess + 0.002, "extra trials add development gain, not assessed gain"
    assert fmt(big["search_gain"]) == "0.032" and fmt(big["assessment_gain"]) == "0.008", "prediction feedback numbers"
    assert fmt(gain_search) == "0.014" and abs(gain_assess) < 0.001, "check answer numbers"
    assert big["search_gain"] > 2 * big["assessment_gain"], "feedback says most of the reported gain is luck"
