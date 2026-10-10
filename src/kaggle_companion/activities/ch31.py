"""Chapter 31: Deep Learning for Tabular Data. A weaker neural model added to a gradient-boosted blend, by training-set size."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from kaggle_companion.activities._common import clean

SEED = 31
REPLICATES = 6          # independent constructed datasets; every estimate is their mean
N_DEV, N_SMALL_DEV, N_NEW = 600, 60, 5000
GRID = np.linspace(0, 1, 21)   # candidate blend weights on the neural model: 0 is the GBM alone, 1 the network alone


def truth(X):
    """Smooth terms (easy for a network) plus two sharp steps and an interaction (easy for trees)."""
    smooth = np.sin(1.5 * X[:, 0]) + 0.6 * X[:, 1] ** 2 + 0.5 * X[:, 2] + 0.4 * np.tanh(2 * X[:, 3])
    steps = 1.2 * ((X[:, 4] > 0.3) & (X[:, 5] > -0.2)) + 0.8 * (X[:, 6] > 0.8)
    return smooth + steps


def generate(n, rng):
    X = rng.normal(size=(n, 8))
    return X, truth(X) + rng.normal(0, 0.5, n)


def mse(pred, y):
    return float(np.mean((pred - y) ** 2))


def best_weight(p_net, p_gbm, y):
    """The weight on the network that minimizes squared error on these rows."""
    return float(GRID[int(np.argmin([mse(w * p_net + (1 - w) * p_gbm, y) for w in GRID]))])


def credible_trees(X, y, X_dev, y_dev):
    """A credible tree baseline: tree size and the number of boosting rounds are chosen on the development rows."""
    best = None
    for leaves in (4, 8):
        trial = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, max_leaf_nodes=leaves, min_samples_leaf=5,
                                              random_state=0).fit(X, y)
        errors = [mse(p, y_dev) for p in trial.staged_predict(X_dev)]     # development error after each boosting round
        rounds = int(np.argmin(errors)) + 1
        if best is None or errors[rounds - 1] < best[0]:
            best = (errors[rounds - 1], leaves, rounds)
    return HistGradientBoostingRegressor(max_iter=best[2], learning_rate=0.08, max_leaf_nodes=best[1], min_samples_leaf=5,
                                         random_state=0).fit(X, y)


def one_replicate(train_rows, seed):
    rng = np.random.default_rng(seed)
    X, y = generate(train_rows, rng)
    X_dev, y_dev = generate(N_DEV, rng)         # development rows: tune the trees and choose the blend weight here
    X_new, y_new = generate(N_NEW, rng)         # untouched rows: score here only
    gbm = credible_trees(X, y, X_dev, y_dev)
    scaler, mean, sd = StandardScaler().fit(X), y.mean(), y.std()      # the network needs scaled inputs and target
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # a small network may stop at the iteration cap; that is part of the trial
        net = MLPRegressor(hidden_layer_sizes=(32,), alpha=0.1, learning_rate_init=0.003, max_iter=300,
                           random_state=seed).fit(scaler.transform(X), (y - mean) / sd)
    predict_net = lambda Z: net.predict(scaler.transform(Z)) * sd + mean
    g_dev, n_dev, g_new, n_new = gbm.predict(X_dev), predict_net(X_dev), gbm.predict(X_new), predict_net(X_new)

    w = best_weight(n_dev, g_dev, y_dev)                                           # weight from 600 development rows
    w_small = best_weight(n_dev[:N_SMALL_DEV], g_dev[:N_SMALL_DEV], y_dev[:N_SMALL_DEV])   # weight from only 60 rows
    blend = lambda weight: weight * n_new + (1 - weight) * g_new
    return {"gbm": mse(g_new, y_new), "net": mse(n_new, y_new), "equal": mse(blend(0.5), y_new),
            "dev_weight": mse(blend(w), y_new), "small_dev_weight": mse(blend(w_small), y_new),
            "weight": w, "small_weight": w_small,
            "residual_corr": float(np.corrcoef(g_new - y_new, n_new - y_new)[0, 1])}


def run(train_rows):
    reps = [one_replicate(train_rows, SEED * 100 + i) for i in range(REPLICATES)]
    mean = lambda key: float(np.mean([r[key] for r in reps]))
    keys = ("gbm", "net", "equal", "dev_weight", "small_dev_weight")
    return clean({
        "train_rows": train_rows,
        **{k: mean(k) for k in keys},
        "weight": mean("weight"), "small_weight": mean("small_weight"),
        "small_weight_range": [min(r["small_weight"] for r in reps), max(r["small_weight"] for r in reps)],
        "residual_corr": mean("residual_corr"),
        "per_replicate_gain": {k: [r["gbm"] - r[k] for r in reps] for k in ("equal", "dev_weight", "small_dev_weight")},
        "replicates": REPLICATES, "development_rows": N_DEV, "small_development_rows": N_SMALL_DEV, "new_rows": N_NEW,
    })
# notebook-end


SPEC = {
    "chapter": 31,
    "chapter_title": "Deep Learning for Tabular Data",
    "subtitle": "Report a neural model's standalone quality and its marginal blend contribution separately.",
    "summary": ("A neural model that loses to a credible boosted-tree baseline on its own can still improve a blend, but how much "
                "depends on the data and on how the weight is chosen. One demonstration measures the standalone and blended error as "
                "the training set grows."),
    "title": "A weaker neural model in a gradient-boosted blend, by training-set size",
    "question": "Does a neural model that is worse on its own lower the blend's error, and does that survive as the tree model gets more data?",
    "why": ("The decision to build a neural model is usually made on its standalone score, and then it loses. Measuring what it adds "
            "to the blend, with the weight chosen on development rows, is the comparison the chapter asks for."),
    "method": ("Six constructed regression datasets with eight standard-normal features. The target has smooth terms, two sharp "
               "steps and an interaction, plus noise of standard deviation 0.5. A histogram gradient boosting regressor, whose tree "
               "size (4 or 8 leaves) and number of boosting rounds are chosen on 600 separate development rows, and a "
               "one-hidden-layer network (32 units, scaled inputs and target, fixed settings) are fitted on the training rows (the control). "
               "The blend weight on the network is chosen on the same 600 development rows, or on only 60 of them, and "
               "everything is scored by mean squared error on 5,000 untouched rows. Equal averaging is also shown."),
    "control": {"key": "train_rows", "label": "Training rows for both models",
                "values": [150, 400, 1000, 3000], "default": 400,
                "value_labels": ["150", "400", "1,000", "3,000"]},
    "source_section": "Adding DL to a GBM Ensemble",
    "symbols": ("p_g and p_n are the boosted-tree and network predictions, w the weight on the network (0 to 1) and "
                "the blend is (1 - w) p_g + w p_n. The weight minimizes squared error on development rows only."),
    "explanation": ("The network is clearly worse than the trees at every size here, but its errors are only partly the same as the trees' "
                    "errors, so a small weight on it (well under one half) beats both. Equal averaging gives the weaker model too "
                    "much say: it never beats the weighted blend, and from 1,000 rows up it is worse than the trees alone. When the training set is small the trees are further "
                    "from the truth and the network's smooth fit repairs more; as the trees get more data their errors shrink and become "
                    "more like the network's, so the benefit shrinks. A weight chosen on a few development rows is noisy and gives some "
                    "or all of the gain back."),
    "application": ("Report standalone quality and the blend gain separately, on the same eligible rows and folds as a credible tree baseline. "
                    "Choose the weight on development rows with nonnegative weights that sum to one, and look at both endpoints "
                    "(each model alone) and equal averaging before keeping the blend."),
    "assumptions": ("Constructed data. A scikit-learn network with one small hidden layer stands in for the chapter's neural models, "
                    "and a histogram gradient boosting regressor for LightGBM; real tabular networks are larger and trained on more rows, "
                    "so the sizes here describe this generator only. The trees are tuned on the development rows and the network is not "
                    "(it has fixed scaling and weight decay), so the comparison is deliberately tough on the network, as the chapter's "
                    "credible-baseline rule asks; a better network would change the blend weights."),
    "prediction": ("With 400 training rows the network's error is about 1.5 times the trees'. Which way of blending the two beats the "
                   "trees alone on new rows?"),
    "prediction_options": ["Neither: a weaker model can only drag a blend down",
                           "Both equal averaging and a weight chosen on 600 development rows",
                           "The development-weighted blend; equal averaging gives the weaker model too much say"],
    "prediction_answer": 2,
    "prediction_feedback": {
        "correct": ("The trees score 0.464 and the network 0.682. Equal averaging scores 0.476, worse than the trees here, while a weight of "
                    "about 0.19 on the network, chosen on 600 development rows, scores 0.446, 0.018 better than the trees."),
        "incorrect": ("The trees score 0.464 and the network 0.682. Equal averaging scores 0.476, worse than the trees here, while a weight of "
                      "about 0.19 on the network, chosen on 600 development rows, scores 0.446, 0.018 better than the trees. The errors "
                      "differ enough to help, but only with a small weight on the weaker model."),
    },
    "check": "How does the blend's advantage over the trees change from 150 to 3,000 training rows, and why?",
    "answer": ("It falls from 0.040 to 0.003 mean squared error (0.649 to 0.609 at 150 rows, 0.296 to 0.293 at 3,000). The trees' "
               "errors shrink with data and become more correlated with the network's (residual correlation 0.63 to 0.81), so "
               "there is less for the network to repair. At 3,000 rows the gain is tiny, and a weight "
               "chosen on 60 rows gives all of it back (0.296)."),
    "provenance": "Constructed example: six seeded synthetic regression datasets, a development-tuned histogram gradient boosting model and a small scikit-learn network, measured by the chapter activity.",
    "apply": [
        "Fit the neural model on the same eligible rows, folds and preprocessing boundary as a credible, tuned tree baseline, and log standalone scores.",
        "Choose blend weights on development rows (nonnegative, summing to one) and score the blend, equal averaging and both endpoints on untouched rows.",
        "Measure the blend gain at your real data size: here it shrank as the trees got more data and the two models' residuals correlated more.",
        "Do not average a weaker model in with equal weight, and do not pick the weight on a few dozen rows: both give part or all of the gain back.",
    ],
    "honesty": "Constructed data; the sizes of these effects are properties of this generator and the small network, not a competition result.",
}

EQUATIONS = [{"tex": r"\hat y = (1-w)\,p_g + w\,p_n, \qquad w^{*} = \arg\min_{w\in[0,1]} \sum_{i \in \text{dev}} \bigl(\hat y_i(w) - y_i\bigr)^2",
              "alt": "the blend prediction is one minus w times the tree prediction plus w times the network prediction, with w chosen to minimize squared error on the development rows",
              "basis": "The development-fitted convex blend the chapter describes in words (Adding DL to a GBM Ensemble); written as an equation by the activity."}]
NCOLS = 2
HEIGHT = 4.4
MODELS = [("gbm", "Trees\nalone"), ("net", "Network\nalone"), ("equal", "Equal\nblend"), ("small_dev_weight", "Weight from\n60 rows"),
          ("dev_weight", "Weight from\n600 rows")]
GAINS = [("equal", "Equal\nblend"), ("small_dev_weight", "Weight from\n60 rows"), ("dev_weight", "Weight from\n600 rows")]


def draw(axes, result, parameter):
    ax, ax2 = axes
    xs = np.arange(len(MODELS))
    values = [result[k] for k, _ in MODELS]
    colors = [COLORS["navy"], COLORS["light"], COLORS["gold"], COLORS["olive"], COLORS["teal"]]
    ax.bar(xs, values, width=0.6, color=colors, edgecolor=COLORS["ink"], lw=0.6)
    for x, v in zip(xs, values):
        ax.text(x, v + 0.01 * max(values), fmt(v), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(xs, [n for _, n in MODELS], fontsize=10)
    ax.set_ylim(0, max(values) * 1.15)
    ax.set_ylabel("Mean squared error on 5,000 new rows")
    ax.set_xlabel(f"Model, {parameter} training rows")

    pos = np.arange(len(GAINS))
    means = [result["gbm"] - result[k] for k, _ in GAINS]
    ax2.bar(pos, means, width=0.55, color=[COLORS["gold"], COLORS["olive"], COLORS["teal"]], edgecolor=COLORS["ink"], lw=0.6)
    for p, (k, _) in zip(pos, GAINS):
        dots = result["per_replicate_gain"][k]
        ax2.scatter([p + (i - (len(dots) - 1) / 2) * 0.05 for i in range(len(dots))], dots, s=14, color=COLORS["ink"], zorder=3)
        ax2.text(p, max(max(dots), result["gbm"] - result[k]) + 0.006, signed(float(fmt(result["gbm"])) - float(fmt(result[k]))), ha="center", va="bottom", fontsize=10)
    ax2.axhline(0, color=COLORS["grey"], lw=0.8)
    ax2.set_xticks(pos, [n for _, n in GAINS], fontsize=10)
    top = max(max(v) for v in result["per_replicate_gain"].values())
    bottom = min(0, min(min(v) for v in result["per_replicate_gain"].values()))
    ax2.set_ylim(bottom - 0.01, top * 1.25 + 0.01)
    ax2.set_ylabel("Squared error removed versus trees alone")
    ax2.set_xlabel("How the blend weight is chosen (dots: one dataset each)")


def explain(result, parameter):
    g, n, e, d, s = (result[k] for k in ("gbm", "net", "equal", "dev_weight", "small_dev_weight"))
    gain = float(fmt(g)) - float(fmt(d))
    gain_small = float(fmt(g)) - float(fmt(s))
    lo, hi = result["small_weight_range"]
    interpretation = (
        f"With {parameter} training rows the trees score {fmt(g)} and the network {fmt(n)}, so the network is "
        f"{'worse' if n > g else 'better'} on its own. A blend with the weight chosen on 600 development rows scores {fmt(d)}, "
        f"and {fmt(g)} - {fmt(d)} = {fmt(gain)} is the error it removes versus the trees. The chosen weight on the network averages "
        f"{fmt(result['weight'], 2)}; equal averaging scores {fmt(e)}. Choosing the weight on 60 rows scores {fmt(s)} with weights "
        f"between {fmt(lo, 2)} and {fmt(hi, 2)} across datasets. The two models' errors correlate at {fmt(result['residual_corr'], 2)}.")
    steps = [
        f"Trees alone: {fmt(g)}; network alone: {fmt(n)} (mean squared error on new rows).",
        f"Blend, weight from 600 rows: {fmt(g)} - {fmt(d)} = {signed(gain)} versus trees.",
        f"Equal blend: {fmt(g)} - {fmt(e)} = {signed(float(fmt(g)) - float(fmt(e)))} versus trees.",
        f"Blend, weight from 60 rows: {fmt(g)} - {fmt(s)} = {signed(gain_small)} versus trees.",
    ]
    metrics = {"Trees alone": fmt(g), "Network alone": fmt(n), "Equal blend": fmt(e),
               "Blend, weight from 600 rows": fmt(d), "Blend, weight from 60 rows": fmt(s),
               "Residual correlation": fmt(result["residual_corr"], 2)}
    alt = (f"Left: bars of mean squared error on new rows for the trees ({fmt(g)}), the network ({fmt(n)}), an equal blend ({fmt(e)}) and "
           f"two development-weighted blends ({fmt(s)} and {fmt(d)}) with {parameter} training rows. Right: error removed versus the trees "
           f"for each blend, with a dot per dataset.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    sizes = (150, 400, 1000, 3000)
    for k, res in results.items():
        assert res["net"] > res["gbm"] + 0.05, f"network should be clearly weaker on its own at {k}"
        assert res["dev_weight"] < res["gbm"] - 0.002, f"development-weighted blend should beat the trees at {k}"
        assert res["dev_weight"] < res["net"] and res["dev_weight"] <= res["equal"], f"weighted blend should beat the network and equal averaging at {k}"
        assert res["small_dev_weight"] >= res["dev_weight"] - 0.003, f"60-row weight should not beat the 600-row weight at {k}"
        assert res["small_weight_range"][1] - res["small_weight_range"][0] > 0.1, f"60-row weights should vary at {k}"
        assert res["weight"] < 0.4, f"chosen weight should be well under one half at {k}"
    for k in (1000, 3000):
        assert results[k]["equal"] > results[k]["gbm"], f"equal averaging should be worse than the trees at {k}"
    gains = [results[k]["gbm"] - results[k]["dev_weight"] for k in sizes]
    assert gains[0] > gains[1] > gains[2] > gains[3] > 0, "blend gain should shrink as training rows grow"
    corr = [results[k]["residual_corr"] for k in sizes]
    assert corr[0] < corr[1] < corr[2] < corr[3], "residual correlation should rise with training rows"
    diff = lambda a, b: fmt(float(fmt(a)) - float(fmt(b)))      # the difference of the two numbers as displayed
    r = results[400]
    assert fmt(r["net"] / r["gbm"], 1) == "1.5", "prediction ratio"
    assert fmt(r["gbm"]) == "0.464" and fmt(r["net"]) == "0.682" and fmt(r["dev_weight"]) == "0.446", "prediction feedback numbers"
    assert fmt(r["equal"]) == "0.476" and fmt(r["weight"], 2) == "0.19", "prediction feedback numbers"
    assert diff(r["gbm"], r["dev_weight"]) == "0.018", "prediction feedback gain"
    assert diff(results[150]["gbm"], results[150]["dev_weight"]) == "0.040", "check answer gain at 150"
    assert fmt(results[150]["gbm"]) == "0.649" and fmt(results[150]["dev_weight"]) == "0.609", "check answer numbers"
    big = results[3000]
    assert diff(big["gbm"], big["dev_weight"]) == "0.003", "check answer gain at 3000"
    assert fmt(big["gbm"]) == "0.296" and fmt(big["dev_weight"]) == "0.293", "check answer numbers"
    assert fmt(big["small_dev_weight"]) == "0.296" and diff(big["gbm"], big["small_dev_weight"]) == "0.000", "check answer: 60-row weight"
    assert fmt(results[150]["residual_corr"], 2) == "0.63" and fmt(big["residual_corr"], 2) == "0.81", "check answer correlations"
