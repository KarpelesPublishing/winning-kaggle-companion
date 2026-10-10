"""Chapter 37: Stacking Deep. A meta-model scored on its own rows, by global OOF CV and by nested assessment, as the library grows."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from kaggle_companion.activities._common import clean

SEED = 37
REPLICATES = 5          # independent worlds per library size; results are means over them
N_DEV, N_FRESH, FEATURES = 250, 1000, 10
RIDGE_ALPHA = 50.0      # the "simple regularized meta model" the chapter prefers


def generate(n, rng):
    """A bent, interacting signal that no single base model captures, plus noise with variance 1."""
    X = rng.normal(size=(n, FEATURES))
    signal = 1.2 * np.sin(1.5 * X[:, 0]) + 0.8 * X[:, 1] * X[:, 2] + 0.6 * X[:, 3] ** 2 + 0.5 * X[:, 4] - 0.4 * X[:, 5]
    return X, signal + rng.normal(0, 1.0, n)


def make_library(size, rng):
    """`size` base models: ridge, kNN and shallow trees, each seeing a random subset of 3 to 6 features."""
    return [(i % 3, np.sort(rng.choice(FEATURES, rng.integers(3, 7), replace=False))) for i in range(size)]


def base_predictions(library, X_fit, y_fit, X_new):
    """Fit every base model on (X_fit, y_fit) and return their predictions for X_new as columns."""
    out = np.empty((len(X_new), len(library)))
    for j, (kind, cols) in enumerate(library):
        model = (Ridge(alpha=1.0) if kind == 0 else KNeighborsRegressor(15) if kind == 1
                 else DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=j))
        out[:, j] = model.fit(X_fit[:, cols], y_fit).predict(X_new[:, cols])
    return out


def oof_matrix(library, X, y, folds):
    matrix = np.zeros((len(y), len(library)))
    for fit, out in folds:
        matrix[out] = base_predictions(library, X[fit], y[fit], X[out])
    return matrix


def mse(pred, y):
    return float(np.mean((pred - y) ** 2))


def one_world(size, seed):
    rng = np.random.default_rng(seed)
    X, y = generate(N_DEV, rng)
    X_fresh, y_fresh = generate(N_FRESH, rng)               # new rows: the reference no estimate may touch
    library = make_library(size, rng)
    outer = list(KFold(5, shuffle=True, random_state=1).split(X))

    # Global scheme: one OOF matrix for the whole development set, then meta-CV on that matrix.
    G = oof_matrix(library, X, y, outer)
    global_cv = np.zeros(N_DEV)
    for fit, out in KFold(5, shuffle=True, random_state=2).split(G):
        global_cv[out] = LinearRegression().fit(G[fit], y[fit]).predict(G[out])

    # Nested scheme: rebuild the base stage inside each outer training partition.
    nested = {k: np.zeros(N_DEV) for k in ("ols", "ridge", "average", "best")}
    own = {k: [] for k in nested}                           # each method scored on the rows it was fitted or selected on
    fresh = {"ols": [], "ridge": []}
    for fit, out in outer:
        inner = list(KFold(3, shuffle=True, random_state=3).split(fit))
        G_in = oof_matrix(library, X[fit], y[fit], inner)    # inner OOF base predictions for outer training
        both = base_predictions(library, X[fit], y[fit], np.vstack([X[out], X_fresh]))
        G_out, G_new = both[:len(out)], both[len(out):]
        best = int(np.argmin(((G_in - y[fit][:, None]) ** 2).mean(axis=0)))
        metas = {"ols": LinearRegression().fit(G_in, y[fit]), "ridge": Ridge(alpha=RIDGE_ALPHA).fit(G_in, y[fit])}
        for k, meta in metas.items():
            nested[k][out] = meta.predict(G_out)
            own[k].append(mse(meta.predict(G_in), y[fit]))
            fresh[k].append(mse(meta.predict(G_new), y_fresh))
        nested["average"][out], own["average"] = G_out.mean(axis=1), own["average"] + [mse(G_in.mean(axis=1), y[fit])]
        nested["best"][out], own["best"] = G_out[:, best], own["best"] + [mse(G_in[:, best], y[fit])]
    return {"global_cv": mse(global_cv, y), "nested": {k: mse(v, y) for k, v in nested.items()},
            "own": {k: float(np.mean(v)) for k, v in own.items()}, "fresh": {k: float(np.mean(v)) for k, v in fresh.items()}}


def run(library_size):
    worlds = [one_world(library_size, SEED * 100 + r) for r in range(REPLICATES)]

    def avg(get):
        return float(np.mean([get(w) for w in worlds]))
    return clean({
        "library_size": library_size,
        "ols": {"own": avg(lambda w: w["own"]["ols"]), "global_cv": avg(lambda w: w["global_cv"]),
                "nested": avg(lambda w: w["nested"]["ols"]), "fresh": avg(lambda w: w["fresh"]["ols"])},
        "methods": {k: {"own": avg(lambda w, k=k: w["own"][k]), "nested": avg(lambda w, k=k: w["nested"][k])}
                    for k in ("best", "average", "ols", "ridge")},
        "ridge_fresh": avg(lambda w: w["fresh"]["ridge"]),
        "ridge_beats_ols_nested": float(np.mean([w["nested"]["ridge"] < w["nested"]["ols"] for w in worlds])),
        "noise_variance": 1.0, "replicates": REPLICATES, "development_rows": N_DEV, "fresh_rows": N_FRESH,
    })
# notebook-end


SPEC = {
    "chapter": 37,
    "chapter_title": "Stacking Deep",
    "subtitle": "A stack's score on its own OOF rows is meta-training, not assessment of the whole stack.",
    "summary": ("A meta-model fitted on out-of-fold predictions is a fitted model with its own degrees of freedom. One demonstration "
                "grows the library of base models and compares the meta-model's score on its own rows, a meta-CV on one global OOF "
                "matrix and a fully nested assessment, against the equal average, the strongest single model and a regularized meta-model."),
    "title": "Scoring a stack on its own rows, by library size",
    "question": "How far does a meta-model's score on its own OOF rows drift from an honest assessment as the library of base models grows, and when does a regularized meta-model overtake the unregularized one?",
    "why": ("Every extra base model is another column for the meta-model to fit. The score on the rows it was fitted to keeps "
            "improving, so it cannot say when to stop; the chapter asks for nested assessment and baselines to decide that."),
    "method": ("A constructed regression task with 250 development rows, 10 features, a bent signal and noise of variance 1. The library "
               "is the control: that many ridge, nearest-neighbor and shallow-tree models on random feature subsets. Five outer folds; "
               "inside each, 3-fold inner OOF predictions build the meta training matrix, the base models refit on the outer training "
               "rows predict the outer fold, and the meta-model (ordinary least squares, or ridge with alpha 50) predicts from those. "
               "A global scheme instead builds one OOF matrix and cross-validates the meta-model on it. Fresh-row scores come from "
               "1,000 new rows. Results are means over 5 independent worlds."),
    "control": {"key": "library_size", "label": "Number of base models in the library",
                "values": [2, 10, 30, 50], "default": 30,
                "value_labels": ["2", "10", "30", "50"]},
    "source_section": "Distinguish Meta Fitting from Outer Assessment",
    "symbols": ("MSE_own is the mean squared error of a meta-model on the OOF rows it was fitted to, MSE_nested the mean squared "
                "error of the same procedure on outer-assessment rows it never saw, and optimism their difference."),
    "explanation": ("An ordinary least squares meta-model gets one free weight per base model. With 50 correlated columns and 200 "
                    "fitting rows it can fit noise, so its own-row score keeps falling while the nested score stops improving and "
                    "then worsens. Meta-CV on one global OOF matrix lands near the nested and fresh-row scores in this generator on "
                    "average, though the chapter warns it can depend on assessment labels indirectly; the own-row score is far off. A ridge meta-model spends its freedom more slowly and keeps improving with the library."),
    "application": ("Compare any stack with the strongest model and the equal average under the same nested assessment, and prefer "
                    "a regularized meta-model once the library is large relative to the rows it is fitted on."),
    "assumptions": ("Constructed data with simple scikit-learn base models, not GBM libraries; one fixed ridge strength (alpha 50) "
                    "and one data size. The library size where ridge overtakes least squares depends on the rows and on how correlated the "
                    "base models are. The nested estimate is slightly pessimistic because its meta-model trains on 80% of the rows."),
    "prediction": ("With 50 base models and an ordinary least squares meta-model, the stack scores 1.27 (mean squared error) on the OOF rows it "
                   "was fitted to. What does nested assessment give?"),
    "prediction_options": ["About 1.3, the same", "About 1.8, somewhat worse", "About 2.4, nearly twice as bad"],
    "prediction_answer": 2,
    "prediction_feedback": {
        "correct": "Nested assessment gives 2.380: optimism of 1.108, almost as large as the whole score.",
        "incorrect": "Nested assessment gives 2.380, not 1.3 or 1.8: optimism of 1.108, almost as large as the whole score.",
    },
    "check": "From 30 to 50 base models the least squares stack's own-row score improves while its nested score worsens. What does the regularized meta-model do, and which score would have told you?",
    "answer": ("Least squares moves from 1.663 to 1.272 on its own rows but from 2.184 to 2.380 nested, because the extra "
               "columns are mostly noise to fit. Ridge improves from 2.133 to 2.006 nested over the same range. Only the nested "
               "score shows the reversal; the own-row score would have recommended the largest library with the freest meta-model."),
    "provenance": "Constructed example: seeded synthetic regression data and scikit-learn base models, measured by the chapter activity.",
    "apply": [
        "Report a stack with nested assessment (base stage rebuilt inside each outer partition), never with its score on its own OOF rows.",
        "Compare against the strongest single model and the equal average under the same outer rows; a stack must beat both to be worth its extra stage.",
        "Limit the meta-model's freedom: ridge, nonnegative weights or a smaller library once the library is large relative to the rows it is fitted on.",
        "Archive the recipe and row IDs, and keep the negative result if the stack does not clear the baselines.",
    ],
    "honesty": ("Constructed data; the library size where regularization wins is a property of this generator. At 50 models the "
                "unregularized stack still beats the strongest single model on average, but by far less than its own-row score suggests."),
}

EQUATIONS = [{"tex": r"\mathrm{optimism}=\mathrm{MSE}_{\mathrm{nested}}-\mathrm{MSE}_{\mathrm{own}}",
              "alt": "Optimism equals the nested mean squared error minus the mean squared error on the meta-model's own rows",
              "basis": "The activity's own measure of the Chapter 37 warning (Distinguish Meta Fitting from Outer Assessment); the chapter has no display equation for it."}]
NCOLS = 2
HEIGHT = 4.4
ESTIMATES = [("own", "Own OOF rows"), ("global_cv", "Meta-CV, one\nglobal matrix"), ("nested", "Nested")]
METHODS = [("best", "Strongest\nmodel"), ("average", "Equal\naverage"), ("ols", "Least squares\nstack"), ("ridge", "Ridge\nstack")]


def draw(axes, result, parameter):
    left, right = axes
    xs = list(range(len(ESTIMATES)))
    values = [result["ols"][k] for k, _ in ESTIMATES]
    left.bar(xs, values, color=[COLORS["light"], COLORS["navy"], COLORS["teal"]], edgecolor=COLORS["ink"], lw=0.6, width=0.6)
    for x, v, c in zip(xs, values, [COLORS["ink"], "white", "white"]):
        left.text(x, v - 0.06, fmt(v, 2), ha="center", va="top", fontsize=10, color=c)
    left.axhline(result["ols"]["fresh"], color=COLORS["terracotta"], ls=(0, (4, 3)), lw=1.4,
                 label=f"Fresh rows: {fmt(result['ols']['fresh'], 2)}")
    left.axhline(result["noise_variance"], color=COLORS["grey"], ls=":", lw=1.2, label="Noise floor: 1.00")
    left.set_xticks(xs, [name for _, name in ESTIMATES], fontsize=10)
    left.set_ylim(0, 3.9)
    left.set_ylabel("Mean squared error (lower is better)")
    left.set_xlabel(f"How the least squares stack is scored, {parameter} base models")
    left.legend(loc="upper right", frameon=False, fontsize=10)

    mx = list(range(len(METHODS)))
    own = [result["methods"][k]["own"] for k, _ in METHODS]
    nested = [result["methods"][k]["nested"] for k, _ in METHODS]
    right.bar([x - 0.2 for x in mx], own, width=0.4, color=COLORS["light"], edgecolor=COLORS["ink"], lw=0.6, label="On the rows it was fitted or chosen on")
    right.bar([x + 0.2 for x in mx], nested, width=0.4, color=COLORS["navy"], edgecolor=COLORS["ink"], lw=0.6, label="Nested assessment")
    for x, a, b in zip(mx, own, nested):
        right.text(x - 0.2, a + 0.04, fmt(a, 2), ha="center", va="bottom", fontsize=10)
        right.text(x + 0.2, b + 0.04, fmt(b, 2), ha="center", va="bottom", fontsize=10)
    right.set_xticks(mx, [name for _, name in METHODS], fontsize=10)
    right.set_ylim(0, 4.6)
    right.set_ylabel("Mean squared error (lower is better)")
    right.set_xlabel("Method")
    right.legend(loc="upper right", frameon=False, fontsize=10)


def explain(result, parameter):
    o, m = result["ols"], result["methods"]
    optimism = o["nested"] - o["own"]
    gain_nested = m["best"]["nested"] - m["ols"]["nested"]
    gain_own = m["best"]["own"] - m["ols"]["own"]
    ridge_edge = m["ols"]["nested"] - m["ridge"]["nested"]
    interpretation = (
        f"With {parameter} base models the least squares stack scores {fmt(o['own'], 2)} on its own OOF rows but "
        f"{fmt(o['nested'], 2)} under nested assessment ({fmt(o['nested'], 2)} - {fmt(o['own'], 2)} = {fmt(optimism, 2)} of optimism); "
        f"the global OOF matrix gives {fmt(o['global_cv'], 2)} and fresh rows {fmt(o['fresh'], 2)}. Against the strongest single "
        f"model the stack gains {fmt(gain_own, 2)} on its own rows but {signed(gain_nested, 2)} nested "
        f"({fmt(m['best']['nested'], 2)} - {fmt(m['ols']['nested'], 2)} = {fmt(gain_nested, 2)}). The equal average scores {fmt(m['average']['nested'], 2)} "
        f"and the ridge stack {fmt(m['ridge']['nested'], 2)}, which is {signed(-ridge_edge, 2)} relative to least squares "
        f"(a negative sign means ridge is better; it was better in {round(100 * result['ridge_beats_ols_nested'])}% of {result['replicates']} worlds).")
    steps = [
        f"Optimism of the least squares stack: {fmt(o['nested'], 2)} - {fmt(o['own'], 2)} = {fmt(optimism, 2)}.",
        f"Honest gain over the strongest single model: {fmt(m['best']['nested'], 2)} - {fmt(m['ols']['nested'], 2)} = {fmt(gain_nested, 2)}.",
        f"Gain the own-row scores would claim: {fmt(m['best']['own'], 2)} - {fmt(m['ols']['own'], 2)} = {fmt(gain_own, 2)}.",
        f"Nested ridge against nested least squares: {fmt(m['ridge']['nested'], 2)} - {fmt(m['ols']['nested'], 2)} = {signed(-ridge_edge, 2)}.",
    ]
    metrics = {"Own OOF rows": fmt(o["own"], 2), "Global-matrix meta-CV": fmt(o["global_cv"], 2), "Nested": fmt(o["nested"], 2),
               "Fresh rows": fmt(o["fresh"], 2), "Equal average, nested": fmt(m["average"]["nested"], 2),
               "Ridge stack, nested": fmt(m["ridge"]["nested"], 2)}
    alt = (f"Left: mean squared error of the least squares stack scored on its own rows ({fmt(o['own'], 2)}), by meta-CV on one global OOF matrix "
           f"({fmt(o['global_cv'], 2)}) and nested ({fmt(o['nested'], 2)}), with {parameter} base models; fresh rows give {fmt(o['fresh'], 2)}. "
           f"Right: own-row and nested errors for the strongest model, equal average, least squares stack and ridge stack.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    own = {k: r["ols"]["own"] for k, r in results.items()}
    nested = {k: r["ols"]["nested"] for k, r in results.items()}
    assert own[2] > own[10] > own[30] > own[50], "own-row score keeps improving with library size"
    assert nested[50] > nested[30], "least squares nested score worsens from 30 to 50"
    optimism = {k: nested[k] - own[k] for k in results}
    assert optimism[2] < optimism[10] < optimism[30] < optimism[50], "optimism grows with the library"
    assert optimism[50] > 1.0 and optimism[2] < 0.1
    for k, r in results.items():
        assert abs(r["ols"]["global_cv"] - r["ols"]["nested"]) < 0.2, f"global OOF matrix is close to nested at {k}"
        assert abs(r["ols"]["nested"] - r["ols"]["fresh"]) < 0.15, f"nested tracks fresh rows at {k}"
        assert r["ols"]["own"] < r["ols"]["fresh"] + 0.1 or k == 2, "own rows are optimistic"
        m = r["methods"]
        assert m["ols"]["own"] <= m["ridge"]["own"] + 1e-9, f"least squares always wins on its own rows at {k}"
    assert results[10]["methods"]["ols"]["nested"] < results[10]["methods"]["best"]["nested"] - 0.3, "stack helps at 10"
    assert results[30]["methods"]["ols"]["nested"] < results[30]["methods"]["best"]["nested"] - 0.3, "stack helps at 30"
    g50 = results[50]["methods"]["best"]["nested"] - results[50]["methods"]["ols"]["nested"]
    assert 0 < g50 < 0.5 * (results[50]["methods"]["best"]["own"] - results[50]["methods"]["ols"]["own"]), "honest gain far below own-row gain at 50"
    assert results[50]["methods"]["ridge"]["nested"] < results[50]["methods"]["ols"]["nested"] - 0.25 and results[50]["ridge_beats_ols_nested"] == 1
    assert results[50]["methods"]["ridge"]["nested"] < results[30]["methods"]["ridge"]["nested"], "ridge keeps improving 30 to 50"
    s = results[50]["ols"]
    assert fmt(s["own"], 2) == "1.27" and fmt(s["nested"], 3) == "2.380" and fmt(s["nested"] - s["own"], 3) == "1.108", "prediction feedback"
    assert fmt(results[30]["ols"]["own"], 3) == "1.663" and fmt(results[30]["ols"]["nested"], 3) == "2.184"
    assert fmt(results[50]["ols"]["own"], 3) == "1.272"
    assert fmt(results[30]["methods"]["ridge"]["nested"], 3) == "2.133" and fmt(results[50]["methods"]["ridge"]["nested"], 3) == "2.006"
    assert abs(results[2]["methods"]["ols"]["nested"] - results[2]["methods"]["average"]["nested"]) < 0.25, "two models: stack about equals average"
