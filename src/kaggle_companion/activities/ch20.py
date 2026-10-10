"""Chapter 20: Horse Health: Small Data, Big Discipline. Per-class factors tuned on OOF probabilities, scored on the same rows and on new rows."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from kaggle_companion.activities._common import clean

SEED = 20
REPLICATES = 14      # independent constructed datasets per development size
N_NEW = 4000         # fresh rows from the same generator, never used for fitting or tuning
GRID = np.exp(np.linspace(np.log(0.5), np.log(2.0), 13))
FACTORS = np.array([(1.0, a, b) for a in GRID for b in GRID])   # divide class 0, 1, 2 probabilities by these, then take argmax


def generate(n, rng, weights, bias):
    """Three-class task with moderate imbalance (about 45% / 33% / 22%) and a few nonlinear terms."""
    X = rng.normal(size=(n, 8))
    z = X @ weights + bias + 0.5 * np.column_stack([np.sin(2 * X[:, 0]), X[:, 1] * X[:, 2], np.zeros(n)])
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return X, np.minimum((rng.random(n)[:, None] > p.cumsum(1)).sum(1), 2)


def model():
    return HistGradientBoostingClassifier(learning_rate=0.1, max_leaf_nodes=6, max_iter=40, early_stopping=False, random_state=0)


def tune(proba, y):
    """Grid-search the factors that maximize accuracy on these rows; return them and that accuracy."""
    accuracy = (np.argmax(proba[:, None, :] / FACTORS[None], 2) == y[:, None]).mean(0)
    best = int(np.argmax(accuracy))
    return FACTORS[best], accuracy[best]


def gain(proba, y, factors):
    """Accuracy of factor-adjusted argmax minus accuracy of plain argmax, on these rows."""
    return float((np.argmax(proba / factors, 1) == y).mean() - (proba.argmax(1) == y).mean())


def one_dataset(n, seed):
    rng = np.random.default_rng(seed)
    weights, bias = rng.normal(0, 0.55, (8, 3)), np.array([0.5, 0.0, -0.5])
    X, y = generate(n, rng, weights, bias)
    X_new, y_new = generate(N_NEW, rng, weights, bias)
    oof = np.zeros((n, 3))
    for fit, out in StratifiedKFold(5, shuffle=True, random_state=1).split(X, y):
        oof[out] = model().fit(X[fit], y[fit]).predict_proba(X[out])
    new = model().fit(X, y).predict_proba(X_new)         # the deployed model, refitted on every development row
    prior = np.bincount(y, minlength=3) / n
    half = rng.permutation(n)
    halves = (half[: n // 2], half[n // 2:])
    out = {}
    # "plain": the model's own probabilities. "rebalanced": divided by training class frequency and renormalized,
    # which distorts them toward the rare classes in the way class-weighted training does.
    for name, (p_oof, p_new) in {"plain": (oof, new), "rebalanced": (oof / prior, new / prior)}.items():
        p_oof, p_new = p_oof / p_oof.sum(1, keepdims=True), p_new / p_new.sum(1, keepdims=True)
        factors, tuned_accuracy = tune(p_oof, y)
        # Same rows: tuned on the OOF rows and scored on those same rows. Cross-half: tuned on one half, scored on the other.
        same_rows = tuned_accuracy - (p_oof.argmax(1) == y).mean()
        cross = np.mean([gain(p_oof[b], y[b], tune(p_oof[a], y[a])[0]) for a, b in (halves, halves[::-1])])
        out[name] = (same_rows, cross, gain(p_new, y_new, factors), (p_oof.argmax(1) == y).mean())
    return out


def run(development_rows):
    sets = [one_dataset(development_rows, SEED * 1000 + i) for i in range(REPLICATES)]
    models = {}
    for name in ("plain", "rebalanced"):
        v = np.array([s[name] for s in sets])
        models[name] = {"same_rows": float(v[:, 0].mean()), "cross_half": float(v[:, 1].mean()), "new_rows": float(v[:, 2].mean()),
                        "new_rows_se": float(v[:, 2].std() / np.sqrt(REPLICATES)), "argmax_accuracy": float(v[:, 3].mean())}
    return clean({"development_rows": development_rows, "new_rows": N_NEW, "replicates": REPLICATES, "models": models})
# notebook-end


def points(x):
    """Accuracy difference shown in percentage points."""
    return fmt(100 * x, 2)


def spoints(x):
    return signed(100 * x, 2)


def par(x):
    """Operand text for a hand calculation: negative numbers in parentheses."""
    return f"({points(x)})" if round(100 * x, 2) < 0 else points(x)


SPEC = {
    "chapter": 20,
    "chapter_title": "Horse Health: Small Data, Big Discipline",
    "subtitle": "Decision-rule factors fitted on OOF predictions need rows they were not fitted on.",
    "summary": ("Per-class factors tuned on out-of-fold probabilities raise the accuracy of the same rows. One demonstration measures how much of "
                "that gain survives on new rows, as the number of development rows grows and for two kinds of probabilities."),
    "title": "Class factors tuned on OOF probabilities, scored on the same rows and on new rows",
    "question": "How much of the accuracy a tuned class factor adds on its own OOF rows survives on new rows?",
    "why": ("With 3 classes and micro-F1 (which equals accuracy for single-label prediction), a per-class factor looks like free accuracy on the "
            "rows that produced it. The chapter says an OOF base matrix does not make the fitted factors' same-row score independent; this measures by how much."),
    "method": ("Fourteen constructed three-class datasets per development size (about 45%, 33% and 22% of rows), eight numeric features. "
               "A small boosted-tree classifier gives 5-fold OOF probabilities. A grid of 169 factor triples is searched for the highest OOF accuracy "
               "(divide each class probability by its factor, take argmax). Same-row gain is that accuracy minus plain argmax accuracy on the OOF rows. "
               "Cross-half gain tunes on one half of the OOF rows and scores the other half. New-row gain applies the tuned factors to the model refitted on all "
               "development rows and scores 4,000 fresh rows. Two kinds of probabilities are compared: the model's own, and the same probabilities "
               "divided by training class frequency and renormalized, which distorts them toward the rare classes as class-weighted training does."),
    "control": {"key": "development_rows", "label": "Development rows",
                "values": [250, 500, 988, 3000], "default": 988,
                "value_labels": ["250", "500", "988: the chapter's development size", "3,000"]},
    "source_section": "Per-Class Threshold Optimization on OOF",
    "symbols": ("p_c is the predicted probability of class c, f_c the factor for class c (f_0 = 1) and the predicted class is the c "
                "that maximizes p_c / f_c. Gain is accuracy with factors minus accuracy of plain argmax."),
    "explanation": ("Plain argmax is already the accuracy-optimal rule when the probabilities are not distorted, so no factor can help in "
                    "expectation and the search only fits noise: it gains on its own rows and loses a little, or nothing, on new ones. When the probabilities are "
                    "distorted, a factor can undo the distortion and the gain is real once there are enough rows to find it (from about 500 here). The same-row gain "
                    "is positive in both cases, so it cannot tell them apart. From 500 rows up the cross-half and new-row gains both separate them; at 250 rows neither does reliably."),
    "application": ("Compare plain argmax first, tune factors only inside an outer assessment or on untouched rows, and keep them only if the "
                    "assessed gain is positive."),
    "assumptions": ("Constructed data and one boosted-tree stand-in for the chapter's gradient boosting libraries. The rebalanced probabilities are a "
                    "constructed distortion, not a recipe. Fourteen datasets per size give new-row gains with standard errors of at most about 0.4 points, "
                    "so smaller differences, including the rebalanced gain at 250 rows, are not resolved."),
    "prediction": "With 988 development rows and the model's own probabilities, factors tuned on the OOF rows raise accuracy on those rows. What do they do to accuracy on new rows?",
    "prediction_options": ["Keep most of the gain", "Lose a little accuracy", "Gain more than on the OOF rows"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "They add 1.03 points on their own rows and change new-row accuracy by -0.52 points.",
        "incorrect": "They add 1.03 points on their own rows and change new-row accuracy by -0.52 points: plain argmax was already the better rule.",
    },
    "check": "Why do the same factors help the rebalanced probabilities at 988 rows but not the model's own?",
    "answer": ("A factor can only repair a distortion. Rebalancing moves the decision boundary away from the accuracy-optimal one, and "
               "tuned factors move it back (+0.72 points on new rows at 988 rows). The model's own probabilities leave nothing to repair, so the same "
               "search only fits noise (-0.52 points). Only rows the factors were not tuned on distinguish the two cases."),
    "provenance": "Constructed example: seeded synthetic three-class datasets and a boosted-tree classifier, measured by the chapter activity.",
    "apply": [
        "Score plain argmax first; treat any tuned factors as a candidate decision rule that must beat it on rows it was not tuned on.",
        "Tune factors inside an outer fold, or on half of the OOF rows and score the other half, or on an untouched holdout; never report the same-row gain.",
        "With few rows, expect the same-row gain to be largest and the least trustworthy; with many rows it shrinks toward the real gain.",
        "If your model's probabilities were rebalanced (class weights, resampling), factors are more likely to pay; re-check them whenever the model changes.",
    ],
    "honesty": ("Constructed data. Whether factors pay depends on the probabilities, the metric and the number of rows; the contrast, not the "
                "size, is the lesson."),
}

EQUATIONS = [{"tex": r"\hat{y} = \arg\max_{c}\; \frac{p_c}{f_c}, \qquad f_0 = 1",
              "alt": "the predicted class is the class c that maximizes the predicted probability p c divided by its factor f c, with the factor for class zero fixed at one",
              "basis": "The chapter's per-class factor rule in prose (Per-Class Threshold Optimization on OOF): divide each class probability by a positive factor and take argmax."}]
NCOLS = 2
HEIGHT = 4.4
KINDS = [("plain", "The model's own probabilities"), ("rebalanced", "Rebalanced probabilities")]
BARS = [("same_rows", "Same rows", COLORS["terracotta"]), ("cross_half", "Cross-half", COLORS["gold"]), ("new_rows", "New rows", COLORS["teal"])]


def draw(axes, result, parameter):
    top = max(abs(result["models"][k][b]) for k, _ in KINDS for b, _, _ in BARS) * 100
    for ax, (kind, title) in zip(axes, KINDS):
        m = result["models"][kind]
        for x, (key, label, color) in enumerate(BARS):
            v = 100 * m[key]
            yerr = 100 * m["new_rows_se"] if key == "new_rows" else None
            ax.bar(x, v, width=0.6, color=color, edgecolor=COLORS["ink"], lw=0.6, yerr=yerr, capsize=4, error_kw={"lw": 1.2})
            offset = (yerr or 0) + top * 0.04
            ax.text(x, v + offset if v >= 0 else v - offset, f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
        ax.axhline(0, color=COLORS["ink"], lw=0.8)
        ax.set_xticks(range(len(BARS)), [label for _, label, _ in BARS], fontsize=10)
        ax.set_ylim(-top * 0.6 - 0.3, top * 1.3 + 0.3)
        ax.set_xlabel(f"{title}, {parameter:,} development rows", fontsize=10)
    axes[0].set_ylabel("Accuracy gain over plain argmax (points)")
    axes[1].set_ylabel("Accuracy gain over plain argmax (points)")


def explain(result, parameter):
    p, r = result["models"]["plain"], result["models"]["rebalanced"]
    over_p, over_r = p["same_rows"] - p["new_rows"], r["same_rows"] - r["new_rows"]
    steps = [
        f"Plain: same-row {spoints(p['same_rows'])}, cross-half {spoints(p['cross_half'])}, new rows {spoints(p['new_rows'])} points.",
        f"Rebalanced: same-row {spoints(r['same_rows'])}, cross-half {spoints(r['cross_half'])}, new rows {spoints(r['new_rows'])} points.",
        f"Plain overstatement: {points(p['same_rows'])} - {par(p['new_rows'])} = {points(over_p)} points.",
        f"Rebalanced overstatement: {points(r['same_rows'])} - {par(r['new_rows'])} = {points(over_r)} points.",
    ]
    pays = "pay" if r["new_rows"] > 2 * r["new_rows_se"] and r["new_rows"] > 0.002 else "do not clearly pay"
    interpretation = (
        f"With {parameter:,} development rows the model's own probabilities gain {points(p['same_rows'])} accuracy points from tuned factors on the rows that "
        f"tuned them and {spoints(p['new_rows'])} on {result['new_rows']:,} new rows, so {points(p['same_rows'])} - {par(p['new_rows'])} = {points(over_p)} points is overstatement. "
        f"For the rebalanced probabilities the same-row gain is {points(r['same_rows'])} and the new-row gain {spoints(r['new_rows'])} points: the factors {pays} there. "
        f"The same-row gain is positive in both cases and cannot tell them apart. The cross-half gains are {spoints(p['cross_half'])} and {spoints(r['cross_half'])}"
        + (", which does separate them." if r["cross_half"] > p["cross_half"] + 0.004 else ", too close to separate them at this size."))
    metrics = {"Plain, same-row gain (points)": points(p["same_rows"]), "Plain, new-row gain (points)": spoints(p["new_rows"]),
               "Rebalanced, same-row gain (points)": points(r["same_rows"]), "Rebalanced, new-row gain (points)": spoints(r["new_rows"]),
               "Plain argmax accuracy on OOF rows": fmt(p["argmax_accuracy"])}
    alt = (f"Two panels of three bars each, accuracy gain of tuned class factors in percentage points on the same rows, in a cross-half check and on new rows, "
           f"with {parameter} development rows. For the model's own probabilities the new-row gain is {spoints(p['new_rows'])} points.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    sizes = sorted(results)
    for n, res in results.items():
        p, r = res["models"]["plain"], res["models"]["rebalanced"]
        for m in (p, r):
            assert m["same_rows"] > 0.002 and m["same_rows"] > m["new_rows"] + 0.002, f"same-row gain should overstate at {n}"
        assert p["new_rows"] < 0.003, f"plain factors should not clearly help at {n}"
        assert res["models"]["plain"]["same_rows"] > 0
    for a, b in zip(sizes, sizes[1:]):
        assert results[b]["models"]["plain"]["same_rows"] < results[a]["models"]["plain"]["same_rows"], "same-row gain falls with n"
    for n in (500, 988, 3000):
        p, r = results[n]["models"]["plain"], results[n]["models"]["rebalanced"]
        assert r["new_rows"] > p["new_rows"] + 0.005, f"rebalanced factors should beat plain factors at {n}"
    for n in (500, 988, 3000):
        reb = results[n]["models"]["rebalanced"]
        assert reb["new_rows"] > 0.003 and reb["new_rows"] > 2 * reb["new_rows_se"], f"explanation: rebalanced factors pay from about 500 rows ({n})"
    assert results[250]["models"]["rebalanced"]["new_rows"] < 0.0075, "rebalanced gain unresolved at 250"
    for n in (500, 988, 3000):
        m = results[n]["models"]
        assert m["rebalanced"]["cross_half"] > m["plain"]["cross_half"] + 0.004, f"cross-half should separate the cases at {n}"
    assert results[988]["models"]["plain"]["new_rows"] < 0, "plain factors lose accuracy on new rows at 988"
    mid = results[988]["models"]
    assert points(mid["plain"]["same_rows"]) == "1.03" and spoints(mid["plain"]["new_rows"]) == "-0.52", "prediction feedback numbers"
    assert spoints(mid["rebalanced"]["new_rows"]) == "+0.72", "check answer number"
    assert all(res["models"][k]["new_rows_se"] < 0.0045 for res in results.values() for k in ("plain", "rebalanced")), "se text says at most about 0.4 points"
