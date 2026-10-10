"""Chapter 62: The Kaggle Landscape, 2024 to 2026. Which inference pipeline finishes inside a runtime cap, measured in counted work."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from kaggle_companion.activities._common import clean

SEED = 62
DATASETS = 5                         # independent constructed tasks; every estimate is their mean
F, N_TRAIN, N_QUERY = 12, 2000, 1000
T_SMALL, T_BIG = 6, 150              # trees in the feasible reference and in the stronger model
LOAD_UNITS = 5.0                     # assumed work to read one tree node when loading a model (inference costs 1 per node visited)


def generate(n, rng, weights):
    """A binary task with interactions, so a bigger forest really is a bit more accurate than a small one."""
    X = rng.normal(size=(n, F))
    z = np.tanh(X @ weights[0]) @ weights[1] + 0.8 * X[:, 0] * X[:, 1] - 0.6 * X[:, 2] * X[:, 3]
    return X, (z + rng.normal(0, 1.0, n) > 0).astype(int)


def fit_forest(trees, seed, X, y):
    return RandomForestClassifier(trees, min_samples_leaf=2, max_features=0.5, random_state=seed, n_jobs=1).fit(X, y)


def load_cost(model):
    """Work to load: every node of every tree is read once."""
    return LOAD_UNITS * sum(t.tree_.node_count for t in model.estimators_)


def query_costs(model, X):
    """Work to answer each query: the number of tree nodes it visits, summed over trees (counted, not timed)."""
    visited, _ = model.decision_path(X)
    return np.asarray(visited.sum(axis=1)).ravel().astype(float)


def run(deadline_percent):
    keys = ("small", "big", "reference_then_upgrade")
    acc = {k: [] for k in keys}
    done = {k: [] for k in keys}
    solo = {"small": [], "big": []}
    majority = []
    cost = {"load_small": [], "run_small": [], "load_big": [], "run_big": []}
    for d in range(DATASETS):
        rng = np.random.default_rng(SEED * 100 + d)
        weights = (rng.normal(size=(F, 4)), rng.normal(size=4))
        Xtr, ytr = generate(N_TRAIN, rng, weights)
        Xq, yq = generate(N_QUERY, rng, weights)
        small, big = fit_forest(T_SMALL, d, Xtr, ytr), fit_forest(T_BIG, d, Xtr, ytr)
        p_small, p_big = small.predict_proba(Xq)[:, 1], big.predict_proba(Xq)[:, 1]
        c_small, c_big = query_costs(small, Xq), query_costs(big, Xq)
        l_small, l_big = load_cost(small), load_cost(big)
        deadline = deadline_percent / 100 * (l_big + c_big.sum())   # a share of what the big model needs end to end
        fallback = int(ytr.mean() > 0.5)                            # a query left unanswered gets the majority class

        def in_order(load, per_query, p):
            """Load, then answer queries in file order until the budget runs out."""
            finished = np.cumsum(per_query) <= deadline - load if deadline > load else np.zeros(N_QUERY, bool)
            return float(np.mean(np.where(finished, p > 0.5, fallback) == yq)), float(finished.mean())

        for name, load, per_query, p in (("small", l_small, c_small, p_small), ("big", l_big, c_big, p_big)):
            accuracy, answered = in_order(load, per_query, p)
            acc[name].append(accuracy)
            done[name].append(answered)
        # Retained reference: answer every query with the small model first, then spend what is left on the big model,
        # most uncertain queries (small-model probability nearest 0.5) first.
        pred = np.where(np.cumsum(c_small) <= deadline - l_small, p_small > 0.5, fallback) if deadline > l_small else np.full(N_QUERY, fallback)
        upgraded = 0
        spare = deadline - l_small - c_small.sum() - l_big
        if spare > 0:
            used = 0.0
            for i in np.argsort(np.abs(p_small - 0.5)):
                if used + c_big[i] > spare:
                    break
                used += c_big[i]
                pred[i] = p_big[i] > 0.5
                upgraded += 1
        acc["reference_then_upgrade"].append(float(np.mean(pred == yq)))
        done["reference_then_upgrade"].append(upgraded / N_QUERY)
        majority.append(float(np.mean(yq == fallback)))
        solo["small"].append(float(np.mean((p_small > 0.5) == yq)))     # unlimited-time accuracy of each model
        solo["big"].append(float(np.mean((p_big > 0.5) == yq)))
        cost["load_small"].append(l_small); cost["run_small"].append(c_small.sum())
        cost["load_big"].append(l_big); cost["run_big"].append(c_big.sum())
    m = {k: float(np.mean(v)) for k, v in cost.items()}
    return clean({
        "deadline_percent": deadline_percent,
        "accuracy": {k: float(np.mean(v)) for k, v in acc.items()},
        "per_dataset": {k: v for k, v in acc.items()},
        "share_answered": {k: float(np.mean(v)) for k, v in done.items()},
        "unlimited_accuracy": {k: float(np.mean(v)) for k, v in solo.items()},
        "total_units": {"small": m["load_small"] + m["run_small"], "big": m["load_big"] + m["run_big"]},
        "load_units": {"small": m["load_small"], "big": m["load_big"]},
        "deadline_units": deadline_percent / 100 * (m["load_big"] + m["run_big"]),
        "majority_class_accuracy": float(np.mean(majority)), "datasets": DATASETS, "queries": N_QUERY, "trees": {"small": T_SMALL, "big": T_BIG},
    })
# notebook-end


SPEC = {
    "chapter": 62,
    "chapter_title": "The Kaggle Landscape, 2024 to 2026",
    "subtitle": "Quality includes completed work: budget loading and inference together, and keep a pipeline that finishes.",
    "summary": ("A stronger model that cannot finish inside the runtime cap does not improve the submission. One demonstration counts the work "
                "of loading and running a small and a large forest against a deadline and scores what each pipeline actually delivers."),
    "title": "Small reference, large model and reference-then-upgrade under a runtime cap",
    "question": "At what deadline does a stronger model stop being the better pipeline, and what does a retained feasible reference recover?",
    "why": ("The chapter's rule is to budget loading, preprocessing, inference and fallbacks together and to retain a feasible reference. "
            "A model with better completed-query accuracy cannot win if the queries it does not finish are scored as misses."),
    "method": ("Five constructed binary tasks with 2,000 training rows and 1,000 queries. A 6-tree random forest is the feasible reference and a "
               "150-tree forest is the stronger model. Work is counted, not timed: loading reads every tree node once at 5 units per node, and each "
               "query costs the number of tree nodes it visits, summed over trees. The deadline (the control) is a share of the stronger model's "
               "end-to-end work. Pipelines: the small model alone; the large model answering in file order until work runs out (unanswered queries "
               "get the majority class); and reference-then-upgrade, which answers every query with the small model first and spends the "
               "remaining work on the large model for the queries where the small model is least sure."),
    "control": {"key": "deadline_percent", "label": "Deadline (percent of the large model's end-to-end work)",
                "values": [120, 80, 40, 10], "default": 80,
                "value_labels": ["120%: comfortable", "80%", "40%", "10%: cannot load the large model"]},
    "source_section": "Quality Includes Completed Work",
    "symbols": ("W is the work a pipeline needs, L the work to load the model, c_i the work to answer query i and D the deadline in the "
                "same units. A pipeline is feasible when W is at most D."),
    "explanation": ("The large forest is about 0.05 more accurate on queries it finishes, but it costs about 25 times the work of the small "
                    "one. Once the deadline falls below its total, queries it never reaches are answered by the majority class. At 40% "
                    "that loss swamps the gain on the rest; at 80% the two roughly cancel, and which side wins depends on the size of the "
                    "model's advantage (it lost here and came out slightly ahead under another seed). The reference-then-upgrade pipeline "
                    "pays a small overhead to answer everything first, so no query is left unanswered, and it spends leftover work where "
                    "the two models disagree most. It scored at least as well as the reference at every deadline here."),
    "application": ("Estimate load and per-query work on a representative slice, scale to the real test size, keep a pipeline that "
                    "finishes under the cap, and upgrade queries with the leftover budget only after measuring the upgrade rule."),
    "assumptions": ("Constructed data and counted work rather than timed work: node visits stand in for seconds on one fixed machine, and "
                    "the 5 units per node for loading is an assumed constant, so the deadlines are shares of one pipeline's cost and not "
                    "a statement about any real GPU or CPU. Measure your own pipeline on your own hardware. Unanswered queries here get "
                    "the majority class, which is kinder than a real timeout that can invalidate a submission. Ordering upgrades by the "
                    "small model's uncertainty is one routing rule, and the chapter asks for each route to be assessed on its own."),
    "prediction": ("At a deadline of 80% of the large model's end-to-end work, the large model answers about three quarters of the queries in "
                   "file order. Which pipeline scores best on all 1,000 queries?"),
    "prediction_options": ["The large model: three quarters of the queries get the better model",
                           "The small reference, because it always finishes",
                           "Reference-then-upgrade"],
    "prediction_answer": 2,
    "prediction_feedback": {
        "correct": ("Reference-then-upgrade scores 0.773, against 0.729 for the small reference and 0.708 for the large model, which answers "
                    "76% of queries and leaves the rest to the majority class."),
        "incorrect": ("Reference-then-upgrade scores 0.773. The large model answers 76% of queries and scores 0.708, because the unanswered "
                      "rows fall back to the majority class; the small reference scores 0.729."),
    },
    "check": "At 80%, reference-then-upgrade answers only 71% of the queries with the large model yet scores 0.773. How does it recover almost all of the large model's gain?",
    "answer": ("It answers every query with the small model first, which costs 4% of the large model's end-to-end work, then upgrades the queries "
               "where the small model is least sure, which is where the larger model's answers differ most. Upgrading 71% of the queries "
               "recovers 95% of the accuracy gain of the large model run on all of them. That route is a measured result for this generator, "
               "and the chapter asks that any such route be assessed with its overhead."),
    "provenance": "Constructed example: seeded synthetic tasks and two random forests, with work counted from the fitted trees by the chapter activity.",
    "apply": [
        "Measure load time and per-query time of every pipeline stage on a slice of the real test data, then scale to the test size and compare with the cap.",
        "Keep a pipeline that finishes comfortably as the reference, and save its predictions before any experiment that risks the cap.",
        "Compare completed answers under the deadline, not completed-query accuracy alone.",
        "Treat a routing rule (which queries get the expensive model) as its own experiment with its own quality and cost.",
    ],
    "honesty": ("Constructed data and counted work, not timed work. The deadlines are shares of one pipeline's cost; no claim is made about any "
                "actual Kaggle runtime limit or hardware, which are competition-specific and change."),
}

EQUATIONS = [{"tex": r"W = L + \sum_{i=1}^{n} c_i \le D",
              "alt": "The work W equals the load work L plus the sum over n queries of c i, and must be at most the deadline D",
              "basis": "The activity's version of the chapter's constructed workload (loading time plus per-query time against a deadline); the manuscript gives it in words, not as a display equation."}]
NCOLS = 2
HEIGHT = 4.4
PIPELINES = [("small", "Small\nreference", COLORS["light"]), ("big", "Large model,\nfile order", COLORS["terracotta"]),
             ("reference_then_upgrade", "Reference,\nthen upgrade", COLORS["teal"])]


def sub(a, b):
    """Difference of two values as the page shows them, so a hand calculation always adds up."""
    return float(fmt(a)) - float(fmt(b))


def draw(axes, result, parameter):
    left, right = axes
    per = result["per_dataset"]
    for x, (key, name, color) in enumerate(PIPELINES):
        left.bar([x], [result["accuracy"][key]], width=0.6, color=color, edgecolor=COLORS["ink"], lw=0.6)
        dots = per[key]
        left.scatter([x + (i - (len(dots) - 1) / 2) * 0.04 for i in range(len(dots))], dots, s=10, color=COLORS["ink"], zorder=3)
        left.text(x, max(dots) + 0.012, fmt(result["accuracy"][key]), ha="center", va="bottom", fontsize=10,
                  bbox={"fc": "white", "ec": "none", "pad": 1})
    left.axhline(result["unlimited_accuracy"]["big"], color=COLORS["gold"], ls=(0, (4, 3)), lw=1.4,
                 label=f"Large model with no deadline: {fmt(result['unlimited_accuracy']['big'])}")
    left.axhline(result["majority_class_accuracy"], color=COLORS["grey"], ls=":", lw=1.2, label="Majority class")
    left.set_xticks(range(3), [p[1] for p in PIPELINES])
    left.set_ylim(0.4, 0.93)
    left.set_ylabel("Accuracy on all 1,000 queries (dots: tasks)")
    left.set_xlabel(f"Pipeline, deadline {parameter}% of large-model work")
    left.legend(loc="upper left", frameon=False, fontsize=10, ncol=1)

    k = 1000.0
    sm, bg = result["total_units"]["small"] / k, result["total_units"]["big"] / k
    lb = result["load_units"]["big"] / k
    right.bar([0], [sm], width=0.6, color=COLORS["light"], edgecolor=COLORS["ink"], lw=0.6)
    right.bar([1], [lb], width=0.6, color=COLORS["navy"], edgecolor=COLORS["ink"], lw=0.6, label="Loading")
    right.bar([1], [bg - lb], bottom=[lb], width=0.6, color=COLORS["terracotta"], edgecolor=COLORS["ink"], lw=0.6, label="Answering 1,000 queries")
    right.axhline(result["deadline_units"] / k, color=COLORS["ink"], ls=(0, (4, 3)), lw=1.6)
    right.text(-0.45, result["deadline_units"] / k, f"Deadline {fmt(result['deadline_units'] / k, 0)}", ha="left", va="bottom", fontsize=10)
    right.text(0.34, sm, fmt(sm, 0), ha="left", va="center", fontsize=10)
    right.text(1, bg + bg * 0.015, fmt(bg, 0), ha="center", va="bottom", fontsize=10)
    right.set_xticks([0, 1], ["Small\nreference", "Large\nmodel"])
    right.set_xlim(-0.5, 1.5)
    right.set_ylim(0, max(bg, result["deadline_units"] / k) * 1.18)
    right.set_ylabel("End-to-end work (thousands of units)")
    right.set_xlabel("Model, whole workload")
    right.legend(loc="center left", frameon=False, fontsize=10)


def explain(result, parameter):
    a, u, s = result["accuracy"], result["unlimited_accuracy"], result["share_answered"]
    gap_big = sub(a["small"], a["big"])
    gain_unlimited = sub(u["big"], u["small"])
    recovered = (a["reference_then_upgrade"] - a["small"]) / (u["big"] - u["small"])
    cost_ratio = result["total_units"]["big"] / result["total_units"]["small"]
    interpretation = (
        f"With a deadline of {parameter}% of the large model's work, the large model in file order answers {round(100 * s['big'])}% of the queries "
        f"and scores {fmt(a['big'])} overall, against {fmt(a['small'])} for the small reference: "
        + (f"{fmt(a['small'])} - {fmt(a['big'])} = {fmt(gap_big)} lower. " if a["big"] < a["small"] else
           f"{fmt(a['big'])} - {fmt(a['small'])} = {fmt(sub(a['big'], a['small']))} higher. ")
        + f"Given unlimited time it would score {fmt(u['big'])}, {fmt(u['big'])} - {fmt(u['small'])} = {fmt(gain_unlimited)} above the reference. "
        f"Reference-then-upgrade scores {fmt(a['reference_then_upgrade'])} and upgrades {round(100 * s['reference_then_upgrade'])}% of queries to the large model, "
        f"recovering {round(100 * recovered)}% of that gain. The large model needs {cost_ratio:.0f} times the small model's work.")
    steps = [
        f"Large model against reference: {fmt(a['big'])} - {fmt(a['small'])} = {fmt(sub(a['big'], a['small']))}.",
        f"Gain with no deadline: {fmt(u['big'])} - {fmt(u['small'])} = {fmt(gain_unlimited)}.",
        f"Reference-then-upgrade against reference: {fmt(a['reference_then_upgrade'])} - {fmt(a['small'])} = {fmt(sub(a['reference_then_upgrade'], a['small']))}.",
        f"Deadline in work units: {fmt(result['deadline_units'] / 1000, 0)} thousand, against {fmt(result['total_units']['big'] / 1000, 0)} thousand needed by the large model.",
    ]
    metrics = {"Small reference accuracy": fmt(a["small"]), "Large model, file order": fmt(a["big"]),
               "Reference then upgrade": fmt(a["reference_then_upgrade"]), "Large model with no deadline": fmt(u["big"]),
               "Queries answered by the large model": f"{round(100 * s['big'])}% in file order, {round(100 * s['reference_then_upgrade'])}% upgraded"}
    alt = (f"Left: bars of whole-test accuracy at a deadline of {parameter}% of the large model's work: small reference {fmt(a['small'])}, large model in "
           f"file order {fmt(a['big'])}, reference then upgrade {fmt(a['reference_then_upgrade'])}; a dashed line marks the large model with no deadline at "
           f"{fmt(u['big'])}. Right: end-to-end work of each model in thousands of units against the deadline line.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for d, res in results.items():
        a = res["accuracy"]
        assert a["reference_then_upgrade"] >= a["small"] - 1e-9, f"reference-then-upgrade should never be worse than the reference at {d}"
    r120, r80, r40, r10 = results[120], results[80], results[40], results[10]
    assert r120["share_answered"]["big"] == 1.0 and r120["accuracy"]["big"] == r120["unlimited_accuracy"]["big"], "large model finishes at 120%"
    assert r120["accuracy"]["big"] > r120["accuracy"]["small"] + 0.03, "text: large model about 0.05 more accurate when it finishes"
    assert r80["accuracy"]["big"] < r80["accuracy"]["small"], "explanation: large model in file order lost to the small one at 80% here"
    assert r80["accuracy"]["reference_then_upgrade"] > max(r80["accuracy"]["big"], r80["accuracy"]["small"]), "prediction: reference-then-upgrade best at 80%"
    assert r40["accuracy"]["big"] < r40["accuracy"]["small"] - 0.1
    assert r10["share_answered"]["big"] == 0.0 and abs(r10["accuracy"]["big"] - r10["majority_class_accuracy"]) < 1e-9, "cannot load at 10%"
    assert r10["accuracy"]["reference_then_upgrade"] == r10["accuracy"]["small"], "no spare work at 10%"
    assert r80["accuracy"]["reference_then_upgrade"] > r80["accuracy"]["small"] and r40["accuracy"]["reference_then_upgrade"] > r40["accuracy"]["small"]
    ratio = r80["total_units"]["big"] / r80["total_units"]["small"]
    assert 20 < ratio < 30, f"text says about 25 times the work: {ratio}"
    assert 0.03 < r80["unlimited_accuracy"]["big"] - r80["unlimited_accuracy"]["small"] < 0.07, "text says about 0.05"
    assert round(100 * r80["share_answered"]["big"]) == 76 and fmt(r80["accuracy"]["big"]) == "0.708" and fmt(r80["accuracy"]["small"]) == "0.729"
    assert fmt(r80["accuracy"]["reference_then_upgrade"]) == "0.773" and round(100 * r80["share_answered"]["reference_then_upgrade"]) == 71
    recovered = (r80["accuracy"]["reference_then_upgrade"] - r80["accuracy"]["small"]) / (r80["unlimited_accuracy"]["big"] - r80["unlimited_accuracy"]["small"])
    assert round(100 * recovered) == 95, "check answer: 95% of the gain recovered"
    overhead = r80["total_units"]["small"] / r80["total_units"]["big"]
    assert round(100 * overhead) == 4, "check answer: reference costs 4% of the large model's work"
