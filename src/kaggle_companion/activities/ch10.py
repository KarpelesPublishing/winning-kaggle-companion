"""Chapter 10: Interactions, Groups, and Text. A stepwise interaction search scored on its own folds and on fresh rows."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from kaggle_companion.activities._common import clean

SEED = 10
REPLICATES = 6         # independent constructed datasets; every estimate is their mean
N_SEARCH = 1200        # rows the stepwise search may use
N_FRESH = 20000        # fresh rows from the same generator: the stand-in for a reserved assessment partition
MIN_GAIN = 1e-5        # the chapter's accept rule: keep a pair if the development AUC rises by more than this


def generate(n, rng):
    """Twelve numeric features. One real interaction, x0 * x1, on top of three main effects; the rest is noise."""
    X = rng.normal(size=(n, 12))
    logit = 0.8 * X[:, 0] + 0.5 * X[:, 1] + 0.4 * X[:, 2] + 1.0 * X[:, 0] * X[:, 1] - 0.2
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def design(X, pairs):
    """Original columns plus one product column per kept pair."""
    return np.column_stack([X] + [X[:, a] * X[:, b] for a, b in pairs])


def cv_auc(X, y, folds):
    """Development score: mean AUC of a logistic regression over the search folds."""
    scores = []
    for fit, val in folds:
        model = LogisticRegression(max_iter=200).fit(X[fit], y[fit])
        scores.append(roc_auc_score(y[val], model.decision_function(X[val])))
    return float(np.mean(scores))


def stepwise_search(X, y, pairs, folds):
    """The chapter's accept rule: try each pair in turn, keep it if the development AUC rises by more than MIN_GAIN."""
    current = X
    base = cv_auc(current, y, folds)
    trace, kept = [base], []
    for a, b in pairs:
        trial = np.column_stack([current, X[:, a] * X[:, b]])
        score = cv_auc(trial, y, folds)
        if score - base > MIN_GAIN:
            base, current = score, trial
            kept.append((a, b))
            trace.append(score)
    return kept, trace


def fresh_auc(X, y, pairs, X_new, y_new):
    """Refit on the search rows with the given pairs, score once on rows the search never saw."""
    model = LogisticRegression(max_iter=200).fit(design(X, pairs), y)
    return roc_auc_score(y_new, model.decision_function(design(X_new, pairs)))


def run(top_features):
    pairs = list(itertools.combinations(range(top_features), 2))
    search_gain, fresh_gain, oracle_gain, noise, found = [], [], [], [], []
    for rep in range(REPLICATES):
        rng = np.random.default_rng(SEED * 100 + rep)
        X, y = generate(N_SEARCH + N_FRESH, rng)
        Xs, ys, Xf, yf = X[:N_SEARCH], y[:N_SEARCH], X[N_SEARCH:], y[N_SEARCH:]
        folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(Xs, ys))
        kept, trace = stepwise_search(Xs, ys, pairs, folds)
        base_fresh = fresh_auc(Xs, ys, [], Xf, yf)
        search_gain.append(trace[-1] - trace[0])
        fresh_gain.append(fresh_auc(Xs, ys, kept, Xf, yf) - base_fresh)
        oracle_gain.append(fresh_auc(Xs, ys, [(0, 1)], Xf, yf) - base_fresh)   # the one pair that is really there
        noise.append(len([p for p in kept if p != (0, 1)]))
        found.append((0, 1) in kept)
        if rep == 0:   # the path of the first dataset: development AUC and fresh AUC after each accepted pair
            path = [[i, trace[i], fresh_auc(Xs, ys, kept[:i], Xf, yf)] for i in range(len(kept) + 1)]
    return clean({
        "top_features": top_features, "pairs_screened": len(pairs),
        "search_gain": float(np.mean(search_gain)), "fresh_gain": float(np.mean(fresh_gain)),
        "oracle_gain": float(np.mean(oracle_gain)), "noise_pairs_kept": float(np.mean(noise)),
        "true_pair_found": float(np.mean(found)),
        "per_replicate": {"search_gain": search_gain, "fresh_gain": fresh_gain},
        "path": path, "replicates": REPLICATES, "search_rows": N_SEARCH, "fresh_rows": N_FRESH,
    })
# notebook-end


SPEC = {
    "chapter": 10,
    "chapter_title": "Interactions, Groups, and Text",
    "subtitle": "A stepwise interaction search produces a development score, not an assessed gain.",
    "summary": ("The chapter's stepwise search keeps any product that raises its own cross-validation score. One demonstration "
                "measures what that score claims, and what the selected recipe earns on fresh rows, as more pairs are screened."),
    "title": "A stepwise interaction search, scored on its own folds and on fresh rows",
    "question": "As more candidate pairs are screened, does the search score keep describing what the selected features earn on new rows?",
    "why": ("Every extra pair screened is another chance for noise to look like signal. The search score is computed on the rows that "
            "chose the pairs, so it can rise while the recipe gets worse on rows that did not."),
    "method": ("A constructed binary task with 12 numeric features, three main effects and exactly one real interaction, x0 times x1. "
               "The chapter's accept rule (keep a pair if development AUC rises by more than 0.00001) runs over every pairwise product "
               "of the first k features, using a logistic regression and 5-fold cross-validation on 1,200 rows. The control is k. "
               "The selected recipe is then refitted on the same 1,200 rows and scored once on 20,000 fresh rows from the same generator, "
               "which stands in for a reserved assessment partition. Six independent datasets are averaged."),
    "control": {"key": "top_features", "label": "Top features whose pairwise products are screened",
                "values": [4, 6, 9, 12], "default": 12,
                "value_labels": ["4 features (6 pairs)", "6 features (15 pairs)", "9 features (36 pairs)", "12 features (66 pairs)"]},
    "source_section": "Interaction Features: The Stepwise Approach",
    "symbols": ("S(F) is the development AUC (mean over the search folds) of a model using feature set F, c = x_a * x_b is a "
                "candidate product, and epsilon is the minimum gain (0.00001). A pair is kept when S(F plus c) - S(F) exceeds epsilon."),
    "explanation": ("The true pair is easy to find and earns a real gain. Every other pair is noise, but some noise products raise "
                    "the development AUC by chance, and the greedy rule keeps each of them. The more pairs are screened, the more "
                    "of these chance gains are kept, so the search score keeps climbing while the fitted noise columns dilute the "
                    "model on rows that were not used for the search."),
    "application": ("Treat the search score as a development result. Reserve a partition before the search, or rerun the whole search "
                    "(including the top-feature screen and the stopping rule) inside each outer training fold, and report the gain "
                    "of the frozen recipe there."),
    "assumptions": ("Constructed data, one real interaction that the search finds every time, and a logistic regression in place of the "
                    "chapter's LightGBM. With a weak or absent true pair the picture is worse, because nothing real offsets the noise "
                    "pairs. A fresh 20,000-row sample from the generator stands in for a reserved assessment partition, which in a "
                    "competition would be far smaller and noisier."),
    "prediction": "When all 66 pairs of 12 features are screened, how does the gain the search reports compare with the gain on fresh rows?",
    "prediction_options": ["About the same", "Search gain is about 0.03 AUC larger", "Fresh gain is larger"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "The search reports +0.071 AUC; the same recipe earns +0.044 on fresh rows.",
        "incorrect": "The search reports +0.071 AUC; the same recipe earns +0.044 on fresh rows. The gap is the selection bias of the search.",
    },
    "check": "The true interaction is found at every screen size. Why does screening more pairs still make the selected recipe worse on fresh rows?",
    "answer": ("The extra pairs are noise, but the greedy rule keeps any product that raises its own folds by more than 0.00001. "
               "At 66 pairs the search kept 14.3 pairs other than the true one on average against 1.7 at 6 pairs, each one a fitted column that "
               "describes chance in the 1,200 search rows. The search score cannot show that cost, because it is computed on the rows that chose them."),
    "provenance": "Constructed example: seeded synthetic data, one built-in interaction and a logistic regression, measured by the chapter activity.",
    "apply": [
        "Reserve an assessment partition before the interaction search, or nest the entire search inside each outer training fold.",
        "Report the frozen recipe's gain on the reserved rows; the search's own final score is a development number.",
        "Keep the screen as small as the domain allows, and log how many pairs were screened and how many were kept.",
        "Add a minimum-gain margin that reflects fold-to-fold noise instead of accepting any positive gain.",
    ],
    "honesty": ("Constructed data; one interaction is real by design, and the sizes of the effects are properties of this generator. "
                "The noise cost is small per pair and shows only in the accumulation."),
}

EQUATIONS = [{"tex": r"\text{keep } c = x_a x_b \iff S(F \cup \{c\}) - S(F) > \varepsilon",
              "alt": "keep c equal to x a times x b if and only if S of F union c minus S of F is greater than epsilon",
              "basis": "The accept rule of the chapter's stepwise_interaction_search (min_gain, Interaction Features: The Stepwise Approach); the chapter has no display equation."}]
NCOLS = 2
HEIGHT = 4.4


def draw(axes, result, parameter):
    left, right = axes
    steps = [p[0] for p in result["path"]]
    left.step(steps, [p[1] for p in result["path"]], where="post", color=COLORS["terracotta"], lw=1.8,
              label="Search score (its own folds)")
    left.step(steps, [p[2] for p in result["path"]], where="post", color=COLORS["teal"], lw=1.8,
              label=f"AUC on {result['fresh_rows']:,} fresh rows")
    left.scatter(steps, [p[1] for p in result["path"]], s=14, color=COLORS["terracotta"], zorder=3)
    left.scatter(steps, [p[2] for p in result["path"]], s=14, color=COLORS["teal"], zorder=3)
    left.set_xlabel("Pairs accepted so far (first dataset)")
    left.set_ylabel("AUC")
    left.set_xlim(-0.3, max(steps) + 0.3)
    low = min(min(p[1], p[2]) for p in result["path"])
    high = max(max(p[1], p[2]) for p in result["path"])
    left.set_ylim(low - 0.006, high + 0.03)
    left.legend(loc="upper left", frameon=False, fontsize=10)

    labels = ["Search\nscore", "Fresh\nrows", "True pair\nonly"]
    values = [result["search_gain"], result["fresh_gain"], result["oracle_gain"]]
    colors = [COLORS["terracotta"], COLORS["teal"], COLORS["light"]]
    right.bar(range(3), values, width=0.55, color=colors, edgecolor=COLORS["ink"], lw=0.6)
    for x, key in enumerate(["search_gain", "fresh_gain"]):
        dots = result["per_replicate"][key]
        right.scatter([x + (i - (len(dots) - 1) / 2) * 0.06 for i in range(len(dots))], dots, s=12, color=COLORS["ink"], zorder=3)
    tops = [max([values[0]] + result["per_replicate"]["search_gain"]),
            max([values[1]] + result["per_replicate"]["fresh_gain"]), values[2]]
    for x, v in enumerate(values):
        right.text(x, tops[x] + 0.004, fmt(v), ha="center", va="bottom", fontsize=10)
    top = max(tops)
    right.set_xticks(range(3), labels)
    right.set_ylim(0, top + 0.02)
    right.set_ylabel("AUC gain over no interactions")
    right.set_xlabel(f"{result['pairs_screened']} pairs screened (dots: one dataset each)")


def diff(a, b):
    """Difference of the displayed (3 decimal) values, so the hand calculation in the text adds up."""
    return fmt(round(a, 3) - round(b, 3))


def explain(result, parameter):
    s, f, o, n = result["search_gain"], result["fresh_gain"], result["oracle_gain"], result["noise_pairs_kept"]
    p = result["pairs_screened"]
    gaps = [a - b for a, b in zip(result["per_replicate"]["search_gain"], result["per_replicate"]["fresh_gain"])]
    se = (sum((g - sum(gaps) / len(gaps)) ** 2 for g in gaps) / (len(gaps) - 1)) ** 0.5 / len(gaps) ** 0.5
    if s - f > 2 * se:
        bias = (f"so {fmt(s)} - {fmt(f)} = {diff(s, f)} of the reported gain is selection bias "
                f"(standard error {fmt(se)} over {result['replicates']} datasets)")
    else:
        bias = (f"a difference of {fmt(s)} - {fmt(f)} = {diff(s, f)}, within two standard errors ({fmt(se)} over "
                f"{result['replicates']} datasets), so at this screen size the search score is not clearly biased")
    noise_cost = round(o, 3) - round(f, 3)
    if abs(noise_cost) < 0.0025:
        noise_text = f"so the noise pairs make no clear difference on fresh rows ({diff(f, o)})"
    else:
        noise_text = f"so the noise pairs {'cost' if noise_cost > 0 else 'add'} {fmt(abs(noise_cost))} on fresh rows"
    interpretation = (
        f"Screening {p} pairs, the search reports a gain of {fmt(s)} AUC and the same recipe earns {fmt(f)} on fresh rows, "
        f"{bias}. It kept {fmt(n, 1)} pairs other than the true one on "
        f"average and found the true pair in {round(100 * result['true_pair_found'])}% of datasets. Adding only the true pair earns {fmt(o)}, "
        f"{noise_text}.")
    steps = [
        f"Search gain minus fresh gain: {fmt(s)} - {fmt(f)} = {diff(s, f)} AUC.",
        f"Fresh gain minus true-pair-only gain: {fmt(f)} - {fmt(o)} = {signed(round(f, 3) - round(o, 3))} AUC.",
        f"Pairs screened: {p}; pairs other than the true one kept: {fmt(n, 1)} on average.",
    ]
    metrics = {"Pairs screened": str(p), "Search score gain": signed(s), "Gain on fresh rows": signed(f),
               "True pair only": signed(o), "Noise pairs kept": fmt(n, 1)}
    alt = (f"Left: development AUC and fresh-row AUC after each pair the search accepted on the first dataset, with {p} pairs screened. "
           f"Right: mean AUC gain by search score ({fmt(s)}), fresh rows ({fmt(f)}) and the true pair alone ({fmt(o)}).")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    order = sorted(results)
    for k in order:
        res = results[k]
        assert res["true_pair_found"] == 1.0, f"true pair not found at {k}"
        assert res["oracle_gain"] > 0.03, f"true pair should be worth a clear gain at {k}"
        if k >= 9:
            assert res["search_gain"] > res["fresh_gain"] + 0.005, f"search gain should exceed fresh gain at {k}"
    for a, b in zip(order, order[1:]):
        assert results[b]["search_gain"] > results[a]["search_gain"], "search gain should rise with the screen"
        assert results[b]["fresh_gain"] < results[a]["fresh_gain"], "fresh gain should fall with the screen"
        assert results[b]["noise_pairs_kept"] > results[a]["noise_pairs_kept"]
    big, small = results[12], results[4]
    assert 0.02 < big["search_gain"] - big["fresh_gain"] < 0.035, "prediction option says about 0.03"
    assert big["oracle_gain"] - big["fresh_gain"] > 0.005, "noise pairs should cost something at 66 pairs"
    assert fmt(big["search_gain"]) == "0.071" and fmt(big["fresh_gain"]) == "0.044", "prediction feedback numbers"
    assert fmt(big["noise_pairs_kept"], 1) == "14.3" and fmt(small["noise_pairs_kept"], 1) == "1.7", "check answer numbers"
    assert abs(small["search_gain"] - small["fresh_gain"]) < 0.01, "small screens roughly agree"
