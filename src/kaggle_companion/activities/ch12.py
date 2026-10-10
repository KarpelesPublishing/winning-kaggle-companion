"""Chapter 12: The GBM Trio. One boosted-tree learner, three category representations, by number of levels."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from kaggle_companion.activities._common import clean

SEED = 12
REPLICATES = 5                 # independent constructed datasets; every estimate is their mean
N_TRAIN, N_NEW = 3000, 6000
NUMERIC = 4                    # numeric feature columns before the category column


def generate(levels, seed):
    """Four numeric features plus one category with `levels` levels. Each level has a random effect on the log odds,
    and the integer code given to a level is arbitrary, as it is for a label-encoded column."""
    rng = np.random.default_rng(seed)
    effect = rng.normal(0.0, 1.0, levels)
    code = rng.permutation(levels)

    def rows(n):
        level = rng.integers(0, levels, n)
        x = rng.normal(size=(n, NUMERIC))
        logit = 0.9 * x[:, 0] - 0.6 * x[:, 1] + 0.5 * x[:, 2] * x[:, 3] + effect[level]
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        return x, level, y

    return rows(N_TRAIN), rows(N_NEW), code


def learner(**extra):
    """One fixed configuration for every representation: the comparison changes the input, not the budget."""
    return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_leaf_nodes=15, random_state=0, **extra)


def run(levels):
    scores = {"none": [], "ordinal": [], "native": [], "onehot": []}
    for rep in range(REPLICATES):
        (x, lv, y), (x_new, lv_new, y_new), code = generate(levels, SEED * 100 + rep)
        ordinal, ordinal_new = np.column_stack([x, code[lv]]), np.column_stack([x_new, code[lv_new]])
        onehot, onehot_new = np.column_stack([x, np.eye(levels)[lv]]), np.column_stack([x_new, np.eye(levels)[lv_new]])
        fits = {"none": (learner(), x, x_new),                                   # no category column at all
                "ordinal": (learner(), ordinal, ordinal_new),                    # codes treated as ordered numbers
                "native": (learner(categorical_features=[NUMERIC]), ordinal, ordinal_new),   # native categorical splits
                "onehot": (learner(), onehot, onehot_new)}                       # one 0/1 column per level
        for name, (model, a, b) in fits.items():
            scores[name].append(roc_auc_score(y_new, model.fit(a, y).predict_proba(b)[:, 1]))
    return clean({
        "levels": levels, "columns": {"none": NUMERIC, "ordinal": NUMERIC + 1, "native": NUMERIC + 1, "onehot": NUMERIC + levels},
        "auc": {k: float(np.mean(v)) for k, v in scores.items()}, "per_replicate": scores,
        "rows_per_level": N_TRAIN / levels, "replicates": REPLICATES, "train_rows": N_TRAIN, "new_rows": N_NEW,
    })
# notebook-end


SPEC = {
    "chapter": 12,
    "chapter_title": "The GBM Trio",
    "subtitle": "Compare configurations on common rows and a stated budget; a framework name settles nothing.",
    "summary": ("The chapter treats the framework as a bundle of choices: how a category is represented, how rounds are chosen, what it costs. "
                "One demonstration holds the boosted-tree budget fixed and measures how three category representations compare as the number of levels grows."),
    "title": "Three ways to feed a category to a boosted-tree learner",
    "question": "How much does it matter whether a category is fed as ordered integer codes, as native categorical splits or as one-hot columns?",
    "why": ("The chapter warns that integer codes impose an order the category does not have, and that native categorical support "
            "has its own requirements. Whether the choice is worth a deadline hour depends on how many levels the column has."),
    "method": ("A constructed binary task with four numeric features and one category whose levels have random effects and arbitrary integer "
               "codes. One scikit-learn histogram gradient boosting configuration (100 rounds, learning rate 0.1, 15 leaves) is fitted on 3,000 "
               "rows and scored by AUC on 6,000 new rows, with the category as ordered integer codes, as native categorical splits, as one-hot "
               "columns, and left out. Five independent datasets per setting; the control is the number of levels."),
    "control": {"key": "levels", "label": "Levels in the category",
                "values": [8, 32, 64, 128], "default": 128,
                "value_labels": ["8", "32", "64", "128 (about 23 rows per level)"]},
    "source_section": "Feature Types and Framework Fit",
    "symbols": ("AUC_rep is the AUC on new rows when the category is represented as rep (ordinal, native or onehot), and the gap is "
                "measured against the native representation."),
    "explanation": ("With few levels, the boosted trees recover every level whichever way it is fed. As the number of levels grows, ordered "
                    "codes force the trees to carve the level effects out of one integer axis, and the score falls behind. Native "
                    "categorical splits group levels by their effect directly. One-hot columns recover most of the loss "
                    "at the price of one column per level."),
    "application": ("Run the representations your library supports on the same folds with the same rounds and leaves, and keep the "
                    "cheapest one that is within noise of the best. For a high-cardinality column, include a native or "
                    "target-aware path in the comparison."),
    "assumptions": ("Constructed data and scikit-learn's HistGradientBoostingClassifier as the stand-in: LightGBM, XGBoost and CatBoost are not "
                    "installed here. The three representations correspond to treating a label-encoded column as numeric, to the native "
                    "categorical paths of LightGBM and CatBoost (and XGBoost with categorical support enabled), and to one-hot encoding. The "
                    "numbers belong to this learner and generator. Fit time is not reported because it varies by machine; the column count "
                    "is the deterministic cost proxy."),
    "prediction": "With 128 levels, how much AUC does treating the integer codes as ordered numbers cost compared with native categorical splits?",
    "prediction_options": ["Nothing measurable", "About 0.03 AUC", "About 0.15 AUC"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "Ordinal codes score 0.751 against 0.782 for native splits, a gap of 0.031.",
        "incorrect": "Ordinal codes score 0.751 against 0.782 for native splits, a gap of 0.031: real, but far from catastrophic, and absent at 8 levels.",
    },
    "check": "Why is the representation choice nearly irrelevant at 8 levels, and what does one-hot buy at 128 levels?",
    "answer": ("With 8 levels each level has hundreds of rows and a few splits on the code isolate it, so every representation reaches the "
               "same score (0.791 for ordinal against 0.792 for native). At 128 levels there are about 23 rows per level and ordered "
               "codes need many splits on one integer axis, so they fall to 0.751 against 0.782 for native. One-hot recovers most of that "
               "(0.769) but feeds the trees 132 columns. Native splits stayed ahead of one-hot in each of the five datasets at 128 levels, "
               "while up to 64 levels the two were within 0.01, so that ordering belongs to this learner and generator rather than to a "
               "general ranking. The mechanism is the pattern measured here, not a proof."),
    "provenance": "Constructed example: seeded synthetic data and scikit-learn histogram gradient boosting, measured by the chapter activity.",
    "apply": [
        "Fix rounds, leaves and folds first, then change only the category representation, so a difference can be attributed to it.",
        "Compare ordered codes, native categorical splits and one-hot on the same rows; the gap grows with the number of levels.",
        "Keep the cheapest representation that is within noise of the best, and record columns and fit time next to the score.",
        "Native handling still needs consistent category meaning at inference and an unknown-level policy; it does not replace the outer assessment.",
    ],
    "honesty": ("Constructed data with random level effects and a stand-in learner. The gap between native splits and one-hot columns "
                "is a few thousandths of AUC up to 64 levels and about 0.01 at 128 levels, and it changed size across seed bases, so no general "
                "ranking between them is claimed."),
}

EQUATIONS = [{"tex": r"\Delta_{\mathrm{rep}} = \mathrm{AUC}_{\mathrm{native}} - \mathrm{AUC}_{\mathrm{rep}}",
              "alt": "Delta for a representation equals AUC with native splits minus AUC with that representation",
              "basis": "The activity's own comparison label; the chapter has no display equation (Feature Types and Framework Fit)."}]
NCOLS = 1
HEIGHT = 4.4
ARMS = [("none", "No category", COLORS["light"]), ("ordinal", "Ordered codes", COLORS["terracotta"]),
        ("native", "Native splits", COLORS["teal"]), ("onehot", "One-hot", COLORS["navy"])]


def draw(ax, result, parameter):
    xs = range(len(ARMS))
    values = [result["auc"][k] for k, _, _ in ARMS]
    ax.bar(xs, values, width=0.6, color=[c for _, _, c in ARMS], edgecolor=COLORS["ink"], lw=0.6)
    for x, (key, _, _) in zip(xs, ARMS):
        dots = result["per_replicate"][key]
        ax.scatter([x + (i - (len(dots) - 1) / 2) * 0.06 for i in range(len(dots))], dots, s=12, color=COLORS["ink"], zorder=3)
        ax.text(x, max(dots + [result["auc"][key]]) + 0.006, fmt(result["auc"][key]), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(list(xs), [f"{name}\n{result['columns'][k]} columns" for k, name, _ in ARMS])
    ax.set_ylim(0.65, 0.84)
    ax.set_ylabel("AUC on 6,000 new rows")
    ax.set_xlabel(f"Category representation, {parameter} levels (dots: one dataset each)")


def diff(a, b):
    """Difference of the displayed (3 decimal) values, so the hand calculation in the text adds up."""
    return fmt(round(a, 3) - round(b, 3))


def explain(result, parameter):
    a = result["auc"]
    gap_ord, gap_hot = a["native"] - a["ordinal"], a["native"] - a["onehot"]
    interpretation = (
        f"With {parameter} levels (about {fmt(result['rows_per_level'], 0)} training rows each), native categorical splits score {fmt(a['native'])} AUC. "
        f"Ordered integer codes score {fmt(a['ordinal'])}, so {fmt(a['native'])} - {fmt(a['ordinal'])} = {diff(a['native'], a['ordinal'])}. "
        f"One-hot columns score {fmt(a['onehot'])} with {result['columns']['onehot']} columns instead of {result['columns']['native']}. "
        f"Leaving the category out scores {fmt(a['none'])}. "
        + ("The representation barely matters at this size." if max(gap_ord, gap_hot) < 0.01 else
           "The choice of representation is worth a comparison at this size."))
    steps = [
        f"Ordered codes behind native: {fmt(a['native'])} - {fmt(a['ordinal'])} = {diff(a['native'], a['ordinal'])} AUC.",
        f"One-hot behind native: {fmt(a['native'])} - {fmt(a['onehot'])} = {diff(a['native'], a['onehot'])} AUC.",
        f"Value of the category: {fmt(a['native'])} - {fmt(a['none'])} = {diff(a['native'], a['none'])} AUC.",
        f"Columns fed to the trees: {result['columns']['ordinal']} (codes, native) against {result['columns']['onehot']} (one-hot).",
    ]
    metrics = {"Native splits": fmt(a["native"]), "Ordered codes": fmt(a["ordinal"]), "One-hot": fmt(a["onehot"]),
               "No category": fmt(a["none"]), "One-hot columns": str(result["columns"]["onehot"])}
    alt = (f"Bars of AUC on new rows for a category with {parameter} levels: no category {fmt(a['none'])}, ordered codes {fmt(a['ordinal'])}, "
           f"native splits {fmt(a['native'])} and one-hot {fmt(a['onehot'])}, with dots for each of five datasets.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    order = sorted(results)
    for lv in order:
        a = results[lv]["auc"]
        assert a["native"] - a["none"] > 0.05 and a["ordinal"] - a["none"] > 0.03, f"category should carry signal at {lv}"
        if lv <= 64:
            assert abs(a["native"] - a["onehot"]) < 0.01, f"one-hot should be within 0.01 of native up to 64 levels, at {lv}"
    gaps = [results[lv]["auc"]["native"] - results[lv]["auc"]["ordinal"] for lv in order]
    assert all(b > a for a, b in zip(gaps, gaps[1:])), "ordinal gap should grow with levels"
    assert gaps[0] < 0.01 and gaps[-1] > 0.02, "gap negligible at 8 and clear at 128"
    big = results[128]["auc"]
    assert big["native"] >= big["onehot"] - 0.001 and big["onehot"] > big["ordinal"] + 0.01, "one-hot between ordinal and native at 128"
    assert 0.02 < big["native"] - big["ordinal"] < 0.05, "prediction option says about 0.03"
    assert fmt(big["ordinal"]) == "0.751" and fmt(big["native"]) == "0.782", "prediction feedback numbers"
    assert fmt(big["native"] - big["ordinal"]) == "0.031", "prediction feedback gap"
    assert fmt(results[8]["auc"]["ordinal"]) == "0.791" and fmt(results[8]["auc"]["native"]) == "0.792", "check answer numbers"
    assert fmt(big["onehot"]) == "0.769" and results[128]["columns"]["onehot"] == 132, "check answer numbers"
    assert all(n > h for n, h in zip(results[128]["per_replicate"]["native"], results[128]["per_replicate"]["onehot"])), \
        "native ahead of one-hot in every dataset at 128 levels"
    assert 0.005 < big["native"] - big["onehot"] < 0.02, "honesty says about 0.01 at 128 levels"
