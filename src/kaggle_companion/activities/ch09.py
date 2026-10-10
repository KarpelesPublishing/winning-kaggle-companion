"""Chapter 9: Target Encoding Without Leakage. Three encoding pipelines, CV estimate against a true holdout."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from kaggle_companion.activities._common import clean

SEED = 9
M = 10.0  # smoothing strength from the chapter's smoothed mean


def generate(category_signal, seed=SEED):
    """Training rows and a large holdout. 300 categories with 2 to 9 training rows each."""
    rng = np.random.default_rng(seed)
    n_cat = 300
    effect = rng.normal(0.0, category_signal, n_cat)          # true category effect (log odds)
    counts = rng.integers(2, 10, n_cat)
    cat_train = np.repeat(np.arange(n_cat), counts)
    cat_hold = rng.integers(0, n_cat, 4000)

    def rows(cat):
        x = rng.normal(size=(len(cat), 2))
        logit = 0.8 * x[:, 0] - 0.5 * x[:, 1] + effect[cat] - 0.3
        y = (rng.random(len(cat)) < 1 / (1 + np.exp(-logit))).astype(int)
        return x, y

    x_train, y_train = rows(cat_train)
    x_hold, y_hold = rows(cat_hold)
    return cat_train, x_train, y_train, cat_hold, x_hold, y_hold


def smoothed_map(cat, y, n_cat=300):
    """Training-only smoothed mean e_c = (s_c + m*mu) / (n_c + m)."""
    mu = y.mean()
    s = np.bincount(cat, weights=y, minlength=n_cat)
    n = np.bincount(cat, minlength=n_cat)
    return (s + M * mu) / (n + M)


def oof_encode(cat, y, seed):
    """Cross-fitted encoding: each row is encoded from the other folds only."""
    enc = np.zeros(len(cat))
    for fit, out in KFold(5, shuffle=True, random_state=seed).split(cat):
        enc[out] = smoothed_map(cat[fit], y[fit])[cat[out]]
    return enc


def fit_score(x_fit, enc_fit, y_fit, x_eval, enc_eval, y_eval):
    model = LogisticRegression(max_iter=1000).fit(np.column_stack([x_fit, enc_fit]), y_fit)
    return roc_auc_score(y_eval, model.predict_proba(np.column_stack([x_eval, enc_eval]))[:, 1])


def run(category_signal):
    cat, x, y, cat_h, x_h, y_h = generate(category_signal)
    outer = list(KFold(5, shuffle=True, random_state=1).split(cat))
    full_map = smoothed_map(cat, y)

    # 1. Naive: one encoding from all training labels, each row's own label included.
    naive_col = full_map[cat]
    naive_cv = np.mean([fit_score(x[a], naive_col[a], y[a], x[b], naive_col[b], y[b]) for a, b in outer])
    naive_hold = fit_score(x, naive_col, y, x_h, full_map[cat_h], y_h)

    # 2. Global OOF table: cross-fitted once, then model CV on that table (fails the second boundary).
    oof_col = oof_encode(cat, y, seed=2)
    global_cv = np.mean([fit_score(x[a], oof_col[a], y[a], x[b], oof_col[b], y[b]) for a, b in outer])

    # 3. Nested: inside each outer fold, cross-fit the training rows and map the validation rows.
    nested_scores = []
    for a, b in outer:
        enc_a = oof_encode(cat[a], y[a], seed=3)
        enc_b = smoothed_map(cat[a], y[a])[cat[b]]
        nested_scores.append(fit_score(x[a], enc_a, y[a], x[b], enc_b, y[b]))
    nested_cv = np.mean(nested_scores)

    # The deployable pipeline for 2 and 3 is the same: cross-fitted training column, full map for new rows.
    honest_hold = fit_score(x, oof_col, y, x_h, full_map[cat_h], y_h)
    no_enc_hold = roc_auc_score(y_h, LogisticRegression(max_iter=1000).fit(x, y).predict_proba(x_h)[:, 1])

    return clean({
        "category_signal": category_signal,
        "naive": {"cv": naive_cv, "holdout": naive_hold},
        "global_oof": {"cv": global_cv, "holdout": honest_hold},
        "nested": {"cv": nested_cv, "holdout": honest_hold},
        "no_encoding_holdout": no_enc_hold,
        "training_rows": len(y), "categories": 300,
    })
# notebook-end


SPEC = {
    "chapter": 9,
    "chapter_title": "Target Encoding Without Leakage",
    "subtitle": "Encode categories from permitted labels, and assess the whole encoding pipeline.",
    "summary": ("A target encoding must be fitted without the row's own label and rebuilt inside every outer fold. "
                "One demonstration measures what each shortcut does to the cross-validation estimate and to the score on new rows."),
    "title": "Three ways to target-encode, scored against new rows",
    "question": "How far does each target-encoding pipeline's cross-validation estimate drift from its score on new rows?",
    "why": ("Target encoding is one of the most common sources of a validation score that does not survive the leaderboard. "
            "Measuring the gap tells you which shortcut is safe and which one quietly rewards a leak."),
    "method": ("A constructed binary task with 300 categories of 2 to 9 training rows each, two numeric features, and a "
               "true category effect whose size is the control. Each pipeline encodes the category with the chapter's smoothed "
               "mean (m = 10), fits a logistic regression and is scored two ways: 5-fold cross-validation on the 1,606 training rows, "
               "and AUC on 4,000 new rows from the same generator."),
    "control": {"key": "category_signal", "label": "True category effect (standard deviation, log odds)",
                "values": [0, 0.5, 1.0, 1.5], "default": 1.0,
                "value_labels": ["0: categories carry no signal", "0.5", "1.0", "1.5: strong category effect"]},
    "source_section": "Two Boundaries, Not One Global OOF Table",
    "symbols": ("e_c is the encoding of category c, s_c the sum and n_c the count of permitted training targets in c, "
                "mu their mean and m the smoothing strength (10 here)."),
    "explanation": ("A naive encoding uses every training label, including the row's own, so the model learns a column that "
                    "partly contains the answer. Cross-fitting removes the row's own label; nesting the encoder inside each outer "
                    "fold also keeps the validation labels out. Change the category effect to see when the encoding is worth having at all."),
    "application": ("Build the encoder inside each outer training fold, cross-fit the training rows, map validation rows from the "
                    "outer-training labels only, and compare against the same model without the encoded column."),
    "assumptions": ("Constructed data and a logistic model. In this generator the global OOF table's estimate is close to the nested one; "
                    "with other models or data the second boundary can matter more, which is why the chapter nests by default."),
    "prediction": "When the categories carry no signal at all, what cross-validation AUC does the naive pipeline report?",
    "prediction_options": ["About the same as on new rows", "About 0.10 higher than on new rows", "Lower than on new rows"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "It reports 0.791 against 0.690 on new rows: the encoded column contains each row's own label.",
        "incorrect": "It reports 0.791 against 0.690 on new rows. Its own label leaks into each row's encoding, so cross-validation rewards the leak.",
    },
    "check": "With no category signal, why does the naive pipeline also score worse on new rows, while with a strong signal it does not?",
    "answer": ("With no signal it learned to trust a column that carried each training row's own label, and new rows' encodings carry "
               "no such label (0.690 against 0.717). With a strong signal the column is genuinely useful, so the deployed models end up "
               "alike (0.773 against 0.772); the damage is then to the estimate you use to choose, not to the model."),
    "provenance": "Constructed example: seeded synthetic categories and a logistic regression, measured by the chapter activity.",
    "apply": [
        "Wrap the encoder in the same outer folds as the model; never encode the full training set once and then cross-validate.",
        "Inside each outer fold, cross-fit the training rows and map validation rows with a map fitted on outer-training labels only.",
        "Always compare against the model without the encoded column: an encoding that adds nothing is a cost, not a feature.",
        "Choose the smoothing strength m with inner validation, not by habit.",
    ],
    "honesty": "Constructed data; the effect sizes are properties of this generator, not a competition result.",
}

EQUATIONS = [{"tex": r"e_c = \frac{s_c + m\mu}{n_c+m}",
              "alt": "e c equals s c plus m times mu, divided by n c plus m",
              "basis": "The chapter's smoothed mean (display equation in Chapter 9, The Smoothed Mean)."}]
NCOLS = 1
HEIGHT = 4.2
PIPELINES = [("naive", "Naive (all labels)"), ("global_oof", "Global OOF table"), ("nested", "Nested (chapter)")]


def draw(ax, result, parameter):
    xs = range(len(PIPELINES))
    cv = [result[k]["cv"] for k, _ in PIPELINES]
    hold = [result[k]["holdout"] for k, _ in PIPELINES]
    ax.bar([x - 0.2 for x in xs], cv, width=0.4, color=COLORS["light"], edgecolor=COLORS["ink"], lw=0.6,
           label="Cross-validation estimate")
    ax.bar([x + 0.2 for x in xs], hold, width=0.4, color=COLORS["teal"], edgecolor=COLORS["ink"], lw=0.6,
           label="AUC on 4,000 new rows")
    for x, a, b in zip(xs, cv, hold):
        ax.text(x - 0.2, a + 0.004, fmt(a), ha="center", va="bottom", fontsize=10)
        ax.text(x + 0.2, b + 0.004, fmt(b), ha="center", va="bottom", fontsize=10)
    ax.axhline(result["no_encoding_holdout"], color=COLORS["gold"], ls=(0, (4, 3)), lw=1.4,
               label=f"No encoding, new rows: {fmt(result['no_encoding_holdout'])}")
    ax.set_xticks(list(xs), [name for _, name in PIPELINES])
    ax.set_ylim(0.6, 0.95)
    ax.set_ylabel("AUC")
    ax.set_xlabel("Encoding pipeline")
    ax.legend(loc="upper right", frameon=False, fontsize=10)


def explain(result, parameter):
    n, g, s = result["naive"], result["global_oof"], result["nested"]
    gap = n["cv"] - n["holdout"]
    nested_gap = s["cv"] - s["holdout"]
    value = s["holdout"] - result["no_encoding_holdout"]
    if value > 0.01:
        worth = f"the encoded column adds {fmt(value)} AUC on new rows"
    elif value < -0.01:
        worth = f"the encoded column costs {fmt(-value)} AUC on new rows"
    else:
        worth = "the encoded column adds nothing on new rows, because the categories carry no usable signal"
    interpretation = (
        f"At category effect {parameter}: the naive pipeline reports {fmt(n['cv'])} but scores {fmt(n['holdout'])} on new rows, "
        f"{fmt(n['cv'])} - {fmt(n['holdout'])} = {fmt(gap)} of optimism. The nested estimate {fmt(s['cv'])} is "
        f"{fmt(abs(nested_gap))} from its new-row score {fmt(s['holdout'])}; the global OOF table reports {fmt(g['cv'])}. "
        f"Compared with no encoding, {worth}.")
    steps = [
        f"Naive: {fmt(n['cv'])} - {fmt(n['holdout'])} = {fmt(gap)} (estimate minus new rows).",
        f"Global OOF table: {fmt(g['cv'])} - {fmt(g['holdout'])} = {signed(g['cv'] - g['holdout'])}.",
        f"Nested: {fmt(s['cv'])} - {fmt(s['holdout'])} = {signed(nested_gap)}.",
        f"Value of the encoding on new rows: {fmt(s['holdout'])} - {fmt(result['no_encoding_holdout'])} = {signed(value)}.",
    ]
    metrics = {"Naive CV estimate": fmt(n["cv"]), "Naive on new rows": fmt(n["holdout"]),
               "Nested CV estimate": fmt(s["cv"]), "Cross-fitted on new rows": fmt(s["holdout"]),
               "No encoding on new rows": fmt(result["no_encoding_holdout"])}
    alt = (f"Paired bars of cross-validation estimate and new-row AUC for three encoding pipelines at category effect {parameter}; "
           f"the naive pipeline's estimate is {fmt(gap)} above its new-row score.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for value, res in results.items():
        assert res["naive"]["cv"] - res["naive"]["holdout"] > 0.05, f"naive optimism too small at {value}"
        assert abs(res["nested"]["cv"] - res["nested"]["holdout"]) < 0.02, f"nested estimate off at {value}"
    zero = results[0]
    assert fmt(zero["naive"]["cv"]) == "0.791" and fmt(zero["naive"]["holdout"]) == "0.690", "prediction feedback numbers"
    assert abs(zero["nested"]["holdout"] - zero["no_encoding_holdout"]) < 0.01, "no-signal case should add nothing"
    assert fmt(zero["nested"]["holdout"]) == "0.717" and zero["naive"]["holdout"] < zero["nested"]["holdout"] - 0.02
    strong = results[1.5]
    assert fmt(strong["naive"]["holdout"]) == "0.773" and fmt(strong["nested"]["holdout"]) == "0.772"
    assert abs(strong["naive"]["holdout"] - strong["nested"]["holdout"]) < 0.01
    assert results[1.5]["nested"]["holdout"] - results[1.5]["no_encoding_holdout"] > 0.05, "strong signal should help"
