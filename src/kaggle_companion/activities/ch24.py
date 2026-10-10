"""Chapter 24: Building a Kaggle Portfolio. A teammate's OOF file joined by position or by ID, as more of its rows are out of order."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from kaggle_companion.activities._common import clean

SEED = 24
DATASETS = 6          # independent constructed tasks; every estimate is their mean
N_ROWS = 2000


def generate(rng):
    """Two feature groups: A is linear in the first six columns, B is nonlinear in the last six. Each model sees only its own group."""
    X = rng.normal(size=(N_ROWS, 12))
    z = (X[:, :6] @ np.array([0.9, -0.7, 0.6, 0.5, -0.4, 0.3]) + np.sin(1.5 * X[:, 6]) + 0.9 * X[:, 7] * X[:, 8]
         + 0.7 * (X[:, 9] > 0.3) - 0.2)
    return X, (rng.random(N_ROWS) < 1 / (1 + np.exp(-z))).astype(int)


def oof_predictions(model, X, y, seed):
    """Out-of-fold probabilities, one per row, in the original row order (row i is predicted by a model that never saw row i)."""
    p = np.zeros(len(y))
    for fit, val in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        p[val] = model.fit(X[fit], y[fit]).predict_proba(X[val])[:, 1]
    return p


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def saved_order(rng, fraction):
    """The order a teammate's file is saved in: `fraction` of the rows are shuffled among themselves, the rest stay in place."""
    order = np.arange(N_ROWS)
    moved = rng.choice(N_ROWS, int(round(fraction * N_ROWS)), replace=False)
    order[moved] = rng.permutation(moved)
    return order


def stacked_auc(p_a, p_b, y):
    """Cross-validated logistic-regression stack of the two logit-transformed columns."""
    features = np.column_stack([logit(p_a), logit(p_b)])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    return roc_auc_score(y, cross_val_predict(LogisticRegression(), features, y, cv=cv, method="predict_proba")[:, 1])


def one_dataset(fraction, seed):
    rng = np.random.default_rng(seed)
    X, y = generate(rng)
    p_a = oof_predictions(LogisticRegression(max_iter=500), X[:, :6], y, 1)
    p_b = oof_predictions(ExtraTreesClassifier(100, min_samples_leaf=3, random_state=0, n_jobs=1), X[:, 6:], y, 2)
    order = saved_order(rng, fraction)
    # The file as saved: row k of the file holds the prediction for row order[k]. A join by ID undoes that; a join by position does not.
    ids_b = order
    p_b_file = p_b[order]
    by_id = np.empty(N_ROWS)
    by_id[ids_b] = p_b_file                      # sort the file back by its ID column
    assert np.allclose(by_id, p_b)
    return {"a": roc_auc_score(y, p_a), "b_by_id": roc_auc_score(y, by_id), "b_by_position": roc_auc_score(y, p_b_file),
            "blend_by_id": roc_auc_score(y, p_a + by_id), "stack_by_id": stacked_auc(p_a, by_id, y),
            "blend_by_position": roc_auc_score(y, p_a + p_b_file), "stack_by_position": stacked_auc(p_a, p_b_file, y),
            "ids_matching": float(np.mean(ids_b == np.arange(N_ROWS)))}


def run(misaligned_fraction):
    sets = [one_dataset(misaligned_fraction, SEED * 1000 + i) for i in range(DATASETS)]
    mean = {k: float(np.mean([s[k] for s in sets])) for k in sets[0]}
    mean["blend_by_position_per_dataset"] = [s["blend_by_position"] for s in sets]
    return clean({"misaligned_fraction": misaligned_fraction, "rows": N_ROWS, "datasets": DATASETS, **mean,
                  "rows_out_of_place": int(round(misaligned_fraction * N_ROWS))})
# notebook-end


SPEC = {
    "chapter": 24,
    "chapter_title": "Building a Kaggle Portfolio",
    "subtitle": "Agree on a prediction-artifact contract (IDs, folds, order) before stacking a teammate's model.",
    "summary": ("A teammate's out-of-fold file that arrives without IDs can be joined by position and silently misalign. One demonstration measures what "
                "the blend, the teammate's model and a stacked blend look like as more of the file's rows are out of order."),
    "title": "A teammate's OOF file joined by position or by ID",
    "question": "How much of a blend's gain is lost, and what does it look like, when a teammate's OOF file is joined by position but is partly out of order?",
    "why": ("The chapter asks teams to agree on sample, class and fold identities and to validate unique IDs, fold membership and join cardinality "
            "before stacking. This measures what skipping the ID check costs and why the failure is easy to miss."),
    "method": ("Six constructed binary tasks of 2,000 rows. Model A (logistic regression on six features) and model B (extremely randomized trees on six "
               "other features, with nonlinear effects) each produce 5-fold out-of-fold probabilities. Teammate B's file is saved with the control's "
               "share of rows shuffled among themselves, a simple stand-in for a file that was partly re-sorted; at 100% almost every row is displaced, as in a file saved in fold order. Each blend is scored by AUC against the labels in "
               "the original order: an average of the two probabilities and a cross-validated logistic stack, joined by ID (the file sorted back by its ID "
               "column) and joined by position."),
    "control": {"key": "misaligned_fraction", "label": "Share of the teammate's rows saved out of order",
                "values": [0, 0.05, 0.25, 1.0], "default": 0.25,
                "value_labels": ["0: same order as the labels", "5%", "25%", "100%: fully shuffled"]},
    "source_section": "Solo or Team",
    "symbols": ("a_i and b_i are the two models' out-of-fold probabilities for row i, pi(i) the row that the file holds at position i, and the "
                "blend is the mean of the two columns."),
    "explanation": ("Both files still have one finite probability per position, so nothing fails to run. A join by position pairs some of "
                    "model A's predictions with predictions for other rows, so teammate B looks weaker than it is and both an average blend and a "
                    "stack gain less; once the file is fully shuffled the stack learns to ignore the column. A join by ID restores every pair, and the full blend gain."),
    "application": ("Agree on one saved format with an ID column, a fold column and class order; assert unique IDs and equal ID sets before every "
                    "join, and join by ID, never by position."),
    "assumptions": ("Constructed data with one linear and one tree model on disjoint features, a deliberately complementary pair; real pairs gain "
                    "less. Misalignment here is a random shuffle of the stated share of rows. This is a data-contract effect only and says nothing "
                    "about portfolios, teams or hiring."),
    "prediction": "A quarter of teammate B's rows are saved out of order and the blend is joined by position. Where does its AUC land compared with the join by ID and with model A alone?",
    "prediction_options": ["Same as the join by ID", "Between model A alone and the join by ID", "Below model A alone"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "It lands at 0.798 against 0.813 by ID and 0.771 for model A alone: about a third of the gain is gone, with no error raised.",
        "incorrect": "It lands at 0.798 against 0.813 by ID and 0.771 for model A alone: about a third of the gain is gone, and the file still ran without error.",
    },
    "check": "With the file fully shuffled, the average blend falls below model A alone while the stacked blend only matches it. Why do they differ?",
    "answer": ("An average gives the shuffled column the same weight as the good one, so noise is added (0.750 against 0.771 for A alone). A stack is "
               "fitted on labels, finds that the misaligned column predicts nothing and gives it near-zero weight, so it recovers model A's score "
               "(0.770) and quietly throws away the teammate. Neither failure raises an error; a one-line ID check does."),
    "provenance": "Constructed example: six seeded synthetic tasks, logistic regression and extremely randomized trees, measured by the chapter activity.",
    "apply": [
        "Agree before the team starts on one OOF file format: sample ID, fold, true label, one probability column per class in a stated order.",
        "Before blending, assert that IDs are unique, complete and equal across files; join by ID and check the join is one to one.",
        "Score each teammate's file alone as soon as it arrives: a model that scores near chance, or below its owner's own report, signals misalignment first.",
        "Keep the fold assignments with the predictions, so the OOF provenance can be checked and the blend repeated.",
    ],
    "honesty": "Constructed data. The size of the loss depends on how complementary the two models are and on how many rows are displaced.",
}

EQUATIONS = [{"tex": r"\hat{p}_i = \tfrac12\,\bigl(a_i + b_{\pi(i)}\bigr), \qquad \pi(i) = i \text{ when joined by ID}",
              "alt": "the blended probability for position i is the mean of A's prediction a i and B's prediction at the row the file holds in that position, which is row i only when joined by ID",
              "basis": "The activity's blend of two OOF columns; the chapter states the ID requirement in prose (Solo or Team, and the prediction-artifact contract check)."}]
NCOLS = 1
HEIGHT = 4.4
BARS = [("a", "Model A\nalone", COLORS["light"]), ("b_by_position", "Model B,\nfile order", COLORS["light"]),
        ("blend_by_id", "Blend\nby ID", COLORS["teal"]), ("blend_by_position", "Average blend,\nby position", COLORS["terracotta"]),
        ("stack_by_position", "Stack,\nby position", COLORS["gold"])]


def draw(ax, result, parameter):
    for x, (key, label, color) in enumerate(BARS):
        ax.bar(x, result[key], width=0.6, color=color, edgecolor=COLORS["ink"], lw=0.6)
        top = max([result[key]] + (result["blend_by_position_per_dataset"] if key == "blend_by_position" else []))
        ax.text(x, top + 0.008, fmt(result[key]), ha="center", va="bottom", fontsize=10)
    dots = result["blend_by_position_per_dataset"]
    ax.scatter([3 + (i - (len(dots) - 1) / 2) * 0.07 for i in range(len(dots))], dots, s=14, color=COLORS["ink"], zorder=3, label="One task (average blend)")
    ax.axhline(result["a"], color=COLORS["grey"], ls=(0, (4, 3)), lw=1.4, label=f"Model A alone: {fmt(result['a'])}")
    ax.set_xticks(range(len(BARS)), [label for _, label, _ in BARS], fontsize=10)
    ax.set_ylim(0.45, 0.95)
    ax.set_ylabel("AUC against the true labels")
    ax.set_xlabel(f"Teammate's file joined as shown, {int(round(100 * parameter))}% of its rows out of order")
    ax.legend(loc="upper right", frameon=False, fontsize=10)


def explain(result, parameter):
    a, by_id, pos, stack = result["a"], result["blend_by_id"], result["blend_by_position"], result["stack_by_position"]
    lost = by_id - pos
    gain = by_id - a
    share = lost / gain if gain else 0.0
    interpretation = (
        f"With {result['rows_out_of_place']} of {result['rows']} rows out of order ({fmt(100 * parameter, 0)}%), the blend joined by ID scores {fmt(by_id)}, "
        f"model A alone {fmt(a)}, and the blend joined by position {fmt(pos)}: {fmt(by_id)} - {fmt(pos)} = {fmt(lost)} of AUC lost, "
        f"{fmt(100 * share, 0)}% of the {fmt(gain)} the blend gains over model A. The teammate's model scores {fmt(result['b_by_position'])} alone in file order "
        f"against {fmt(result['b_by_id'])} by ID, and the stacked blend by position scores {fmt(stack)} against {fmt(result['stack_by_id'])} by ID. "
        + ("Positions and IDs agree, so nothing is lost." if parameter == 0 else
           f"Only {fmt(100 * result['ids_matching'], 0)}% of positions match their IDs, which an ID check would have reported."))
    steps = [
        f"Blend by ID: {fmt(by_id)}; model A alone: {fmt(a)}; gain {fmt(by_id)} - {fmt(a)} = {fmt(gain)}.",
        f"Average blend by position: {fmt(pos)}; lost {fmt(by_id)} - {fmt(pos)} = {fmt(lost)}.",
        f"Stack by position: {fmt(stack)} against {fmt(result['stack_by_id'])} by ID.",
        f"Teammate B alone: {fmt(result['b_by_position'])} in file order, {fmt(result['b_by_id'])} by ID.",
    ]
    metrics = {"Blend by ID": fmt(by_id), "Average blend by position": fmt(pos), "Model A alone": fmt(a),
               "Teammate B alone, file order": fmt(result["b_by_position"]), "Positions matching their IDs": fmt(100 * result["ids_matching"], 0) + "%"}
    alt = (f"Bars of AUC for model A alone, teammate B in file order, the blend by ID, the average blend by position and the stack by position "
           f"with {fmt(100 * parameter, 0)}% of rows out of order, and a dashed line at model A alone, {fmt(a)}.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for q, res in results.items():
        assert res["blend_by_id"] > res["a"] + 0.03, f"blend by ID should gain over A at {q}"
        assert abs(res["ids_matching"] - (1 - q)) < 0.06, f"ids matching should be about 1 - q at {q}"
    zero = results[0]
    assert zero["blend_by_position"] == zero["blend_by_id"] and zero["b_by_position"] == zero["b_by_id"], "no misalignment, no difference"
    qs = sorted(results)
    pos = [results[q]["blend_by_position"] for q in qs]
    assert all(b < a for a, b in zip(pos, pos[1:])), "blend by position falls as more rows are out of order"
    quarter = results[0.25]
    assert quarter["blend_by_position"] > quarter["a"] + 0.01, "25%: still above A alone"
    assert 0.2 < (quarter["blend_by_id"] - quarter["blend_by_position"]) / (quarter["blend_by_id"] - quarter["a"]) < 0.5, "about a third of the gain at 25%"
    assert quarter["stack_by_position"] < quarter["stack_by_id"] - 0.01, "25%: the stack also gains less"
    full = results[1.0]
    assert full["blend_by_position"] < full["a"] - 0.01, "100%: average blend below A alone"
    assert abs(full["stack_by_position"] - full["a"]) < 0.01, "100%: the stack recovers about A alone"
    assert abs(full["b_by_position"] - 0.5) < 0.03, "100%: B alone near chance"
    assert fmt(quarter["blend_by_position"]) == "0.798" and fmt(quarter["blend_by_id"]) == "0.813" and fmt(quarter["a"]) == "0.771", "prediction feedback numbers"
    assert fmt(full["blend_by_position"]) == "0.750" and fmt(full["a"]) == "0.771" and fmt(full["stack_by_position"]) == "0.770", "check answer numbers"
