"""Chapter 28: Nearest-Neighbor Features. A neighbour target-mean feature built three ways, scored by CV and on new rows, by K."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from kaggle_companion.activities._common import clean

SEED = 28
REPLICATES = 6          # independent constructed datasets; every estimate is their mean
DIMS, BLOBS = 5, 40
N_TRAIN, N_NEW = 1500, 4000


def generate(rng):
    """Forty tight clusters in five dimensions, each with its own event rate (0.12 or 0.88).
    The label is a property of the neighbourhood, so neighbours' labels carry real signal."""
    centers = rng.uniform(-3, 3, (BLOBS, DIMS))
    rate = np.where(rng.random(BLOBS) < 0.5, 0.12, 0.88)

    def rows(n):
        b = rng.integers(0, BLOBS, n)
        return centers[b] + rng.normal(0, 0.55, (n, DIMS)), (rng.random(n) < rate[b]).astype(int)

    return rows(N_TRAIN), rows(N_NEW)


def neighbour_mean(X_db, y_db, X_query, k, exclude_self):
    """Mean label of the k nearest database rows. The scaler and the database come from X_db only.
    With exclude_self the query rows ARE the database rows, so each one's first neighbour is itself and is dropped."""
    scaler = StandardScaler().fit(X_db)
    nn = NearestNeighbors(n_neighbors=k + int(exclude_self)).fit(scaler.transform(X_db))
    ids = nn.kneighbors(scaler.transform(X_query))[1]
    return y_db[ids[:, int(exclude_self):]].mean(axis=1)


def auc(X_fit, y_fit, X_eval, y_eval):
    model = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=8, random_state=0).fit(X_fit, y_fit)
    return roc_auc_score(y_eval, model.predict_proba(X_eval)[:, 1])


def with_feature(X, f):
    return np.column_stack([X, f])


def one_replicate(k, seed):
    (X, y), (X_new, y_new) = generate(np.random.default_rng(seed))
    outer = list(KFold(5, shuffle=True, random_state=1).split(X))
    cv = lambda col: np.mean([auc(with_feature(X[a], col[a]), y[a], with_feature(X[b], col[b]), y[b]) for a, b in outer])

    baseline = (np.mean([auc(X[a], y[a], X[b], y[b]) for a, b in outer]), auc(X, y, X_new, y_new))
    query_new = neighbour_mean(X, y, X_new, k, exclude_self=False)    # new rows are queried against the training rows

    # 1. Self included: the training column is built from the same rows it describes, each row's own label among them.
    col_self = neighbour_mean(X, y, X, k, exclude_self=False)
    # 2. Self excluded once on the full training set, then ordinary cross-validation on that column.
    col_loo = neighbour_mean(X, y, X, k, exclude_self=True)
    # 3. Nested: inside each outer fold, exclude self among the outer-training rows and query the validation rows from them.
    nested = np.mean([auc(with_feature(X[a], neighbour_mean(X[a], y[a], X[a], k, True)), y[a],
                          with_feature(X[b], neighbour_mean(X[a], y[a], X[b], k, False)), y[b]) for a, b in outer])
    deployed_loo = auc(with_feature(X, col_loo), y, with_feature(X_new, query_new), y_new)
    return {"baseline": baseline,
            "self_included": (cv(col_self), auc(with_feature(X, col_self), y, with_feature(X_new, query_new), y_new)),
            "self_excluded": (cv(col_loo), deployed_loo),
            "nested": (nested, deployed_loo)}      # pipelines 2 and 3 deploy the same model; they differ in the estimate


def run(k):
    reps = [one_replicate(k, SEED * 100 + i) for i in range(REPLICATES)]
    out = {"neighbors": k, "replicates": REPLICATES, "training_rows": N_TRAIN, "new_rows": N_NEW}
    for name in ("baseline", "self_included", "self_excluded", "nested"):
        out[name] = {"cv": float(np.mean([r[name][0] for r in reps])), "new_rows": float(np.mean([r[name][1] for r in reps]))}
    out["per_replicate_cv"] = {n: [r[n][0] for r in reps] for n in ("self_included", "self_excluded", "nested")}
    return clean(out)
# notebook-end


SPEC = {
    "chapter": 28,
    "chapter_title": "Nearest-Neighbor Features",
    "subtitle": "A neighbour target summary is a label-derived feature: exclude the row itself and build it inside the fold.",
    "summary": ("A feature built from the labels of the K nearest training rows leaks the row's own label if the row is its own "
                "neighbour. One demonstration builds the feature three ways and measures how far each cross-validation estimate "
                "sits from the score on new rows, for K from 1 to 50."),
    "title": "A neighbour target-mean feature, scored by cross-validation and on new rows",
    "question": "How many neighbours does it take before including the row itself stops distorting the estimate, and what does the legitimate feature gain?",
    "why": ("Neighbour target features look like a free gain, and a self-included version looks like a miracle. Measuring "
            "the three constructions against new rows shows where the shine is leakage and what remains once it is removed."),
    "method": ("Six constructed datasets, each with 40 tight clusters in five dimensions with event rates "
               "of 0.12 or 0.88, 1,500 training rows and 4,000 new rows from the same generator. The feature is the mean label of "
               "the K nearest training rows after a training-fitted scaler. It is built three ways: with the row itself among its "
               "neighbours, with the row excluded once on the full training set, and nested inside each of five outer folds. A histogram "
               "gradient boosting classifier is scored by AUC, in 5-fold cross-validation and on the new rows, and compared "
               "with the same model without the feature."),
    "control": {"key": "neighbors", "label": "Neighbours K in the target-mean feature",
                "values": [1, 5, 20, 50], "default": 5,
                "value_labels": ["1: the single nearest row", "5", "20", "50"]},
    "source_section": "A Three-Row Query",
    "symbols": ("f_i is the neighbour feature of row i, N_K(i) the K database rows nearest to it, y_j the label of row j and "
                "K the number of neighbours. If row i is in its own database, its label y_i is one of the K terms and carries weight 1/K."),
    "explanation": ("A row stored in its own database is its own nearest neighbour at distance zero, so its label supplies 1/K of "
                    "the feature. At K = 1 the feature is the label. A tree model finds that immediately: cross-validation reads "
                    "near-perfect, but new rows have no such column and the deployed model leans on it. As K grows the row's own "
                    "share shrinks and the distortion fades. Excluding the row once on the full set already removes most of it; "
                    "nesting also keeps the validation labels out of the training rows' features."),
    "application": ("Build neighbour target features like a target encoding: exclude the row itself, cross-fit inside each outer "
                    "training fold, query validation rows from the outer-training database only, and keep a no-feature baseline "
                    "on matched folds."),
    "assumptions": ("Constructed data with strong local structure, so the legitimate gain is real here and may be smaller on a "
                    "competition table. In this generator excluding the row once on the full training set and nesting give "
                    "estimates within about 0.01 AUC of each other and of the new-row score; with other data the second boundary "
                    "can matter more, which is why the chapter nests by default. A histogram gradient boosting classifier stands in for LightGBM."),
    "prediction": "With K = 1 and the row included among its own neighbours, what does cross-validation report, and what does the model score on new rows?",
    "prediction_options": ["About 0.84 on both, like the model without the feature",
                           "About 1.00 in cross-validation, and below the no-feature model on new rows",
                           "About 0.92 in cross-validation and above the no-feature model on new rows"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "Cross-validation reports 1.000, but the model scores 0.754 on new rows, below the 0.836 of the model without the feature.",
        "incorrect": "Cross-validation reports 1.000, but the model scores 0.754 on new rows, below the 0.836 of the model without the feature. The feature is the row's own label.",
    },
    "check": "At K = 5, how much does the self-included estimate overstate the score on new rows, and what does the honest feature add?",
    "answer": ("The self-included estimate is 0.921 against 0.856 on new rows, an overstatement of 0.065, while the nested estimate (0.853) "
               "is within 0.01 of the new-row score (0.858). Compared with 0.836 for the model without the feature, the honest feature "
               "adds 0.022 AUC here. The leak made it look like 0.085."),
    "provenance": "Constructed example: six seeded synthetic datasets with clustered labels and a histogram gradient boosting model, measured by the chapter activity.",
    "apply": [
        "Never encode training rows in place with the database that contains them; drop each row's own entry, as a leave-one-out query does.",
        "Inside every outer fold, build the training rows' feature from the outer-training rows, then query the validation rows from the same database.",
        "Compare against the same model without the feature on matched rows, and try several K: a tiny K can be pure noise once the leak is gone.",
        "Treat a near-perfect estimate from a neighbour feature as a signal to check the exclusion, not as a result.",
    ],
    "honesty": "Constructed data with strong local structure; the sizes of these effects are properties of this generator, not a competition result.",
}

EQUATIONS = [{"tex": r"f_i = \frac{1}{K}\sum_{j \in N_K(i)} y_j",
              "alt": "f i equals one over K times the sum of y j over the K nearest database rows j of row i",
              "basis": "The neighbour target mean the chapter's helper computes (The Implementation); written as an equation by the activity."}]
NCOLS = 1
HEIGHT = 4.4
PIPELINES = [("self_included", "Self included"), ("self_excluded", "Self excluded,\nthen CV"), ("nested", "Nested\n(chapter)")]


def draw(ax, result, parameter):
    xs = np.arange(len(PIPELINES))
    cv = [result[k]["cv"] for k, _ in PIPELINES]
    new = [result[k]["new_rows"] for k, _ in PIPELINES]
    ax.bar(xs - 0.2, cv, width=0.4, color=COLORS["light"], edgecolor=COLORS["ink"], lw=0.6, label="Cross-validation estimate")
    ax.bar(xs + 0.2, new, width=0.4, color=COLORS["teal"], edgecolor=COLORS["ink"], lw=0.6, label="AUC on 4,000 new rows")
    for x, a, b in zip(xs, cv, new):
        ax.text(x - 0.2, a + 0.006, fmt(a), ha="center", va="bottom", fontsize=10)
        ax.text(x + 0.2, b + 0.006, fmt(b), ha="center", va="bottom", fontsize=10)
    ax.axhline(result["baseline"]["new_rows"], color=COLORS["gold"], ls=(0, (4, 3)), lw=1.5,
               label=f"No neighbour feature, new rows: {fmt(result['baseline']['new_rows'])}")
    ax.set_xticks(xs, [name for _, name in PIPELINES])
    ax.set_ylim(0.6, 1.22)
    ax.set_yticks(np.arange(0.6, 1.01, 0.1))
    ax.set_ylabel("AUC")
    ax.set_xlabel(f"How the neighbour feature is built, K = {parameter}")
    ax.legend(loc="upper right", frameon=False, fontsize=10)


def explain(result, parameter):
    s, e, n, b = result["self_included"], result["self_excluded"], result["nested"], result["baseline"]
    inflation = float(fmt(s["cv"])) - float(fmt(s["new_rows"]))
    gain = float(fmt(n["new_rows"])) - float(fmt(b["new_rows"]))
    nested_gap = float(fmt(n["cv"])) - float(fmt(n["new_rows"]))
    if gain > 0.01:
        value = f"the honest feature adds {fmt(gain)} AUC on new rows"
    elif gain < -0.01:
        value = f"the honest feature costs {fmt(-gain)} AUC on new rows"
    else:
        value = "the honest feature adds nothing measurable on new rows"
    if inflation > 0.003:
        optimism = " of optimism"
        position = f"{fmt(inflation)} above"
    else:
        optimism = ", so at this K including the row no longer inflates the estimate measurably"
        position = f"{fmt(abs(inflation))} below" if inflation < -0.0005 else "level with"
    interpretation = (
        f"With K = {parameter}: the self-included pipeline reports {fmt(s['cv'])} but scores {fmt(s['new_rows'])} on new rows, "
        f"{fmt(s['cv'])} - {fmt(s['new_rows'])} = {fmt(inflation)}{optimism}. Excluding the row once reports {fmt(e['cv'])}; the nested "
        f"estimate {fmt(n['cv'])} differs from the new-row score {fmt(n['new_rows'])} by {signed(nested_gap)}. Against {fmt(b['new_rows'])} without the feature, {value}. "
        f"In a self-included feature the row's own label is 1/{parameter} of the mean.")
    steps = [
        f"Self included: {fmt(s['cv'])} - {fmt(s['new_rows'])} = {fmt(inflation)} (estimate minus new rows).",
        f"Self excluded, then CV: {fmt(e['cv'])} - {fmt(e['new_rows'])} = {signed(float(fmt(e['cv'])) - float(fmt(e['new_rows'])))}.",
        f"Nested: {fmt(n['cv'])} - {fmt(n['new_rows'])} = {signed(nested_gap)}.",
        f"Value of the honest feature on new rows: {fmt(n['new_rows'])} - {fmt(b['new_rows'])} = {signed(gain)}.",
    ]
    metrics = {"Self-included CV": fmt(s["cv"]), "Self-included, new rows": fmt(s["new_rows"]),
               "Nested CV": fmt(n["cv"]), "Honest feature, new rows": fmt(n["new_rows"]),
               "No feature, new rows": fmt(b["new_rows"])}
    alt = (f"Paired bars of cross-validation AUC and new-row AUC for three neighbour-feature constructions at K = {parameter}, "
           f"with a dashed line at the no-feature score {fmt(b['new_rows'])}; the self-included estimate is {position} its new-row score.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    r1, r5, r20, r50 = (results[k] for k in (1, 5, 20, 50))
    gaps = [r["self_included"]["cv"] - r["self_included"]["new_rows"] for r in (r1, r5, r20, r50)]
    assert gaps[0] > gaps[1] > gaps[2] > gaps[3] - 0.002, "self-inclusion optimism should fall as K grows"
    assert gaps[0] > 0.2 and gaps[1] > 0.04 and gaps[3] < 0.02, "optimism sizes"
    assert fmt(r1["self_included"]["cv"]) == "1.000" and fmt(r1["self_included"]["new_rows"]) == "0.754", "prediction feedback numbers"
    assert fmt(r1["baseline"]["new_rows"]) == "0.836", "prediction feedback numbers"
    assert r1["self_included"]["new_rows"] < r1["baseline"]["new_rows"] - 0.05, "K=1 self-included should hurt on new rows"
    assert abs(r1["nested"]["new_rows"] - r1["baseline"]["new_rows"]) < 0.01, "K=1 honest feature adds nothing"
    for k, r in results.items():
        assert abs(r["nested"]["cv"] - r["nested"]["new_rows"]) < 0.015, f"nested estimate off at K={k}"
        assert abs(r["self_excluded"]["cv"] - r["self_excluded"]["new_rows"]) < 0.015, f"self-excluded estimate off at K={k}"
        assert r["self_excluded"]["new_rows"] == r["nested"]["new_rows"], "pipelines 2 and 3 deploy the same model"
    for r in (r5, r20, r50):
        assert r["nested"]["new_rows"] - r["baseline"]["new_rows"] > 0.01, "honest feature should help for K >= 5"
    assert fmt(r5["self_included"]["cv"]) == "0.921" and fmt(r5["self_included"]["new_rows"]) == "0.856", "check answer numbers"
    assert fmt(r5["nested"]["cv"]) == "0.853" and fmt(r5["nested"]["new_rows"]) == "0.858", "check answer numbers"
    assert fmt(r5["nested"]["new_rows"] - r5["baseline"]["new_rows"]) == "0.022", "check answer: honest gain"
    assert fmt(r5["self_included"]["cv"] - r5["baseline"]["new_rows"]) == "0.085", "check answer: apparent gain"
    assert fmt(gaps[1]) == "0.065", "check answer: overstatement"
