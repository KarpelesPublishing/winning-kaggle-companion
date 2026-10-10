"""Chapter 33: Data Augmentation for Tabular Data. SMOTE before the split against inside the fold, by minority prevalence."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from kaggle_companion.activities._common import clean

SEED = 33
REPLICATES = 8          # independent constructed datasets; every estimate is their mean
N_TRAIN, N_NEW, FEATURES = 4000, 8000, 10


def generate(n, prevalence, rng):
    """Ten standard-normal features; the minority class is shifted by 1.0, 0.8 and 0.6 standard deviations on the first three."""
    y = (rng.random(n) < prevalence).astype(int)
    X = rng.normal(size=(n, FEATURES))
    X[:, :3] += y[:, None] * np.array([1.0, 0.8, 0.6])
    return X, y


def smote(X, y, rng, k=5):
    """Interpolate between each minority row and one of its k nearest minority neighbours until the classes are balanced."""
    minority = X[y == 1]
    need = int((y == 0).sum() - len(minority))
    if need <= 0 or len(minority) < 2:
        return X, y
    k = min(k, len(minority) - 1)
    neighbours = NearestNeighbors(n_neighbors=k + 1).fit(minority).kneighbors(minority)[1][:, 1:]   # column 0 is the row itself
    start = rng.integers(0, len(minority), need)
    partner = neighbours[start, rng.integers(0, k, need)]
    synthetic = minority[start] + rng.random((need, 1)) * (minority[partner] - minority[start])
    return np.vstack([X, synthetic]), np.concatenate([y, np.ones(need, int)])


def model(class_weight=None):
    return HistGradientBoostingClassifier(max_iter=30, max_leaf_nodes=8, max_bins=32, learning_rate=0.1,
                                          class_weight=class_weight, random_state=0)


def one_replicate(prevalence, seed):
    rng = np.random.default_rng(seed)
    X, y = generate(N_TRAIN, prevalence, rng)
    X_new, y_new = generate(N_NEW, prevalence, rng)             # new rows from the same generator: the "truth"
    folds = list(StratifiedKFold(3, shuffle=True, random_state=1).split(X, y))

    def pooled_cv(fit_predict):
        """Pool the held-out predictions of all folds and compute one average precision over them."""
        pooled = np.zeros(N_TRAIN)
        for train, valid in folds:
            pooled[valid] = fit_predict(train, valid)
        return average_precision_score(y, pooled)

    out = {}
    out["none"] = (pooled_cv(lambda a, b: model().fit(X[a], y[a]).predict_proba(X[b])[:, 1]),
                   average_precision_score(y_new, model().fit(X, y).predict_proba(X_new)[:, 1]))
    out["weight"] = (pooled_cv(lambda a, b: model("balanced").fit(X[a], y[a]).predict_proba(X[b])[:, 1]),
                     average_precision_score(y_new, model("balanced").fit(X, y).predict_proba(X_new)[:, 1]))

    def smote_in_fold(train, valid):                            # resample the training fold only; validation rows stay untouched
        Xr, yr = smote(X[train], y[train], np.random.default_rng(seed + int(train[0])))
        return model().fit(Xr, yr).predict_proba(X[valid])[:, 1]

    X_all, y_all = smote(X, y, np.random.default_rng(seed + 1))   # SMOTE on the whole training set: the shortcut
    deployed = average_precision_score(y_new, model().fit(X_all, y_all).predict_proba(X_new)[:, 1])
    out["inside"] = (pooled_cv(smote_in_fold), deployed)

    # SMOTE before the split: cross-validate the resampled table, so synthetic rows built from a validation row's neighbours
    # sit in the training folds. Scoring only the original rows keeps the class balance honest; scoring everything is worse.
    pooled = np.zeros(len(y_all))
    for train, valid in StratifiedKFold(3, shuffle=True, random_state=1).split(X_all, y_all):
        pooled[valid] = model().fit(X_all[train], y_all[train]).predict_proba(X_all[valid])[:, 1]
    original = np.arange(len(y_all)) < N_TRAIN
    out["before"] = (average_precision_score(y_all[original], pooled[original]), deployed)
    out["before_all_rows"] = (average_precision_score(y_all, pooled), deployed)
    return out


def run(prevalence):
    reps = [one_replicate(prevalence, SEED * 100 + i) for i in range(REPLICATES)]
    result = {"prevalence": prevalence, "replicates": REPLICATES, "training_rows": N_TRAIN, "new_rows": N_NEW,
              "training_positives": round(N_TRAIN * prevalence)}
    for name in ("none", "weight", "inside", "before", "before_all_rows"):
        result[name] = {"cv": float(np.mean([r[name][0] for r in reps])), "new_rows": float(np.mean([r[name][1] for r in reps]))}
    return clean(result)
# notebook-end


SPEC = {
    "chapter": 33,
    "chapter_title": "Data Augmentation for Tabular Data",
    "subtitle": "Resample inside the training fold, and compare with weighting and with no resampling.",
    "summary": ("SMOTE builds synthetic minority rows between real ones. Applied before the split, those rows carry information about "
                "validation rows into training. One demonstration measures how far each resampling policy's cross-validation "
                "estimate sits from the score on new rows, and what resampling adds at all, by minority prevalence."),
    "title": "SMOTE before the split against inside the fold, by minority prevalence",
    "question": "How much does resampling before the split inflate cross-validation, and does resampling inside the fold beat weighting or doing nothing?",
    "why": ("Imbalanced tasks tempt a quick fix, and the quick version (resample the table, then cross-validate) produces a large, convincing "
            "gain that is not real. Measuring the estimate against new rows shows how large the artefact is and whether any real gain remains."),
    "method": ("Eight constructed datasets of 4,000 rows and ten features; the minority class (prevalence is the control) is shifted by "
               "0.6 to 1.0 standard deviations on three features. A histogram gradient boosting classifier is scored by average "
               "precision (AP), pooled over 3-fold cross-validation and measured on 8,000 new rows, under four policies: no "
               "resampling, class weights, SMOTE inside each training fold, and SMOTE on the full training set before the folds are "
               "drawn (validation rows then include synthetic neighbours in training; only original rows are scored). SMOTE here is "
               "written out in numpy: interpolate to a random one of the 5 nearest minority neighbours until the classes balance."),
    "control": {"key": "prevalence", "label": "Minority class share of the rows",
                "values": [0.02, 0.05, 0.15, 0.30], "default": 0.05,
                "value_labels": ["0.02: 2% positives", "0.05", "0.15", "0.30: 30% positives"]},
    "source_section": "SMOTE and Oversampling: Handle With Care",
    "symbols": ("AP is average precision, the area under the precision-recall curve; a random ranking scores the prevalence. A synthetic row is "
                "x_new = x_a + u (x_b - x_a), where x_a is a minority row, x_b one of its nearest minority neighbours and u is uniform on [0, 1]."),
    "explanation": ("A synthetic minority row lies on a segment between two real minority rows. If the table is resampled before the folds are "
                    "drawn, the segments that end at a validation row sit in the training folds, so the model has effectively seen the "
                    "answer. The rarer the minority, the more synthetic rows are built from each real one and the larger the artefact; "
                    "once the classes are close to balanced there is little to synthesize and the artefact disappears. Resampling inside the "
                    "fold removes the leak, and then the estimate describes new rows but resampling itself adds no AP."),
    "application": ("Put resampling inside the training fold (a pipeline that fits it on training rows only), score untouched validation rows, "
                    "and always keep the no-resampling baseline and a class-weight arm. If SMOTE does not beat them on the official metric, drop it."),
    "assumptions": ("Constructed data and one gradient boosting model, scored by average precision; the chapter's weighting options are "
                    "LightGBM and XGBoost parameters, represented here by the class weights of a histogram gradient boosting model. "
                    "Scoring the synthetic rows too (what a careless pipeline reports) is higher still and is reported beside the main estimate. "
                    "Three-fold models train on two thirds of the rows, so honest estimates read a little low (0.202 against 0.226 for SMOTE "
                    "inside the fold at 5% positives). In this generator SMOTE inside the fold did not improve on no resampling; with other data and metrics the "
                    "result can differ, and the chapter's advice is to test it against weighting."),
    "prediction": "With 5% positives, SMOTE before the split reports what cross-validation AP, compared with the AP the final model scores on new rows?",
    "prediction_options": ["About the same as on new rows", "About 0.08 higher than on new rows", "About 0.5 higher than on new rows"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "It reports 0.302 against 0.226 on new rows, 0.076 higher, and that is with only the original rows scored.",
        "incorrect": "It reports 0.302 against 0.226 on new rows, 0.076 higher, with only the original rows scored (scoring all rows reports 0.879).",
    },
    "check": "Does SMOTE inside the fold improve the score on new rows over no resampling, and when does the pre-split artefact vanish?",
    "answer": ("No: at 5% positives it scores 0.226 against 0.234 with no resampling, and class weights score 0.246. The pre-split artefact is "
               "0.076 at 5% and 0.021 at 15%, and it is gone at 30% (0.685 against 0.686), where the classes are nearly balanced."),
    "provenance": "Constructed example: eight seeded synthetic imbalanced datasets and a histogram gradient boosting model, measured by the chapter activity.",
    "apply": [
        "Never resample the table before the folds are drawn; put the resampler in a pipeline so it fits on the training fold only.",
        "Score validation folds on untouched rows and keep the no-resampling baseline on the same folds.",
        "Compare against class weights or loss reweighting first: they add no synthetic rows and are cheaper to validate.",
        "Check calibration and threshold choice separately, because resampling and weighting change the predicted probabilities.",
    ],
    "honesty": "Constructed data; the sizes of these effects are properties of this generator, not a competition result. Resampling did not help here, and the text does not claim it never does.",
}

EQUATIONS = [{"tex": r"x_{\text{new}} = x_a + u\,(x_b - x_a), \quad u \sim \mathrm{Uniform}(0,1)",
              "alt": "a synthetic row x new equals the minority row x a plus u times the difference between the neighbour x b and x a, with u uniform between zero and one",
              "basis": "The interpolation the chapter describes in words (SMOTE and Oversampling); written as an equation by the activity."}]
NCOLS = 1
HEIGHT = 4.5
POLICIES = [("none", "No\nresampling"), ("weight", "Class\nweights"), ("inside", "SMOTE inside\nthe fold"), ("before", "SMOTE before\nthe split")]


def draw(ax, result, parameter):
    xs = np.arange(len(POLICIES))
    cv = [result[k]["cv"] for k, _ in POLICIES]
    new = [result[k]["new_rows"] for k, _ in POLICIES]
    ax.bar(xs - 0.2, cv, width=0.4, color=COLORS["light"], edgecolor=COLORS["ink"], lw=0.6, label="Cross-validation estimate")
    ax.bar(xs + 0.2, new, width=0.4, color=COLORS["teal"], edgecolor=COLORS["ink"], lw=0.6, label="AP on 8,000 new rows")
    top = max(cv + new)
    for x, a, b in zip(xs, cv, new):
        ax.text(x - 0.2, a + 0.01 * top, fmt(a), ha="center", va="bottom", fontsize=10)
        ax.text(x + 0.2, b + 0.01 * top, fmt(b), ha="center", va="bottom", fontsize=10)
    ax.axhline(parameter, color=COLORS["gold"], ls=(0, (4, 3)), lw=1.5, label=f"Random ranking (prevalence): {fmt(parameter, 2)}")
    ax.set_xticks(xs, [n for _, n in POLICIES])
    ax.set_ylim(0, top * 1.4)
    ax.set_ylabel("Average precision (AP)")
    ax.set_xlabel(f"Resampling policy, {fmt(100 * parameter, 0)}% positives")
    ax.legend(loc="upper right", frameon=False, fontsize=10)


def explain(result, parameter):
    b, i, w, n = result["before"], result["inside"], result["weight"], result["none"]
    inflation = float(fmt(b["cv"])) - float(fmt(b["new_rows"]))
    inside_gap = float(fmt(i["cv"])) - float(fmt(i["new_rows"]))
    value = float(fmt(i["new_rows"])) - float(fmt(n["new_rows"]))
    weight_value = float(fmt(w["new_rows"])) - float(fmt(n["new_rows"]))
    if inflation > 0.003:
        optimism = " of optimism"
        position = f"{fmt(inflation)} above"
    else:
        optimism = ", so on the original rows the pre-split artefact has gone at this prevalence"
        position = f"{fmt(abs(inflation))} below" if inflation < -0.0005 else "level with"
    interpretation = (
        f"With {fmt(100 * parameter, 0)}% positives, SMOTE before the split reports AP {fmt(b['cv'])} but the model scores {fmt(b['new_rows'])} on new rows, "
        f"{fmt(b['cv'])} - {fmt(b['new_rows'])} = {fmt(inflation)}{optimism} (scoring the synthetic rows too reports {fmt(result['before_all_rows']['cv'])}). "
        f"Inside the fold the estimate is {fmt(i['cv'])} against {fmt(i['new_rows'])} on new rows. Against {fmt(n['new_rows'])} for no resampling, "
        f"SMOTE inside the fold changes AP by {signed(value)} and class weights by {signed(weight_value)}.")
    steps = [
        f"SMOTE before the split: {fmt(b['cv'])} - {fmt(b['new_rows'])} = {fmt(inflation)} (estimate minus new rows).",
        f"SMOTE inside the fold: {fmt(i['cv'])} - {fmt(i['new_rows'])} = {signed(inside_gap)}.",
        f"Value of SMOTE on new rows: {fmt(i['new_rows'])} - {fmt(n['new_rows'])} = {signed(value)}.",
        f"Value of class weights on new rows: {fmt(w['new_rows'])} - {fmt(n['new_rows'])} = {signed(weight_value)}.",
    ]
    metrics = {"Pre-split CV AP": fmt(b["cv"]), "Inside-fold CV AP": fmt(i["cv"]), "SMOTE, new rows": fmt(i["new_rows"]),
               "No resampling, new rows": fmt(n["new_rows"]), "Class weights, new rows": fmt(w["new_rows"])}
    alt = (f"Paired bars of cross-validation AP and new-row AP for four resampling policies at {fmt(100 * parameter, 0)}% positives, with a dashed line at "
           f"the random-ranking level {fmt(parameter, 2)}; SMOTE before the split reports {position} its new-row score.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    rates = (0.02, 0.05, 0.15, 0.3)
    gaps = [results[p]["before"]["cv"] - results[p]["before"]["new_rows"] for p in rates]
    assert gaps[0] > gaps[1] > gaps[2] > gaps[3] - 0.003, "pre-split optimism should fall as prevalence rises"
    assert gaps[0] > 0.08 and gaps[1] > 0.05 and gaps[2] > 0.01 and abs(gaps[3]) < 0.01, "optimism sizes"
    for p, res in results.items():
        assert res["before_all_rows"]["cv"] > res["before"]["cv"] + 0.1, f"scoring synthetic rows should inflate further at {p}"
        assert abs(res["inside"]["cv"] - res["inside"]["new_rows"]) < 0.05, f"inside-fold estimate should describe new rows at {p}"
        assert abs(res["none"]["cv"] - res["none"]["new_rows"]) < 0.05, f"baseline estimate should describe new rows at {p}"
        assert res["inside"]["new_rows"] < res["none"]["new_rows"] + 0.01, f"SMOTE should not clearly beat no resampling at {p}"
    r = results[0.05]
    assert fmt(r["before"]["cv"]) == "0.302" and fmt(r["before"]["new_rows"]) == "0.226", "prediction feedback numbers"
    assert fmt(float(fmt(r["before"]["cv"])) - float(fmt(r["before"]["new_rows"]))) == "0.076", "prediction feedback gap"
    assert fmt(r["before_all_rows"]["cv"]) == "0.879", "prediction feedback all-rows number"
    assert fmt(r["inside"]["cv"]) == "0.202" and fmt(r["inside"]["new_rows"]) == "0.226", "assumption numbers"
    assert fmt(r["none"]["new_rows"]) == "0.234" and fmt(r["weight"]["new_rows"]) == "0.246", "check answer numbers"
    assert fmt(float(fmt(results[0.15]["before"]["cv"])) - float(fmt(results[0.15]["before"]["new_rows"]))) == "0.021", "check answer gap at 0.15"
    assert fmt(results[0.3]["before"]["cv"]) == "0.685" and fmt(results[0.3]["before"]["new_rows"]) == "0.686", "check answer numbers at 0.30"
