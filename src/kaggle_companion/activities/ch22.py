"""Chapter 22: Vision and Multimodal. PCA on an embedding before a second stage: explained variance against held-out accuracy."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from kaggle_companion.activities._common import clean

SEED = 22
REPLICATES = 24            # independent constructed datasets; every estimate is their mean
D = 40                     # embedding dimension
VARIANCE = np.exp(-0.15 * np.arange(D))     # variance of each embedding direction, largest first
KS = [2, 4, 8, 16, 24, 32]  # candidate numbers of principal components
N_TRAIN, N_NEW, WIDTH = 600, 4000, 5
SEPARATION = 0.75          # class shift per signal direction, in standard deviations of that direction


def generate(n, rng, signal_rank):
    """Embedding plus a noisy VLM soft label. Five embedding directions, starting at variance rank `signal_rank`, carry the class.

    Every signal direction is shifted by the same number of its own standard deviations, so the information in the embedding
    is identical wherever the signal sits; only its place in the variance ranking changes."""
    y = rng.integers(0, 2, n)
    sign = 2 * y - 1
    z = rng.normal(size=(n, D)) * np.sqrt(VARIANCE)
    rows = np.arange(signal_rank, signal_rank + WIDTH)
    z[:, rows] += (SEPARATION / 2) * np.sqrt(VARIANCE[rows]) * sign[:, None]
    soft = sign * 0.55 + rng.normal(size=n)            # the VLM's output as a logit-like score, informative but noisy
    return z, soft, y


def second_stage(train_features, y_train, new_features):
    """Standardize, fit logistic regression and return predicted classes for the new features."""
    scaler = StandardScaler().fit(train_features)
    model = LogisticRegression(C=1.0, max_iter=500).fit(scaler.transform(train_features), y_train)
    return model.predict(scaler.transform(new_features))


def features(kind, z_fit, soft_fit, z_apply, soft_apply):
    """Second-stage inputs for one candidate: soft label only, soft label + raw embedding, or soft label + k principal components."""
    if kind == "soft":
        return soft_fit[:, None], soft_apply[:, None]
    if kind == "raw":
        return np.column_stack([soft_fit, z_fit]), np.column_stack([soft_apply, z_apply])
    pca = PCA(kind).fit(z_fit)                          # fitted on the fitting rows only
    return np.column_stack([soft_fit, pca.transform(z_fit)]), np.column_stack([soft_apply, pca.transform(z_apply)])


def one_dataset(signal_rank, seed):
    rng = np.random.default_rng(seed)
    z, soft, y = generate(N_TRAIN, rng, signal_rank)
    z_new, soft_new, y_new = generate(N_NEW, rng, signal_rank)
    candidates = ["soft"] + KS + ["raw"]
    new_accuracy = {}
    for kind in candidates:
        a, b = features(kind, z, soft, z_new, soft_new)
        new_accuracy[kind] = (second_stage(a, y, b) == y_new).mean()
    # Choose among the candidates by 5-fold cross-validation inside the development rows; ties keep the earlier (simpler) one.
    cv = {kind: 0.0 for kind in candidates}
    for fit, val in StratifiedKFold(5, shuffle=True, random_state=0).split(z, y):
        for kind in candidates:
            a, b = features(kind, z[fit], soft[fit], z[val], soft[val])
            cv[kind] += (second_stage(a, y[fit], b) == y[val]).mean() / 5
    chosen = max(candidates, key=lambda c: (round(cv[c], 6), -candidates.index(c)))
    # The habit the chapter warns about: keep enough components for 90% of the explained variance.
    k90 = PCA(0.90).fit(z).n_components_
    a, b = features(int(k90), z, soft, z_new, soft_new)
    variance90 = (second_stage(a, y, b) == y_new).mean()
    explained = np.cumsum(PCA(D).fit(z).explained_variance_ratio_)
    return new_accuracy, chosen, k90, variance90, explained


def run(signal_rank):
    sets = [one_dataset(signal_rank, SEED * 1000 + i) for i in range(REPLICATES)]
    candidates = ["soft"] + KS + ["raw"]
    mean_new = {str(c): float(np.mean([s[0][c] for s in sets])) for c in candidates}
    chosen_accuracy = float(np.mean([s[0][s[1]] for s in sets]))
    explained = np.mean([s[4] for s in sets], axis=0)
    return clean({
        "signal_rank": signal_rank, "replicates": REPLICATES, "embedding_dimension": D, "train_rows": N_TRAIN, "new_rows": N_NEW,
        "soft_only": mean_new["soft"], "raw_embedding": mean_new["raw"],
        "pca_by_k": {str(k): mean_new[str(k)] for k in KS},
        "explained_by_k": {str(k): float(explained[k - 1]) for k in KS},
        "variance_rule": {"accuracy": float(np.mean([s[3] for s in sets])), "components": float(np.mean([s[2] for s in sets])),
                          "explained": float(explained[int(round(np.mean([s[2] for s in sets]))) - 1])},
        "cv_choice": {"accuracy": chosen_accuracy,
                      "share_soft_only": float(np.mean([s[1] == "soft" for s in sets])),
                      "share_raw": float(np.mean([s[1] == "raw" for s in sets]))},
    })
# notebook-end


SPEC = {
    "chapter": 22,
    "chapter_title": "Vision and Multimodal: When You Need a VLM",
    "subtitle": "A pretrained embedding and its reduction are separate choices; explained variance does not say what task information is kept.",
    "summary": ("PCA keeps variance, not necessarily the directions that carry the label. One demonstration places the label signal at "
                "different places in an embedding's variance ranking and measures what a variance rule and a cross-validated choice each keep."),
    "title": "PCA on an embedding before a second stage: the 90% variance rule against cross-validation",
    "question": "If the label signal sits low in the embedding's variance ranking, what does a PCA that keeps 90% of the variance leave for the second stage?",
    "why": ("The chapter says PCA preserves variance under its fitted distribution, not necessarily task information, and that a percentage of lost "
            "predictive signal cannot be inferred from explained variance. Measuring a constructed case makes that concrete and shows the safer routine."),
    "method": ("Twenty-four constructed binary tasks per setting. A 40-dimensional embedding has directions with geometrically decaying variance. Five "
               "directions starting at the control's variance rank carry the class, each shifted by the same number of its own standard deviations, so "
               "the information is identical wherever the signal sits. A noisy soft label stands in for the VLM output. A logistic regression second stage "
               "is fitted on 600 rows and scored on 4,000 new rows with: the soft label alone, the soft label plus the raw embedding, the soft label plus k principal "
               "components (k from 2 to 32, PCA fitted on the training rows only), a 90% variance rule, and the candidate chosen by 5-fold cross-validation "
               "among all of these inside the 600 rows."),
    "control": {"key": "signal_rank", "label": "Variance rank where the label signal starts (0 is the largest-variance direction)",
                "values": [0, 10, 20, 30], "default": 20,
                "value_labels": ["0: largest-variance directions", "10", "20", "30: very low-variance directions"]},
    "source_section": "When VLMs Beat Traditional CV Pipelines",
    "symbols": ("lambda_j is the variance of embedding direction j (largest first), k the number of principal components kept and EV(k) "
                "the fraction of total variance they explain."),
    "explanation": ("Each principal component is ranked by the variance it explains, which is a property of the embedding alone and never "
                    "sees the label. When the signal sits among the top directions a few components keep it. When it sits lower, the components "
                    "that explain 90% of the variance leave it out and the second stage falls back to the soft label. Keeping every component "
                    "avoids the loss, and cross-validating the choice finds that without needing to know where the signal is."),
    "application": ("Treat the reduction as a candidate: compare no reduction, the soft outputs alone and a few component counts inside the "
                    "development folds, fit the projection on training rows only, and assess the frozen choice on untouched rows."),
    "assumptions": ("Constructed data with a linear, equal-strength signal in five directions, a logistic regression stand-in for the chapter's "
                    "LightGBM second stage, and an independent soft label. Real embeddings do not place their signal on convenient axes, and this "
                    "activity does not show that this mechanism caused the PCA loss in the Autopilot comparison, which the chapter says was not isolated."),
    "prediction": "With the signal starting at variance rank 20, how well does a second stage on a PCA that keeps 90% of the variance do, compared with the soft label alone?",
    "prediction_options": ["Clearly better than the soft label alone", "No better than the soft label alone", "Clearly worse than the soft label alone"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "It scores 0.695 against 0.707 for the soft label alone, while the raw embedding reaches 0.823.",
        "incorrect": "It scores 0.695 against 0.707 for the soft label alone, while the raw embedding reaches 0.823: the kept components carry none of the signal.",
    },
    "check": "The 90% rule keeps the same fraction of variance wherever the signal sits. Why does its accuracy change so much?",
    "answer": ("Explained variance counts how much the embedding varies along a direction, not whether the label varies with it. The signal here "
               "is equally strong at every rank, but the 90% rule keeps about 15 or 16 components, which contain all five signal directions only when the signal starts at rank 10 "
               "or earlier. Cross-validation measures what each candidate delivers on held-out rows, so it keeps the signal wherever it sits."),
    "provenance": "Constructed example: twenty-four seeded synthetic embeddings with a planted signal, PCA and logistic regression, measured by the chapter activity.",
    "apply": [
        "Compare the soft outputs alone, the raw embedding and a few PCA sizes inside the development folds before adding any reduction.",
        "Fit the projection on training rows only, inside each fold, and assess the frozen choice on rows it never saw.",
        "Never read explained variance as signal kept: a projection that keeps 90% of the variance can keep none of the label information.",
        "Record the comparison, including the failed reduction, as the chapter does for the Autopilot pipeline.",
    ],
    "honesty": ("Constructed data. Where the signal sits is a choice made by the generator; the activity measures what each rule does for "
                "each choice and does not claim real embeddings look like this."),
}

EQUATIONS = [{"tex": r"\mathrm{EV}(k) = \frac{\sum_{j=1}^{k} \lambda_j}{\sum_{j=1}^{D} \lambda_j}",
              "alt": "EV of k equals the sum of the k largest variances lambda j divided by the sum of all D variances",
              "basis": "The activity's explained-variance fraction for the PCA the chapter discusses in its opening and in the Autopilot callout; not a display equation in the manuscript."}]
NCOLS = 2
HEIGHT = 4.4


def draw(axes, result, parameter):
    left, right = axes
    ks = sorted(int(k) for k in result["pca_by_k"])
    acc = [result["pca_by_k"][str(k)] for k in ks]
    left.plot(range(len(ks)), acc, "o-", color=COLORS["teal"], lw=2, ms=6, label="Soft label + k components")
    left.axhline(result["soft_only"], color=COLORS["grey"], ls=(0, (4, 3)), lw=1.4, label=f"Soft label alone: {fmt(result['soft_only'])}")
    left.axhline(result["raw_embedding"], color=COLORS["gold"], ls=(0, (4, 3)), lw=1.4, label=f"Raw embedding: {fmt(result['raw_embedding'])}")
    left.set_xticks(range(len(ks)), [f"{k}\n({100 * result['explained_by_k'][str(k)]:.0f}%)" for k in ks], fontsize=10)
    left.set_ylim(0.55, 0.9)
    left.set_xlabel("Components kept (variance explained)")
    left.set_ylabel("Accuracy on 4,000 new rows")
    left.legend(loc="lower right", frameon=False, fontsize=10)
    names = ["Soft label\nalone", "90% variance\nrule", "CV choice", "Raw\nembedding"]
    vals = [result["soft_only"], result["variance_rule"]["accuracy"], result["cv_choice"]["accuracy"], result["raw_embedding"]]
    right.bar(range(4), vals, width=0.6, color=[COLORS["light"], COLORS["terracotta"], COLORS["teal"], COLORS["gold"]], edgecolor=COLORS["ink"], lw=0.6)
    for x, v in enumerate(vals):
        right.text(x, v + 0.004, fmt(v), ha="center", va="bottom", fontsize=10)
    right.set_xticks(range(4), names, fontsize=10)
    right.set_ylim(0.6, 0.9)
    right.set_xlabel(f"Second stage, signal starting at rank {parameter}")
    right.set_ylabel("Accuracy on 4,000 new rows")


def explain(result, parameter):
    soft, raw = result["soft_only"], result["raw_embedding"]
    rule, cv = result["variance_rule"], result["cv_choice"]
    kept = "carries the signal" if rule["accuracy"] > soft + 0.02 else "has lost the signal"
    interpretation = (
        f"With the signal starting at variance rank {parameter}, the 90% rule keeps about {fmt(rule['components'], 0)} components "
        f"({fmt(100 * rule['explained'], 0)}% of the variance) and scores {fmt(rule['accuracy'])}, against {fmt(soft)} for the soft label alone and {fmt(raw)} "
        f"for the raw embedding: {fmt(rule['accuracy'])} - {fmt(soft)} = {fmt(rule['accuracy'] - soft)}, so the reduced embedding {kept}. "
        f"Choosing by cross-validation scores {fmt(cv['accuracy'])}, and {fmt(cv['accuracy'])} - {fmt(rule['accuracy'])} = {fmt(cv['accuracy'] - rule['accuracy'])} "
        + ("is its margin over the rule." if cv["accuracy"] - rule["accuracy"] >= 0.0005 else
           "is within noise of the rule: here the rule happens to keep the signal."))
    steps = [
        f"Soft label alone: {fmt(soft)}. Raw embedding plus soft label: {fmt(raw)}.",
        f"90% variance rule: {fmt(rule['accuracy'])} with about {fmt(rule['components'], 0)} components.",
        f"Variance rule against soft label alone: {fmt(rule['accuracy'])} - {fmt(soft)} = {fmt(rule['accuracy'] - soft)}.",
        f"Cross-validated choice: {fmt(cv['accuracy'])}; it picked the soft label alone in {fmt(cv['share_soft_only'], 2)} of datasets.",
    ]
    metrics = {"Soft label alone": fmt(soft), "Raw embedding": fmt(raw), "90% variance rule": fmt(rule["accuracy"]),
               "Cross-validated choice": fmt(cv["accuracy"]), "Components in the 90% rule": fmt(rule["components"], 0)}
    alt = (f"Left, accuracy of a second stage using k principal components with the variance explained under each k, with dashed lines "
           f"for the soft label alone and the raw embedding; right, bars for four choices with the signal starting at rank {parameter}.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for rank, res in results.items():
        assert res["raw_embedding"] > res["soft_only"] + 0.07, f"raw embedding should add signal at rank {rank}"
        assert 14 <= res["variance_rule"]["components"] <= 18, "the 90% rule keeps about 16 components"
        assert res["cv_choice"]["accuracy"] > res["variance_rule"]["accuracy"] - 0.005, f"cv choice should not trail the rule at {rank}"
        assert res["cv_choice"]["accuracy"] > res["raw_embedding"] - 0.02, f"cv choice should be near the best at {rank}"
        assert all(0.85 < ev < 1.0 for k, ev in res["explained_by_k"].items() if int(k) >= 16), "16 or more components explain over 85%"
    for rank in (20, 30):
        res = results[rank]
        assert abs(res["variance_rule"]["accuracy"] - res["soft_only"]) < 0.02, f"90% rule should add nothing at rank {rank}"
        assert res["cv_choice"]["accuracy"] > res["variance_rule"]["accuracy"] + 0.08, f"cv choice should beat the rule at rank {rank}"
    assert results[0]["variance_rule"]["accuracy"] > results[0]["soft_only"] + 0.07, "rule keeps the signal at rank 0"
    assert results[10]["variance_rule"]["accuracy"] > results[10]["soft_only"] + 0.07, "rule keeps the signal at rank 10"
    mid = results[20]
    assert fmt(mid["variance_rule"]["accuracy"]) == "0.695" and fmt(mid["soft_only"]) == "0.707" and fmt(mid["raw_embedding"]) == "0.823", "prediction feedback numbers"
    assert results[10]["pca_by_k"]["16"] > results[10]["soft_only"] + 0.07 and results[20]["pca_by_k"]["16"] < results[20]["soft_only"] + 0.02, "k=16 reaches rank 10 not rank 20"
