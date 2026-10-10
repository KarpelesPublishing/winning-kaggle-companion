"""Chapter 11: When Feature Engineering Fails. Fold-fitted PCA on a constructed embedding: variance kept against label kept."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from kaggle_companion.activities._common import clean

SEED = 11
REPLICATES = 12                      # independent constructed embeddings; every estimate is their mean
DIM, N_TRAIN, N_NEW = 64, 600, 6000
GRID = [1, 2, 4, 8, 12, 16, 24, 64]  # component counts for the variance-against-AUC curve


def make_rows(rng, n, scales, signal, rotation):
    """Rows of a 64-dimensional 'embedding'. The label depends on two coordinates (`signal`), then everything is rotated."""
    z = rng.normal(size=(n, DIM)) * scales
    s = z[:, signal] / scales[signal]                       # the two signal coordinates, standardized
    y = (rng.random(n) < 1 / (1 + np.exp(-1.6 * (s[:, 0] + s[:, 1]) / np.sqrt(2)))).astype(int)
    return z @ rotation.T, y


def one_embedding(signal_is_high_variance, seed):
    """Fit on 600 rows, score on 6,000 new rows. Everything fitted (PCA, LDA, logistic regression) sees training rows only."""
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(DIM, DIM)))
    scales = np.full(DIM, 0.8)                              # 54 weak background directions
    scales[:8] = 8.0                                        # 8 strong directions that carry no label
    if signal_is_high_variance:
        scales[:2], signal = 12.0, [0, 1]                   # world B: the label sits in the two strongest directions
    else:
        scales[8:10], signal = 1.2, [8, 9]                  # world A: the label sits just below the 8 strong directions
    X, y = make_rows(rng, N_TRAIN, scales, signal, rotation)
    X_new, y_new = make_rows(rng, N_NEW, scales, signal, rotation)
    pca = PCA(DIM).fit(X)                                   # fitted on training rows only
    train, new = pca.transform(X), pca.transform(X_new)
    kept = np.cumsum(pca.explained_variance_ratio_)
    auc = {}
    for k in GRID:
        model = LogisticRegression(max_iter=500).fit(train[:, :k], y)
        auc[k] = roc_auc_score(y_new, model.decision_function(new[:, :k]))
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)   # supervised, fitted inside training
    return {"kept": {k: kept[k - 1] for k in GRID}, "auc": auc, "lda": roc_auc_score(y_new, lda.decision_function(X_new)),
            "all": auc[DIM]}                                # k = 64 keeps every direction: the unreduced regularized model


def run(components):
    world_a = [one_embedding(False, SEED * 100 + i) for i in range(REPLICATES)]
    world_b = [one_embedding(True, SEED * 100 + 50 + i) for i in range(REPLICATES)]

    def mean(runs, getter):
        return float(np.mean([getter(r) for r in runs]))

    def curve(runs):
        return [[k, mean(runs, lambda r: r["kept"][k]), mean(runs, lambda r: r["auc"][k])] for k in GRID]

    return clean({
        "components": components,
        "variance_kept": mean(world_a, lambda r: r["kept"][components]),
        "label_low_variance": {"pca": mean(world_a, lambda r: r["auc"][components]), "all": mean(world_a, lambda r: r["all"]),
                               "lda": mean(world_a, lambda r: r["lda"]),
                               "per_replicate": [r["auc"][components] for r in world_a]},
        "label_high_variance": {"pca": mean(world_b, lambda r: r["auc"][components]), "all": mean(world_b, lambda r: r["all"]),
                                "lda": mean(world_b, lambda r: r["lda"]),
                                "variance_kept": mean(world_b, lambda r: r["kept"][components]),
                                "per_replicate": [r["auc"][components] for r in world_b]},
        "curve_low": curve(world_a), "curve_high": curve(world_b),
        "replicates": REPLICATES, "train_rows": N_TRAIN, "new_rows": N_NEW, "dimension": DIM,
    })
# notebook-end


SPEC = {
    "chapter": 11,
    "chapter_title": "When Feature Engineering Fails",
    "subtitle": "Diagnose the representation, boundary or selection dependency before changing the algorithm.",
    "summary": ("PCA keeps the directions with the most variance, which need not be the directions that carry the label. One "
                "demonstration keeps k components of a constructed embedding and measures variance kept against AUC on new rows."),
    "title": "How much variance PCA keeps, and how much label it keeps",
    "question": "How many principal components does a reduction need before it keeps the label, and what does the variance kept say about that?",
    "why": ("A reduction that keeps 93% of the variance sounds safe. The share of variance says nothing about where the label "
            "sits, so the check is the downstream score of the whole reduced pipeline, against the unreduced representation."),
    "method": ("Constructed 64-dimensional embeddings, 600 training rows and 6,000 new rows, twelve independent embeddings per world. "
               "Eight strong directions carry no label. In world A the label sits in two weaker directions just below them; in "
               "world B it sits in the two strongest directions. PCA is fitted on training rows, the first k components feed a "
               "logistic regression, and AUC is measured on the new rows. The unreduced 64-dimension logistic regression and a "
               "shrinkage LDA fitted inside training are the comparators."),
    "control": {"key": "components", "label": "Principal components kept (k)",
                "values": [2, 8, 12, 24], "default": 8,
                "value_labels": ["2", "8: the strong directions", "12", "24"]},
    "source_section": "Failure 1: PCA on Pretrained Features",
    "symbols": ("lambda_j is the variance of the j-th principal component, k the number of components kept, and V(k) the share of "
                "total variance those k components keep."),
    "explanation": ("Components are ordered by variance, not by use. When the label lives in weak directions, the first eight "
                    "components capture almost all the variance and none of the label, and the model scores about chance. Once k "
                    "reaches the label's variance rank the score jumps. When the label sits in the strongest directions, PCA "
                    "is fine from the start. The same reduction fails in one world and works in the other."),
    "application": ("Compare the unreduced representation with fold-fitted PCA at several k, scored on the same assessment. "
                    "Keep any supervised reduction inside the training boundary, and record the failure as a note about this "
                    "artifact, not a rule against reduction."),
    "assumptions": ("Constructed embeddings with a label that depends on exactly two directions, a logistic regression, and PCA "
                    "fitted on 600 rows. The chapter's failed reduction used a real pretrained artifact that is not reproduced "
                    "here. In this generator a reduction that keeps the label also beats the unreduced model, because it drops "
                    "52 directions that only add estimation noise; real embeddings need not behave that way."),
    "prediction": "With 8 principal components, which keep about 93% of the variance, what AUC does the model reach when the label sits in weaker directions?",
    "prediction_options": ["About 0.78, close to the unreduced model", "About 0.65", "About 0.51, chance level"],
    "prediction_answer": 2,
    "prediction_feedback": {
        "correct": "PCA(8) keeps 0.933 of the variance and scores 0.507; the unreduced model scores 0.784.",
        "incorrect": "PCA(8) keeps 0.933 of the variance and scores 0.507, chance level; the unreduced model scores 0.784.",
    },
    "check": "Why does PCA with 12 components beat the unreduced model here, and what does that say about banning PCA?",
    "answer": ("Twelve components include the two label directions and drop most of the 52 directions that carry no label, which an "
               "unreduced fit has to estimate from 600 rows (0.806 against 0.784). The same reduction scores chance at 8 components, "
               "so the verdict belongs to this artifact and this k, as the chapter says: it is a comparison to run, not a ban."),
    "provenance": "Constructed example: seeded synthetic embeddings, PCA, LDA and logistic regression, measured by the chapter activity.",
    "apply": [
        "Score the unreduced representation with a regularized model first; it is the comparator every reduction must beat.",
        "Fit PCA inside the training boundary and compare several k on the same assessment, instead of picking k by variance kept.",
        "Treat variance kept as a size report, never as evidence that the label survived.",
        "Write the failure down with its artifact, k and score, and name the next test (more components, a supervised reduction fitted inside training).",
    ],
    "honesty": ("Constructed embeddings; where the label sits is chosen by the generator, and the sizes of these effects are "
                "properties of it, not of any pretrained model."),
}

EQUATIONS = [{"tex": r"V(k) = \frac{\sum_{j=1}^{k} \lambda_j}{\sum_{j=1}^{64} \lambda_j}",
              "alt": "V of k equals the sum of lambda j for j from 1 to k, divided by the sum of lambda j for j from 1 to 64",
              "basis": "The share of variance kept, the activity's own measure; the chapter has no display equation (Failure 1: PCA on Pretrained Features)."}]
NCOLS = 2
HEIGHT = 4.4


def draw(axes, result, parameter):
    left, right = axes
    for key, name, color in (("curve_low", "Label in weaker directions (world A)", COLORS["terracotta"]),
                             ("curve_high", "Label in strongest directions (world B)", COLORS["teal"])):
        pts = result[key]
        left.plot([100 * p[1] for p in pts], [p[2] for p in pts], marker="o", ms=4, lw=1.4, color=color, label=name)
    k = result["components"]
    for key, color in (("curve_low", COLORS["terracotta"]), ("curve_high", COLORS["teal"])):
        point = [p for p in result[key] if p[0] == k][0]
        left.scatter([100 * point[1]], [point[2]], s=80, facecolor="none", edgecolor=COLORS["ink"], lw=1.6, zorder=4)
    left.axhline(result["label_low_variance"]["all"], color=COLORS["gold"], ls=(0, (4, 3)), lw=1.4,
                 label=f"Unreduced 64 dimensions: {fmt(result['label_low_variance']['all'])}")
    left.axhline(0.5, color=COLORS["grey"], lw=0.6)
    left.set_xlabel("Variance kept by the first k components (%), ringed: current k")
    left.set_ylabel("AUC on new rows")
    left.set_ylim(0.15, 0.9)
    left.legend(loc="lower left", frameon=False, fontsize=10)

    a, b = result["label_low_variance"], result["label_high_variance"]
    names = [f"PCA({k})\nworld A", f"PCA({k})\nworld B", "Unreduced\n(world A)", "LDA\n(world A)"]
    values = [a["pca"], b["pca"], a["all"], a["lda"]]
    colors = [COLORS["terracotta"], COLORS["teal"], COLORS["light"], COLORS["navy"]]
    right.bar(range(4), values, width=0.6, color=colors, edgecolor=COLORS["ink"], lw=0.6)
    for x, (v, dots) in enumerate(zip(values, [a["per_replicate"], b["per_replicate"], None, None])):
        if dots:
            right.scatter([x + (i - (len(dots) - 1) / 2) * 0.035 for i in range(len(dots))], dots, s=10, color=COLORS["ink"], zorder=3)
        right.text(x, max([v] + (dots or [])) + 0.012, fmt(v), ha="center", va="bottom", fontsize=10)
    right.axhline(0.5, color=COLORS["grey"], lw=0.6)
    right.set_xticks(range(4), names)
    right.set_ylim(0.4, 1.0)
    right.set_ylabel("AUC on new rows")
    right.set_xlabel("Representation (dots: one embedding each)")


def diff(a, b):
    """Difference of the displayed (3 decimal) values, so the hand calculation in the text adds up."""
    return fmt(round(a, 3) - round(b, 3))


def explain(result, parameter):
    a, b, v = result["label_low_variance"], result["label_high_variance"], result["variance_kept"]
    k = result["components"]
    verdict = ("is no better than chance" if a["pca"] < 0.55 else
               f"beats the unreduced model by {diff(a['pca'], a['all'])}" if a["pca"] > a["all"] + 0.005 else
               "matches the unreduced model")
    interpretation = (
        f"With {k} components the reduction keeps {fmt(100 * v, 1)}% of the variance in world A ({fmt(100 * b['variance_kept'], 1)}% in world B). When the label sits in weaker directions it "
        f"scores {fmt(a['pca'])} AUC on new rows, which {verdict}; the unreduced model scores {fmt(a['all'])}, "
        f"so {fmt(a['pca'])} - {fmt(a['all'])} = {diff(a['pca'], a['all'])}. When the label sits in the strongest directions the same "
        f"reduction scores {fmt(b['pca'])}. A shrinkage LDA fitted inside training scores {fmt(a['lda'])} without choosing k.")
    steps = [
        f"Variance kept by {k} components, world A: {fmt(100 * v, 1)}%.",
        f"World A, PCA minus unreduced: {fmt(a['pca'])} - {fmt(a['all'])} = {diff(a['pca'], a['all'])} AUC.",
        f"World B, PCA minus unreduced: {fmt(b['pca'])} - {fmt(b['all'])} = {diff(b['pca'], b['all'])} AUC.",
        f"World A, LDA minus PCA: {fmt(a['lda'])} - {fmt(a['pca'])} = {diff(a['lda'], a['pca'])} AUC.",
    ]
    metrics = {"Components kept": str(k), "Variance kept (world A)": f"{fmt(100 * v, 1)}%", "PCA, label in weaker directions": fmt(a["pca"]),
               "PCA, label in strongest directions": fmt(b["pca"]), "Unreduced, 64 dimensions": fmt(a["all"])}
    alt = (f"Left: AUC against variance kept for PCA with the label in weaker or strongest directions, with the unreduced model as a "
           f"dashed line. Right: AUC at {k} components, {fmt(a['pca'])} in world A and {fmt(b['pca'])} in world B, against the "
           f"unreduced {fmt(a['all'])} and LDA {fmt(a['lda'])}.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for k, res in results.items():
        a, b = res["label_low_variance"], res["label_high_variance"]
        assert res["variance_kept"] > 0.9 or k == 2, f"variance kept should be high at {k}"
        assert b["pca"] > b["all"] + 0.01 or k == 2, f"world B reduction should help at {k}"
    r2, r8, r12, r24 = results[2], results[8], results[12], results[24]
    assert r8["variance_kept"] > 0.9 and abs(r8["label_low_variance"]["pca"] - 0.5) < 0.03, "k=8 keeps variance, loses label"
    assert r2["label_high_variance"]["pca"] > r2["label_high_variance"]["all"], "k=2 should already work in world B"
    for res in (r12, r24):
        a = res["label_low_variance"]
        assert a["pca"] > 0.7 and a["pca"] > a["all"] + 0.01, "reduction that keeps the label should beat unreduced"
    assert r8["label_low_variance"]["all"] > 0.7, "the unreduced model should keep the label"
    assert fmt(r8["variance_kept"]) == "0.933" and fmt(r8["label_low_variance"]["pca"]) == "0.507", "prediction feedback numbers"
    assert fmt(r8["label_low_variance"]["all"]) == "0.784", "prediction feedback numbers"
    assert fmt(r12["label_low_variance"]["pca"]) == "0.806", "check answer numbers"
