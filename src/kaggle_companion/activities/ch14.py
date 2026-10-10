"""Chapter 14: What Doesn't Work. A K-means column added to boosted trees, against supplying the true grouping directly."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from kaggle_companion.activities._common import clean

SEED = 14
REPLICATES = 16              # independent constructed datasets; every estimate is their mean
GROUPS, DIM = 6, 8
N_TRAIN, N_NEW = 800, 6000
EFFECT_SD = 1.5              # how strongly the true group moves the log odds


def generate(spread, seed):
    """Six hidden groups in eight dimensions. The label depends on three features and on the hidden group."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, spread, (GROUPS, DIM))               # `spread` sets how far apart the groups sit
    effect = rng.normal(size=GROUPS)
    effect = (effect - effect.mean()) / effect.std() * EFFECT_SD   # group effect on the log odds
    weights = np.array([0.7, -0.5, 0.4])

    def rows(n):
        group = rng.integers(0, GROUPS, n)
        X = centers[group] + rng.normal(size=(n, DIM))
        y = (rng.random(n) < 1 / (1 + np.exp(-(X[:, :3] @ weights + effect[group])))).astype(int)
        return X, y, group

    return rows(N_TRAIN), rows(N_NEW)


def booster(**extra):
    return HistGradientBoostingClassifier(max_iter=60, learning_rate=0.1, max_leaf_nodes=8, min_samples_leaf=15,
                                          random_state=0, **extra)


def cluster_features(X_fit, *others):
    """Scale and cluster on training rows only; return the training columns and the same columns for other rows."""
    scaler = StandardScaler().fit(X_fit)
    km = KMeans(GROUPS, n_init=5, random_state=0).fit(scaler.transform(X_fit))
    def build(X):
        Z = scaler.transform(X)
        return np.column_stack([X, km.predict(Z), km.transform(Z)])   # raw columns, cluster id, distance to each centre
    return [build(X_fit)] + [build(X) for X in others], km, scaler


def run(spread):
    gain_learned, gain_known, recovery, stability, base_auc = [], [], [], [], []
    for rep in range(REPLICATES):
        (X, y, g), (X_new, y_new, g_new) = generate(spread, SEED * 100 + rep)
        score = lambda model, a, b: roc_auc_score(y_new, model.fit(a, y).predict_proba(b)[:, 1])
        base = score(booster(), X, X_new)
        (F, F_new), km, scaler = cluster_features(X, X_new)
        gain_learned.append(score(booster(categorical_features=[DIM]), F, F_new) - base)   # K-means id learned from training rows
        gain_known.append(score(booster(categorical_features=[DIM]),               # the true group id, supplied directly
                                np.column_stack([X, g]), np.column_stack([X_new, g_new])) - base)
        recovery.append(adjusted_rand_score(g_new, km.predict(scaler.transform(X_new))))   # known only because the data are constructed
        half = N_TRAIN // 2                                                         # observable: do two halves agree on new rows?
        labels = []
        for part in (slice(0, half), slice(half, N_TRAIN)):
            half_scaler = StandardScaler().fit(X[part])
            half_km = KMeans(GROUPS, n_init=5, random_state=0).fit(half_scaler.transform(X[part]))
            labels.append(half_km.predict(half_scaler.transform(X_new)))
        stability.append(adjusted_rand_score(labels[0], labels[1]))
        base_auc.append(base)
    return clean({
        "spread": spread, "baseline_auc": float(np.mean(base_auc)),
        "gain_learned": float(np.mean(gain_learned)), "gain_known": float(np.mean(gain_known)),
        "recovery": float(np.mean(recovery)), "stability": float(np.mean(stability)),
        "standard_error_learned": float(np.std(gain_learned) / np.sqrt(REPLICATES)),
        "per_replicate": {"gain_learned": gain_learned, "gain_known": gain_known},
        "replicates": REPLICATES, "train_rows": N_TRAIN, "new_rows": N_NEW,
    })
# notebook-end


SPEC = {
    "chapter": 14,
    "chapter_title": "What Doesn't Work",
    "subtitle": "A failed addition is a scoped observation: measure it against the unchanged baseline and say what it did not show.",
    "summary": ("K-means is a common feature idea that the chapter lists among additions that did not help. One demonstration adds a "
                "K-means column to boosted trees and measures it against the unchanged baseline and against the true grouping supplied directly."),
    "title": "A K-means column on boosted trees, against the true grouping supplied directly",
    "question": "When the hidden grouping matters to the label, how much of that value does a K-means column recover for a boosted-tree model?",
    "why": ("The chapter says K-means geometry need not align with the target, and asks for the addition to be compared with the original "
            "features and checked for stable assignments. This measures how that comparison comes out when the answer is known."),
    "method": ("Constructed binary task: six hidden groups in eight dimensions whose centres sit a distance set by the control, "
               "and a label that depends on three features and on the group (group effect standard deviation 1.5 log odds). One scikit-learn histogram "
               "gradient boosting model scores AUC on 6,000 new rows with the original features, with a K-means column and distances "
               "(scaling and clusters fitted on the 800 training rows, six clusters), and with the true group id supplied directly; "
               "both group ids enter as native categorical columns. "
               "Stability is the agreement (adjusted Rand index) between clusterings fitted on two halves of the training rows; "
               "recovery is the agreement with the true groups. Sixteen independent datasets are averaged."),
    "control": {"key": "spread", "label": "Spread of the group centres (standard deviations of the within-group noise)",
                "values": [0.4, 0.8, 1.2, 1.8], "default": 0.8,
                "value_labels": ["0.4: groups overlap heavily", "0.8", "1.2", "1.8: groups well separated"]},
    "source_section": "K-Means Clustering as a Feature",
    "symbols": ("Gain is the AUC on new rows of the model with the added column minus the AUC of the unchanged baseline; "
                "ARI is the adjusted Rand index, 1 for identical groupings and 0 for chance-level agreement."),
    "explanation": ("When the groups overlap, they matter a great deal to the label, but K-means cannot find them, so the column adds "
                    "nothing and the clusters are unstable. When the groups are well separated, K-means finds them, but boosted trees "
                    "already separate them from the raw features, so there is little left to add. Between the two, the column "
                    "recovers part of the grouping and adds a small gain."),
    "application": ("Fit scaling and clusters inside training, add the column to the unchanged baseline, and score both on the same "
                    "assessment. Check that two training subsets give the same clusters before trusting the column, and supply a "
                    "known grouping directly instead of re-deriving it."),
    "assumptions": ("Constructed Gaussian groups with equal sizes, six clusters chosen to match the true number, one scaling "
                    "choice (standardization) and a gradient boosting stand-in for LightGBM. Real data rarely has a true number of "
                    "clusters, and a different base model (for example a linear one) would gain more from cluster columns. The "
                    "K-means gains are only a few standard errors from zero, so the activity claims only that they stay small."),
    "prediction": "With group centres spread 0.8 apart, supplying the true groups would add about 0.07 AUC. How much does a K-means column add?",
    "prediction_options": ["About 0.07, most of it", "About 0.03", "Under 0.01"],
    "prediction_answer": 2,
    "prediction_feedback": {
        "correct": "The true groups add +0.069 AUC; the K-means column adds +0.004, about 7% of that.",
        "incorrect": "The true groups add +0.069 AUC; the K-means column adds +0.004. It matches the true groups at an adjusted Rand index of only 0.49, so most of the value is out of reach.",
    },
    "check": "Why does the K-means column help least where the grouping matters most, and why does it still add little where K-means finds the groups?",
    "answer": ("When the centres are close (0.4), the true groups add +0.143 AUC, but the groups overlap so much that K-means recovers almost "
               "none of them (adjusted Rand index 0.10) and the column adds -0.001. When the centres are far apart (1.8), K-means recovers them "
               "(0.96) but the boosted trees already separate them from the raw features, so even the true group id adds only +0.008. "
               "In between, the column recovers part of the grouping and adds at most about 0.01 AUC, well below the true grouping."),
    "provenance": "Constructed example: seeded Gaussian groups, K-means and scikit-learn histogram gradient boosting, measured by the chapter activity.",
    "apply": [
        "Compare the K-means addition with the unchanged baseline on the same assessment rows, and report the paired difference with its spread.",
        "Fit scaling and clusters inside the training partition, and check that clusters from two training subsets agree before keeping the column.",
        "If a natural grouping exists, supply it directly as a categorical column rather than hoping clustering rediscovers it.",
        "Write the failure down as a scoped observation: data, scaling, k, base model and the gain, not as a ban on clustering features.",
    ],
    "honesty": ("Constructed data. The K-means gain is at most about 0.01 AUC, a few standard errors above zero at the middle "
                "settings, so the ordering between settings is not claimed; only that it stays far below the value of the true grouping."),
}

EQUATIONS = [{"tex": r"\mathrm{gain} = \mathrm{AUC}_{\text{with column}} - \mathrm{AUC}_{\text{baseline}}",
              "alt": "gain equals AUC with the added column minus AUC of the unchanged baseline",
              "basis": "The activity's own label for the paired comparison the chapter asks for (K-Means Clustering as a Feature); the chapter has no display equation."}]
NCOLS = 2
HEIGHT = 4.4


def draw(axes, result, parameter):
    left, right = axes
    names = ["K-means column\n(learned)", "True group id\n(supplied)"]
    values = [result["gain_learned"], result["gain_known"]]
    dots = [result["per_replicate"]["gain_learned"], result["per_replicate"]["gain_known"]]
    left.bar([0, 1], values, width=0.5, color=[COLORS["terracotta"], COLORS["teal"]], edgecolor=COLORS["ink"], lw=0.6)
    for x, d in enumerate(dots):
        left.scatter([x + (i - (len(d) - 1) / 2) * 0.025 for i in range(len(d))], d, s=10, color=COLORS["ink"], zorder=3)
        left.text(x, max([values[x]] + d) + 0.006, signed(values[x]), ha="center", va="bottom", fontsize=10)
    left.axhline(0, color=COLORS["grey"], lw=0.8)
    left.set_xticks([0, 1], names)
    low = min(min(d) for d in dots)
    left.set_ylim(min(low, -0.01) - 0.01, max(max(d) for d in dots) + 0.05)
    left.set_ylabel("AUC gain over the baseline (new rows)")
    left.set_xlabel(f"Added column, centre spread {parameter} (dots: one dataset each)")

    right.bar([0, 1], [result["stability"], result["recovery"]], width=0.5, color=[COLORS["navy"], COLORS["gold"]],
              edgecolor=COLORS["ink"], lw=0.6)
    for x, v in enumerate([result["stability"], result["recovery"]]):
        right.text(x, v + 0.02, fmt(v, 2), ha="center", va="bottom", fontsize=10)
    right.set_xticks([0, 1], ["Stability\n(two halves agree)", "Recovery\n(true groups)"])
    right.set_ylim(0, 1.1)
    right.set_ylabel("Adjusted Rand index (1 = identical)")
    right.set_xlabel("K-means assignments on new rows")


def diff(a, b):
    """Difference of the displayed (3 decimal) values, so the hand calculation in the text adds up."""
    return fmt(round(a, 3) - round(b, 3))


def sub(x):
    """A subtrahend for a hand calculation: a negative value is written in parentheses."""
    return fmt(x) if round(x, 3) >= 0 else f"({fmt(x)})"


def explain(result, parameter):
    l, k = result["gain_learned"], result["gain_known"]
    se = result["standard_error_learned"]
    share = ("none of it" if l <= 0.002 else f"{round(100 * l / k)}% of it" if k > 0.002 else "a small gain")
    interpretation = (
        f"With centre spread {parameter}, the true group id adds {signed(k)} AUC to the baseline of {fmt(result['baseline_auc'])}. "
        f"The K-means column adds {signed(l)} (standard error {fmt(se)}), which is {share}, so {fmt(k)} - {sub(l)} = {diff(k, l)} of the "
        f"value stays out of reach. Two halves of the training rows agree on the clusters at ARI {fmt(result['stability'], 2)}, and "
        f"the clusters match the true groups at {fmt(result['recovery'], 2)}.")
    steps = [
        f"Value of the true grouping: {fmt(result['baseline_auc'] + k)} - {fmt(result['baseline_auc'])} = {diff(result['baseline_auc'] + k, result['baseline_auc'])} AUC.",
        f"Value of the K-means column: {fmt(result['baseline_auc'] + l)} - {fmt(result['baseline_auc'])} = {diff(result['baseline_auc'] + l, result['baseline_auc'])} AUC (standard error {fmt(se)}).",
        f"Left out of reach: {fmt(k)} - {sub(l)} = {diff(k, l)} AUC.",
    ]
    metrics = {"Baseline AUC": fmt(result["baseline_auc"]), "K-means column gain": signed(l), "True group id gain": signed(k),
               "Stability (ARI)": fmt(result["stability"], 2), "Recovery (ARI)": fmt(result["recovery"], 2)}
    alt = (f"Left: AUC gain over the baseline at centre spread {parameter}, {signed(l)} for the K-means column and {signed(k)} for the "
           f"true group id, with a dot per dataset. Right: adjusted Rand index for clustering stability {fmt(result['stability'], 2)} "
           f"and recovery of the true groups {fmt(result['recovery'], 2)}.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    order = sorted(results)
    for s in order:
        assert results[s]["gain_learned"] < 0.012, f"K-means gain should stay small at {s}"
        assert results[s]["gain_known"] > results[s]["gain_learned"] - 0.003, f"true grouping should be worth at least as much at {s}"
    for a, b in zip(order, order[1:]):
        assert results[b]["recovery"] > results[a]["recovery"], "recovery should rise with spread"
        assert results[b]["gain_known"] < results[a]["gain_known"], "the true grouping matters less as trees can find it"
    assert results[0.4]["gain_known"] > 0.1 and results[0.4]["recovery"] < 0.25, "overlap: grouping matters, K-means cannot find it"
    assert results[0.4]["gain_learned"] < 0.005, "overlap: K-means column adds nothing"
    assert results[1.8]["recovery"] > 0.9 and results[1.8]["gain_known"] < 0.02, "separated: found, but trees already have it"
    mid = results[0.8]
    assert mid["gain_known"] > 0.05 and mid["gain_learned"] < 0.01, "prediction option: under 0.01 against about 0.07"
    assert fmt(mid["gain_known"]) == "0.069" and fmt(mid["gain_learned"]) == "0.004" and fmt(mid["recovery"], 2) == "0.49", "feedback"
    assert round(100 * mid["gain_learned"] / mid["gain_known"]) == 7, "feedback says about 7%"
    assert fmt(results[0.4]["gain_known"]) == "0.143" and fmt(results[0.4]["recovery"], 2) == "0.10", "check answer numbers"
    assert fmt(results[0.4]["gain_learned"]) == "-0.001", "check answer numbers"
    assert fmt(results[1.8]["recovery"], 2) == "0.96" and fmt(results[1.8]["gain_known"]) == "0.008", "check answer numbers"
