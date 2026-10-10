"""Chapter 29: Advanced Feature Selection. Null importance against a raw importance ranking, by null percentile."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from kaggle_companion.activities._common import clean

SEED = 29
REPLICATES = 6          # independent constructed datasets; every estimate is their mean
PERMUTATIONS = 12       # label permutations that make the null importance of each feature
N_TRAIN, N_HELD = 700, 4000
# Column groups, in order: strong signal, weak signal (binary), noise, binary noise, ID-like noise.
GROUPS = [("strong", 3), ("weak", 4), ("noise", 12), ("binary_noise", 4), ("id_like", 2)]
KIND = np.array([name for name, count in GROUPS for _ in range(count)])
N_FEATURES = len(KIND)


def generate(n, rng):
    """Three continuous strong features, four weak binary features, and 18 columns that carry nothing:
    12 continuous, 4 binary and 2 integer ID-like columns with about n/4 distinct values."""
    strong = rng.normal(size=(n, 3))
    weak = (rng.random((n, 4)) < 0.5).astype(float)
    noise = rng.normal(size=(n, 12))
    binary_noise = (rng.random((n, 4)) < 0.5).astype(float)
    id_like = rng.integers(0, n // 4, (n, 2)).astype(float)
    logit = 0.9 * strong.sum(axis=1) + 0.7 * (weak.sum(axis=1) - 2)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return np.column_stack([strong, weak, noise, binary_noise, id_like]), y


def forest(seed):
    return RandomForestClassifier(n_estimators=30, min_samples_leaf=3, max_features="sqrt", random_state=seed, n_jobs=1)


def importance(X, y, seed):
    """Impurity importance: the share of split gain each column earns. It is biased toward columns with many distinct values."""
    return forest(seed).fit(X, y).feature_importances_


def held_out_auc(columns, X, y, X_held, y_held, seed):
    model = forest(seed).fit(X[:, columns], y)
    return roc_auc_score(y_held, model.predict_proba(X_held[:, columns])[:, 1])


def one_replicate(percentile, seed):
    rng = np.random.default_rng(seed)
    X, y = generate(N_TRAIN, rng)
    X_held, y_held = generate(N_HELD, rng)
    real = importance(X, y, seed)
    # Null: refit after shuffling the labels, so every column's importance is what chance alone earns it.
    null = np.array([importance(X, rng.permutation(y), seed + 1 + i) for i in range(PERMUTATIONS)])
    threshold = np.percentile(null, percentile, axis=0)          # each column is compared with its own null
    kept = np.where(real > threshold)[0]
    raw_top = np.argsort(-real)[:len(kept)]                       # same number of columns, chosen by raw importance
    everything = np.arange(N_FEATURES)
    return {"real": real, "threshold": threshold, "kept": kept, "raw_top": raw_top,
            "auc_all": held_out_auc(everything, X, y, X_held, y_held, seed),
            "auc_null": held_out_auc(kept, X, y, X_held, y_held, seed) if len(kept) else 0.5,
            "auc_raw": held_out_auc(raw_top, X, y, X_held, y_held, seed) if len(kept) else 0.5}


def run(percentile):
    reps = [one_replicate(percentile, SEED * 100 + i) for i in range(REPLICATES)]
    count = lambda key, kind: float(np.mean([np.sum(KIND[r[key]] == kind) for r in reps]))
    first = reps[0]
    return clean({
        "percentile": percentile,
        "kept_null": {name: count("kept", name) for name, _ in GROUPS},
        "kept_raw": {name: count("raw_top", name) for name, _ in GROUPS},
        "columns_kept": float(np.mean([len(r["kept"]) for r in reps])),
        "auc_all": float(np.mean([r["auc_all"] for r in reps])),
        "auc_null": float(np.mean([r["auc_null"] for r in reps])),
        "auc_raw": float(np.mean([r["auc_raw"] for r in reps])),
        "per_replicate_auc": {k: [r["auc_" + k] for r in reps] for k in ("all", "null", "raw")},
        "first_dataset": {"kind": KIND.tolist(), "real": first["real"], "threshold": first["threshold"],
                          "kept": [int(i in first["kept"]) for i in range(N_FEATURES)]},
        "replicates": REPLICATES, "permutations": PERMUTATIONS, "columns": N_FEATURES, "training_rows": N_TRAIN,
    })
# notebook-end


SPEC = {
    "chapter": 29,
    "chapter_title": "Advanced Feature Selection",
    "subtitle": "Compare each importance with its own label-permutation null before screening columns out.",
    "summary": ("Raw importance favours columns with many distinct values, and 'drop the bottom N' is arbitrary. One demonstration "
                "compares a raw importance ranking with a null-importance screen on a table where the truth is known, across "
                "null percentiles, and scores the selected columns on new rows."),
    "title": "Raw importance against null importance, by null percentile",
    "question": "Which columns does each screen keep, and at what null percentile does a strict screen start discarding real signal?",
    "why": ("Feature selection is a fitted procedure whose output is a smaller table. Knowing which columns it keeps, and what "
            "that costs on new rows, is the only way to tell a useful screen from an arbitrary one."),
    "method": ("Six constructed datasets of 700 rows with 25 columns whose roles are known: 3 continuous strong features, 4 weak binary "
               "features, and 18 columns that carry nothing (12 continuous, 4 binary, 2 integer ID-like with about 175 values). A random "
               "forest gives each column's impurity importance. The null is the same importance after shuffling the labels, "
               "repeated 12 times. A column is kept when its real importance exceeds its own null percentile (the control). "
               "The same number of columns is also chosen by raw importance rank, and all three tables (everything, null screen, raw "
               "top columns) are refitted and scored by AUC on 4,000 new rows."),
    "control": {"key": "percentile", "label": "Null percentile a column's importance must exceed",
                "values": [50, 75, 90, 99], "default": 75,
                "value_labels": ["50: the null median", "75", "90", "99: above nearly every shuffle"]},
    "source_section": "Null Importance: Distinguishing Real Signal from Chance Correlation",
    "symbols": ("I_j is column j's importance on the real labels, N_j^(q) the q-th percentile of its importance over the "
                "label permutations, and a column is kept when I_j exceeds N_j^(q). The percentile rule is a screening heuristic, "
                "not a probability that the column is real."),
    "explanation": ("A column with many distinct values offers a tree many places to split, so it earns importance even when it carries "
                    "nothing: here the noise and ID-like columns outrank the weak binary features that really matter. Shuffling the "
                    "labels measures what each column earns by chance, so comparing a column with its own null removes that "
                    "advantage. Raising the percentile removes more chance winners but starts to cut weak real columns."),
    "application": ("Fit the screen on development rows only, compare each column with its own permutation null, choose the percentile by "
                    "held-out loss rather than by habit, and assess the selected table separately. Keep a raw-rank selection of the "
                    "same size as the comparison."),
    "assumptions": ("Constructed data and one forest as the importance model; the chapter's BorutaShap and LightGBM are not installed. "
                    "With only 12 permutations the 99th percentile is close to the largest null value. In this generator the real "
                    "labels also shrink the importance of the noise columns below their null, so the null screen is conservative; "
                    "differences in AUC between percentiles are small and are properties of this generator."),
    "prediction": "A raw-importance ranking and a 75th-percentile null screen each keep the same number of columns. Which table scores higher on new rows?",
    "prediction_options": ["The raw ranking, because the strongest columns lead it",
                           "They score the same, because both keep the strong features",
                           "The null screen, because the raw ranking admits high-value noise columns and drops weak real ones"],
    "prediction_answer": 2,
    "prediction_feedback": {
        "correct": "The null screen scores 0.786 against 0.759 for the raw ranking, which keeps noise columns in place of weak binary ones.",
        "incorrect": "The null screen scores 0.786 against 0.759 for the raw ranking: the raw ranking keeps continuous and ID-like noise columns in place of weak binary ones.",
    },
    "check": "The 99th percentile keeps fewer noise columns than the median. Why does it score lower on new rows?",
    "answer": ("At the 50th percentile 0.7 noise columns survive on average along with 3.3 of the 4 weak real columns, and the screen scores "
               "0.792; at the 99th the weak real columns fall to 2.0 kept and the score drops to 0.780. A stricter rule removes chance "
               "winners and real but weak columns together, and here the weak columns were worth more than the noise cost, so the "
               "percentile is a choice to validate, not a constant."),
    "provenance": "Constructed example: six seeded synthetic datasets, a random forest and twelve label permutations, measured by the chapter activity.",
    "apply": [
        "Compute each feature's importance on the real labels and on shuffled labels, then compare a feature with its own null, not with other features.",
        "Do the whole screen inside development rows, with a permutation that respects groups or time if the data has them.",
        "Choose the percentile by held-out loss on matched folds and keep the same-size raw-rank selection as a baseline.",
        "Before removing a column, also test removing it with its correlated substitutes, and log each removal result.",
    ],
    "honesty": "Constructed data where the true role of every column is known; the sizes of these effects are properties of this generator, not a competition result.",
}

EQUATIONS = [{"tex": r"\text{keep } j \iff I_j > N_j^{(q)}",
              "alt": "keep column j if and only if its importance I j exceeds the q-th percentile of its null importance N j",
              "basis": "The percentile rule the chapter describes in words (Null Importance); written as a rule by the activity."}]
NCOLS = 2
HEIGHT = 4.5
TYPE_NAMES = {"strong": "Strong", "weak": "Weak\n(binary)", "noise": "Noise", "binary_noise": "Binary\nnoise", "id_like": "ID-like\nnoise"}
TYPE_COLORS = {"strong": COLORS["teal"], "weak": COLORS["gold"], "noise": COLORS["grey"],
               "binary_noise": COLORS["light"], "id_like": COLORS["terracotta"]}


def draw(axes, result, parameter):
    ax, ax2 = axes
    d = result["first_dataset"]
    kind, real, thr, kept = d["kind"], d["real"], d["threshold"], d["kept"]
    centers, labels = [], []
    start = 0
    for name, count in GROUPS:
        centers.append(start + (count - 1) / 2)
        labels.append(TYPE_NAMES[name])
        start += count
    for i in range(len(kind)):
        ax.plot([i - 0.35, i + 0.35], [thr[i], thr[i]], color=COLORS["ink"], lw=1.4, zorder=2,
                label="Null threshold" if i == 0 else None)
        ax.scatter([i], [real[i]], s=46, color=TYPE_COLORS[kind[i]] if kept[i] else "white",
                   edgecolor=TYPE_COLORS[kind[i]] if kind[i] != "binary_noise" else COLORS["grey"], lw=1.6, zorder=3)
    ax.scatter([], [], s=46, color=COLORS["ink"], label="Kept (filled)")
    ax.scatter([], [], s=46, color="white", edgecolor=COLORS["ink"], lw=1.6, label="Dropped (hollow)")
    ax.set_xticks(centers, labels, fontsize=10)
    ax.set_ylabel("Impurity importance (first dataset)")
    ax.set_xlabel("Column group")
    ax.set_ylim(0, max(max(real), max(thr)) * 1.45)
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    names = [("all", "All 25\ncolumns"), ("raw", "Raw top-k"), ("null", f"Null screen,\n{parameter}th pct.")]
    colors = [COLORS["light"], COLORS["gold"], COLORS["teal"]]
    values = [result["auc_" + k] for k, _ in names]
    pos = np.arange(3)
    ax2.bar(pos, values, width=0.55, color=colors, edgecolor=COLORS["ink"], lw=0.6)
    for p, (k, _) in zip(pos, names):
        dots = result["per_replicate_auc"][k]
        ax2.scatter([p + (i - (len(dots) - 1) / 2) * 0.05 for i in range(len(dots))], dots, s=14, color=COLORS["ink"], zorder=3,
                    label="One dataset" if p == 0 else None)
        ax2.text(p, max(max(dots), result["auc_" + k]) + 0.006, fmt(result["auc_" + k]), ha="center", va="bottom", fontsize=10)
    ax2.set_xticks(pos, [n for _, n in names])
    ax2.set_ylim(0.65, 0.88)
    ax2.set_ylabel("AUC on 4,000 new rows")
    ax2.set_xlabel(f"Table fitted, top-k = {fmt(result['columns_kept'], 1)} columns")
    ax2.legend(loc="upper left", frameon=False, fontsize=10)


def _kept_text(counts):
    parts = [f"{fmt(counts['strong'], 1)} of 3 strong", f"{fmt(counts['weak'], 1)} of 4 weak"]
    false_keeps = counts["noise"] + counts["binary_noise"] + counts["id_like"]
    parts.append(f"{fmt(false_keeps, 1)} of 18 noise")
    return ", ".join(parts)


def explain(result, parameter):
    a, n, w = result["auc_all"], result["auc_null"], result["auc_raw"]
    kn, kr = result["kept_null"], result["kept_raw"]
    gain = float(fmt(n)) - float(fmt(a))
    interpretation = (
        f"At the {parameter}th null percentile the screen keeps {fmt(result['columns_kept'], 1)} columns on average: {_kept_text(kn)}. "
        f"The same number chosen by raw importance rank holds {_kept_text(kr)}. Scored on new rows the null screen reaches {fmt(n)}, the raw "
        f"top columns {fmt(w)} and the full table {fmt(a)}, so {fmt(n)} - {fmt(a)} = {fmt(gain)} is the change from screening with the null "
        f"(negative means it cost AUC).")
    steps = [
        f"Null screen: {fmt(n)} - {fmt(a)} = {fmt(gain)} AUC against keeping every column.",
        f"Raw ranking of the same size: {fmt(w)} - {fmt(a)} = {fmt(float(fmt(w)) - float(fmt(a)))}.",
        f"Kept by the null screen: {_kept_text(kn)}.",
        f"Kept by the raw ranking: {_kept_text(kr)}.",
    ]
    metrics = {"Columns kept": fmt(result["columns_kept"], 1), "Weak real kept (of 4)": fmt(kn["weak"], 1),
               "Noise kept (of 18)": fmt(kn["noise"] + kn["binary_noise"] + kn["id_like"], 1),
               "Null screen AUC": fmt(n), "Raw top-k AUC": fmt(w), "All columns AUC": fmt(a)}
    alt = (f"Left: real impurity importance of 25 columns in one dataset with a null-threshold tick for each, filled when kept at the "
           f"{parameter}th percentile. Right: AUC on new rows for all columns ({fmt(a)}), a raw top-k ranking ({fmt(w)}) and the null screen ({fmt(n)}).")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for q, res in results.items():
        assert res["kept_null"]["strong"] > 2.9, f"strong columns should survive at {q}"
        assert res["auc_null"] - res["auc_raw"] > 0.01, f"null screen should beat raw ranking at {q}"
        false_keeps = res["kept_null"]["noise"] + res["kept_null"]["binary_noise"] + res["kept_null"]["id_like"]
        assert false_keeps < 2.0, f"null screen should keep few noise columns at {q}"
        raw_noise = res["kept_raw"]["noise"] + res["kept_raw"]["id_like"] + res["kept_raw"]["binary_noise"]
        assert raw_noise > 1.0 and res["kept_raw"]["weak"] < res["kept_null"]["weak"], f"raw ranking should admit noise at {q}"
    weak = [results[q]["kept_null"]["weak"] for q in (50, 75, 90, 99)]
    assert weak[0] >= weak[1] >= weak[2] >= weak[3] and weak[0] - weak[3] > 0.5, "stricter percentile should drop weak real columns"
    assert results[99]["auc_null"] < results[75]["auc_null"] - 0.003, "99th percentile should score lower than 75th"
    assert results[50]["kept_null"]["binary_noise"] + results[50]["kept_null"]["noise"] + results[50]["kept_null"]["id_like"] > 0.2, \
        "median screen should let a noise column through"
    assert fmt(results[75]["auc_null"]) == "0.786" and fmt(results[75]["auc_raw"]) == "0.759", "prediction feedback numbers"
    assert fmt(results[50]["auc_null"]) == "0.792" and fmt(results[99]["auc_null"]) == "0.780", "check answer numbers"
    k50 = results[50]["kept_null"]
    assert fmt(k50["noise"] + k50["binary_noise"] + k50["id_like"], 1) == "0.7" and fmt(k50["weak"], 1) == "3.3", "check answer counts"
    assert fmt(results[99]["kept_null"]["weak"], 1) == "2.0", "check answer weak count"
    assert results[99]["auc_null"] > results[99]["auc_all"], "even the strictest screen beats the full table here"
