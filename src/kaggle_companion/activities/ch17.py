"""Chapter 17: Stacking and Hill-Climbing. The chapter's greedy_brier_weights as the candidate library fills with near-duplicates."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.linear_model import LogisticRegression

from kaggle_companion.activities._common import clean

SEED = 17
REPLICATES = 25                    # independent constructed datasets; every estimate is their mean
FEATURES, FAMILIES = 20, 4         # 20 features, 6 of them informative; 4 genuinely different model families
N_TRAIN, N_DEV, N_ASSESS = 400, 200, 5000


def generate(rng, n, coef):
    X = rng.normal(size=(n, FEATURES))
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ coef)))).astype(int)
    return X, y


def greedy_brier_weights(y_development, candidate_predictions, step=0.1, max_iterations=30):
    """The chapter's hill-climbing function: start from the best candidate, then repeatedly move `step` of the weight
    to whichever candidate lowers the development Brier loss most, and stop when nothing helps."""
    y = np.asarray(y_development, dtype=float)
    p = np.asarray(candidate_predictions, dtype=float)
    if p.ndim != 2 or p.shape[0] != len(y) or p.shape[1] == 0:
        raise ValueError('rows by candidate matrix required')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('finite binary probabilities required')
    if not np.isin(y, [0, 1]).all() or not 0 < step <= 1:
        raise ValueError('binary labels and step in (0,1] required')
    loss = lambda q: np.mean((q - y) ** 2)
    start = int(np.argmin(np.mean((p - y[:, None]) ** 2, axis=0)))
    weights = np.eye(p.shape[1])[start]
    current = p @ weights
    for _ in range(max_iterations):
        choices = (1 - step) * current[:, None] + step * p
        losses = np.mean((choices - y[:, None]) ** 2, axis=0)
        winner = int(np.argmin(losses))
        if losses[winner] >= loss(current) - 1e-12:
            break
        weights *= 1 - step
        weights[winner] += step
        current = p @ weights
    return weights


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def run(library_size):
    rows = []
    for rep in range(REPLICATES):
        rng = np.random.default_rng(SEED * 100 + rep)
        coef = np.zeros(FEATURES)
        coef[:6] = rng.normal(0, 0.8, 6)
        (X, y), (Xd, yd), (Xa, ya) = [generate(rng, n, coef) for n in (N_TRAIN, N_DEV, N_ASSESS)]
        columns = [rng.choice(FEATURES, 5, replace=False) for _ in range(FAMILIES)]   # each family reads five features
        Pd, Pa = np.zeros((N_DEV, library_size)), np.zeros((N_ASSESS, library_size))
        for j in range(library_size):
            cols = columns[j % FAMILIES]
            rows_used = np.arange(N_TRAIN) if j < FAMILIES else rng.integers(0, N_TRAIN, N_TRAIN)  # extras: bootstrap refits
            model = LogisticRegression(max_iter=300).fit(X[rows_used][:, cols], y[rows_used])
            Pd[:, j], Pa[:, j] = model.predict_proba(Xd[:, cols])[:, 1], model.predict_proba(Xa[:, cols])[:, 1]
        best = int(np.argmin([brier(yd, Pd[:, j]) for j in range(library_size)]))   # best single candidate, chosen on development rows
        w_all = greedy_brier_weights(yd, Pd)
        w_start = greedy_brier_weights(yd, Pd[:, :FAMILIES])                        # the same search on the four original models
        rows.append({
            "best": (brier(yd, Pd[:, best]), brier(ya, Pa[:, best])),
            "equal": (brier(yd, Pd.mean(axis=1)), brier(ya, Pa.mean(axis=1))),
            "greedy": (brier(yd, Pd @ w_all), brier(ya, Pa @ w_all)),
            "greedy_start": (brier(yd, Pd[:, :FAMILIES] @ w_start), brier(ya, Pa[:, :FAMILIES] @ w_start)),
            "kept": int((w_all > 0).sum()),
        })
    pick = lambda name, i: np.array([r[name][i] for r in rows])
    gain_dev = pick("greedy_start", 0) - pick("greedy", 0)        # what the extra candidates add on the rows the search used
    gain_assess = pick("greedy_start", 1) - pick("greedy", 1)     # what they add on assessment rows
    return clean({
        "library_size": library_size, "candidates_kept": float(np.mean([r["kept"] for r in rows])),
        "development": {k: float(pick(k, 0).mean()) for k in ("best", "equal", "greedy", "greedy_start")},
        "assessment": {k: float(pick(k, 1).mean()) for k in ("best", "equal", "greedy", "greedy_start")},
        "added_gain_development": float(gain_dev.mean()), "added_gain_assessment": float(gain_assess.mean()),
        "per_replicate": {"development": gain_dev, "assessment": gain_assess},
        "replicates": REPLICATES, "development_rows": N_DEV, "assessment_rows": N_ASSESS,
    })
# notebook-end


SPEC = {
    "chapter": 17,
    "chapter_title": "Stacking and Hill-Climbing",
    "subtitle": "Hill-climbing's development score is a search result; judge the candidate library on rows the search never used.",
    "summary": ("Greedy ensemble search keeps whatever lowers its development loss. One demonstration runs the chapter's hill-climbing "
                "function as the library grows from four distinct models to many near-duplicates, and measures what the extra candidates add on development and assessment rows."),
    "title": "What extra candidates add to a hill-climbing ensemble, on development and assessment rows",
    "question": "As the candidate library fills with near-duplicate models, how much of the hill-climbing improvement shows up on assessment rows?",
    "why": ("The chapter warns that greedy search can overfit repeated development comparisons, and that an old score does not "
            "show the benefit. Measuring development against assessment gain shows what extra candidates are worth."),
    "method": ("A constructed binary task with 20 features, six of them informative. Four logistic-regression families each read five "
               "features and are fitted on 400 rows; the library then grows by bootstrap refits of those same four, so the extras are "
               "near-duplicates. The chapter's greedy_brier_weights runs on 200 development rows and is scored by Brier loss on "
               "those rows and on 5,000 assessment rows, against the same search on the four original models. Twenty-five independent datasets are averaged."),
    "control": {"key": "library_size", "label": "Candidates in the library",
                "values": [4, 12, 40, 100], "default": 100,
                "value_labels": ["4 (the originals)", "12", "40", "100"]},
    "source_section": "Hill-Climbing Ensemble",
    "symbols": ("B_dev and B_assess are the Brier loss on development and assessment rows. The added gain of a library is the Brier "
                "loss of the search on the four original models minus the loss of the search on the whole library; positive is better."),
    "explanation": ("Every extra near-duplicate gives the search another small, mostly noise-driven way to lower its development loss, "
                    "so the development improvement keeps growing. The assessment improvement grows far more slowly, because the "
                    "extras carry almost no new information. The gap between the two is the search's optimism."),
    "application": ("Freeze the candidate library and the search recipe, score the frozen ensemble on rows the search never used, "
                    "and compare it with the best single candidate, equal averaging and the same search on a smaller library."),
    "assumptions": ("Constructed data, logistic regressions standing in for gradient boosting and other model families, and extras that are "
                    "bootstrap refits of four originals, which is deliberately the near-duplicate case. A library of diverse candidates "
                    "would show a larger real gain. Assessment loss did not get worse as the library grew here, so no claim is made that "
                    "adding models must hurt, only that the development score overstates what they add."),
    "prediction": "Growing the library from 4 to 100 near-duplicates lowers the search's development Brier loss by 0.007. How much of that appears on assessment rows?",
    "prediction_options": ["Nearly all of it", "Less than a third of it", "None: assessment loss rises"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "Development loss falls by 0.007 and assessment loss by 0.002, about 28% of the development gain.",
        "incorrect": "Development loss falls by 0.007 and assessment loss by 0.002, about 28% of the development gain. It does not rise; it just improves far less.",
    },
    "check": "The assessment loss barely improves with 100 candidates. Why does the development loss keep falling, and what should stop the search?",
    "answer": ("The 96 extra candidates are bootstrap refits of the same four models, so they differ only by noise. Each one gives hill-climbing "
               "another chance to match noise in the 200 development rows, and the search kept 4.4 candidates on average to do it. "
               "The search should stop when the assessment gain, not the development gain, stops moving: here the 4-model search "
               "already earns most of what the 100-model library does."),
    "provenance": "Constructed example: seeded synthetic data, logistic regressions and the chapter's hill-climbing function, measured by the chapter activity.",
    "apply": [
        "Freeze the library and the search recipe before looking at assessment rows, then score the frozen ensemble once.",
        "Compare the search with equal averaging, the best single candidate and the same search on a smaller library.",
        "Drop near-duplicate candidates; seeds and refits of one model add development optimism much faster than information.",
        "Stop adding candidates when the assessment gain stops moving, even while the development score keeps improving.",
    ],
    "honesty": ("Constructed data with deliberate near-duplicates. The assessment loss did not rise with library size, so the extra "
                "candidates were harmless here but also nearly useless; the sizes are properties of this generator."),
}

EQUATIONS = [{"tex": r"\mathrm{Brier} = \frac{1}{n}\sum_{i}(p_i - y_i)^2",
              "alt": "Brier loss equals the mean over rows of the squared difference between predicted probability p i and label y i",
              "basis": "The loss minimized inside the chapter's greedy_brier_weights function (Hill-Climbing Ensemble); the chapter shows it as code, not a display equation."}]
NCOLS = 2
HEIGHT = 4.4
METHODS = [("best", "Best single"), ("equal", "Equal average"), ("greedy", "Hill-climbing")]


def draw(axes, result, parameter):
    left, right = axes
    xs = range(len(METHODS))
    dev = [result["development"][k] for k, _ in METHODS]
    new = [result["assessment"][k] for k, _ in METHODS]
    left.bar([x - 0.2 for x in xs], dev, width=0.4, color=COLORS["light"], edgecolor=COLORS["ink"], lw=0.6, label="Development rows (200)")
    left.bar([x + 0.2 for x in xs], new, width=0.4, color=COLORS["navy"], edgecolor=COLORS["ink"], lw=0.6, label="Assessment rows (5,000)")
    for x, a, b in zip(xs, dev, new):
        left.text(x - 0.2, a + 0.002, fmt(a), ha="center", va="bottom", fontsize=10)
        left.text(x + 0.2, b + 0.002, fmt(b), ha="center", va="bottom", fontsize=10)
    left.set_xticks(list(xs), [name for _, name in METHODS])
    left.set_ylim(0.17, 0.25)
    left.set_ylabel("Brier loss (lower is better)")
    left.set_xlabel(f"Method, library of {parameter} candidates")
    left.legend(loc="upper left", frameon=False, fontsize=10)

    values = [result["added_gain_development"], result["added_gain_assessment"]]
    right.bar([0, 1], values, width=0.5, color=[COLORS["terracotta"], COLORS["teal"]], edgecolor=COLORS["ink"], lw=0.6)
    for x, key in enumerate(["development", "assessment"]):
        dots = result["per_replicate"][key]
        right.scatter([x + (i - (len(dots) - 1) / 2) * 0.03 for i in range(len(dots))], dots, s=10, color=COLORS["ink"], zorder=3)
        right.text(x, max([values[x]] + dots) + 0.0015, fmt(values[x], 4), ha="center", va="bottom", fontsize=10)
    right.axhline(0, color=COLORS["grey"], lw=0.8)
    right.set_xticks([0, 1], ["Development\nrows", "Assessment\nrows"])
    top = max(max(result["per_replicate"]["development"]), 0.01)
    low = min(min(result["per_replicate"]["assessment"]), min(result["per_replicate"]["development"]), 0)
    right.set_ylim(low - 0.002, top + 0.01)
    right.set_ylabel("Brier loss gained from the extra candidates")
    right.set_xlabel("Where the gain is measured (dots: one dataset each)")


def diff(a, b, digits=3):
    """Difference of the displayed values, so the hand calculation in the text adds up."""
    return fmt(round(a, digits) - round(b, digits), digits)


def explain(result, parameter):
    d, a = result["development"], result["assessment"]
    gd, ga = result["added_gain_development"], result["added_gain_assessment"]
    extra = parameter - FAMILIES
    share = f"{round(100 * ga / gd)}% of it" if gd > 0.001 else "none of it"
    interpretation = (
        f"With {parameter} candidates the search kept {fmt(result['candidates_kept'], 1)} on average. Its development Brier loss is "
        f"{fmt(d['greedy'])}; on assessment rows it is {fmt(a['greedy'])}, so {fmt(a['greedy'])} - {fmt(d['greedy'])} = {diff(a['greedy'], d['greedy'])} of optimism "
        f"(the equal average, which chooses nothing on the development rows, differs by {diff(a['equal'], d['equal'])}). "
        + (f"The {extra} extra candidates lowered the development loss by {fmt(gd, 4)} and the assessment loss by {fmt(ga, 4)}, {share}. "
           if extra > 0 else "These are the four original models, the reference that the larger libraries are compared with. ")
        + f"The best single candidate scores {fmt(a['best'])} on assessment rows and the equal average {fmt(a['equal'])}.")
    steps = [
        f"Optimism of the search: {fmt(a['greedy'])} - {fmt(d['greedy'])} = {diff(a['greedy'], d['greedy'])} Brier (assessment minus development).",
        f"Development gain from {extra} extra candidates: {fmt(d['greedy_start'])} - {fmt(d['greedy'])} = {diff(d['greedy_start'], d['greedy'])}.",
        f"Assessment gain from the same candidates: {fmt(a['greedy_start'])} - {fmt(a['greedy'])} = {diff(a['greedy_start'], a['greedy'])}.",
        f"Hill-climbing against the best single candidate on assessment rows: {fmt(a['best'])} - {fmt(a['greedy'])} = {diff(a['best'], a['greedy'])}.",
    ]
    metrics = {"Candidates": str(parameter), "Hill-climbing, development": fmt(d["greedy"]), "Hill-climbing, assessment": fmt(a["greedy"]),
               "Added gain, development": fmt(gd, 4), "Added gain, assessment": fmt(ga, 4)}
    alt = (f"Left: Brier loss on development and assessment rows for the best single candidate, the equal average and hill-climbing with a "
           f"library of {parameter}. Right: loss gained from the {extra} extra candidates, {fmt(gd, 4)} on development rows and {fmt(ga, 4)} on assessment rows.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    order = sorted(results)
    assert abs(results[4]["added_gain_development"]) < 1e-9 and abs(results[4]["added_gain_assessment"]) < 1e-9, "library of 4 is the reference"
    dev = [results[k]["added_gain_development"] for k in order]
    assert all(b > a for a, b in zip(dev, dev[1:])), "development gain should keep rising with the library"
    for k in order[1:]:
        res = results[k]
        assert res["added_gain_assessment"] < res["added_gain_development"] * 0.6, f"assessment gain should lag development at {k}"
        assert res["added_gain_assessment"] > -0.002, f"extra candidates should not hurt assessment loss much at {k}"
    opt = [results[k]["assessment"]["greedy"] - results[k]["development"]["greedy"] for k in order]
    assert all(b > a for a, b in zip(opt, opt[1:])), "optimism should grow with the library"
    big = results[100]
    assert big["assessment"]["greedy"] < big["assessment"]["best"] - 0.003, "the search should beat the best single candidate"
    assert big["assessment"]["greedy"] < big["assessment"]["equal"] - 0.005, "the search should beat the equal average here"
    assert 0.1 < big["added_gain_assessment"] / big["added_gain_development"] < 1 / 3, "prediction option says less than a third"
    assert (big["assessment"]["best"] - big["assessment"]["greedy_start"]) > 0.5 * (big["assessment"]["best"] - big["assessment"]["greedy"]), \
        "answer: the 4-model search earns most of what the 100-model library does"
    assert 0.005 < big["added_gain_development"] < 0.008, "prediction text says 0.007"
    assert fmt(big["added_gain_development"], 3) == "0.007" and fmt(big["added_gain_assessment"], 3) == "0.002", "prediction feedback numbers"
    assert round(100 * big["added_gain_assessment"] / big["added_gain_development"]) == 28, "prediction feedback percentage"
    assert fmt(big["candidates_kept"], 1) == "4.4", "check answer numbers"
