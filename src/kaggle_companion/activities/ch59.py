"""Chapter 59: Threshold Optimization. How far a tuned F1 threshold's development score overstates its score on fresh labels, by development-set size, and what a prevalence shift does to it."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.linear_model import LogisticRegression

from kaggle_companion.activities._common import clean

SEED = 59
PREVALENCE = 0.05            # share of positive rows when the model and the threshold are fitted
DRAWS = 300                  # independent development sets per size; results are means over draws
AUDIT_ROWS = 50000           # fresh rows used only for scoring
SHIFTS = [0.01, 0.05, 0.20]  # audit prevalences used to test the frozen threshold


def generate(n, prevalence, rng):
    """Five noisy features; each shifts a little for positive rows."""
    y = (rng.random(n) < prevalence).astype(int)
    X = rng.normal(size=(n, 5)) + y[:, None] * np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    return X, y


class Scored:
    """Rows sorted by score, so F1 at any threshold costs one search instead of one pass."""
    def __init__(self, p, y):
        order = np.argsort(p)
        self.p, self.y = p[order], y[order]
        self.positives_from = np.r_[np.cumsum(self.y[::-1])[::-1], 0]   # positives among rows at or above each position
        self.total = int(y.sum())

    def f1(self, threshold):
        """F1 when every row with score >= threshold is called positive."""
        i = np.searchsorted(self.p, threshold, side="left")
        called, hits = len(self.p) - i, self.positives_from[i]
        return 2 * hits / (called + self.total) if called + self.total else 0.0

    def best_threshold(self):
        """The threshold that maximizes F1 on these rows: a midpoint between two neighbouring scores."""
        k = np.arange(len(self.p), 0, -1)                       # rows called positive when cutting at each position
        f1 = 2 * self.positives_from[:-1] / (k + self.total)
        i = int(np.argmax(f1))
        return (self.p[i - 1] + self.p[i]) / 2 if i > 0 else self.p[0] - 1e-9, float(f1[i])


def averaged_fold_threshold(p, y, rng, folds=5):
    """A variant of the chapter's fold aggregation: tune on each fold's training part (four fifths of the rows), then average."""
    order = rng.permutation(len(p))
    found = []
    for k in range(folds):
        keep = np.setdiff1d(order, order[k::folds])
        if y[keep].sum() > 0:
            found.append(Scored(p[keep], y[keep]).best_threshold()[0])
    return float(np.mean(found))


def run(development_rows):
    rng = np.random.default_rng(SEED)
    X, y = generate(5000, PREVALENCE, rng)
    model = LogisticRegression(max_iter=500).fit(X, y)             # fitted once; never sees development or audit rows
    audits = {}
    for prevalence in SHIFTS:
        Xa, ya = generate(AUDIT_ROWS, prevalence, rng)
        audits[prevalence] = Scored(model.predict_proba(Xa)[:, 1], ya)
    audit = audits[PREVALENCE]

    tuned_dev, tuned_audit, fold_audit, thresholds = [], [], [], []
    shifted = {s: [] for s in SHIFTS}
    for _ in range(DRAWS):
        Xd, yd = generate(development_rows, PREVALENCE, rng)
        if yd.sum() == 0:                                           # a draw with no positive row cannot be tuned: give it one
            yd[rng.integers(development_rows)] = 1
        dev = Scored(model.predict_proba(Xd)[:, 1], yd)
        threshold, dev_f1 = dev.best_threshold()                    # fitted on the development rows' own labels
        tuned_dev.append(dev_f1)
        tuned_audit.append(audit.f1(threshold))                     # frozen, then scored on fresh labels
        thresholds.append(threshold)
        fold_audit.append(audit.f1(averaged_fold_threshold(dev.p, dev.y, rng)))
        for s in SHIFTS:
            shifted[s].append(audits[s].f1(threshold))
    return clean({
        "development_rows": development_rows,
        "development_f1": float(np.mean(tuned_dev)), "development_f1_sd": float(np.std(tuned_dev)),
        "audit_f1": float(np.mean(tuned_audit)), "audit_f1_sd": float(np.std(tuned_audit)),
        "fold_average_audit_f1": float(np.mean(fold_audit)),
        "default_audit_f1": audit.f1(0.5), "best_possible_audit_f1": audit.best_threshold()[1],
        "best_possible_threshold": audit.best_threshold()[0],
        "threshold_mean": float(np.mean(thresholds)), "threshold_sd": float(np.std(thresholds)),
        "tuned_beats_default": float(np.mean(np.array(tuned_audit) > audit.f1(0.5))),
        "shift": {str(s): {"frozen": float(np.mean(shifted[s])), "default": audits[s].f1(0.5),
                           "best_possible": audits[s].best_threshold()[1]} for s in SHIFTS},
        "draws": DRAWS, "audit_rows": AUDIT_ROWS,
    })
# notebook-end


SPEC = {
    "chapter": 59,
    "chapter_title": "Threshold Optimization",
    "subtitle": "A tuned threshold's search score is a development result: freeze it and measure it on different labels.",
    "summary": ("Threshold search fits the labels it is scored on. One demonstration tunes an F1 threshold on development sets of "
                "different sizes, freezes it, and measures how much the development score overstates the score on 50,000 fresh rows, "
                "and what happens when the prevalence changes."),
    "title": "Tuned F1 threshold: development score against fresh-label score, by development-set size",
    "question": "How far does the development F1 of a tuned threshold overstate its F1 on fresh labels, and how does that depend on the size of the development set?",
    "why": ("A threshold is a fitted parameter chosen to maximize the metric on rows whose labels it can exploit. The best search score is "
            "a development result, optimistic by an amount that depends on how many positives the search saw."),
    "method": ("A constructed binary task with 5% positives and five noisy features. A logistic regression is fitted once on 5,000 rows. For each "
               "development-set size, 300 independent development sets are drawn; on each, the threshold that maximizes F1 is found by "
               "scanning every cut of the sorted scores, then frozen and scored on 50,000 fresh rows with the same prevalence. Also reported: "
               "the default 0.5, the best possible threshold (tuned on the 50,000 audit rows, a ceiling no real search sees), the average "
               "of five thresholds each tuned on the development set with one fifth held back, and the frozen threshold on audits with 1% and 20% positives."),
    "control": {"key": "development_rows", "label": "Rows in the development set used to tune the threshold",
                "values": [100, 300, 1000, 5000], "default": 100,
                "value_labels": ["100 (about 5 positives)", "300", "1,000", "5,000"]},
    "source_section": "When Threshold Optimization Overfits",
    "symbols": ("F1 = 2TP / (2TP + FP + FN) for the rows called positive at threshold t. The development score is the maximum of F1 "
                "over t on the development rows; the audit score is F1 at that frozen t on fresh rows. Optimism is the difference."),
    "explanation": ("A search over every cut of a small development set picks the threshold that happens to rank its handful of positives best, "
                    "so its score is inflated and the threshold itself is noisy. More rows shrink both. Tuning still beats the default "
                    "0.5 because the default is badly matched to a rare class. A frozen threshold also carries the prevalence it was tuned at: "
                    "with 1% positives instead of 5% it no longer beats the default 0.5 (here it scores below it), and with 20% it falls well "
                    "short of the best cut for that audit."),
    "application": ("Tune the threshold on development predictions, freeze it, report the frozen score on separate labels, and "
                    "check it under the prevalence you expect rather than quoting the search score."),
    "assumptions": ("Constructed data with one logistic model and a stable relationship, so the audit set comes from the same generator. "
                    "A development draw with no positive row is given one so a threshold can be tuned; that affects about 0.6% of draws at 100 rows. "
                    "The best possible threshold is tuned on the audit labels themselves and is a ceiling, not an achievable score. "
                    "The prevalence-shift audits assume the same feature distribution and only a different share of positives."),
    "prediction": "With a development set of 100 rows, by how much does the tuned threshold's development F1 exceed its F1 on 50,000 fresh rows?",
    "prediction_options": ["Under 0.03", "About 0.19", "About 0.40"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "The development F1 averages 0.502 and the frozen threshold scores 0.315 on fresh rows: 0.187 of optimism.",
        "incorrect": "The development F1 averages 0.502 and the frozen threshold scores 0.315 on fresh rows: 0.187 of optimism, from tuning on about five positives.",
    },
    "check": "The development score is badly optimistic at 100 rows, yet the tuned threshold still beats the default 0.5. Are those two facts in conflict?",
    "answer": ("No. The development score is a biased estimate of the threshold's quality, not a measure of whether tuning helps. The "
               "tuned threshold scores 0.315 on fresh rows against 0.215 for the default 0.5, and beats it in 93% of the 300 draws. The "
               "bias is the reason to freeze the threshold and score it on other labels before quoting any number."),
    "provenance": "Constructed example: seeded synthetic data with 5% positives, a logistic regression and 300 development sets per size, measured by the chapter activity.",
    "apply": [
        "Tune the threshold on development predictions only, freeze it, and report its score on labels the search never saw.",
        "Expect the development score to overstate the frozen score, most of all when the development set holds few positives, and count positives, not rows.",
        "Compare the tuned threshold with the default and with a simple aggregate such as the average of fold thresholds before keeping a complicated recipe.",
        "Check the frozen threshold at the prevalence you expect in the test data; a threshold carries the class balance it was tuned on.",
    ],
    "honesty": ("Constructed data with a stable relationship. The gain from averaging fold thresholds is small here (about 0.01 at 100 rows); sizes depend on the model's strength and the metric."),
}

EQUATIONS = [{"tex": r"\mathrm{F1}(t) = \frac{2\,\mathrm{TP}(t)}{2\,\mathrm{TP}(t) + \mathrm{FP}(t) + \mathrm{FN}(t)}, \qquad \text{optimism} = \max_t \mathrm{F1}_{\mathrm{dev}}(t) - \mathrm{F1}_{\mathrm{audit}}(\hat t)",
              "alt": "F1 at threshold t equals twice the true positives over twice the true positives plus false positives plus false negatives; optimism equals the maximum over t of the development F1 minus the audit F1 at the chosen threshold t hat",
              "basis": "F1 as used in the chapter's search function; the optimism definition is the activity's own measure (the chapter has no display equation)."}]
NCOLS = 2
HEIGHT = 4.4
BARS = [("development_f1", "Tuned,\ndevelopment\nrows", COLORS["gold"]), ("audit_f1", "Tuned,\nfresh\nrows", COLORS["teal"]),
        ("fold_average_audit_f1", "Fold-average,\nfresh\nrows", COLORS["navy"]), ("default_audit_f1", "Default 0.5,\nfresh\nrows", COLORS["light"])]


def draw(axes, result, parameter):
    left, right = axes
    xs = list(range(len(BARS)))
    spreads = {"development_f1": result["development_f1_sd"], "audit_f1": result["audit_f1_sd"]}
    for x, (key, _, color) in zip(xs, BARS):
        left.bar(x, result[key], width=0.6, color=color, edgecolor=COLORS["ink"], lw=0.6)
        sd = spreads.get(key)
        if sd is not None:
            left.errorbar(x, result[key], yerr=sd, color=COLORS["ink"], capsize=4, lw=1.2, label="One standard deviation across draws" if x == 0 else None)
        left.text(x, result[key] + (sd or 0) + 0.015, fmt(result[key]), ha="center", va="bottom", fontsize=10,
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5}, zorder=5)
    left.axhline(result["best_possible_audit_f1"], color=COLORS["terracotta"], ls=(0, (4, 3)), lw=1.6,
                 label=f"Best possible threshold: {fmt(result['best_possible_audit_f1'])}")
    left.set_xticks(xs, [name for _, name, _ in BARS])
    left.set_xlim(-0.6, len(BARS) - 0.4)
    left.set_ylim(0, max(result["development_f1"] + result["development_f1_sd"], result["best_possible_audit_f1"]) * 1.5)
    left.set_ylabel("F1 (5% positives)")
    left.set_xlabel(f"Threshold and where it is scored, {parameter} development rows")
    left.legend(loc="upper right", frameon=False, fontsize=10)

    groups = list(result["shift"])
    width = 0.26
    series = [("default", "Default 0.5", COLORS["light"]), ("frozen", "Frozen tuned threshold", COLORS["teal"]),
              ("best_possible", "Best for that audit", COLORS["terracotta"])]
    for j, (key, label, color) in enumerate(series):
        values = [result["shift"][g][key] for g in groups]
        right.bar([i + (j - 1) * width for i in range(len(groups))], values, width=width, color=color, edgecolor=COLORS["ink"], lw=0.6, label=label)
        for i, v in enumerate(values):
            right.text(i + (j - 1) * width, v + 0.01, fmt(v, 2), ha="center", va="bottom", fontsize=10)
    right.set_xticks(range(len(groups)), [f"{round(100 * float(g))}%" for g in groups])
    right.set_ylim(0, 1.3 * max(result["shift"][g]["best_possible"] for g in groups))
    right.set_ylabel("F1 on 50,000 fresh rows")
    right.set_xlabel("Share of positives in the fresh rows (tuned at 5%)")
    right.legend(loc="upper left", frameon=False, fontsize=10)


def diff(a, b, digits=3):
    """Difference of the two numbers as displayed, so the hand calculation on the page adds up."""
    return float(fmt(a, digits)) - float(fmt(b, digits))


def explain(result, parameter):
    dev, aud, fold, dflt, best = (result[k] for k in ("development_f1", "audit_f1", "fold_average_audit_f1", "default_audit_f1", "best_possible_audit_f1"))
    optimism = diff(dev, aud)
    gain = diff(aud, dflt)
    low, high = result["shift"]["0.01"], result["shift"]["0.2"]
    interpretation = (
        f"With {parameter:,} development rows, the tuned threshold reaches an F1 of {fmt(dev)} on those rows but {fmt(aud)} on 50,000 fresh rows, "
        f"so {fmt(dev)} - {fmt(aud)} = {fmt(optimism)} is the optimism. The default 0.5 scores {fmt(dflt)} on the fresh rows, so tuning gains "
        f"{fmt(aud)} - {fmt(dflt)} = {fmt(gain)} and beats the default in {round(100 * result['tuned_beats_default'])}% of {result['draws']} draws. "
        f"The best possible threshold scores {fmt(best)}. The tuned threshold itself varies by {fmt(result['threshold_sd'])} (one standard deviation) around {fmt(result['threshold_mean'])}. "
        f"Frozen at 5% positives and scored where only 1% are positive it gives {fmt(low['frozen'])} (default 0.5: {fmt(low['default'])}), "
        f"and at 20% positive {fmt(high['frozen'])} (best cut for that audit: {fmt(high['best_possible'])}).")
    steps = [
        f"Optimism: {fmt(dev)} - {fmt(aud)} = {signed(optimism)} (development minus fresh rows).",
        f"Gain over the default 0.5: {fmt(aud)} - {fmt(dflt)} = {signed(gain)}.",
        f"Shortfall against the best possible threshold: {fmt(best)} - {fmt(aud)} = {signed(diff(best, aud))}.",
        f"Fold-averaged threshold minus single tuned threshold on fresh rows: {fmt(fold)} - {fmt(aud)} = {signed(diff(fold, aud))}.",
    ]
    metrics = {"Development F1 (tuned)": fmt(dev), "Fresh-rows F1 (same threshold)": fmt(aud), "Optimism": fmt(optimism),
               "Default 0.5 F1": fmt(dflt), "Tuned beats default": f"{round(100 * result['tuned_beats_default'])}% of draws"}
    alt = (f"Left: with {parameter:,} development rows, the tuned threshold scores {fmt(dev)} on development rows and {fmt(aud)} on fresh rows; "
           f"the default 0.5 scores {fmt(dflt)} and the best possible threshold {fmt(best)}. Right: F1 on audits with 1%, 5% and 20% positives for the "
           f"default, the frozen threshold and the best threshold for each audit.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def _qualitative(results):
    """Directional claims; checked again under a different seed base."""
    optimism = {n: res["development_f1"] - res["audit_f1"] for n, res in results.items()}
    assert all(optimism[n] > 0 for n in optimism), "development score overstates at every size"
    assert optimism[100] > optimism[300] > optimism[1000] > optimism[5000] >= 0, "optimism shrinks with size"
    assert optimism[100] > 0.1 and optimism[5000] < 0.04, "size of the optimism at the ends"
    for n, res in results.items():
        assert res["audit_f1"] > res["default_audit_f1"] + 0.05, f"tuning should beat the default 0.5 at {n}"
        assert res["audit_f1"] <= res["best_possible_audit_f1"] + 1e-9 and res["tuned_beats_default"] > 0.85, f"tuned vs default at {n}"
        assert res["threshold_sd"] > 0, "threshold varies"
    assert results[100]["threshold_sd"] > 2 * results[5000]["threshold_sd"], "small development sets give noisy thresholds"
    assert results[100]["audit_f1_sd"] > 2 * results[5000]["audit_f1_sd"], "fresh-row score is more variable at 100 rows"
    assert -0.005 < results[100]["fold_average_audit_f1"] - results[100]["audit_f1"] < 0.05, "fold averaging gain is small"
    shift = results[5000]["shift"]
    assert shift["0.01"]["best_possible"] < shift["0.05"]["best_possible"] < shift["0.2"]["best_possible"], "harder at low prevalence"
    assert shift["0.05"]["frozen"] > shift["0.05"]["default"] and shift["0.2"]["frozen"] > shift["0.2"]["default"], "frozen beats default at 5% and 20%"
    for n, res in results.items():
        assert res["shift"]["0.01"]["frozen"] < res["shift"]["0.01"]["default"], f"at 1% positives the frozen threshold scores below the default, {n}"
        assert res["shift"]["0.2"]["best_possible"] - res["shift"]["0.2"]["frozen"] > 0.05, f"at 20% it falls well short of the best cut, {n}"
    assert shift["0.05"]["frozen"] - shift["0.05"]["default"] > 0.1, "at the tuned prevalence the advantage is large"
    assert shift["0.2"]["best_possible"] - shift["0.2"]["frozen"] > shift["0.05"]["best_possible"] - shift["0.05"]["frozen"], \
        "the frozen threshold loses more at a shifted prevalence than at the tuned one"


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    _qualitative(results)
    r = results[100]
    assert fmt(r["development_f1"]) == "0.502" and fmt(r["audit_f1"]) == "0.315" and fmt(diff(r["development_f1"], r["audit_f1"])) == "0.187", "feedback"
    assert 0.17 < r["development_f1"] - r["audit_f1"] < 0.21, "option says about 0.19"
    assert fmt(r["default_audit_f1"]) == "0.215" and round(100 * r["tuned_beats_default"]) == 93, "answer numbers"
    assert fmt(r["fold_average_audit_f1"] - r["audit_f1"], 2) == "0.01", "honesty: fold averaging gain about 0.01"
