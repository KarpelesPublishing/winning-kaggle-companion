"""Chapter 26: Advanced Target Encoding. How much shrinkage a category encoding needs, by how much categories really differ."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

from kaggle_companion.activities._common import clean

SEED = 26
REPLICATES = 30       # independent constructed datasets per setting; every number is their mean
K, N_TRAIN, N_NEW = 300, 6000, 20000
PSEUDOCOUNTS = [0, 1, 3, 10, 30, 100, 300]   # smoothing strength m; 0 means no smoothing
WOE_SMOOTH = 0.5      # the chapter's Weight of Evidence default
RARE = 5              # a category with fewer than 5 development rows is rare


def generate(spread, seed):
    """300 categories with Zipf-like frequencies, so a few are common and many are rare. True category effects ~ Normal(0, spread)."""
    rng = np.random.default_rng(seed)
    effect = rng.normal(0, spread, K)
    freq = 1 / np.arange(1, K + 1)
    freq /= freq.sum()
    cat, cat_new = rng.choice(K, N_TRAIN, p=freq), rng.choice(K, N_NEW, p=freq)

    def rows(c):
        x = rng.normal(size=(len(c), 2))
        logit = 0.6 * x[:, 0] - 0.4 * x[:, 1] + effect[c] - 0.3
        return x, (rng.random(len(c)) < 1 / (1 + np.exp(-logit))).astype(int)

    (x, y), (x_new, y_new) = rows(cat), rows(cat_new)
    return cat, x, y, cat_new, x_new, y_new


def smoothed_mean(cat, y, m):
    """The chapter 9 encoding e_c = (s_c + m*mu) / (n_c + m); with m = 0 it is each category's own mean (the global mean if unseen)."""
    mu = y.mean()
    s, n = np.bincount(cat, weights=y, minlength=K), np.bincount(cat, minlength=K)
    return (s + m * mu) / (n + m) if m > 0 else np.where(n > 0, s / np.maximum(n, 1), mu)


def weight_of_evidence(cat, y, smooth=WOE_SMOOTH):
    """The chapter's fit_woe and transform_woe: log of (smoothed share among positives) over (smoothed share among negatives).

    As in the chapter, the pseudocount goes to every category seen in fitting (k_seen*smooth in each denominator), and a
    category unseen in fitting gets the neutral value 0."""
    seen = np.bincount(cat, minlength=K) > 0
    k_seen = seen.sum()
    pos = np.bincount(cat, weights=y, minlength=K) + smooth
    neg = np.bincount(cat, weights=1 - y, minlength=K) + smooth
    woe = np.log((pos / (y.sum() + k_seen * smooth)) / (neg / ((1 - y).sum() + k_seen * smooth)))
    return np.where(seen, woe, 0.0)


def to_logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def evaluate(data, encoder, transform=lambda v: v):
    """The deployable pipeline: cross-fit the training column, fit the model, encode new rows with the map from all training rows."""
    cat, x, y, cat_new, x_new, y_new = data
    column = np.zeros(len(cat))
    for fit, out in KFold(5, shuffle=True, random_state=2).split(cat):
        column[out] = encoder(cat[fit], y[fit])[cat[out]]
    full_map = encoder(cat, y)
    model = LogisticRegression(C=10, max_iter=500).fit(np.column_stack([x, transform(column)]), y)
    p = model.predict_proba(np.column_stack([x_new, transform(full_map[cat_new])]))[:, 1]
    rare = np.bincount(cat, minlength=K)[cat_new] < RARE
    return [log_loss(y_new, p), log_loss(y_new[rare], p[rare], labels=[0, 1]), log_loss(y_new[~rare], p[~rare], labels=[0, 1]), rare.mean()]


def run(category_spread):
    sets = [generate(category_spread, SEED * 1000 + i) for i in range(REPLICATES)]
    by_m = {str(m): np.mean([evaluate(d, lambda c, y, m=m: smoothed_mean(c, y, m), to_logit) for d in sets], axis=0) for m in PSEUDOCOUNTS}
    woe = np.mean([evaluate(d, weight_of_evidence) for d in sets], axis=0)
    best = min(PSEUDOCOUNTS, key=lambda m: by_m[str(m)][0])
    return clean({
        "category_spread": category_spread, "replicates": REPLICATES, "categories": K, "training_rows": N_TRAIN, "new_rows": N_NEW,
        "rare_share_of_new_rows": float(woe[3]), "best_m": best,
        "overall": {str(m): by_m[str(m)][0] for m in PSEUDOCOUNTS}, "rare": {str(m): by_m[str(m)][1] for m in PSEUDOCOUNTS},
        "common": {str(m): by_m[str(m)][2] for m in PSEUDOCOUNTS},
        "woe": {"overall": woe[0], "rare": woe[1], "common": woe[2]},
    })
# notebook-end


SPEC = {
    "chapter": 26,
    "chapter_title": "Advanced Target Encoding",
    "subtitle": "Shrinkage shares information across categories; how much it should share depends on how much categories truly differ.",
    "summary": ("The smoothed mean from Chapter 9 and the Weight of Evidence map are both shrinkage rules with a smoothing strength. One demonstration "
                "measures log loss on new rows for a range of smoothing strengths as the true spread of category effects changes."),
    "title": "How much to smooth a target encoding, by how much categories really differ",
    "question": "How much smoothing does a target encoding need, and does the answer change with how strongly categories really differ?",
    "why": ("The chapter says to start from the smoothed mean and reach for a richer encoder only when it supplies a missing function. "
            "Measuring the cost of too little and too much smoothing shows whether the simple default is enough and what the richer rules would have to beat."),
    "method": ("Thirty constructed binary tasks per setting with 300 categories of Zipf-like frequency (many categories have fewer than 5 of the 6,000 "
               "training rows), two numeric features and true category effects with the control's standard deviation. Each pipeline cross-fits the encoded training column in "
               "5 folds, fits a logistic regression and encodes 20,000 new rows with a map fitted on all training rows. Smoothed means with pseudocount m "
               "from 0 to 300 are compared with the chapter's Weight of Evidence at smoothing 0.5, by log loss on all new rows and on those whose category has fewer than 5 training rows."),
    "control": {"key": "category_spread", "label": "True spread of category effects (standard deviation, log odds)",
                "values": [0.2, 0.5, 1.0, 1.5], "default": 1.0,
                "value_labels": ["0.2: categories barely differ", "0.5", "1.0", "1.5: categories differ strongly"]},
    "source_section": "Which One to Use When",
    "symbols": ("e_c is the encoding of category c, s_c the sum and n_c the count of its training labels, mu the global mean and m the "
                "pseudocount; WoE(c) is the log ratio of the category's share among positives to its share among negatives."),
    "explanation": ("A category's raw mean is noisy when it has few rows, and the noise grows when the model trusts the column. A pseudocount pulls "
                    "each mean toward the global mean in proportion to how few rows support it. When categories barely differ, strong pulling costs nothing "
                    "and removes noise; when they differ strongly, strong pulling erases real effects. The best m therefore falls as the true spread rises, and no smoothing at all "
                    "hurts rare categories most."),
    "application": ("Keep Chapter 9's smoothed mean as the baseline, choose m by validation inside the outer boundary rather than by habit, and judge any richer "
                    "encoder by whether it beats the best smoothed mean on rows it never saw, with rare categories reported separately."),
    "assumptions": ("Constructed data with normally distributed category effects, a logistic model and cross-fitted columns. Weight of Evidence is the chapter's fit_woe "
                    "formula with smoothing 0.5. No James-Stein, mixed-model or native-categorical encoder is run, so this does not rank those. When "
                    "categories barely differ the differences between m values are within about 0.002 log loss and should not be over-read."),
    "prediction": "With a moderate true spread of 1.0, how does an encoding with no smoothing (m = 0) compare with the Chapter 9 default m = 10 on log loss for new rows?",
    "prediction_options": ["No measurable difference", "About 0.04 worse", "Better, because it keeps each category's own mean"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "No smoothing scores 0.619 against 0.580 for m = 10, and on rare categories 0.695 against 0.621.",
        "incorrect": "No smoothing scores 0.619 against 0.580 for m = 10: a rare category's own mean is mostly noise, and the model trusts it.",
    },
    "check": "Why does the best pseudocount fall as the true spread of category effects grows, and why does skipping smoothing hurt rare categories most?",
    "answer": ("Smoothing trades noise for bias. When categories barely differ there is little real effect to erase, so a large m (30 or more) removes noise at no cost; "
               "when they differ strongly (spread 1.5) the best m is 3, because heavy pulling would flatten real differences. A rare category's own mean rests on "
               "a handful of rows, so with m = 0 the model trusts noise (rare-category log loss 0.750 at spread 1.5 against 0.588 at m = 3)."),
    "provenance": "Constructed example: thirty seeded synthetic tasks per setting with Zipf-like category frequencies and a logistic regression, measured by the chapter activity.",
    "apply": [
        "Start from the smoothed mean and pick m by inner validation, never m = 0 for a category with few rows.",
        "Cross-fit the training column and map new rows with a table fitted on all training rows, as Chapter 9 requires; return and store the map.",
        "Report log loss for rare and common categories separately: the gain from shrinkage sits in the rare ones.",
        "Replace the baseline with Weight of Evidence, a mixed model or a native encoder only if it beats the best smoothed mean on untouched rows.",
    ],
    "honesty": ("Constructed data. The best m depends on the generator's frequencies and spread; the pattern (best m falls as the spread rises, no "
                "smoothing hurts rare categories) is the claim, not the particular values."),
}

EQUATIONS = [{"tex": r"e_c = \frac{s_c + m\mu}{n_c + m}, \qquad \operatorname{WoE}(c)=\log\frac{P(X=c\mid Y=1)}{P(X=c\mid Y=0)}",
              "alt": "e c equals s c plus m times mu, divided by n c plus m; and the weight of evidence of category c is the log of the ratio of the probability of c among positives to the probability of c among negatives",
              "basis": "The smoothed mean from Chapter 9 (used as the baseline in Advanced Target Encoding) and the display equation in Weight of Evidence (WoE): Built for Binary Targets."}]
NCOLS = 2
HEIGHT = 4.4


def _curve(ax, values, woe, best, ylabel, xlabel):
    ms = [str(m) for m in PSEUDOCOUNTS]
    ys = [values[m] for m in ms]
    ax.plot(range(len(ms)), ys, "o-", color=COLORS["teal"], lw=2, ms=6, label="Smoothed mean")
    i = ms.index(str(best))
    ax.plot([i], [ys[i]], "o", ms=13, mfc="none", mec=COLORS["ink"], mew=2, label=f"Lowest at m = {best}")
    ax.axhline(woe, color=COLORS["gold"], ls=(0, (4, 3)), lw=1.6, label=f"Weight of evidence: {fmt(woe)}")
    ax.set_xticks(range(len(ms)), ms, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)


def draw(axes, result, parameter):
    left, right = axes
    best = result["best_m"]
    _curve(left, result["overall"], result["woe"]["overall"], best, "Log loss on 20,000 new rows", f"Pseudocount m, category spread {parameter}")
    left.legend(loc="upper right", frameon=False, fontsize=10)
    rare_best = min(PSEUDOCOUNTS, key=lambda m: result["rare"][str(m)])
    _curve(right, result["rare"], result["woe"]["rare"], rare_best, "Log loss, rare categories only", "Pseudocount m")
    right.legend(loc="upper right", frameon=False, fontsize=10)


def explain(result, parameter):
    o, r = result["overall"], result["rare"]
    best = str(result["best_m"])
    gain0 = o["0"] - o[best]
    gain10 = o["10"] - o[best]
    interpretation = (
        f"With category spread {parameter}, the lowest new-row log loss is {fmt(o[best])} at m = {best}. No smoothing scores {fmt(o['0'])}, "
        f"so {fmt(o['0'])} - {fmt(o[best])} = {fmt(gain0)} is what skipping smoothing costs; "
        + ("the Chapter 9 default m = 10 is the best setting here. " if best == "10" else f"the Chapter 9 default m = 10 is {fmt(gain10)} above the best. ")
        + f"On rare categories ({fmt(100 * result['rare_share_of_new_rows'], 1)}% of new rows) no smoothing scores {fmt(r['0'])} against {fmt(r['10'])} at m = 10. "
        f"Weight of evidence at smoothing 0.5 scores {fmt(result['woe']['overall'])}.")
    steps = [
        f"Best pseudocount: m = {best} with log loss {fmt(o[best])}.",
        f"No smoothing: {fmt(o['0'])} - {fmt(o[best])} = {fmt(gain0)} worse than the best.",
        f"Default m = 10: {fmt(o['10'])} - {fmt(o[best])} = {fmt(gain10)} worse than the best.",
        f"Rare categories: m = 0 gives {fmt(r['0'])}, m = 10 gives {fmt(r['10'])}; weight of evidence {fmt(result['woe']['rare'])}.",
    ]
    metrics = {"Best pseudocount m": best, "Log loss at best m": fmt(o[best]), "Log loss with no smoothing": fmt(o["0"]),
               "Log loss at m = 10": fmt(o["10"]), "Weight of evidence": fmt(result["woe"]["overall"])}
    alt = (f"Two curves of log loss against pseudocount at category spread {parameter}: all new rows on the left, rare categories on the right, "
           f"each with a circle at its lowest point and a dashed line for weight of evidence; the lowest overall is at m = {best}.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    spreads = sorted(results)
    best = [results[s]["best_m"] for s in spreads]
    assert best == sorted(best, reverse=True) and best[0] >= 100 and best[-1] <= 10, f"best m falls from large to small: {best}"
    for s in spreads[1:]:
        res = results[s]
        assert res["overall"]["0"] - min(res["overall"].values()) > 0.005, f"no smoothing should cost something at {s}"
        assert res["rare"]["0"] > res["rare"]["10"] + (0.05 if s >= 1.0 else 0.005), f"no smoothing should hurt rare categories at {s}"
        assert res["overall"]["10"] - min(res["overall"].values()) < 0.012, f"m = 10 should be near the best at {s}"
    weak = results[0.2]
    assert max(weak["overall"].values()) - min(weak["overall"].values()) < 0.004, "at spread 0.2 the choice of m barely matters"
    assert abs(results[1.5]["woe"]["overall"] - min(results[1.5]["overall"].values())) < 0.01, "WoE close to the best smoothed mean at strong spread"
    mid = results[1.0]
    assert fmt(mid["overall"]["0"]) == "0.619" and fmt(mid["overall"]["10"]) == "0.580", "prediction feedback numbers"
    assert fmt(mid["rare"]["0"]) == "0.695" and fmt(mid["rare"]["10"]) == "0.621", "prediction feedback numbers"
    strong = results[1.5]
    assert strong["best_m"] == 3 and weak["best_m"] in (30, 100, 300), "check answer: best m at the extremes"
    assert fmt(strong["rare"]["0"]) == "0.750" and fmt(strong["rare"]["3"]) == "0.588", "check answer numbers"
