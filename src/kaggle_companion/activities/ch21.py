"""Chapter 21: Financial Competitions. Shuffled, walk-forward and purged validation against the future, by label horizon."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

from kaggle_companion.activities._common import clean

SEED = 21
REPLICATES = 8          # independent constructed series; every estimate is their mean
N_DEV, N_FUTURE = 1500, 8000
BLOCK, ORIGINS = 150, 5  # walk-forward: five chronological origins, 150-day validation blocks


def generate(horizon, seed):
    """One daily series. A slow hidden state drives returns; features are backward-looking and known at the close."""
    rng = np.random.default_rng(seed)
    burn = 100
    T = burn + N_DEV + N_FUTURE + horizon + 25
    state = np.zeros(T)
    for t in range(1, T):
        state[t] = 0.97 * state[t - 1] + rng.normal(0, 0.25)
    ret = 0.12 * state + rng.normal(0, 1.0, T)            # daily return
    observed = state + rng.normal(0, 0.5, T)              # noisy reading of the state, published at the close
    csum = np.concatenate([[0.0], np.cumsum(ret)])

    def trailing_mean(w):                                 # mean return over the w days ending today
        out = np.full(T, np.nan)
        out[w:] = (csum[w + 1:T + 1] - csum[1:T - w + 1]) / w
        return out

    X = np.column_stack([observed, trailing_mean(5), trailing_mean(20), np.r_[np.nan, ret[:-1]]])
    y = np.full(T, np.nan)
    y[:T - horizon] = csum[1 + horizon:T + 1] - csum[1:T - horizon + 1]   # label: sum of the next `horizon` returns
    rows = np.arange(burn, burn + N_DEV + N_FUTURE)
    return X[rows], y[rows]


def model():
    return ExtraTreesRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.75, random_state=0, n_jobs=1)


def ic(pred, y):
    """Information coefficient: correlation between prediction and realized label."""
    return float(np.corrcoef(pred, y)[0, 1])


def one_series(horizon, seed):
    X, y = generate(horizon, seed)
    Xd, yd = X[:N_DEV], y[:N_DEV]
    Xf, yf = X[N_DEV + horizon:], y[N_DEV + horizon:]     # the future starts after the last development label resolves

    # Every scheme pools its held-out predictions and scores them once, as a final out-of-sample IC.
    shuffled = np.zeros(N_DEV)
    for a, b in KFold(5, shuffle=True, random_state=0).split(Xd):
        shuffled[b] = model().fit(Xd[a], yd[a]).predict(Xd[b])
    held = np.arange(N_DEV - ORIGINS * BLOCK, N_DEV)
    walk, purged = np.zeros(len(held)), np.zeros(len(held))
    for i, start in enumerate(range(held[0], N_DEV, BLOCK)):
        b = np.arange(start, start + BLOCK)
        walk[i * BLOCK:(i + 1) * BLOCK] = model().fit(Xd[:start], yd[:start]).predict(Xd[b])
        cut = start - horizon                             # purge: drop training rows whose label overlaps the block
        purged[i * BLOCK:(i + 1) * BLOCK] = model().fit(Xd[:cut], yd[:cut]).predict(Xd[b])
    future = ic(model().fit(Xd[:N_DEV - horizon], yd[:N_DEV - horizon]).predict(Xf), yf)
    return {"shuffled": ic(shuffled, yd), "walk_forward": ic(walk, yd[held]), "purged": ic(purged, yd[held]), "future": future}


def run(horizon):
    series = [one_series(horizon, SEED * 100 + i) for i in range(REPLICATES)]
    keys = ["shuffled", "walk_forward", "purged", "future"]
    return clean({
        "horizon": horizon,
        **{k: float(np.mean([s[k] for s in series])) for k in keys},
        "per_series": {k: [s[k] for s in series] for k in keys},
        "development_days": N_DEV, "future_days": N_FUTURE - horizon, "replicates": REPLICATES,
    })
# notebook-end


SPEC = {
    "chapter": 21,
    "chapter_title": "Financial Competitions: Time, Regime, and Leakage",
    "subtitle": "Choose validation from the forecast contract, and keep overlapping labels out of the assessment.",
    "summary": ("When labels span several future days, neighbouring rows share most of their answer. One demonstration measures "
                "how far shuffled, walk-forward and purged validation each drift from the score a model actually earns on the future."),
    "title": "Three validation schemes against the future, as the label horizon grows",
    "question": "How far does each validation scheme's estimate drift from the model's score on the future as the label horizon grows?",
    "why": ("Forward-return targets overlap: today's 20-day label shares 19 days with tomorrow's. A shuffled split puts those "
            "near-copies on both sides of the boundary, and the estimate you use to choose models stops describing the future."),
    "method": ("Eight constructed daily series in which a slow hidden state drives returns. Features are backward-looking and known at "
               "the close: a noisy reading of the state, trailing 5 and 20 day mean returns, and yesterday's return. The label is the "
               "sum of the next h returns, where h is the control. An extremely randomized trees regressor is scored by information "
               "coefficient (IC, the correlation of prediction and label, computed once over all held-out predictions) four ways on 1,500 development days: shuffled 5-fold "
               "cross-validation, walk-forward over five 150-day blocks, the same walk-forward with the last h training rows purged, "
               "and finally on about 8,000 later days the model never saw."),
    "control": {"key": "horizon", "label": "Label horizon (days of future return in each label)",
                "values": [1, 5, 10, 20], "default": 20,
                "value_labels": ["1: next-day return", "5", "10", "20: four-week return"]},
    "source_section": "Failure Mode 2: Regime Change",
    "symbols": ("y_t is the label for day t, r_t the return on day t, h the label horizon in days and v the first day of a "
                "validation block. IC is the Pearson correlation between predictions and realized labels on the "
                "scored rows; higher is better and 0 means no skill."),
    "explanation": ("With h = 1 each label is its own day, so every scheme estimates the future about equally well. As h grows, "
                    "a shuffled split trains on rows whose labels contain most of each validation row's answer, and the trees "
                    "memorize them. Chronological schemes keep training rows in the past, so only the first few rows of each "
                    "block can share a label with training data. They come out a little low instead, because the early "
                    "origins train on less history than the final model."),
    "application": ("Validate forward-return targets chronologically, purge at least the label horizon before every validation "
                    "block, and judge model choices by the chronological estimate, never by a shuffled one."),
    "assumptions": ("Constructed series with a stable relationship (no regime change), one model family and eight replicates; "
                    "per-series estimates vary, so the chart shows each series as a dot. In this generator the purge gap moved "
                    "the walk-forward estimate by less than 0.02 in either direction: the boundary leak touches only the first h "
                    "rows of each block. Shuffling, not the gap, is the large error here."),
    "prediction": "With a 20-day label horizon, how does shuffled 5-fold cross-validation compare with the score on the future?",
    "prediction_options": ["About the same", "Higher, by about 0.09 IC", "Lower"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "Shuffled CV reports 0.346 against 0.256 on the future: neighbouring rows share 19 of 20 label days.",
        "incorrect": "Shuffled CV reports 0.346 against 0.256 on the future. Each validation row's neighbours, which share 19 of its 20 label days, sit in training.",
    },
    "check": "Why do the walk-forward and purged walk-forward estimates come out almost the same here, when the chapter insists on purging?",
    "answer": ("Without a gap, only the training rows just before each block have labels that reach into it, so the leak touches "
               "the first h of 150 validation rows and was within the noise of this generator. The purge costs only h "
               "training rows and removes that leak by construction, which is why the chapter makes it a rule; the shuffled "
               "split is the error that moved the estimate by 0.090 at h = 20."),
    "provenance": "Constructed example: eight seeded synthetic daily series and an extremely randomized trees model, measured by the chapter activity.",
    "apply": [
        "Write down the forecast time, each feature's publication time and when each label resolves before choosing a split.",
        "Validate forward-return targets with chronological blocks; never shuffle rows whose labels overlap in time.",
        "Purge at least the label horizon of training rows before every validation block, and start the final holdout after the last development label resolves.",
        "If a shuffled estimate and a chronological estimate disagree, trust the chronological one and look for overlapping labels or slow features.",
    ],
    "honesty": "Constructed data with a stable relationship; the sizes of these effects are properties of this generator, not a competition result.",
}

EQUATIONS = [{"tex": r"y_t = \sum_{k=1}^{h} r_{t+k}",
              "alt": "y t equals the sum over k from 1 to h of r t plus k",
              "basis": "The activity's forward-return label; not a display equation in the manuscript."},
             {"tex": r"\text{keep training row } t \text{ only if } t + h < v",
              "alt": "keep training row t only if t plus h is less than v",
              "basis": "The purge rule the activity applies before each validation block starting at v (Failure Mode 2)."}]
NCOLS = 1
HEIGHT = 4.4
SCHEMES = [("shuffled", "Shuffled 5-fold"), ("walk_forward", "Walk-forward"), ("purged", "Purged walk-forward")]


def draw(ax, result, parameter):
    xs = list(range(len(SCHEMES)))
    means = [result[k] for k, _ in SCHEMES]
    colors = [COLORS["terracotta"], COLORS["light"], COLORS["teal"]]
    ax.bar(xs, means, width=0.55, color=colors, edgecolor=COLORS["ink"], lw=0.6)
    for x, (k, _) in zip(xs, SCHEMES):
        dots = result["per_series"][k]
        ax.scatter([x + (i - (len(dots) - 1) / 2) * 0.05 for i in range(len(dots))], dots, s=14, color=COLORS["ink"], zorder=3,
                   label="One series" if x == 0 else None)
        ax.text(x, max(result[k], max(dots)) + 0.015, fmt(result[k]), ha="center", va="bottom", fontsize=10)
    ax.axhline(result["future"], color=COLORS["gold"], ls=(0, (4, 3)), lw=1.6,
               label=f"Score on the future: {fmt(result['future'])}")
    ax.axhline(0, color=COLORS["grey"], lw=0.6)
    ax.set_xticks(xs, [name for _, name in SCHEMES])
    top = max(max(max(result["per_series"][k]) for k, _ in SCHEMES), result["future"])
    ax.set_ylim(min(0, min(min(result["per_series"][k]) for k, _ in SCHEMES)) - 0.02, top + 0.1)
    ax.set_ylabel("Information coefficient (IC)")
    ax.set_xlabel(f"Validation scheme, label horizon {parameter} days")
    ax.legend(loc="upper right", frameon=False, fontsize=10)


def explain(result, parameter):
    s, w, p, f = result["shuffled"], result["walk_forward"], result["purged"], result["future"]
    gap = s - f
    interpretation = (
        f"With a {parameter}-day label horizon the model's IC on the future is {fmt(f)}. Shuffled cross-validation reports "
        f"{fmt(s)}, and {fmt(s)} - {fmt(f)} = {fmt(gap)} is its error. Walk-forward reports {fmt(w)} and purged walk-forward {fmt(p)}, "
        f"{signed(w - f)} and {signed(p - f)} from the future. "
        + ("The shuffled estimate is the one that misleads." if gap > 0.05 else
           "At this horizon every scheme estimates the future about equally well."))
    steps = [
        f"Shuffled: {fmt(s)} - {fmt(f)} = {signed(gap)} (estimate minus future).",
        f"Walk-forward: {fmt(w)} - {fmt(f)} = {signed(w - f)}.",
        f"Purged walk-forward: {fmt(p)} - {fmt(f)} = {signed(p - f)}.",
        f"Effect of the purge gap: {fmt(p)} - {fmt(w)} = {signed(p - w)}.",
    ]
    metrics = {"Shuffled 5-fold IC": fmt(s), "Walk-forward IC": fmt(w), "Purged walk-forward IC": fmt(p),
               "IC on the future": fmt(f), "Shuffled minus future": signed(gap)}
    alt = (f"Bars of three validation estimates of IC at a {parameter}-day label horizon, dots for each of eight series, and a "
           f"dashed line at the future IC of {fmt(f)}; the shuffled estimate is {signed(gap)} from it.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for h, res in results.items():
        assert res["walk_forward"] - res["future"] < 0.02, f"walk-forward should not be optimistic at h={h}"
        assert abs(res["walk_forward"] - res["future"]) < 0.08, f"walk-forward far from future at h={h}"
        assert abs(res["purged"] - res["walk_forward"]) < 0.02, f"purge effect not small at h={h}: text says < 0.02"
        if h >= 5:
            assert res["shuffled"] - res["future"] > 0.05, f"shuffled not optimistic at h={h}"
    assert abs(results[1]["shuffled"] - results[1]["future"]) < 0.02, "h=1 should be about equal"
    r20 = results[20]
    assert fmt(r20["shuffled"]) == "0.346" and fmt(r20["future"]) == "0.256", "prediction feedback numbers"
    assert fmt(r20["shuffled"] - r20["future"]) == "0.090", "check answer number"
    assert 0.07 < r20["shuffled"] - r20["future"] < 0.11, "prediction option says about 0.09"
