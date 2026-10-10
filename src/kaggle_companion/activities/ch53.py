"""Chapter 53: Financial Competition Patterns. Feature neutralization scored against raw and factor-neutral returns, as the factor premium changes."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.linear_model import Ridge

from kaggle_companion.activities._common import clean

SEED = 53
REPLICATES = 8                  # independent constructed markets; every estimate is their mean
ASSETS, TRAIN_DATES, TEST_DATES = 120, 200, 200
LOAD = np.array([0.8, 0.6, 0.5, 0.4, 0.3])      # how strongly each feature carries the market exposure (beta)
SIGNAL = np.array([0.5, 0.4, 0.4, 0.3, 0.3])    # how strongly each feature carries the true alpha signal
TRAIN_PREMIUM = 0.5             # mean daily factor return while the model is trained
FRACTIONS = [0.0, 0.5, 1.0]     # share of the exposure removed from each feature


def make_market(rng, beta, dates, premium):
    """Features, exposures and next-day returns for `dates` dates. A common factor return moves every asset by its beta."""
    alpha = rng.normal(size=(dates, ASSETS))                       # hidden idiosyncratic signal
    exposure = np.tile(beta, (dates, 1))
    X = np.stack([SIGNAL[k] * alpha + LOAD[k] * exposure + rng.normal(size=(dates, ASSETS)) for k in range(len(LOAD))], axis=-1)
    factor = rng.normal(premium, 0.5, size=(dates, 1))             # the factor premium: mean `premium`, daily noise 0.5
    ret = 0.06 * alpha + 0.5 * exposure * factor + 0.5 * rng.normal(size=(dates, ASSETS))
    return X, exposure, ret


def neutralize(X, exposure, fraction):
    """Cross-sectional regression per date: subtract `fraction` of each feature's projection on the exposure."""
    b = exposure - exposure.mean(axis=1, keepdims=True)
    slope = ((X - X.mean(axis=1, keepdims=True)) * b[..., None]).sum(axis=1) / (b ** 2).sum(axis=1, keepdims=True)
    return X - fraction * b[..., None] * slope[:, None, :]


def factor_neutral(ret, exposure):
    """The return left after the day's regression on exposure: what a factor-neutral metric scores."""
    b = exposure - exposure.mean(axis=1, keepdims=True)
    centered = ret - ret.mean(axis=1, keepdims=True)
    return centered - b * ((b * centered).sum(axis=1, keepdims=True) / (b ** 2).sum(axis=1, keepdims=True))


def ranks(a):
    """Per-date ranks scaled to [-0.5, 0.5]."""
    return (a.argsort(axis=1).argsort(axis=1) + 0.5) / a.shape[1] - 0.5


def date_correlations(pred, target):
    """Spearman correlation between prediction and target on each date."""
    p, t = ranks(pred), ranks(target)
    return (p * t).sum(axis=1) / np.sqrt((p ** 2).sum(axis=1) * (t ** 2).sum(axis=1))


def one_market(test_premium, replicate):
    rng = np.random.default_rng(SEED * 100 + replicate)
    beta = rng.normal(size=ASSETS)
    X_train, e_train, r_train = make_market(rng, beta, TRAIN_DATES, TRAIN_PREMIUM)
    X_test, e_test, r_test = make_market(rng, beta, TEST_DATES, test_premium)
    targets = {"raw": r_test, "neutral": factor_neutral(r_test, e_test)}
    out = {}
    for fraction in FRACTIONS:
        # Features are neutralized with the same rule in training and in the later period; the target is never altered.
        model = Ridge(alpha=100).fit(neutralize(X_train, e_train, fraction).reshape(-1, len(LOAD)), ranks(r_train).reshape(-1))
        pred = model.predict(neutralize(X_test, e_test, fraction).reshape(-1, len(LOAD))).reshape(TEST_DATES, ASSETS)
        for name, target in targets.items():
            c = date_correlations(pred, target)
            out[(fraction, name)] = (float(c.mean()), float(c.std()))
    return out


def run(test_premium):
    markets = [one_market(test_premium, i) for i in range(REPLICATES)]
    score = {}
    for name in ("raw", "neutral"):
        score[name] = {str(f): {"mean": float(np.mean([m[(f, name)][0] for m in markets])),
                               "daily_sd": float(np.mean([m[(f, name)][1] for m in markets])),
                               "per_market": [m[(f, name)][0] for m in markets]} for f in FRACTIONS}
    return clean({"test_premium": test_premium, "score": score, "train_premium": TRAIN_PREMIUM,
                  "replicates": REPLICATES, "assets": ASSETS, "test_dates": TEST_DATES})
# notebook-end


SPEC = {
    "chapter": 53,
    "chapter_title": "Financial Competition Patterns",
    "subtitle": "Whether to neutralize factor exposure depends on the target and the official metric.",
    "summary": ("Factor exposure is not leakage, and removing it is a modelling choice that depends on what is scored. One "
                "demonstration measures a model built on raw and on neutralized features against raw returns and against "
                "factor-neutral returns, while the factor premium after training changes."),
    "title": "Raw and neutralized features, scored against raw and factor-neutral returns",
    "question": "When does removing factor exposure from the features help the score, and when does it cost correlation?",
    "why": ("A feature that carries market exposure is legal, but whether it helps depends on the scored target and on whether the "
            "factor premium seen in training continues. Neutralizing is a candidate representation to compare under the same "
            "metric, not a rule."),
    "method": ("Eight constructed markets of 120 assets. Five features each mix a hidden alpha signal with the asset's market "
               "exposure (beta). Returns are a small alpha term, beta times a daily factor return, and noise. Ridge regression "
               "is trained on 200 dates while the factor premium averages 0.5, then scored on 200 later dates in which the premium "
               "is the control. Features are neutralized by a per-date cross-sectional regression on beta, removing 0%, 50% or "
               "100% of the exposure. Each model is scored by the mean per-date Spearman correlation against the raw returns "
               "and against the returns left after removing the day's beta effect."),
    "control": {"key": "test_premium", "label": "Mean factor return in the scored period (training period: 0.5)",
                "values": [0.5, 0.25, 0.0, -0.5], "default": -0.5,
                "value_labels": ["0.5: premium persists", "0.25: half as strong", "0: premium gone", "-0.5: premium flips sign"]},
    "source_section": "Feature Neutralization: Removing Market Exposure",
    "symbols": ("x is a feature, b the asset's exposure to the factor, and the neutralized feature x* subtracts a fraction q of "
                "the day's regression of x on b. The score is the mean over dates of the Spearman correlation between a "
                "prediction and a target."),
    "explanation": ("Raw features carry exposure to the factor, and the model learns that exposure pays because it did during "
                    "training. While the premium persists, that exposure earns correlation on raw returns. When it fades or flips, the "
                    "same exposure turns into a loss, and the daily correlation swings with the factor. Neutralized features "
                    "give up the exposure bet in every regime and keep only the alpha signal, which is all a factor-neutral "
                    "metric can reward."),
    "application": ("Read the official metric before choosing a representation: score raw and neutralized versions of the same "
                    "model on the exact target and compare them on later periods, not only on the training regime."),
    "assumptions": ("Constructed markets with one factor, a static beta per asset and an alpha signal that does not change; "
                    "Ridge regression stands in for any model and eight markets give the means. Exposure here is observed without "
                    "error, which is generous: with a noisy exposure estimate, neutralization would remove less. The premium "
                    "in the scored period is set by hand to show the regimes. The result concerns the scoring target, and is not a "
                    "claim about any real competition or about trading profit."),
    "prediction": "If the factor premium flips sign after training, what happens to the raw-feature model's correlation with raw returns?",
    "prediction_options": ["It stays positive but smaller", "It turns negative, about -0.19", "It is unchanged"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "The raw-feature model scores -0.186 against raw returns, while the fully neutralized model scores 0.061.",
        "incorrect": "The raw-feature model scores -0.186 against raw returns: the exposure it learned is now paid the wrong way. The fully neutralized model scores 0.061.",
    },
    "check": "Neutralization lowers the score on raw returns when the premium persists. Does that make it the wrong choice?",
    "answer": ("Not by itself. The score on raw returns at a persistent premium (0.274 for raw features, 0.063 for fully neutralized ones) rewards a "
               "bet on the factor, and whether the competition scores that bet is a property of the official metric, not of the "
               "recipe. On the factor-neutral target the neutralized model matches or beats the raw one in every regime. The choice "
               "follows the scored target and the plausibility that the premium persists."),
    "provenance": "Constructed example: eight seeded synthetic markets of 120 assets and a Ridge model, measured by the chapter activity.",
    "apply": [
        "Find out whether the official score is on raw returns or on a factor-neutral or residual target before neutralizing anything.",
        "Compare the raw and neutralized versions of the same model on the exact metric, on later periods, and keep the unneutralized comparator.",
        "Look at the per-date scores as well as the mean: a model whose daily correlation swings with the factor is betting on the premium.",
        "Never alter the requested target to follow a recipe; neutralize inputs or predictions only when the comparison supports it.",
    ],
    "honesty": ("Constructed data with a strong factor; the sizes are properties of this generator. Neutralization gains on the "
                "factor-neutral target are small here (about 0.02); the large effects are the regime-dependent ones on raw returns."),
}

EQUATIONS = [{"tex": r"x^{*}_{i,d} = x_{i,d} - q\,\hat{\gamma}_d\,(b_i - \bar b)",
              "alt": "the neutralized feature x star for asset i on date d equals the raw feature minus q times the date's fitted slope gamma hat d times the asset's exposure b i minus the mean exposure",
              "basis": "The per-date cross-sectional projection the chapter's neutralization code applies; q is the activity's removal fraction (the chapter removes the whole projection, q = 1)."}]
NCOLS = 2
HEIGHT = 4.4
LEVELS = [("0.0", "None", COLORS["terracotta"]), ("0.5", "Half", COLORS["gold"]), ("1.0", "Full", COLORS["teal"])]


def draw(axes, result, parameter):
    panels = [("raw", "Scored on raw returns"), ("neutral", "Scored on factor-neutral returns")]
    lo = min(min(result["score"][n][k]["per_market"]) for n, _ in panels for k, _, _ in LEVELS)
    hi = max(max(result["score"][n][k]["per_market"]) for n, _ in panels for k, _, _ in LEVELS)
    pad = (hi - lo) * 0.18 + 0.02
    for ax, (name, label) in zip(axes, panels):
        for x, (k, _, color) in enumerate(LEVELS):
            s = result["score"][name][k]
            ax.bar(x, s["mean"], width=0.6, color=color, edgecolor=COLORS["ink"], lw=0.6)
            dots = s["per_market"]
            ax.scatter([x + (i - (len(dots) - 1) / 2) * 0.05 for i in range(len(dots))], dots, s=12, color=COLORS["ink"], zorder=3,
                       label="One market" if (x == 0 and name == "raw") else None)
            top = max(dots + [s["mean"]])
            bottom = min(dots + [s["mean"]])
            if s["mean"] >= 0:
                ax.text(x, top + pad * 0.12, fmt(s["mean"]), ha="center", va="bottom", fontsize=10)
            else:
                ax.text(x, bottom - pad * 0.12, fmt(s["mean"]), ha="center", va="top", fontsize=10)
        ax.axhline(0, color=COLORS["grey"], lw=0.8)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(range(len(LEVELS)), [n for _, n, _ in LEVELS])
        ax.set_xlabel(f"Exposure removed from features\n{label}")
    for ax in axes:
        ax.set_ylabel("Mean daily Spearman correlation")
    axes[0].legend(loc="upper right", frameon=False, fontsize=10)


def diff(a, b):
    """Difference of the two numbers as displayed (3 decimals), so the hand calculation on the page adds up."""
    return float(fmt(a)) - float(fmt(b))


def explain(result, parameter):
    raw, neu = result["score"]["raw"], result["score"]["neutral"]
    r0, r1 = raw["0.0"]["mean"], raw["1.0"]["mean"]
    n0, n1 = neu["0.0"]["mean"], neu["1.0"]["mean"]
    cost = diff(r0, r1)
    gain = diff(n1, n0)
    if cost > 0:
        change = f"{fmt(r0)} - {fmt(r1)} = {fmt(cost)} is what neutralizing costs there"
    else:
        sub = f"({fmt(r0)})" if float(fmt(r0)) < 0 else fmt(r0)   # a negative subtrahend is written in parentheses
        change = f"{fmt(r1)} - {sub} = {fmt(abs(cost))} is what neutralizing gains there"
    interpretation = (
        f"With a premium of {fmt(parameter, 2)} in the scored period (0.5 in training), raw features score {fmt(r0)} against raw returns and "
        f"fully neutralized features {fmt(r1)}, so {change}. Against factor-neutral returns the scores are {fmt(n0)} and {fmt(n1)}. "
        f"The daily correlation on raw returns has a standard deviation of {fmt(raw['0.0']['daily_sd'])} with raw features and "
        f"{fmt(raw['1.0']['daily_sd'])} with neutralized ones.")
    steps = [
        f"Cost of full neutralization on raw returns: {fmt(r0)} - {fmt(r1)} = {signed(cost)}.",
        f"Gain on factor-neutral returns: {fmt(n1)} - {fmt(n0)} = {signed(gain)}.",
        f"Half neutralization on raw returns: {fmt(raw['0.5']['mean'])}, between the two ends.",
        f"Daily swing on raw returns, raw features minus neutralized: {fmt(raw['0.0']['daily_sd'])} - {fmt(raw['1.0']['daily_sd'])} = {signed(diff(raw['0.0']['daily_sd'], raw['1.0']['daily_sd']))}.",
    ]
    metrics = {"Raw features, raw returns": fmt(r0), "Neutralized, raw returns": fmt(r1),
               "Raw features, neutral returns": fmt(n0), "Neutralized, neutral returns": fmt(n1),
               "Cost of neutralizing (raw returns)": signed(cost)}
    alt = (f"Two panels of mean daily Spearman correlation for no, half and full removal of exposure at a premium of {fmt(parameter, 2)}. "
           f"On raw returns the scores are {fmt(r0)}, {fmt(raw['0.5']['mean'])} and {fmt(r1)}; on factor-neutral returns "
           f"{fmt(n0)}, {fmt(neu['0.5']['mean'])} and {fmt(n1)}.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def _qualitative(results):
    """Directional claims; checked again under a different seed base."""
    for p, res in results.items():
        raw, neu = res["score"]["raw"], res["score"]["neutral"]
        assert neu["1.0"]["mean"] > neu["0.0"]["mean"] - 0.005, f"neutralizing must not hurt the neutral target at premium {p}"
        assert neu["1.0"]["mean"] - neu["0.0"]["mean"] < 0.03, f"honesty: neutral-target gain stays small at premium {p}"
        assert raw["1.0"]["daily_sd"] < raw["0.0"]["daily_sd"], f"neutralized daily scores should be steadier at {p}"
        assert abs(raw["1.0"]["mean"] - neu["1.0"]["mean"]) < 0.03, f"fully neutralized features score alike on both targets at {p}"
        assert min(raw["0.0"]["mean"], raw["1.0"]["mean"]) <= raw["0.5"]["mean"] <= max(raw["0.0"]["mean"], raw["1.0"]["mean"]), \
            f"steps: half neutralization lies between the two ends at {p}"
    assert results[0.5]["score"]["raw"]["0.0"]["mean"] - results[0.5]["score"]["raw"]["1.0"]["mean"] > 0.1, "persistent premium: neutralizing costs"
    assert results[-0.5]["score"]["raw"]["0.0"]["mean"] < -0.1 and results[-0.5]["score"]["raw"]["1.0"]["mean"] > 0, "flip: raw model negative"
    assert results[0.0]["score"]["raw"]["1.0"]["mean"] > results[0.0]["score"]["raw"]["0.0"]["mean"], "no premium: neutralized is better or equal"
    assert results[0.25]["score"]["raw"]["0.0"]["mean"] > results[0.25]["score"]["raw"]["1.0"]["mean"], "half premium: raw still ahead"


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    _qualitative(results)
    flip, keep = results[-0.5]["score"], results[0.5]["score"]
    assert fmt(flip["raw"]["0.0"]["mean"]) == "-0.186" and fmt(flip["raw"]["1.0"]["mean"]) == "0.061", "prediction feedback numbers"
    assert -0.22 < flip["raw"]["0.0"]["mean"] < -0.16, "option says about -0.19"
    assert fmt(keep["raw"]["0.0"]["mean"]) == "0.274" and fmt(keep["raw"]["1.0"]["mean"]) == "0.063", "answer numbers"
    for res in results.values():
        n = res["score"]["neutral"]
        assert n["1.0"]["mean"] >= n["0.0"]["mean"] - 0.005, "answer: neutralized matches or beats raw on the neutral target"
        assert fmt(n["1.0"]["mean"] - n["0.0"]["mean"], 2) == "0.02", "honesty: gain is about 0.02"
