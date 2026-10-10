"""Chapter 57: PatchTST for Time Series and HAR. A patch-token stand-in against a feature baseline at matched forward origins, by patch length."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
from functools import lru_cache

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor

from kaggle_companion.activities._common import clean

SEED = 57
REPLICATES = 6                       # independent sets of series; every estimate is their mean
SERIES, LENGTH, SPLIT = 30, 640, 420  # series per set, steps per series, last training step
WINDOW, HORIZON = 96, 12             # the model sees 96 steps and forecasts the value 12 steps after the last one
PULSE_RATE = 0.02                    # chance per step that a pulse starts
EMBED = 4                            # width of the learned embedding of each patch


def generate(rng):
    """Daily-style cycle (period 24 and 12) plus random pulses and AR noise. A pulse is a 10-step damped oscillation
    followed 16 steps after its start by a rebound whose height depends on |pulse size|, a nonlinear, local-shape effect."""
    t = np.arange(LENGTH)
    series = []
    for _ in range(SERIES):
        phase = rng.uniform(0, 24)
        y = rng.uniform(0.8, 1.2) * np.sin(2 * np.pi * (t + phase) / 24) + 0.4 * np.sin(2 * np.pi * (t + phase) / 12 + 1)
        for s in np.flatnonzero(rng.random(LENGTH) < PULSE_RATE):
            size = rng.choice([-1, 1]) * rng.uniform(1.0, 2.0)
            n = min(30, LENGTH - s)
            k = np.arange(n)
            y[s:s + n] += size * np.exp(-k / 10) * np.sin(2 * np.pi * k / 10)
            m = min(12, LENGTH - (s + 16))
            if m > 0:
                y[s + 16:s + 16 + m] += abs(size) * np.exp(-0.5 * ((np.arange(m) - 4) / 3.0) ** 2)
        noise = np.zeros(LENGTH)
        for i in range(1, LENGTH):
            noise[i] = 0.5 * noise[i - 1] + rng.normal(0, 0.15)
        series.append(y + noise)
    return np.array(series)


def make_windows(series, first, last, stride=3):
    """Input windows of 96 steps and the target 12 steps after the last input, all inside [first, last)."""
    X, y = [], []
    for s in series:
        for end in range(first + WINDOW, last - HORIZON + 1, stride):
            X.append(s[end - WINDOW:end])
            y.append(s[end - 1 + HORIZON])
    return np.array(X), np.array(y)


def patches(X, patch_length, stride):
    """(windows, tokens, patch_length): the same unfold as the chapter's temporal_patches, on a plain array."""
    out = sliding_window_view(X, patch_length, axis=1)[:, ::stride, :]
    assert out.shape[1] == (WINDOW - patch_length) // stride + 1, "token count must be floor((T-P)/S)+1"
    return out


def feature_baseline(X):
    """Lags and short rolling statistics: the 'feature-engineered' comparator."""
    cols = [X[:, -k] for k in (1, 2, 3, 6, 12, 24, 48)]
    cols += [X[:, -24:].mean(1), X[:, -24:].std(1), X[:, -12:].mean(1), X[:, -12:].std(1)]
    return np.column_stack(cols)


def patch_tokens(X_train, X_test, patch_length):
    """Stand-in for a patch embedding: a projection shared by all patches (PCA fitted on training patches only),
    applied to every patch, with the tokens concatenated. There is no attention layer."""
    stride = patch_length // 2                                    # 50% overlap
    train, test = patches(X_train, patch_length, stride), patches(X_test, patch_length, stride)
    pca = PCA(EMBED, random_state=0).fit(train.reshape(-1, patch_length))
    embed = lambda Z: pca.transform(Z.reshape(-1, patch_length)).reshape(len(Z), -1)
    return embed(train), embed(test), train.shape[1]


@lru_cache(maxsize=None)
def make_split(seed, replicate):
    """One set of series cut into training windows (targets before step 420) and forward test windows (after it)."""
    series = generate(np.random.default_rng(seed * 100 + replicate))
    return (*make_windows(series, 0, SPLIT), *make_windows(series, SPLIT, LENGTH))


@lru_cache(maxsize=None)
def baselines(seed, replicate):
    """Comparators that do not depend on the patch length, computed once per set."""
    X_train, y_train, X_test, y_test = make_split(seed, replicate)
    mae = lambda pred: float(np.mean(np.abs(pred - y_test)))
    return {"seasonal_naive": mae(X_test[:, WINDOW - 1 + HORIZON - 24]),   # the value one cycle (24 steps) before the target
            "features": mae(new_model().fit(feature_baseline(X_train), y_train).predict(feature_baseline(X_test))),
            "raw_window": mae(new_model().fit(X_train, y_train).predict(X_test))}


def new_model():
    return HistGradientBoostingRegressor(max_iter=60, learning_rate=0.12, max_leaf_nodes=15, random_state=0)


def one_set(patch_length, replicate):
    X_train, y_train, X_test, y_test = make_split(SEED, replicate)
    mae = lambda pred: float(np.mean(np.abs(pred - y_test)))
    P_train, P_test, tokens = patch_tokens(X_train, X_test, patch_length)
    return {**baselines(SEED, replicate),
            "patch": mae(new_model().fit(P_train, y_train).predict(P_test)), "tokens": tokens, "windows": len(y_test)}


def run(patch_length):
    sets = [one_set(patch_length, i) for i in range(REPLICATES)]
    keys = ["seasonal_naive", "features", "raw_window", "patch"]
    return clean({
        "patch_length": patch_length, "stride": patch_length // 2, "tokens": sets[0]["tokens"],
        "token_pairs": sets[0]["tokens"] ** 2,                  # attention cost grows with tokens squared
        **{k: float(np.mean([s[k] for s in sets])) for k in keys},
        "per_set": {k: [s[k] for s in sets] for k in keys},
        "replicates": REPLICATES, "test_windows": sets[0]["windows"],
    })
# notebook-end


SPEC = {
    "chapter": 57,
    "chapter_title": "PatchTST for Time Series and HAR",
    "subtitle": "Compare patching with a feature baseline at matched forward origins, and choose patch length from the phenomena.",
    "summary": ("Patching turns short blocks into tokens, and the patch length decides what a token can see. One demonstration "
                "compares a patch-token stand-in with a feature baseline, the raw window and a seasonal-naive forecast on the same forward "
                "origins, and counts the tokens that each patch length creates."),
    "title": "A patch-token stand-in against a feature baseline, by patch length",
    "question": "Does patching beat a feature baseline at the same forward origins, and how does the patch length change the answer and the token count?",
    "why": ("A sequence representation is only worth its cost if it beats a credible feature baseline on matched origins. The patch length "
            "decides how many tokens the model processes and whether one token can hold the shape that matters."),
    "method": ("Six constructed sets of 30 series of 640 steps: a 24-step and 12-step cycle, random pulses (a 10-step damped oscillation "
               "followed 16 steps later by a rebound that depends on the pulse size) and autocorrelated noise. Every model sees a "
               "96-step window and forecasts the value 12 steps after it, and is scored by mean absolute error (MAE) on windows taken "
               "after the last training step. The patch model unfolds the window into patches of length P with 50% overlap, embeds "
               "every patch with one shared PCA projection to 4 numbers, and feeds the concatenated tokens to gradient boosting; there "
               "is no attention. The comparators are a seasonal-naive forecast, gradient boosting on lag and rolling features, "
               "and gradient boosting on the raw 96 values."),
    "control": {"key": "patch_length", "label": "Patch length P in steps (stride is P/2; the pulse oscillation lasts 10)",
                "values": [4, 12, 24, 48], "default": 12,
                "value_labels": ["4: 47 tokens", "12: 15 tokens", "24: 7 tokens", "48: 3 tokens"]},
    "source_section": "PatchTST vs. Feature-Engineered GBM: When to Use Each",
    "symbols": ("T is the window length (96), P the patch length, S the stride (P/2) and the number of tokens without padding is "
                "floor((T - P)/S) + 1. MAE is the mean absolute forecast error, so lower is better."),
    "explanation": ("Short patches keep the local shape of a pulse, so the embedded tokens expose the oscillation that predicts "
                    "the rebound. Long patches average that shape away inside one token, and with few tokens the model sees "
                    "little more than a coarse summary, which the feature baseline also provides. Shorter patches cost more tokens, and "
                    "attention cost grows with the square of the token count."),
    "application": ("Compare a patch model with a feature baseline and the raw window on identical forward origins, pick the patch "
                    "length from the duration of the events that matter, and check the token count and cost before a long run."),
    "assumptions": ("A stand-in, not PatchTST: patches share a PCA projection and feed gradient boosting, with no positional encoding, "
                    "attention or end-to-end training, because the neural library is not installed here. The rebound makes the "
                    "target depend on local pulse shape by construction, which favours a representation that sees shape; a signal that "
                    "is a pure cycle gives the patch and feature models no such gap. One window length, one horizon and six sets; "
                    "the raw-window comparator already uses all 96 values."),
    "prediction": "With a patch length of 24 steps, how does the patch-token model compare with the feature baseline?",
    "prediction_options": ["Better, by the same margin as a short patch", "Worse than the feature baseline", "About the same"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "At P = 24 the patch model scores an MAE of 0.461 against 0.447 for the feature baseline, while P = 12 scores 0.414.",
        "incorrect": "At P = 24 the patch model scores an MAE of 0.461 against 0.447 for the feature baseline, while P = 12 scores 0.414. A token that spans two pulse lengths averages the shape away.",
    },
    "check": "A patch length of 4 scores best here. Why is that not simply the answer to use?",
    "answer": ("P = 4 creates 47 tokens against 15 for P = 12, about ten times as many token pairs for attention to relate, and its error "
               "gain is 0.025 (0.389 against 0.414). Whether that is worth the cost depends on the budget; the chapter asks for the "
               "actual token count and cost to be measured alongside the score. It also depends on the phenomena: the pulses here last about 10 steps."),
    "provenance": "Constructed example: six seeded sets of 30 synthetic series, a patch-token stand-in with gradient boosting, measured by the chapter activity.",
    "apply": [
        "Compute the exact token count floor((T - P)/S) + 1 for each candidate patch length before any long run.",
        "Compare the sequence model with a feature baseline and a raw-window baseline on the same forward origins and the same metric.",
        "Choose the patch length from the duration of the events and the sampling rate, then test one longer and one shorter patch.",
        "Report the cost (tokens, feature extraction, inference time) next to the score; a small gain from many more tokens is a decision, not a free win.",
    ],
    "honesty": ("Constructed series in which a local shape matters by design, and a stand-in model. The advantage of patching over the feature "
                "baseline here is modest (about 7% in MAE at P = 12), and gradient boosting on the raw 96 values matches it (0.409 against "
                "0.414), so the gain comes from seeing the shape of the whole window, which the lag features miss, rather than from patching "
                "itself. This is not evidence about PatchTST on any real competition."),
}

EQUATIONS = [{"tex": r"N_{\mathrm{tokens}} = \left\lfloor \frac{T - P}{S} \right\rfloor + 1",
              "alt": "the number of tokens equals the floor of the quantity T minus P, divided by S, plus 1",
              "basis": "The chapter's no-padding patch count (display formula in Chapter 57, The Architecture: Patches, Position, and Representation)."}]
NCOLS = 1
HEIGHT = 4.4
BARS = [("seasonal_naive", "Seasonal\nnaive", COLORS["light"]), ("features", "Lag and rolling\nfeatures", COLORS["gold"]),
        ("raw_window", "Raw 96-step\nwindow", COLORS["navy"]), ("patch", "Patch tokens", COLORS["teal"])]


def draw(ax, result, parameter):
    xs = list(range(len(BARS)))
    ax.bar(xs, [result[k] for k, _, _ in BARS], width=0.6, color=[c for _, _, c in BARS], edgecolor=COLORS["ink"], lw=0.6)
    for x, (k, _, _) in zip(xs, BARS):
        dots = result["per_set"][k]
        ax.scatter([x + (i - (len(dots) - 1) / 2) * 0.05 for i in range(len(dots))], dots, s=12, color=COLORS["ink"], zorder=3,
                   label="One set of series" if x == 0 else None)
        ax.text(x, max(dots) + 0.02, fmt(result[k]), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(xs, [name for _, name, _ in BARS])
    ax.set_ylim(0, max(max(v) for v in result["per_set"].values()) * 1.3)
    ax.set_ylabel("Forecast error (MAE, lower is better)")
    ax.set_xlabel(f"Model, patch length {parameter} ({result['tokens']} tokens per window)")
    ax.legend(loc="upper right", frameon=False, fontsize=10)


def diff(a, b, digits=3):
    """Difference of the two numbers as displayed, so the hand calculation on the page adds up."""
    return float(fmt(a, digits)) - float(fmt(b, digits))


def explain(result, parameter):
    p, f, r, n = result["patch"], result["features"], result["raw_window"], result["seasonal_naive"]
    gain = diff(f, p)
    if p < f:
        verdict = f"beats the feature baseline at {fmt(f)}, so {fmt(f)} - {fmt(p)} = {fmt(gain)} separates them"
    else:
        verdict = f"does not beat the feature baseline at {fmt(f)}: {fmt(p)} - {fmt(f)} = {fmt(-gain)} is what it gives up"
    interpretation = (
        f"With patches of {parameter} steps (stride {result['stride']}) each 96-step window becomes {result['tokens']} tokens, "
        f"floor((96 - {parameter}) / {result['stride']}) + 1 = {result['tokens']}. The patch model scores an MAE of {fmt(p)} and {verdict}. The raw 96-step window scores {fmt(r)} and the seasonal-naive "
        f"forecast {fmt(n)}. Attention would relate {result['token_pairs']:,} pairs of tokens per window.")
    steps = [
        f"Tokens: floor((96 - {parameter}) / {result['stride']}) + 1 = {result['tokens']}.",
        f"Feature baseline minus patch model: {fmt(f)} - {fmt(p)} = {signed(gain)} (positive means patching is better).",
        f"Raw window minus patch model: {fmt(r)} - {fmt(p)} = {signed(diff(r, p))}.",
        f"Token pairs for attention: {result['tokens']} x {result['tokens']} = {result['token_pairs']:,}.",
    ]
    metrics = {"Patch-token MAE": fmt(p), "Feature baseline MAE": fmt(f), "Raw window MAE": fmt(r), "Seasonal naive MAE": fmt(n),
               "Tokens per window": str(result["tokens"])}
    alt = (f"Bars of forecast error with patch length {parameter}: seasonal naive {fmt(n)}, feature baseline {fmt(f)}, raw window {fmt(r)}, "
           f"patch tokens {fmt(p)}, with a dot for each of six sets of series.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def _qualitative(results):
    """Directional claims; checked again under a different seed base."""
    for p, res in results.items():
        assert res["tokens"] == (96 - p) // (p // 2) + 1, "token count formula"
        assert res["features"] < res["seasonal_naive"] and res["patch"] < res["seasonal_naive"], f"learned models beat naive at {p}"
    short, mid, long_ = results[4], results[12], results[24]
    assert mid["patch"] < mid["features"], "a pulse-length patch beats the feature baseline"
    assert short["patch"] < mid["patch"] < long_["patch"], "shorter patch, lower error here"
    assert long_["patch"] > long_["features"], "a patch of 24 loses to the feature baseline"
    assert 0.03 < (mid["features"] - mid["patch"]) / mid["features"] < 0.15, "honesty: patch advantage at P=12 is modest"
    assert mid["raw_window"] < mid["features"] and abs(mid["raw_window"] - mid["patch"]) < 0.015, "honesty: raw window matches patching at P=12"
    assert short["token_pairs"] > 8 * mid["token_pairs"], "token pairs grow fast"


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    _qualitative(results)
    assert fmt(results[24]["patch"]) == "0.461" and fmt(results[24]["features"]) == "0.447" and fmt(results[12]["patch"]) == "0.414", "feedback"
    assert results[4]["tokens"] == 47 and results[12]["tokens"] == 15 and results[24]["tokens"] == 7 and results[48]["tokens"] == 3, "labels"
    assert fmt(diff(results[12]["patch"], results[4]["patch"]), 3) == "0.025" and fmt(results[4]["patch"]) == "0.389", "answer"
    assert round(100 * (results[12]["features"] - results[12]["patch"]) / results[12]["features"]) == 7, "honesty: about 7%"
    assert fmt(results[12]["raw_window"]) == "0.409", "honesty: raw window number"
    assert 9 < results[4]["token_pairs"] / results[12]["token_pairs"] < 11, "about ten times"
