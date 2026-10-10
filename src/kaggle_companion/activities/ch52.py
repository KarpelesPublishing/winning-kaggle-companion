"""Chapter 52: Time Series Feature Engineering. A shift-by-one feature builder backtested with true lags, against features built at the forecast origin."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from kaggle_companion.activities._common import clean

SEED = 52
REPLICATES = 10             # independent panels of constructed series; every estimate is their mean
N_SERIES, DAYS, TRAIN_END = 30, 450, 300
LAGS = [1, 2, 3, 7, 14, 21]  # the lag set a shift-based builder would create


def generate(rng):
    """Daily series with a weekday profile and autocorrelated noise (AR coefficient 0.8), one row per series."""
    series = []
    for _ in range(N_SERIES):
        level = rng.uniform(50, 150)
        profile = rng.normal(0, 1, 7)
        profile = (profile - profile.mean()) * level * 0.06
        noise = np.zeros(DAYS)
        for t in range(1, DAYS):
            noise[t] = 0.8 * noise[t - 1] + rng.normal(0, level * 0.04)
        series.append(level + profile[np.arange(DAYS) % 7] + noise)
    return np.array(series)


def features(y, target_days, horizon, kind):
    """Feature columns for the target days. The forecast origin is `horizon` days earlier.

    leaky:  lag k reads day t-k, which is what shift(k) does; at horizon > k that day is still in the future.
    stale:  the same columns as production would fill them, with the latest value known at the origin.
    honest: built at the origin: keep only lags k >= horizon, and a 7-day mean ending at the origin.
    """
    origin = target_days - horizon
    cols = []
    for k in LAGS:
        if kind == "honest" and k < horizon:
            continue
        source = np.minimum(target_days - k, origin) if kind == "stale" else target_days - k
        cols.append(y[:, source])
    end = target_days - 1 if kind == "leaky" else origin   # the rolling mean ends yesterday, or at the origin
    total = np.concatenate([np.zeros((len(y), 1)), np.cumsum(y, axis=1)], axis=1)
    cols.append((total[:, end + 1] - total[:, end - 6]) / 7)
    cols.append(np.tile(target_days % 7, (len(y), 1)))     # weekday is known in advance
    return np.column_stack([c.reshape(-1) for c in cols])


def new_model():
    return HistGradientBoostingRegressor(max_iter=80, learning_rate=0.1, max_leaf_nodes=12, random_state=0)


def one_panel(horizon, replicate):
    rng = np.random.default_rng(SEED * 1000 + replicate)
    y = generate(rng)
    train = np.arange(30, TRAIN_END)
    test = np.arange(TRAIN_END + horizon, DAYS)             # every test origin is after the last training day
    y_train, y_test = y[:, train].reshape(-1), y[:, test].reshape(-1)
    leaky = new_model().fit(features(y, train, horizon, "leaky"), y_train)
    honest = new_model().fit(features(y, train, horizon, "honest"), y_train)
    mae = lambda pred: float(np.mean(np.abs(pred - y_test)))
    out = {"backtest": mae(leaky.predict(features(y, test, horizon, "leaky"))),
           "deployed": mae(leaky.predict(features(y, test, horizon, "stale"))),
           "honest": mae(honest.predict(features(y, test, horizon, "honest"))),
           "seasonal_naive": mae(y[:, test - 7].reshape(-1))}  # same weekday last week, usable at every horizon here
    # Origin test from the chapter: change every value published after the origin and see which features move.
    target = np.array([test[0]])
    shifted = y.copy()
    shifted[:, target[0] - horizon + 1:] += rng.normal(0, 10, shifted[:, target[0] - horizon + 1:].shape)
    for kind in ("leaky", "honest"):
        before, after = features(y, target, horizon, kind), features(shifted, target, horizon, kind)
        out["moved_" + kind] = int((np.abs(before - after) > 1e-9).any(axis=0).sum())
        out["columns_" + kind] = before.shape[1]
    return out


def run(horizon):
    panels = [one_panel(horizon, i) for i in range(REPLICATES)]
    mean = lambda key: float(np.mean([p[key] for p in panels]))
    keys = ["backtest", "deployed", "honest", "seasonal_naive"]
    return clean({
        "horizon": horizon,
        **{k: mean(k) for k in keys},
        "per_panel": {k: [p[k] for p in panels] for k in keys},
        "moved_leaky": panels[0]["moved_leaky"], "columns_leaky": panels[0]["columns_leaky"],
        "moved_honest": panels[0]["moved_honest"], "columns_honest": panels[0]["columns_honest"],
        "replicates": REPLICATES, "series": N_SERIES, "test_days": DAYS - TRAIN_END - horizon,
    })
# notebook-end


SPEC = {
    "chapter": 52,
    "chapter_title": "Time Series Feature Engineering",
    "subtitle": "A lag is useful only when it describes information available at the forecast origin.",
    "summary": ("A feature builder that shifts by one row is correct for a next-day forecast and quietly wrong for a three-day one. "
                "One demonstration measures how far a backtest built with shift-by-one features drifts from the error of "
                "features built at the forecast origin, as the horizon grows."),
    "title": "Backtest error with shift-by-one features, against features built at the forecast origin",
    "question": "How much does a backtest flatter a model whose lag features read days that would not yet exist at the forecast origin?",
    "why": ("A perfectly aligned array can still contain values that are unavailable when the forecast is made. A lag-1 column "
            "is safe for a next-day forecast and a leak for a three-day one, and the backtest cannot tell the difference "
            "because it holds the true values."),
    "method": ("Ten constructed panels of 30 daily series, each with a weekday profile and autocorrelated noise. A gradient boosting "
               "regressor forecasts the value h days after the origin, where h is the control, and is scored by mean absolute error "
               "(MAE) on the 150 days after the training period. Two pipelines are built: a shift-based builder that creates "
               "lags 1, 2, 3, 7, 14 and 21 and a 7-day mean ending yesterday, and a direct builder that keeps only lags of at least h "
               "days and a 7-day mean ending at the origin. The shift-based model is scored twice: in a backtest with the true lag "
               "values, and as production would run it, with each lag that is still in the future filled by the latest value known at the origin. "
               "An origin test then perturbs every value after the origin and counts the feature columns that move."),
    "control": {"key": "horizon", "label": "Forecast horizon (days between the origin and the target day)",
                "values": [1, 3, 5, 7], "default": 3,
                "value_labels": ["1: next day", "3: three days ahead", "5", "7: a week ahead"]},
    "source_section": "One-Step and Three-Step Forecasts Use Different Histories",
    "symbols": ("t is the target day, h the horizon in days, o = t - h the forecast origin, and k the lag in days. A lag k is "
                "usable at the origin only when day t - k is at or before o, which holds exactly when k is at least h. "
                "MAE is the mean of the absolute forecast errors."),
    "explanation": ("At h = 1 every lag the shift-based builder creates is already known at the origin, so the backtest, the "
                    "production score and the direct model agree. As h grows, lag 1 and its neighbours read days after the origin. "
                    "The backtest keeps scoring with the true values, so its error does not move, while the direct model loses "
                    "the most informative lags and its error grows, levelling off at 5 and 7 days, where it keeps the same "
                    "lags (7, 14 and 21). The direct model also beats the shift-based model run "
                    "as production would run it, because it was trained on columns it will actually have."),
    "application": ("Store the forecast origin with every row, build features from history up to that origin only, keep lags "
                    "of at least the horizon, and test the builder by changing values after the origin."),
    "assumptions": ("Constructed series with a stable weekday pattern and AR(1) noise, one boosting model "
                    "(scikit-learn's histogram gradient boosting stands in for LightGBM) and ten panels. The size of the "
                    "gap depends on how strong the short-range autocorrelation is. Production-side filling of a missing lag with "
                    "the last known value is one common choice; other fills would give other deployed scores, and the origin test, "
                    "not the fill, is what reveals the problem. The seasonal-naive line repeats the value from the same weekday the week before."),
    "prediction": "At a three-day horizon, how does the backtest MAE of the shift-based pipeline compare with the MAE of the model built at the origin?",
    "prediction_options": ["About the same", "Lower by about 0.7, so the shift-based pipeline looks better than it is", "Higher by about 0.7"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "The backtest reports an MAE of 4.96 against 5.69 for the direct model at three days, a gap of 0.73, and the shift-based model scores 5.95 when it has to run at the origin.",
        "incorrect": "The backtest reports an MAE of 4.96 against 5.69 for the direct model at three days, a gap of 0.73. The backtest reads lag 1 and lag 2 days that do not exist yet at the origin, and the shift-based model scores 5.95 when it has to run there.",
    },
    "check": "Why does the direct model's error grow with the horizon, while the shift-based backtest stays flat?",
    "answer": ("Autocorrelation decays: the nearest known day says less about a target that is further away. The direct model can only use "
               "lags of at least h, so its information shrinks as h grows, and its error rises until only the weekly lags remain (6.03 at both "
               "5 and 7 days). The shift-based backtest keeps reading lag 1 "
               "from the true series at every horizon, so it never loses information, and its error stays near 5.0. The gap between the "
               "two is the error the backtest hides."),
    "provenance": "Constructed example: thirty seeded synthetic daily series per panel, ten panels and a gradient boosting model, measured by the chapter activity.",
    "apply": [
        "Write down the forecast origin and the reporting delay of each input before writing any shift, and store forecast_origin with every training row.",
        "Build features from history at or before the origin: keep lags of at least the horizon, and shift before rolling or differencing.",
        "Run the origin test on the feature builder: change values published after the origin and assert that the inputs stay unchanged.",
        "If a shift-based backtest looks far better than a seasonal-naive baseline that works at the real horizon, suspect the lags before celebrating the model.",
    ],
    "honesty": ("Constructed data with strong short-range autocorrelation; the sizes of these gaps are properties of this generator, "
                "not a competition result. The production-style score is only modestly worse than the direct model, so the "
                "main damage of a leaky builder is the false backtest, not the final forecast."),
}

EQUATIONS = [{"tex": r"o = t - h, \qquad \text{lag } k \text{ is usable} \iff t - k \le o \iff k \ge h",
              "alt": "the forecast origin o equals t minus h, and lag k is usable if and only if t minus k is at most o, which is the same as k being at least h",
              "basis": "The activity's own availability rule, stated from the chapter's one-step and three-step example; not a display equation in the manuscript."}]
NCOLS = 1
HEIGHT = 4.4
BARS = [("backtest", "Shift-based\nbacktest\n(true lags)", COLORS["terracotta"]),
        ("deployed", "Shift-based\nrun at the\norigin", COLORS["light"]),
        ("honest", "Direct model\nbuilt at the\norigin", COLORS["teal"])]


def draw(ax, result, parameter):
    xs = list(range(len(BARS)))
    ax.bar(xs, [result[k] for k, _, _ in BARS], width=0.55, color=[c for _, _, c in BARS], edgecolor=COLORS["ink"], lw=0.6)
    for x, (k, _, _) in zip(xs, BARS):
        dots = result["per_panel"][k]
        ax.scatter([x + (i - (len(dots) - 1) / 2) * 0.035 for i in range(len(dots))], dots, s=12, color=COLORS["ink"], zorder=3,
                   label="One panel" if x == 0 else None)
        ax.text(x, 0.25, fmt(result[k], 2), ha="center", va="bottom", fontsize=10,   # inside the bar, clear of the dots and line
                color=COLORS["ink"] if k == "deployed" else "white")
    ax.axhline(result["seasonal_naive"], color=COLORS["gold"], ls=(0, (4, 3)), lw=1.6,
               label=f"Seasonal naive: {fmt(result['seasonal_naive'], 2)}")
    ax.set_xticks(xs, [name for _, name, _ in BARS])
    top = max(max(max(result["per_panel"][k]) for k, _, _ in BARS), result["seasonal_naive"])
    ax.set_ylim(0, top * 1.3)
    ax.set_ylabel("Forecast error (MAE, lower is better)")
    ax.set_xlabel(f"Pipeline, horizon {parameter} day{'s' if parameter != 1 else ''}")
    ax.legend(loc="upper left", frameon=False, fontsize=10, ncol=2)


def diff(a, b):
    """Difference of the two numbers as displayed (2 decimals), so the hand calculation on the page adds up."""
    return float(fmt(a, 2)) - float(fmt(b, 2))


def explain(result, parameter):
    b, d, h, n = result["backtest"], result["deployed"], result["honest"], result["seasonal_naive"]
    gap = diff(h, b)
    plural = "s" if parameter != 1 else ""
    interpretation = (
        f"At a horizon of {parameter} day{plural} the shift-based backtest reports an MAE of {fmt(b, 2)}. The direct model, which uses only "
        f"history available at the origin, scores {fmt(h, 2)}, so {fmt(h, 2)} - {fmt(b, 2)} = {fmt(gap, 2)} is the error the backtest hides. "
        f"Run at the origin, the shift-based model scores {fmt(d, 2)}. The origin test moves {result['moved_leaky']} of "
        f"{result['columns_leaky']} shift-based columns and {result['moved_honest']} of {result['columns_honest']} direct columns. "
        + ("At this horizon every lag is already known, so the pipelines agree." if parameter == 1 else
           f"The seasonal-naive forecast scores {fmt(n, 2)}."))
    steps = [
        f"Hidden error: {fmt(h, 2)} - {fmt(b, 2)} = {fmt(gap, 2)} (direct model minus shift-based backtest).",
        f"Production penalty of the shift-based model: {fmt(d, 2)} - {fmt(h, 2)} = {signed(diff(d, h), 2)} (run at the origin minus direct).",
        f"Share of the shift-based columns that read after the origin: {result['moved_leaky']} of {result['columns_leaky']}.",
        f"Direct model against seasonal naive: {fmt(h, 2)} - {fmt(n, 2)} = {signed(diff(h, n), 2)}.",
    ]
    metrics = {"Shift-based backtest MAE": fmt(b, 2), "Shift-based, run at the origin": fmt(d, 2),
               "Direct model MAE": fmt(h, 2), "Hidden error (direct - backtest)": fmt(gap, 2),
               "Columns that move after the origin": f"{result['moved_leaky']} of {result['columns_leaky']}"}
    alt = (f"Bars of forecast error at a {parameter}-day horizon: shift-based backtest {fmt(b, 2)}, shift-based run at the origin "
           f"{fmt(d, 2)} and direct model {fmt(h, 2)}, with a dashed line at the seasonal-naive error of {fmt(n, 2)}.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def _qualitative(results):
    """Directional claims; checked again under a different seed base."""
    one = results[1]
    assert abs(one["backtest"] - one["honest"]) < 0.05 and abs(one["deployed"] - one["honest"]) < 0.05, "h=1: pipelines agree"
    assert one["moved_leaky"] == 0 and one["moved_honest"] == 0, "h=1: nothing reads after the origin"
    for h, res in results.items():
        assert res["moved_honest"] == 0, f"direct features must not move at h={h}"
        if h > 1:
            assert res["moved_leaky"] > 0, f"shift-based features must move at h={h}"
            assert res["honest"] - res["backtest"] > 0.5, f"backtest should flatter at h={h}"
            assert res["deployed"] > res["honest"], f"direct should beat the shift-based model at the origin, h={h}"
            assert res["honest"] < res["seasonal_naive"], f"direct model should beat seasonal naive at h={h}"
            assert all(p > b for p, b in zip(res["per_panel"]["honest"], res["per_panel"]["backtest"])), f"every panel flattered at h={h}"
    spread = max(res["backtest"] for res in results.values()) - min(res["backtest"] for res in results.values())
    assert spread < 0.5, "backtest should stay about flat across horizons"
    assert results[5]["honest"] > results[3]["honest"] > results[1]["honest"], "honest error should grow with the horizon"
    assert abs(results[7]["honest"] - results[5]["honest"]) < 0.1, "honest error levels off at 5 and 7 days"


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    _qualitative(results)
    r3 = results[3]
    assert fmt(r3["backtest"], 2) == "4.96" and fmt(r3["honest"], 2) == "5.69" and fmt(r3["deployed"], 2) == "5.95", "feedback numbers"
    assert fmt(diff(r3["honest"], r3["backtest"]), 2) == "0.73", "gap number"
    assert fmt(results[5]["honest"], 2) == "6.03" and fmt(results[7]["honest"], 2) == "6.03", "answer: 6.03 at 5 and 7 days"
    assert 0.65 < r3["honest"] - r3["backtest"] < 0.8, "option says about 0.7"
    assert 4.9 < min(r["backtest"] for r in results.values()) and max(r["backtest"] for r in results.values()) < 5.1, "answer: flat near 5.0"
    assert max(r["deployed"] - r["honest"] for r in results.values()) < 0.6, "honesty: production penalty is modest"
