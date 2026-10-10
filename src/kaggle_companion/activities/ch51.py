"""Chapter 51: Time Series Cross-Validation. Which recipe wins a backtest that respects label availability and one that ignores it, by label delay."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.linear_model import Ridge

from kaggle_companion.activities._common import clean

SEED = 51
DRAWS = 160                      # independent series; every estimate is a mean over draws (40 was too few to rank the recipes)
DAYS, ROWS, FEATURES = 420, 10, 5
BLOCK = 30                       # each assessment block covers the 30 days after its forecast origin
ORIGINS = [100, 130, 160, 190, 220]
DRIFT = 0.04                     # daily pull of the coefficients toward a fresh random draw (about 25 days of memory)
SHORT_WINDOW = 30                # the "recent history only" recipe trains on the last 30 usable days


def generate(rng, drift):
    """Ten rows a day. The target is linear in five features whose coefficients drift slowly (a stationary process), plus noise of sd 1."""
    beta = np.zeros((DAYS, FEATURES))
    beta[0] = rng.normal(size=FEATURES)
    for t in range(1, DAYS):
        beta[t] = (1 - drift) * beta[t - 1] + np.sqrt(1 - (1 - drift) ** 2) * rng.normal(size=FEATURES)
    X = rng.normal(size=(DAYS, ROWS, FEATURES))
    return X, (beta[:, None, :] * X).sum(2) + rng.normal(size=(DAYS, ROWS))


def backtest_mae(X, y, delay, window, respect_availability):
    """Mean absolute error over the assessment blocks. A row observed on day d has its label available on day d + delay.
    Honest rule: train on rows with d + delay < origin. Naive rule: train on every row observed before the origin,
    including rows whose labels would still be on the way. `window` keeps only the most recent usable days (None keeps all)."""
    errors = []
    for origin in ORIGINS:
        last = origin - delay if respect_availability else origin
        first = 0 if window is None else max(0, last - window)
        model = Ridge(alpha=1.0).fit(X[first:last].reshape(-1, FEATURES), y[first:last].reshape(-1))
        block = slice(origin, origin + BLOCK)
        errors.append(np.mean(np.abs(model.predict(X[block].reshape(-1, FEATURES)) - y[block].reshape(-1))))
    return float(np.mean(errors))


def run(delay):
    rng = np.random.default_rng(SEED)
    recipes = {"short": SHORT_WINDOW, "expanding": None}
    rows = {f"{rule}_{name}": [] for rule in ("naive", "honest") for name in recipes}
    stable = []
    for _ in range(DRAWS):
        X, y = generate(rng, DRIFT)
        for name, window in recipes.items():
            for rule in ("naive", "honest"):
                rows[f"{rule}_{name}"].append(backtest_mae(X, y, delay, window, rule == "honest"))
        X, y = generate(rng, 0.0)                              # control world: coefficients never change
        stable.append(backtest_mae(X, y, delay, None, True) - backtest_mae(X, y, delay, None, False))
    mean = {k: float(np.mean(v)) for k, v in rows.items()}
    # Optimism of the naive backtest: how much lower its error is than the honest one, per recipe (paired by series).
    optimism = {name: np.array(rows[f"honest_{name}"]) - np.array(rows[f"naive_{name}"]) for name in recipes}
    # Difference between recipes under each rule; negative means the short-window recipe has the lower error.
    naive_diff = np.array(rows["naive_short"]) - np.array(rows["naive_expanding"])
    honest_diff = np.array(rows["honest_short"]) - np.array(rows["honest_expanding"])
    se = lambda a: np.std(a, ddof=1) / np.sqrt(DRAWS)
    return clean({
        "delay": delay, **mean,
        **{f"optimism_{name}": optimism[name].mean() for name in recipes},
        **{f"optimism_{name}_se": se(optimism[name]) for name in recipes},
        "naive_diff": naive_diff.mean(), "naive_diff_se": se(naive_diff),
        "honest_diff": honest_diff.mean(), "honest_diff_se": se(honest_diff),
        "stable_gap": np.mean(stable), "stable_gap_se": se(stable),
        "draws": DRAWS, "origins": len(ORIGINS), "noise_floor": 0.7979,
    })
# notebook-end


SPEC = {
    "chapter": 51,
    "chapter_title": "Time Series Cross-Validation",
    "subtitle": "At a forecast origin, train only on labels that would already exist, and watch which recipes the shortcut flatters.",
    "summary": ("A row observed before the origin is not usable if its label arrives after it. One demonstration varies the label "
                "delay and compares a backtest that respects label availability with one that ignores it, for a recent-history recipe "
                "and an expanding-history recipe."),
    "title": "Backtests that respect and ignore label availability, for two recipes, by label delay",
    "question": "How much does a backtest that ignores label availability flatter each recipe, and does it still rank the recipes correctly?",
    "why": ("The chapter's rule is to exclude training rows whose labels would arrive after the validation origin, and its six-row "
            "example excludes row D for exactly that reason. Measuring the flattery as the delay grows shows who benefits from the "
            "shortcut and whether a recipe choice made with it would survive."),
    "method": ("Constructed daily data: ten rows a day, five features, and a target that is linear in them with coefficients that drift "
               "slowly (about 25 days of memory) plus unit noise. Two ridge regression recipes are backtested at five forecast origins, "
               "each scored on the next 30 days: one trains on only the last 30 usable days, one on all usable history. "
               "A row's label is available delay days after it is observed. The honest backtest trains on rows with observation day plus delay "
               "before the origin, as the chapter's splitter does; the naive one trains on every row observed before the origin. "
               "A control world with fixed coefficients checks the leak without drift. Everything is a mean over 160 series."),
    "control": {"key": "delay", "label": "Label delay (days between observing a row and receiving its label)",
                "values": [0, 10, 30, 60], "default": 30,
                "value_labels": ["0: labels arrive at once", "10", "30", "60"]},
    "source_section": "Define the Forecast Origin",
    "symbols": ("d is the day a row is observed, L the label delay and v the forecast origin. A row is usable only if d + L < v. "
                "MAE is the mean absolute error on the next 30 days; optimism is the honest backtest's MAE minus the naive backtest's."),
    "explanation": ("The naive backtest never looks at the delay, so its error is the same at every value. The honest backtest cannot "
                    "use the newest rows, and the world has moved by the time the model is used, so its error rises with the delay. A "
                    "recipe that relies on the most recent rows loses the most, which is why the shortcut flatters recency-heavy recipes "
                    "most. Here it ranks the 30-day window first at every delay, which is right only when labels arrive at once; "
                    "respecting availability puts the expanding recipe ahead from a 10-day delay on."),
    "application": ("Record for every row when its label becomes available, build the training set at each origin from labels that "
                    "exist by then, and compare recipes only under that rule."),
    "assumptions": ("Constructed linear data with drifting coefficients, ridge regression instead of a gradient-boosted model and "
                    "five origins sharing the same series; fold results are correlated, so the standard errors describe the 160 series, "
                    "not a confidence interval for a competition. In the control world with fixed coefficients the two rules differ by "
                    "at most 0.002 MAE, so the effect needs a relationship that changes. The sizes depend on the drift speed chosen here."),
    "prediction": "With a 30-day label delay, which recipe does the backtest that ignores label availability prefer?",
    "prediction_options": ["The 30-day window, by a small margin", "Neither: they tie within noise", "The expanding history, clearly"],
    "prediction_answer": 0,
    "prediction_feedback": {
        "correct": "It scores the 30-day window 1.919 and the expanding recipe 1.957, a lead of 0.038 (standard error 0.013) for the window, while the honest backtest reverses the ranking, with the expanding recipe ahead by 0.161.",
        "incorrect": "It scores the 30-day window 1.919 and the expanding recipe 1.957, a lead of 0.038 (standard error 0.013) for the window. That is the right ranking only when labels arrive at once; the honest backtest has the expanding recipe ahead by 0.161.",
    },
    "check": "The naive error is 1.919 for the short window at every delay. What does that constancy tell you, and which recipe does the shortcut flatter more?",
    "answer": ("The naive backtest cannot see the delay at all, so its number carries no information about it. The honest MAE of the short window "
               "grows from 1.919 at no delay to 2.241 at 30 days, an optimism of 0.322, against 0.123 for the expanding recipe, because "
               "the short recipe depends only on the newest rows, which are exactly the ones whose labels have not arrived."),
    "provenance": "Constructed example: seeded synthetic daily regression data with drifting coefficients and ridge models, measured by the chapter activity.",
    "apply": [
        "Store the date each label becomes available next to the observation date, and build every training set from labels known by the origin.",
        "Use the chapter's forward_date_splits (or the same rule) with label-availability dates, and verify it on a six-row example first.",
        "Do not choose between a recent-window recipe and a long-history recipe with a backtest that ignores label delay.",
        "Report the delay you assumed and re-run the comparison if the delay changes in production.",
    ],
    "honesty": "Constructed data and models; the sizes of these effects depend on the drift speed chosen here, not on a competition result.",
}

EQUATIONS = [{"tex": r"\text{train on row } i \iff d_i + L < v",
              "alt": "train on row i if and only if its observation day d i plus the label delay L is less than the origin v",
              "basis": "The label-availability rule of Define the Forecast Origin, as implemented by forward_date_splits; the manuscript states it in prose."}]
NCOLS = 2
HEIGHT = 4.4


def draw(axes, result, parameter):
    ax, bx = axes
    names = [("short", "30-day window"), ("expanding", "Expanding history")]
    colors = {"short": COLORS["terracotta"], "expanding": COLORS["teal"]}
    for x, (k, _) in enumerate(names):
        naive, honest = result[f"naive_{k}"], result[f"honest_{k}"]
        ax.plot([x - 0.12, x + 0.12], [naive, honest], color=colors[k], lw=2.5, zorder=1)
        ax.scatter([x - 0.12], [naive], s=110, marker="o", color="white", edgecolor=colors[k], linewidth=2, zorder=3,
                   label="Backtest ignoring label delay" if x == 0 else None)
        ax.scatter([x + 0.12], [honest], s=110, marker="D", color=colors[k], zorder=3, edgecolor="white", linewidth=1,
                   label="Backtest respecting label delay" if x == 0 else None)
        ax.annotate(fmt(naive), (x - 0.12, naive), xytext=(-12, 0), textcoords="offset points", ha="right", va="center", fontsize=10)
        ax.annotate(fmt(honest), (x + 0.12, honest), xytext=(12, 0), textcoords="offset points", ha="left", va="center", fontsize=10)
    ax.axhline(result["noise_floor"], color=COLORS["gold"], ls=(0, (4, 3)), lw=1.4, label=f"Noise floor: {fmt(result['noise_floor'])}")
    ax.set_xticks([0, 1], [n for _, n in names])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0.6, 3.4)
    ax.set_ylabel("Mean absolute error, next 30 days")
    ax.set_xlabel(f"Recipe, label delay {parameter} days")
    ax.legend(loc="upper left", frameon=False, fontsize=10)

    values = [result["optimism_short"], result["optimism_expanding"]]
    errs = [result["optimism_short_se"], result["optimism_expanding_se"]]
    bx.bar([0, 1], values, width=0.5, color=[colors["short"], colors["expanding"]], edgecolor=COLORS["ink"], lw=0.6,
           yerr=errs, error_kw={"ecolor": COLORS["ink"], "capsize": 4, "lw": 1.2})
    for x, (v, e) in enumerate(zip(values, errs)):
        bx.text(x, v + e + 0.012, fmt(v), ha="center", va="bottom", fontsize=10)
    bx.set_xticks([0, 1], [n for _, n in names])
    bx.set_ylim(0, 0.6)
    bx.set_ylabel("Optimism, honest minus naive MAE")
    bx.set_xlabel("Recipe (error bar: 1 standard error)")


def explain(result, parameter):
    os_, oe = result["optimism_short"], result["optimism_expanding"]
    nd, hd = result["naive_diff"], result["honest_diff"]
    shown = lambda a, b: round(result[a], 3) - round(result[b], 3)     # differences of the displayed numbers
    if parameter == 0:
        lead = ("so the 30-day window is ahead when labels arrive at once" if hd < -2 * result["honest_diff_se"] else
                "a difference within two standard errors")
        interpretation = (
            f"With no label delay the two rules use the same rows, so the backtests agree: the 30-day window scores {fmt(result['honest_short'])} "
            f"and the expanding recipe {fmt(result['honest_expanding'])}: {fmt(max(result['honest_short'], result['honest_expanding']))} - "
            f"{fmt(min(result['honest_short'], result['honest_expanding']))} = {fmt(abs(shown('honest_short', 'honest_expanding')))}, "
            f"with a standard error of {fmt(result['honest_diff_se'])}, {lead}. "
            f"Optimism is {fmt(os_)} for both. Raise the delay to see the shortcut flatter the recipes.")
    else:
        naive_verdict = ("so it prefers the 30-day window" if nd < -2 * result["naive_diff_se"] else
                         "so it prefers the expanding recipe" if nd > 2 * result["naive_diff_se"] else "within noise of a tie")
        verdict = (f"the honest backtest reverses that, with the expanding recipe ahead by {fmt(hd)} (standard error {fmt(result['honest_diff_se'])})"
                   if hd > 2 * result["honest_diff_se"] else f"the honest backtest gives a difference of {signed(hd)}")
        interpretation = (
            f"With a {parameter}-day delay the naive backtest scores the 30-day window {fmt(result['naive_short'])} and the expanding recipe "
            f"{fmt(result['naive_expanding'])}, a gap of {signed(nd)} (standard error {fmt(result['naive_diff_se'])}), {naive_verdict}; {verdict}. "
            f"Respecting availability raises the short window's error to {fmt(result['honest_short'])}: {fmt(result['honest_short'])} - "
            f"{fmt(result['naive_short'])} = {fmt(shown('honest_short', 'naive_short'))} of optimism, "
            f"against {fmt(oe)} for the expanding recipe. With fixed coefficients the two rules differ by {fmt(result['stable_gap'])} MAE, "
            f"so the effect comes from a relationship that changes.")
    steps = [f"Short window optimism: {fmt(result['honest_short'])} - {fmt(result['naive_short'])} = {signed(shown('honest_short', 'naive_short'))} (standard error {fmt(result['optimism_short_se'])}).",
             f"Expanding optimism: {fmt(result['honest_expanding'])} - {fmt(result['naive_expanding'])} = {signed(shown('honest_expanding', 'naive_expanding'))} (standard error {fmt(result['optimism_expanding_se'])}).",
             f"Naive ranking gap, short minus expanding: {fmt(result['naive_short'])} - {fmt(result['naive_expanding'])} = {signed(shown('naive_short', 'naive_expanding'))}.",
             f"Honest ranking gap, short minus expanding: {fmt(result['honest_short'])} - {fmt(result['honest_expanding'])} = {signed(shown('honest_short', 'honest_expanding'))}."]
    metrics = {"Naive MAE, 30-day window": fmt(result["naive_short"]), "Honest MAE, 30-day window": fmt(result["honest_short"]),
               "Naive MAE, expanding": fmt(result["naive_expanding"]), "Honest MAE, expanding": fmt(result["honest_expanding"]),
               "Optimism, 30-day window": signed(os_), "Optimism, expanding": signed(oe)}
    alt = (f"Two panels at a {parameter}-day label delay. Left, mean absolute error of the 30-day-window and expanding recipes under a backtest that "
           f"ignores label delay and one that respects it, with a dashed noise floor at {fmt(result['noise_floor'])}. Right, the optimism of the naive backtest for "
           f"each recipe, {fmt(os_)} for the short window and {fmt(oe)} for the expanding recipe.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    # The chapter's six-row example: row D is observed on day 4 but its label arrives on day 6; the origin is day 5.
    import pandas as pd
    from kaggle_companion.evaluation import forward_date_splits
    day = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"])
    avail = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-06", "2024-01-06", "2024-01-07"])
    train, assess = next(forward_date_splits(day, [("2024-01-05", "2024-01-07")], label_available=avail))
    assert list(train) == [0, 1, 2], "rows A to C only: row D's label is not yet available"
    # The activity's integer rule (d + L < v) matches the chapter's splitter.
    obs, origin, delay = np.arange(40), 30, 7
    ts = pd.Timestamp("2024-01-01") + pd.to_timedelta(obs, "D")
    tr, _ = next(forward_date_splits(ts, [(ts[origin], ts[origin + 5])], label_available=ts + pd.Timedelta(days=delay)))
    assert list(tr) == list(obs[obs + delay < origin]), "activity rule equals forward_date_splits"
    zero, ten, thirty, sixty = results[0], results[10], results[30], results[60]
    assert zero["optimism_short"] == 0 and zero["optimism_expanding"] == 0, "no delay, no optimism"
    for a, b in ((zero, ten), (ten, thirty), (thirty, sixty)):
        assert a["optimism_short"] < b["optimism_short"] and a["optimism_expanding"] < b["optimism_expanding"], "optimism grows with delay"
    for r in (ten, thirty, sixty):
        assert r["optimism_short"] > r["optimism_expanding"] + 0.08, "the recency recipe is flattered more"
        assert r["honest_diff"] > 3 * r["honest_diff_se"], "honest backtest: expanding clearly ahead from 10 days on"
        assert r["naive_diff"] < -2 * r["naive_diff_se"], "naive backtest prefers the 30-day window"
        assert r["naive_short"] == thirty["naive_short"] and r["naive_expanding"] == thirty["naive_expanding"], "naive error does not depend on the delay"
        assert r["stable_gap"] < 0.002, "no drift, no leak: at most 0.002 (assumptions text)"
    assert zero["honest_diff"] < -2 * zero["honest_diff_se"], "at delay 0 the 30-day window is ahead"
    assert thirty["honest_diff"] > 0.1 and sixty["honest_diff"] > 0.1
    assert fmt(thirty["naive_short"]) == "1.919" and fmt(thirty["naive_expanding"]) == "1.957"
    assert fmt(-thirty["naive_diff"]) == "0.038" and fmt(thirty["naive_diff_se"]) == "0.013"
    assert fmt(thirty["honest_diff"]) == "0.161" and fmt(thirty["honest_short"]) == "2.241"
    assert fmt(thirty["optimism_short"]) == "0.322" and fmt(thirty["optimism_expanding"]) == "0.123"
