"""Chapter 3: Building a CV That Matches the Test Set. Leave-one-season-out against past-only validation, by drift."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.linear_model import LogisticRegression

from kaggle_companion.activities._common import clean

SEED = 3
REPLICATES = 100      # independent constructed histories; every estimate is their mean
SEASONS, GAMES, FEATURES = 8, 400, 4
FIRST_SCORED = 4      # score seasons 5, 6 and 7 (0-based 4, 5, 6); season 8 is the future
FUTURE_GAMES = 4000


def generate(drift, seed):
    """Seven development seasons and a large season 8. The weights that turn features into win odds
    wander from season to season; `drift` is the size of the wander (their length stays constant)."""
    rng = np.random.default_rng(seed)
    weights = np.zeros((SEASONS, FEATURES))
    weights[0] = rng.normal(0, 1.0, FEATURES)
    for s in range(1, SEASONS):
        step = weights[s - 1] + rng.normal(0, drift, FEATURES)
        weights[s] = step * np.linalg.norm(weights[0]) / np.linalg.norm(step)

    def games(s, n):
        x = rng.normal(size=(n, FEATURES))
        win = rng.random(n) < 1 / (1 + np.exp(-x @ weights[s]))
        return x, win.astype(int)

    dev = [games(s, GAMES) for s in range(SEASONS - 1)]
    return dev, games(SEASONS - 1, FUTURE_GAMES)


def fit(seasons):
    X = np.vstack([s[0] for s in seasons])
    y = np.concatenate([s[1] for s in seasons])
    return LogisticRegression(max_iter=500).fit(X, y)


def brier(y, p):
    return float(np.mean((y - p) ** 2))


def one_history(drift, seed):
    dev, (X_next, y_next) = generate(drift, seed)
    held, loso, past = [], [], []
    for s in range(FIRST_SCORED, SEASONS - 1):
        X, y = dev[s]
        held.append(y)
        # Leave-one-season-out: train on every other development season, later ones included.
        loso.append(fit([d for k, d in enumerate(dev) if k != s]).predict_proba(X)[:, 1])
        # Past-only: train on the seasons before s, as a real forecast would.
        past.append(fit(dev[:s]).predict_proba(X)[:, 1])
    y = np.concatenate(held)
    # Truth: the final recipe (all seven development seasons) on the next season, 4,000 fresh games.
    truth = brier(y_next, fit(dev).predict_proba(X_next)[:, 1])
    return brier(y, np.concatenate(loso)), brier(y, np.concatenate(past)), truth


def run(drift):
    runs = np.array([one_history(drift, SEED * 1000 + i) for i in range(REPLICATES)])
    loso, past, truth = runs.T
    return clean({
        "drift": drift,
        "loso": float(loso.mean()), "past_only": float(past.mean()), "next_season": float(truth.mean()),
        # optimism = Brier on the next season minus the estimate (positive: the estimate looked better than reality)
        "optimism": {"loso": float((truth - loso).mean()), "past_only": float((truth - past).mean())},
        "per_history": {"loso": (truth - loso).tolist(), "past_only": (truth - past).tolist()},
        "loso_more_optimistic": float(np.mean((truth - loso) > (truth - past))),
        "replicates": REPLICATES, "scored_games": GAMES * (SEASONS - 1 - FIRST_SCORED), "future_games": FUTURE_GAMES,
    })
# notebook-end


SPEC = {
    "chapter": 3,
    "chapter_title": "Building a CV That Matches the Test Set",
    "subtitle": "A split defines what unseen means; match it to how the test set is built.",
    "summary": ("Leave-one-season-out trains on later seasons, a real forecast cannot. One demonstration measures how far "
                "leave-one-season-out and past-only validation each sit from the score on the next season as the relationship drifts."),
    "title": "Leave-one-season-out against past-only validation, as the relationship drifts",
    "question": ("When the relationship between features and outcomes changes from season to season, how far does each "
                 "validation scheme's Brier score sit from the score on the next season?"),
    "why": ("Leave-one-season-out is a respectable group assessment, but its training set contains seasons from after the "
            "held-out one. For a competition that forecasts a future season, the estimate you use to choose models then "
            "describes a different question from the one the leaderboard asks."),
    "method": ("A constructed league of 8 seasons with 400 games each and 4 features. A logistic model gives win odds from "
               "season-specific weights; the weights take a random step from one season to the next, and the size of the step "
               "is the control (0 means no drift). Seasons 1 to 7 are development. Both schemes score seasons 5 to 7 (1,200 "
               "games, pooled into one Brier score): leave-one-season-out trains on the other six development seasons, "
               "past-only trains on earlier seasons only. The reference is the final recipe, fitted on all seven development "
               "seasons, scored on 4,000 fresh games of season 8. Every number is the mean of 100 independent histories."),
    "control": {"key": "drift", "label": "Season-to-season drift (standard deviation of the weight step)",
                "values": [0, 0.25, 0.5, 1.0], "default": 0.5,
                "value_labels": ["0: no drift", "0.25: mild", "0.5: moderate", "1.0: strong"]},
    "source_section": "March Mania as a Special Case",
    "symbols": ("Brier = the mean of (p - y)^2 over scored games, with p the predicted win probability and y the result "
                "(lower is better). Optimism is the Brier score on the next season minus the validation estimate: positive means "
                "validation looked better than the future turned out."),
    "explanation": ("With no drift, seasons are exchangeable and both schemes describe the next season equally well. With drift, "
                    "a model trained on seasons on both sides of the held-out one has seen a close relative of its test season, so "
                    "leave-one-season-out looks better than a forecast can be. Past-only validation scores a genuine forecast, so "
                    "it sits closer to the next-season score, though in this generator it is still a little optimistic once there "
                    "is drift."),
    "application": ("If the test set is a future season, validate with past-only seasons and keep leave-one-season-out, if you "
                    "use it, as a labelled historical grouped check."),
    "assumptions": ("Constructed league with a random-walk relationship, one logistic model and 100 histories. Real drift may be "
                    "abrupt, seasonal or absent, and past-only validation trains on only four to six seasons "
                    "where leave-one-season-out trains on six. At no drift the two schemes are tied and both match the next-season score."),
    "prediction": "At moderate drift (0.5), how does leave-one-season-out's Brier score compare with the score on the next season?",
    "prediction_options": ["About the same", "Lower (better) by about 0.01", "Higher (worse) by about 0.01"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "Leave-one-season-out reports a Brier of 0.184 against 0.196 on the next season: about 0.012 of optimism.",
        "incorrect": ("Leave-one-season-out reports a Brier of 0.184 against 0.196 on the next season, about 0.012 of optimism. "
                      "Its training set holds seasons from after the one it scores."),
    },
    "check": "With no drift both schemes agree, so why does the chapter still tell a forecaster to use past-only seasons?",
    "answer": ("Because the agreement is a property of a stable relationship, which you cannot know in advance. Here drift of "
               "0.5 left leave-one-season-out 0.012 from the next-season score while past-only was 0.006 away, "
               "leave-one-season-out reached 0.017 at drift 1.0, and at zero drift the two tied. Past-only costs nothing when seasons are exchangeable and "
               "protects the decision when they are not, though it is not a guarantee: it was still optimistic here."),
    "provenance": "Constructed example: seeded synthetic seasons of games and a logistic model, measured by the chapter activity. The seasons are not NCAA results.",
    "apply": [
        "Write down whether the test is a future season, a new group or independent rows before choosing a splitter.",
        "For a future season, train only on earlier seasons whose labels would have been revealed, and report which seasons receive predictions.",
        "Label leave-one-season-out as a historical grouped check, and compare it with the past-only estimate: a large difference points to drift.",
        "Expect even past-only validation to be somewhat optimistic when the relationship moves, and keep a margin before trusting a small gain.",
    ],
    "honesty": ("Constructed data with a random-walk relationship; the sizes are properties of this generator, not a competition "
                "result. Past-only is closer to the next season, not exact."),
}

EQUATIONS = [{"tex": r"\mathrm{Brier} = \frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)^2",
              "alt": "Brier equals one over n times the sum over games i of p i minus y i, squared",
              "basis": "The Brier score used by March Mania (Chapter 6); the activity's own metric label, not a display equation in Chapter 3."},
             {"tex": r"\text{train on seasons } k < s \quad\text{(past-only)} \qquad \text{train on seasons } k \neq s \quad\text{(leave-one-season-out)}",
              "alt": "past-only trains on seasons k earlier than s; leave-one-season-out trains on every season k other than s",
              "basis": "The two training rules compared for validation season s (March Mania as a Special Case)."}]
NCOLS = 2
HEIGHT = 4.4
SCHEMES = [("loso", "Leave-one-season-out", COLORS["terracotta"]), ("past_only", "Past-only", COLORS["teal"])]
KEYS = {"loso": "loso", "past_only": "past_only"}


def draw(axes, result, parameter):
    left, right = axes
    xs = list(range(len(SCHEMES)))
    est = [result[k] for k, _, _ in SCHEMES]
    for x, v, (_, _, c) in zip(xs, est, SCHEMES):
        left.vlines(x, v, result["next_season"], color=c, lw=3, alpha=0.5)   # stem: the distance to the next-season score
        left.scatter([x], [v], s=70, color=c, edgecolor=COLORS["ink"], zorder=3)
        left.text(x + 0.1, v, fmt(v), ha="left", va="center", fontsize=10)
    left.axhline(result["next_season"], color=COLORS["gold"], ls=(0, (4, 3)), lw=1.6,
                 label=f"Season 8, fresh games: {fmt(result['next_season'])}")
    low = min(est + [result["next_season"]])
    left.set_ylim(low - 0.01, max(est + [result["next_season"]]) + 0.01)
    left.set_xlim(-0.5, len(SCHEMES) - 0.4)
    left.set_xticks(xs, [n for _, n, _ in SCHEMES])
    left.set_ylabel("Brier score (lower is better)")
    left.set_xlabel(f"Validation scheme, drift {parameter}")
    left.legend(loc="upper left", frameon=False, fontsize=10)

    for x, (k, _, c) in zip(xs, SCHEMES):
        vals = np.array(result["per_history"][k])
        lo, mid, hi = np.percentile(vals, [10, 50, 90])
        right.vlines(x, lo, hi, color=c, lw=3, alpha=0.6)
        right.scatter([x], [result["optimism"][k]], s=60, color=c, edgecolor=COLORS["ink"], zorder=3)
        right.text(x + 0.12, result["optimism"][k], fmt(result["optimism"][k]), va="center", fontsize=10)
    right.axhline(0, color=COLORS["grey"], lw=0.8)
    right.set_xlim(-0.5, len(SCHEMES) - 0.4)
    right.set_xticks(xs, [n for _, n, _ in SCHEMES])
    right.set_ylabel("Optimism: next-season Brier minus estimate")
    right.set_xlabel("Dot: mean of 100 histories; bar: middle 80%")


def explain(result, parameter):
    l, p, t = result["loso"], result["past_only"], result["next_season"]
    ol, op = result["optimism"]["loso"], result["optimism"]["past_only"]
    if ol - op > 0.003:
        verdict = "Leave-one-season-out is the more optimistic scheme."
    elif abs(ol - op) <= 0.003:
        verdict = "The two schemes are about equally far from the next season."
    else:
        verdict = "Past-only is the more optimistic scheme here."
    interpretation = (
        f"At drift {parameter} the final recipe scores a Brier of {fmt(t)} on the next season. Leave-one-season-out reports "
        f"{fmt(l)}, so {fmt(t)} - {fmt(l)} = {fmt(ol)} of optimism. Past-only reports {fmt(p)}, {fmt(t)} - {fmt(p)} = {fmt(op)}. "
        f"{verdict} Leave-one-season-out was the more optimistic in {round(100 * result['loso_more_optimistic'])}% of "
        f"{result['replicates']} histories.")
    steps = [
        f"Leave-one-season-out: {fmt(t)} - {fmt(l)} = {fmt(ol)} (next season minus estimate).",
        f"Past-only: {fmt(t)} - {fmt(p)} = {fmt(op)}.",
        f"Difference between the schemes: {fmt(p)} - {fmt(l)} = {fmt(p - l)} of Brier.",
        f"Share of histories where leave-one-season-out was more optimistic: {round(100 * result['loso_more_optimistic'])}%.",
    ]
    metrics = {"Leave-one-season-out Brier": fmt(l), "Past-only Brier": fmt(p), "Next-season Brier": fmt(t),
               "Optimism, leave-one-season-out": fmt(ol), "Optimism, past-only": fmt(op)}
    alt = (f"Left: Brier score of the two validation schemes with a dashed line at the next-season score {fmt(t)}. Right: optimism "
           f"of each scheme, {fmt(ol)} for leave-one-season-out and {fmt(op)} for past-only, at drift {parameter}.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    zero = results[0]
    assert abs(zero["optimism"]["loso"] - zero["optimism"]["past_only"]) < 0.003, "no drift: the schemes should tie"
    assert abs(zero["optimism"]["loso"]) < 0.004 and abs(zero["optimism"]["past_only"]) < 0.004, "no drift: both near the future"
    for d in (0.25, 0.5, 1.0):
        res = results[d]
        assert res["optimism"]["loso"] > res["optimism"]["past_only"] > 0, f"loso more optimistic than past-only at {d}"
        assert res["loso_more_optimistic"] > 0.6, f"loso more optimistic in most histories at {d}"
    assert results[1.0]["optimism"]["loso"] > results[0.5]["optimism"]["loso"] > results[0.25]["optimism"]["loso"]
    assert results[0.5]["optimism"]["past_only"] > results[0.25]["optimism"]["past_only"]
    assert results[1.0]["optimism"]["past_only"] < 0.6 * results[1.0]["optimism"]["loso"], "past-only about a third of the gap"
    mod = results[0.5]
    assert fmt(mod["loso"]) == "0.184" and fmt(mod["next_season"]) == "0.196", "prediction feedback numbers"
    assert fmt(mod["optimism"]["loso"]) == "0.012", "prediction feedback optimism"
    assert fmt(mod["optimism"]["past_only"]) == "0.006" and fmt(results[1.0]["optimism"]["loso"]) == "0.017", "check answer, past-only optimism"
    assert 0.008 < mod["optimism"]["loso"] < 0.015, "prediction option says about 0.01"
