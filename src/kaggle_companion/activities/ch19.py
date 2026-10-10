"""Chapter 19: March Mania: Prediction at Scale. The chapter's season forecast, rerun with leaked history and the label-flip test."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from kaggle_companion.activities._common import clean

SEED = 19
FIXTURES = 40       # independent constructed seasons-of-games; every estimate is their mean
CHAPTER_SEED = 42   # the chapter's own fixture, reported separately
TEAMS = 12


def fixture(seed):
    """Eight seasons of round-robin games between 12 teams whose strengths drift (the chapter's generator)."""
    rng = np.random.default_rng(seed)
    strength = rng.normal(size=TEAMS)
    season, a, b, y, p = [], [], [], [], []
    for s in range(1, 9):
        strength = 0.85 * strength + rng.normal(0, 0.25, size=TEAMS)
        for i in range(TEAMS):
            for j in range(i + 1, TEAMS):
                prob = 1 / (1 + np.exp(-(strength[i] - strength[j])))
                season.append(s); a.append(i); b.append(j); p.append(prob); y.append(int(rng.random() < prob))
    return {"season": np.array(season), "a": np.array(a), "b": np.array(b), "y": np.array(y), "p": np.array(p)}


def rate_difference(fx, leak):
    """Smoothed win rate of team A minus team B, from the history visible at the start of each game's season.

    Legal history is every earlier season. `leak` adds that many LATER seasons, as a leave-one-season-out
    feature build would. leak = 0 reproduces the chapter's pipeline."""
    season, a, b, y = fx["season"], fx["a"], fx["b"], fx["y"]
    out = np.zeros(len(y))
    for s in range(1, 9):
        use = (season < s) | ((season > s) & (season <= s + leak))
        wins, games = np.zeros(TEAMS), np.zeros(TEAMS)
        np.add.at(wins, a[use], y[use]); np.add.at(wins, b[use], 1 - y[use])
        np.add.at(games, a[use], 1); np.add.at(games, b[use], 1)
        rate = (wins + 1) / (games + 2)                 # Beta(1, 1) smoothing, as in the chapter
        here = season == s
        out[here] = rate[a[here]] - rate[b[here]]
    return out


def candidates():
    """The chapter's two fixed candidates; a constant 0.5 is the third."""
    return {"logistic": LogisticRegression(C=1.0, random_state=42, max_iter=1000),
            "small_tree": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, l2_regularization=1.0,
                                                         early_stopping=False, random_state=42)}


def pipeline(fx, leak):
    """Fit on seasons 1-4, select on 5-6 by Brier, refit on 1-6, then forecast 7-8 with legal history only."""
    x = rate_difference(fx, leak)[:, None]
    season, y = fx["season"], fx["y"]
    train, dev, fit, test = season <= 4, (season == 5) | (season == 6), season <= 6, season >= 7
    scores = {"constant": brier_score_loss(y[dev], np.full(dev.sum(), 0.5))}
    for name, model in candidates().items():
        scores[name] = brier_score_loss(y[dev], model.fit(x[train], y[train]).predict_proba(x[dev])[:, 1])
    order = ["constant", "logistic", "small_tree"]
    selected = min(order, key=lambda n: (scores[n], order.index(n)))
    if selected == "constant":
        p = np.full(test.sum(), 0.5)
    else:   # at forecast time only legal history exists, whatever history built the training rows
        p = candidates()[selected].fit(x[fit], y[fit]).predict_proba(rate_difference(fx, 0)[test][:, None])[:, 1]
    return scores, selected, brier_score_loss(y[test], p)


def flip_assessment_labels(fx):
    """The chapter's test: reverse every season 7 and 8 outcome and see whether anything earlier moves."""
    return {**fx, "y": np.where(fx["season"] >= 7, 1 - fx["y"], fx["y"])}


def run(leaked_seasons):
    fixtures = [fixture(SEED * 1000 + i) for i in range(FIXTURES)]
    estimate, delivered, change, reselected, shift = [], [], [], [], []
    for fx in fixtures:
        scores, selected, outcome = pipeline(fx, leaked_seasons)
        legal_scores, legal_selected, _ = pipeline(fx, 0)
        flipped_scores, flipped_selected, _ = pipeline(flip_assessment_labels(fx), leaked_seasons)
        estimate.append(scores[selected])
        delivered.append(outcome)
        change.append(abs(flipped_scores[selected] - scores[selected]))   # selected recipe's development Brier
        reselected.append(flipped_selected != selected)
        shift.append(scores[selected] - legal_scores[legal_selected])
    ref = fixture(CHAPTER_SEED)
    ref_scores, ref_selected, ref_outcome = pipeline(ref, leaked_seasons)
    ref_flip = pipeline(flip_assessment_labels(ref), leaked_seasons)[0]
    dev, test = np.isin(fixtures[0]["season"], [5, 6]), fixtures[0]["season"] >= 7
    best = lambda m: float(np.mean([(f["p"][m] * (1 - f["p"][m])).mean() for f in fixtures]))   # Brier of the true probabilities
    return clean({
        "leaked_seasons": leaked_seasons, "fixtures": FIXTURES,
        "estimate": float(np.mean(estimate)), "delivered": float(np.mean(delivered)),
        "best_possible_development": best(dev), "best_possible_assessment": best(test),
        "estimate_shift": float(np.mean(shift)), "estimate_shift_se": float(np.std(shift) / np.sqrt(FIXTURES)),
        "flip_change": float(np.mean(change)), "flip_change_per_fixture": change,
        "recipe_changed": float(np.mean(reselected)),
        "chapter_fixture": {"scores": ref_scores, "selected": ref_selected, "delivered": ref_outcome,
                            "flip_change": abs(ref_flip[ref_selected] - ref_scores[ref_selected])},
    })
# notebook-end


SPEC = {
    "chapter": 19,
    "chapter_title": "March Mania: Prediction at Scale",
    "subtitle": "Protect the forecast boundary, select on development seasons, and prove the cutoff with a label-flip test.",
    "summary": ("The chapter's constructed season forecast is rerun forty times while its history is allowed to peek at later seasons. "
                "One demonstration measures how little the score comparison reveals and how clearly the chapter's label-flip test does."),
    "title": "Leaked later seasons: what the scores hide and the label-flip test shows",
    "question": "If the team history quietly includes later seasons, would the development score tell you, and would the label-flip test?",
    "why": ("A leave-one-season-out feature build is an easy mistake and rarely inflates the score by much. The chapter's test, reversing "
            "the later labels and checking that nothing earlier moves, detects it exactly. Measuring both shows why the chapter relies on the test."),
    "method": ("Forty constructed fixtures from the chapter's generator (12 teams, 8 seasons of round-robin games, drifting strengths), plus "
               "the chapter's own seed 42. Each pipeline fits logistic regression and a small boosted tree on seasons 1 to 4, selects among them "
               "and a constant 0.5 by Brier loss on seasons 5 and 6, refits on seasons 1 to 6 and forecasts seasons 7 and 8 using only legal "
               "history. The control adds that many later seasons to every team's history when the features are built, as a leave-one-season-out "
               "build would. At 0 the pipeline is the chapter's. The label-flip test reverses every season 7 and 8 outcome and measures how much "
               "the selected recipe's development Brier moves."),
    "control": {"key": "leaked_seasons", "label": "Later seasons included in each game's team history",
                "values": [0, 1, 2, 3], "default": 1,
                "value_labels": ["0: legal, past-only history", "1: one later season", "2: two later seasons", "3: three later seasons"]},
    "source_section": "Run the Complete Constructed Forecast",
    "symbols": ("p_i is the predicted probability and y_i the outcome of game i, n the games scored, s a game's season, L the number of "
                "later seasons visible in its team history and H_s the set of outcomes used to build its features."),
    "explanation": ("With L = 0 the history is exactly what exists at each season's start, so reversing the season 7 and 8 labels cannot move "
                    "any development score: the test passes. With L at least 1, the history for the development seasons reads later outcomes, "
                    "so the development score moves when those labels change. The leak improves the development score by only a few thousandths, and later "
                    "seasons are harder in this generator whatever the pipeline does, so the gap between the development "
                    "and forecast scores is no reliable signal: with one to three leaked seasons it changes by less than 0.004 from the legal pipeline's gap, too little to tell apart from season difficulty and selection noise."),
    "application": ("Run the label-flip test on the real pipeline: change the assessment labels, rebuild everything and confirm that the development "
                    "scores, the selected recipe and the frozen model do not change."),
    "assumptions": ("Constructed fixtures in which team strengths drift and the spread of strengths shrinks, so later seasons are inherently "
                    "harder; two fixed candidates and a single feature. In a real task the size of a leak depends on how much the later "
                    "outcomes say about the earlier rows. The flip test measures a dependency, not its cost."),
    "prediction": "With one later season visible in the history, how far does reversing the season 7 and 8 labels move the development Brier of the selected recipe, on average?",
    "prediction_options": ["Not at all", "A few thousandths", "More than 0.05"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "It moves by 0.004 on average over the forty fixtures, and the selected recipe changes in 0.075 of them.",
        "incorrect": "It moves by 0.004 on average over the forty fixtures, and the selected recipe changes in 0.075 of them. A legal pipeline would move by exactly 0.000.",
    },
    "check": "The leaky pipelines report a development Brier barely lower than the legal one. Why is the label-flip test still reliable?",
    "answer": ("The score comparison asks whether a number looks too good, and the leak moves that number by only 0.0026 at L = 1, "
               "smaller than the 0.0095 gap that harder later seasons and ordinary selection produce with no leak at all. The flip test asks a yes-or-no question about dependence: a legal pipeline moves by exactly 0.000, so "
               "any movement proves that later labels reach the development scores."),
    "provenance": "Constructed example: forty seeded synthetic fixtures from the chapter's generator, logistic regression and a boosted tree, measured by the chapter activity.",
    "apply": [
        "Build features for every row from only the outcomes revealed before that row's forecast time; never build them once from the whole table.",
        "Run the label-flip test after each pipeline change: reverse the assessment labels and confirm that development scores and the selected recipe stay identical.",
        "Do not use the gap between development and later scores to detect leakage; seasons differ in difficulty and the gap is noisy.",
        "Report the development-to-assessment difference in the handover record, as the chapter does, rather than hiding it behind the development score.",
    ],
    "honesty": ("Constructed data. The leak here is small and the activity says so: the chapter's claim is that the boundary can be tested "
                "exactly, not that leaked history ruins a score."),
}

EQUATIONS = [{"tex": r"\mathrm{BS} = \frac{1}{n}\sum_{i=1}^{n} (p_i - y_i)^2",
              "alt": "Brier score equals the mean over n games of the squared difference between predicted probability p i and outcome y i",
              "basis": "Brier loss as defined in What the Executed Run Shows (Chapter 19); the chapter states it in words."},
             {"tex": r"H_s = \{\, \text{outcomes of seasons } t : t < s \text{ or } s < t \le s + L \,\}",
              "alt": "the history H s for season s holds outcomes of seasons t earlier than s, plus, in a leaky build, the L seasons after s",
              "basis": "The activity's leak control; L = 0 is the chapter's past-only history (Run the Complete Constructed Forecast)."}]
NCOLS = 2
HEIGHT = 4.4


def draw(axes, result, parameter):
    left, right = axes
    estimate, delivered = result["estimate"], result["delivered"]
    left.bar([0, 1], [estimate, delivered], width=0.55, color=[COLORS["light"], COLORS["teal"]], edgecolor=COLORS["ink"], lw=0.6)
    for x, v in enumerate([estimate, delivered]):
        left.text(x, v + 0.0012, fmt(v, 4), ha="center", va="bottom", fontsize=10)
    left.hlines(result["best_possible_development"], -0.4, 0.4, color=COLORS["gold"], ls=(0, (4, 3)), lw=1.6,
                label=f"Best possible, seasons 5-6: {fmt(result['best_possible_development'], 3)}")
    left.hlines(result["best_possible_assessment"], 0.6, 1.4, color=COLORS["navy"], ls=(0, (4, 3)), lw=1.6,
                label=f"Best possible, seasons 7-8: {fmt(result['best_possible_assessment'], 3)}")
    left.set_xticks([0, 1], ["Development\nestimate", "Forecast of\nseasons 7-8"])
    left.set_ylim(0.2, 0.26)
    left.set_ylabel("Brier loss (lower is better)")
    left.set_xlabel("Selected recipe, mean of 40 fixtures")
    left.legend(loc="upper left", frameon=False, fontsize=10)
    dots = result["flip_change_per_fixture"]
    right.scatter([(i - (len(dots) - 1) / 2) * 0.006 for i in range(len(dots))], sorted(dots), s=16, color=COLORS["ink"], zorder=3,
                  label="One fixture")
    right.axhline(result["flip_change"], color=COLORS["terracotta"], lw=1.8, label=f"Mean: {fmt(result['flip_change'], 4)}")
    right.axhline(0, color=COLORS["grey"], lw=0.8)
    right.set_xlim(-0.14, 0.14)
    right.set_xticks([])
    right.set_ylim(-0.003, max(0.03, max(dots) * 1.15))
    right.set_ylabel("Change in development Brier after flipping labels")
    right.set_xlabel(f"Label-flip test, {parameter} later season(s) in history")
    right.legend(loc="upper left", frameon=False, fontsize=10)


def explain(result, parameter):
    e, d = result["estimate"], result["delivered"]
    chap = result["chapter_fixture"]
    shift, flip = result["estimate_shift"], result["flip_change"]
    passes = flip == 0
    interpretation = (
        f"With {parameter} later season(s) in the history, the development estimate averages {fmt(e, 4)} and the forecast of seasons 7 and 8 "
        f"scores {fmt(d, 4)}, so {fmt(d, 4)} - {fmt(e, 4)} = {fmt(d - e, 4)}. "
        + ("This is the legal pipeline itself. " if parameter == 0 else
           f"Against the legal pipeline the estimate moves {signed(result['estimate_shift'], 4)} "
           f"(standard error {fmt(result['estimate_shift_se'], 4)}). ")
        + ("The label-flip test passes: reversing the season 7 and 8 labels moves nothing earlier, and the recipe never changes."
           if passes else
           f"The label-flip test fails: reversing the season 7 and 8 labels moves the development Brier by {fmt(flip, 4)} on average and changes "
           f"the selected recipe in {fmt(result['recipe_changed'], 3)} of fixtures. The chapter's own fixture moves by {fmt(chap['flip_change'], 4)}."))
    steps = [
        f"Development estimate: {fmt(e, 4)}; forecast of seasons 7 and 8: {fmt(d, 4)}.",
        f"Gap: {fmt(d, 4)} - {fmt(e, 4)} = {fmt(d - e, 4)} (later seasons are harder here: best possible {fmt(result['best_possible_development'], 3)} then {fmt(result['best_possible_assessment'], 3)}).",
        f"Estimate against the legal pipeline: {signed(shift, 4)}.",
        f"Label-flip change in the development Brier: {fmt(flip, 4)} - 0 = {fmt(flip, 4)} (a legal pipeline gives 0).",
    ]
    metrics = {"Development estimate": fmt(e, 4), "Forecast of seasons 7-8": fmt(d, 4),
               "Label-flip change": fmt(flip, 4), "Recipe changed by flip": fmt(result["recipe_changed"], 3),
               "Chapter fixture, logistic dev Brier": fmt(chap["scores"]["logistic"], 6)}
    alt = (f"Left, bars of the development estimate {fmt(e, 4)} and the forecast Brier {fmt(d, 4)} with dashed lines for the best possible "
           f"Brier; right, the change in development Brier after flipping later labels for forty fixtures, mean {fmt(flip, 4)}.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    from kaggle_companion import walkthrough
    fx, official = fixture(CHAPTER_SEED), walkthrough.make_fixture(CHAPTER_SEED)
    assert (fx["y"] == official.target.to_numpy()).all() and (fx["season"] == official.season.to_numpy()).all(), "generator matches the chapter's"
    zero = results[0]
    assert zero["flip_change"] == 0 and zero["recipe_changed"] == 0 and max(zero["flip_change_per_fixture"]) == 0, "legal pipeline must pass the flip test"
    assert zero["chapter_fixture"]["flip_change"] == 0
    report, _ = walkthrough.run_walkthrough(official)      # the chapter's own pipeline, for comparison
    ref = zero["chapter_fixture"]
    assert abs(report["development_brier"]["logistic"] - ref["scores"]["logistic"]) < 1e-4 and fmt(ref["scores"]["logistic"], 4) == "0.2174"
    assert abs(report["development_brier"]["small_tree"] - ref["scores"]["small_tree"]) < 1e-4 and fmt(ref["scores"]["small_tree"], 4) == "0.2488"
    assert abs(report["assessment_selected_brier"] - ref["delivered"]) < 1e-4 and fmt(ref["delivered"], 4) == "0.2376", "chapter's 0.237633"
    assert zero["chapter_fixture"]["selected"] == "logistic"
    assert zero["estimate_shift"] == 0
    leaks = [results[k] for k in (1, 2, 3)]
    for L, res in zip((1, 2, 3), leaks):
        assert res["flip_change"] > 0.002, f"flip test should detect leak at L={L}"
        assert sum(c > 0 for c in res["flip_change_per_fixture"]) > 30, f"most fixtures should move at L={L}"
        assert res["recipe_changed"] > 0.05, f"recipe should change sometimes at L={L}"
        assert abs(res["estimate_shift"]) < 0.006, f"estimate should barely move at L={L}"
        assert abs(res["estimate_shift"]) < zero["delivered"] - zero["estimate"], f"leak shift should be under the legal gap at L={L}"
        assert abs(res["delivered"] - res["estimate"] - (zero["delivered"] - zero["estimate"])) < 0.004, f"explanation: gap changes by less than 0.004 at L={L}"
    assert leaks[0]["flip_change"] < leaks[1]["flip_change"] < leaks[2]["flip_change"], "flip change grows with the leak"
    assert zero["best_possible_assessment"] > zero["best_possible_development"], "later seasons harder"
    one = results[1]
    assert fmt(one["flip_change"], 3) == "0.004" and fmt(one["recipe_changed"], 3) == "0.075", "prediction feedback numbers"
    assert 0.001 < one["flip_change"] < 0.02, "prediction option says a few thousandths"
    assert fmt(one["estimate_shift"], 4) == "-0.0026", "check answer: estimate shift at L=1"
    assert fmt(zero["delivered"] - zero["estimate"], 4) == "0.0095", "check answer: legal gap"
