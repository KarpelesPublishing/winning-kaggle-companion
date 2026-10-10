"""Chapter 6: Running Case, March Mania. Choosing the Elo offseason carryover on past seasons, by how much strength truly persists."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np

from kaggle_companion.activities._common import clean

SEED = 6
REPLICATES = 20        # independent leagues; every number is their mean
TEAMS, SEASONS, GAMES = 24, 10, 360     # 360 games a season is 30 per team
K = 24.0                                # the chapter's Elo update factor
STRENGTH_SD = 150.0                     # spread of true team strength, in Elo points
EARLY = 0.34                            # score the first third of each season, where the carried rating matters most
DECAYS = [round(0.05 * i, 2) for i in range(21)]      # candidate carryover fractions, 0 to 1
DEVELOPMENT, ASSESSMENT = range(1, 6), range(6, 10)   # seasons 2 to 6 choose the decay; seasons 7 to 10 judge it
CHAPTER_DECAY = 0.65                    # the chapter's example value, a hypothesis to compare


def generate(persistence, seed):
    """Ten seasons of games between 24 teams. True strength is an AR(1) process: a team keeps `persistence` of last year's deviation."""
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, STRENGTH_SD, TEAMS)
    seasons = []
    for season in range(SEASONS):
        if season:
            strength = persistence * strength + np.sqrt(1 - persistence ** 2) * rng.normal(0, STRENGTH_SD, TEAMS)
        pairs = np.array([rng.choice(TEAMS, 2, replace=False) for _ in range(GAMES)])
        a, b = pairs[:, 0], pairs[:, 1]
        true_p = 1 / (1 + 10 ** ((strength[b] - strength[a]) / 400))      # the chapter's Elo win probability, at true strength
        won = (rng.random(GAMES) < true_p).astype(float)
        seasons.append((a, b, won, true_p))
    return seasons


def elo_brier(seasons, decay):
    """The chapter's Elo update, with the offseason regression `decay`. Returns, per season, the Brier score of the
    pre-game probabilities on the first third of games, and on the last third (a check, nearer tournament time)."""
    ratings = [1500.0] * TEAMS
    cutoff, late_start = int(EARLY * GAMES), GAMES - int(EARLY * GAMES)
    early, late = [], []
    for k, (a, b, won, _) in enumerate(seasons):
        if k:
            ratings = [1500.0 + decay * (r - 1500.0) for r in ratings]
        err, err_late = 0.0, 0.0
        for g in range(GAMES):
            ra, rb = ratings[a[g]], ratings[b[g]]
            expected = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))      # pre-game probability that A wins
            if g < cutoff:
                err += (expected - won[g]) ** 2
            elif g >= late_start:
                err_late += (expected - won[g]) ** 2
            delta = K * (won[g] - expected)
            ratings[a[g]], ratings[b[g]] = ra + delta, rb - delta
        early.append(err / cutoff)
        late.append(err_late / (GAMES - late_start))
    return np.array(early), np.array(late)


def one_league(persistence, seed):
    seasons = generate(persistence, seed)
    both = [elo_brier(seasons, d) for d in DECAYS]
    table = np.array([e for e, _ in both])                                 # decays x seasons, first third
    late = np.array([l for _, l in both])                                  # decays x seasons, last third
    cutoff = int(EARLY * GAMES)
    floor = np.mean([np.mean((p[:cutoff] - w[:cutoff]) ** 2) for _, _, w, p in seasons[ASSESSMENT.start:]])
    return (table[:, list(DEVELOPMENT)].mean(axis=1), table[:, list(ASSESSMENT)].mean(axis=1), floor,
            late[:, list(ASSESSMENT)].mean(axis=1))


def run(persistence):
    leagues = [one_league(persistence, SEED * 1000 + i) for i in range(REPLICATES)]
    dev = np.array([l[0] for l in leagues])
    ass = np.array([l[1] for l in leagues])
    floor = float(np.mean([l[2] for l in leagues]))
    pick = dev.argmin(axis=1)                                   # each league chooses its own decay on seasons 2 to 6
    tuned = ass[np.arange(REPLICATES), pick]
    default = ass[:, DECAYS.index(CHAPTER_DECAY)]
    late = np.array([l[3] for l in leagues])                    # the same choices, judged on the last third of seasons 7 to 10
    late_gain = late[:, DECAYS.index(CHAPTER_DECAY)] - late[np.arange(REPLICATES), pick]
    return clean({
        "persistence": persistence, "decays": DECAYS,
        "development_curve": dev.mean(axis=0), "assessment_curve": ass.mean(axis=0),
        "true_probability_floor": floor,
        "chosen_decays": [DECAYS[i] for i in pick], "mean_chosen_decay": float(np.mean([DECAYS[i] for i in pick])),
        "tuned": float(tuned.mean()), "chapter_default": float(default.mean()),
        "best_in_hindsight": float(ass.min(axis=1).mean()),
        "reset_each_season": float(ass[:, 0].mean()), "no_regression": float(ass[:, -1].mean()),
        "tuned_beats_default": float(np.mean(tuned < default)),
        "tuned_gain_over_default": float(np.mean(default - tuned)),
        "gain_sd": float(np.std(default - tuned)),
        "late_season_gain_over_default": float(late_gain.mean()),
        "replicates": REPLICATES, "games_per_season": GAMES, "teams": TEAMS,
    })
# notebook-end


SPEC = {
    "chapter": 6,
    "chapter_title": "Running Case: March Mania 2026",
    "subtitle": "Treat the Elo offseason decay as a hypothesis, and choose it on past seasons.",
    "summary": ("The chapter's Elo ratings regress toward the mean each offseason by a fixed fraction. One demonstration builds leagues "
                "in which the true carryover is known, picks the decay on past seasons only, and judges the choice on later seasons."),
    "title": "Choosing the Elo offseason decay on past seasons, against how much strength truly persists",
    "question": "Does an Elo decay chosen on past seasons find how much team strength truly carries over, and does it beat the chapter's 0.65?",
    "why": ("The decay is a basketball hypothesis: how much of a team's rating survives graduation and recruiting. Choosing it on past "
            "seasons and judging it on later ones is the chapter's season-appropriate comparison, and the measurement shows when it "
            "pays and when the example value was already good enough."),
    "method": ("Constructed leagues of 24 teams playing 360 games a season for 10 seasons. True team strength, in Elo points, follows an "
               "AR(1) process whose persistence is the control, so the best carryover is known. The chapter's Elo update (K = 24) runs "
               "with 21 candidate decays from 0 to 1. Each candidate is scored by Brier on the first third of each season's games, where "
               "the carried rating matters most. Seasons 2 to 6 choose the decay; seasons 7 to 10 judge it. Every number is a mean over 20 independent leagues."),
    "control": {"key": "persistence", "label": "True persistence of team strength from one season to the next",
                "values": [0.3, 0.65, 0.9], "default": 0.9,
                "value_labels": ["0.3: strength mostly resets", "0.65: the chapter's example value", "0.9: strength mostly persists"]},
    "source_section": "The Feature Set: Domain Knowledge Encoded as Numbers",
    "symbols": ("r is a team's Elo rating, d the offseason decay (the fraction of its deviation from 1500 that carries into the next "
                "season), r_A and r_B the two teams' pre-game ratings and e_A the probability the chapter's Elo assigns to team A winning."),
    "explanation": ("With persistence p, a team's best forecast for next season keeps a fraction p of its deviation from average, so the "
                    "decay that minimizes Brier should sit near p. A decay that is too low throws away real information and too high "
                    "keeps stale ratings. Choosing on seasons 2 to 6 finds a value near p; whether that beats a fixed example value depends on "
                    "how far p is from it."),
    "application": ("Treat decay, K and any similar constant as hypotheses. Tune them on earlier seasons with past-only folds, judge the "
                    "choice on later seasons, and keep the example value when the gain over it is smaller than the noise of the choice."),
    "assumptions": ("Constructed leagues: random pairings, neutral sites, no home advantage, strength following an AR(1) process, and "
                    "ratings that all start at 1500. Real tournament teams, schedules and transfer behaviour differ, so 0.65 or any other "
                    "value is not a recommendation. The Brier floor is the score of the true win probabilities, which no rating can reach. "
                    "The first third of each season is scored because the carried rating matters most there; the chapter's matchup "
                    "feature is read at tournament start, after a full season of updates. On the last third of each season the same "
                    "tuned decay gained at most 0.0012 Brier over 0.65, under half of its first-third gain at persistence 0.3 and 0.9."),
    "prediction": ("True persistence is 0.9 and the chapter's example decay is 0.65. If each league picks its decay on seasons 2 to 6, "
                   "what decay does it pick on average?"),
    "prediction_options": ["About 0.65, the chapter's value", "About 0.92, close to the true persistence", "About 0.5, halfway between the extremes"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "The mean pick is 0.92. On seasons 7 to 10 the tuned decay scores 0.2077 Brier against 0.2104 for 0.65.",
        "incorrect": "The mean pick is 0.92, close to the true 0.9. On seasons 7 to 10 the tuned decay scores 0.2077 Brier against 0.2104 for 0.65.",
    },
    "check": "When true persistence is 0.65, the chapter's own example value, does tuning on past seasons still help?",
    "answer": ("No. The chosen decay averages 0.67, but the tuned and default scores differ by only 0.0001 Brier and tuning wins in only half of the leagues (50% here): "
               "five seasons of games cannot separate nearby decays, so the choice adds noise about as large as the gain. At persistence 0.3 or 0.9 "
               "the same procedure beats 0.65 in most leagues (85% here) by about 0.0027 Brier."),
    "provenance": "Constructed example: seeded synthetic leagues with known persistence and the chapter's Elo update, measured by the chapter activity.",
    "apply": [
        "Choose offseason decay, K and the like on earlier seasons only, then score the chosen setting on later seasons you did not tune on.",
        "Plot the score against the candidate value: a flat valley means the exact choice hardly matters, a steep one means it does.",
        "Compare against the example value on the later seasons; adopt the tuned one only when its gain exceeds the spread across leagues or seasons.",
        "Re-tune when the population changes (a new rule, a league merger), because the right carryover is a property of the competition.",
    ],
    "honesty": ("Constructed leagues with a known AR(1) strength process. The sizes are properties of this generator; at the chapter's "
                "own value tuning did not help, and no claim is made about real basketball."),
}

EQUATIONS = [{"tex": r"r \leftarrow 1500 + d\,(r - 1500), \qquad e_A = \frac{1}{1 + 10^{(r_B - r_A)/400}}",
              "alt": "r becomes 1500 plus d times r minus 1500; e A equals one over one plus ten to the power r B minus r A over 400",
              "basis": "The offseason regression and win probability in the chapter's build_elo_ratings code (Elo Ratings); the chapter gives them as code, not as display equations."}]
NCOLS = 2
HEIGHT = 4.5
POLICIES = [("reset_each_season", "Reset\n(0)"), ("chapter_default", "Chapter\n(0.65)"), ("tuned", "Tuned on\n2 to 6"),
            ("best_in_hindsight", "Best in\nhindsight"), ("no_regression", "No\nregression\n(1)")]


def draw(axes, result, parameter):
    left, right = axes
    xs, floor = result["decays"], result["true_probability_floor"]
    left.plot(xs, result["development_curve"], color=COLORS["gold"], lw=1.8, label="Seasons 2 to 6 (choose)")
    left.plot(xs, result["assessment_curve"], color=COLORS["teal"], lw=1.8, label="Seasons 7 to 10 (judge)")
    left.axhline(floor, color=COLORS["grey"], ls=(0, (4, 3)), lw=1.2, label=f"True probabilities: {fmt(floor, 3)}")
    left.axvline(result["persistence"], ymax=0.72, color=COLORS["terracotta"], lw=1.4, label=f"True persistence: {result['persistence']}")
    bottom = floor - 0.006
    chosen = result["chosen_decays"]
    left.scatter([c + (i % 5 - 2) * 0.004 for i, c in enumerate(chosen)], [bottom + 0.0012 * (i % 4) for i in range(len(chosen))],
                 s=14, marker="v", color=COLORS["ink"], zorder=3, label="Decay chosen by one league")
    top = max(max(result["development_curve"]), max(result["assessment_curve"]))
    left.set_ylim(bottom - 0.002, top + 0.024)
    left.set_xlim(-0.02, 1.02)
    left.set_xlabel("Offseason decay (fraction of rating kept)")
    left.set_ylabel("Brier score, first third of games")
    left.legend(loc="upper center", frameon=False, fontsize=10, ncol=2, columnspacing=1.0, handlelength=1.6)

    ys = [result[k] - floor for k, _ in POLICIES]
    colors = [COLORS["light"], COLORS["light"], COLORS["teal"], COLORS["gold"], COLORS["light"]]
    right.bar(range(len(ys)), ys, width=0.62, color=colors, edgecolor=COLORS["ink"], lw=0.6)
    for x, y in enumerate(ys):
        right.text(x, y + 0.001, fmt(y, 4), ha="center", va="bottom", fontsize=10)
    right.set_xticks(range(len(ys)), [n for _, n in POLICIES])
    right.set_ylim(0, max(ys) * 1.15)
    right.set_ylabel("Brier above the true-probability floor")
    right.set_xlabel("Decay policy, scored on seasons 7 to 10")


def explain(result, parameter):
    tuned, default, hind = result["tuned"], result["chapter_default"], result["best_in_hindsight"]
    diff = default - tuned
    if diff > 0:
        calc = f"{fmt(default, 4)} - {fmt(tuned, 4)} = {fmt(diff, 4)} in the tuned decay's favour"
    else:
        calc = f"{fmt(tuned, 4)} - {fmt(default, 4)} = {fmt(-diff, 4)} in the chapter value's favour"
    win = result["tuned_beats_default"]
    if abs(diff) < 0.0015 and win < 0.65:
        verdict = "The two are within the noise of the choice, so tuning is not worth trusting here."
    elif diff > 0:
        verdict = "Tuning on past seasons recovers a real gain over the example value."
    else:
        verdict = "The example value did better than the tuned one on average."
    interpretation = (
        f"True persistence {parameter}: leagues choosing on seasons 2 to 6 pick a decay of {fmt(result['mean_chosen_decay'], 2)} on average. "
        f"On seasons 7 to 10 the tuned decay scores {fmt(tuned, 4)} Brier and the chapter's 0.65 scores {fmt(default, 4)}: {calc}. "
        f"Tuning beats 0.65 in {fmt(100 * win, 0)}% of the {result['replicates']} leagues. {verdict}")
    steps = [
        f"Mean decay chosen on seasons 2 to 6: {fmt(result['mean_chosen_decay'], 2)} (true persistence {parameter}).",
        f"Chapter value minus tuned, seasons 7 to 10: {fmt(default, 4)} - {fmt(tuned, 4)} = {signed(diff, 4)}.",
        f"Tuned minus best in hindsight: {fmt(tuned, 4)} - {fmt(hind, 4)} = {signed(tuned - hind, 4)}.",
        f"Tuned decay beats 0.65 in {fmt(100 * win, 0)}% of leagues; spread of the gain across leagues {fmt(result['gain_sd'], 4)}.",
    ]
    metrics = {"Mean decay chosen": fmt(result["mean_chosen_decay"], 2), "Tuned Brier": fmt(tuned, 4),
               "Chapter 0.65 Brier": fmt(default, 4), "Best in hindsight": fmt(hind, 4),
               "Leagues where tuning wins": f"{fmt(100 * win, 0)}%"}
    alt = (f"Left: Brier against offseason decay for the seasons that choose and the seasons that judge, with the true persistence "
           f"{parameter} marked and each league's chosen decay as a triangle. Right: bars of Brier above the true-probability floor for five "
           f"decay policies; tuned {fmt(tuned - result['true_probability_floor'], 4)}, chapter value {fmt(default - result['true_probability_floor'], 4)}.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for p, res in results.items():
        assert abs(res["mean_chosen_decay"] - p) < 0.1, f"chosen decay should track persistence at {p}"
        best = res["decays"][int(min(range(len(res["decays"])), key=lambda i: res["assessment_curve"][i]))]
        assert abs(best - p) < 0.15, f"assessment curve should bottom out near persistence at {p}"
        assert res["tuned"] - res["best_in_hindsight"] < 0.003, f"tuned should be near the hindsight best at {p}"
    low, mid, high = results[0.3], results[0.65], results[0.9]
    for res in (low, high):
        assert res["tuned_gain_over_default"] > 0.001 and res["tuned_beats_default"] > 0.5, "tuning should beat 0.65 when persistence is far from it"
    assert abs(mid["tuned_gain_over_default"]) < 0.0015 and mid["tuned_beats_default"] < 0.65, "at 0.65 tuning should not reliably help"
    assert fmt(high["mean_chosen_decay"], 2) == "0.92", "prediction text: mean pick"
    assert fmt(high["tuned"], 4) == "0.2077" and fmt(high["chapter_default"], 4) == "0.2104", "prediction text: Brier numbers"
    assert fmt(mid["mean_chosen_decay"], 2) == "0.67" and fmt(abs(mid["tuned"] - mid["chapter_default"]), 4) == "0.0001", "answer text at 0.65"
    assert fmt(100 * mid["tuned_beats_default"], 0) == "50", "answer text: wins in half of the leagues"
    assert fmt(100 * low["tuned_beats_default"], 0) == "85" and fmt(100 * high["tuned_beats_default"], 0) == "85", "answer text: 85% of leagues"
    assert fmt(low["tuned_gain_over_default"], 4) == "0.0027" and fmt(high["tuned_gain_over_default"], 4) == "0.0027", "answer text: gain"
    late = {p: res["late_season_gain_over_default"] for p, res in results.items()}
    assert fmt(max(late.values()), 4) == "0.0012", "assumptions: late-season gain at most 0.0012"
    for res in (low, high):
        assert res["late_season_gain_over_default"] < 0.5 * res["tuned_gain_over_default"], "late gain under half the early gain"
