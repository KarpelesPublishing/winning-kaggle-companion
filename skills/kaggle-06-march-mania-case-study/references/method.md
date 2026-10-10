# Chapter 6: Running Case: March Mania 2026

**Does an Elo decay chosen on past seasons find how much team strength truly carries over, and does it beat the chapter's 0.65?**

The decay is a basketball hypothesis: how much of a team's rating survives graduation and recruiting. Choosing it on past seasons and judging it on later ones is the chapter's season-appropriate comparison, and the measurement shows when it pays and when the example value was already good enough.

## The experiment

Constructed leagues of 24 teams playing 360 games a season for 10 seasons. True team strength, in Elo points, follows an AR(1) process whose persistence is the control, so the best carryover is known. The chapter's Elo update (K = 24) runs with 21 candidate decays from 0 to 1. Each candidate is scored by Brier on the first third of each season's games, where the carried rating matters most. Seasons 2 to 6 choose the decay; seasons 7 to 10 judge it. Every number is a mean over 20 independent leagues.

Control: True persistence of team strength from one season to the next (0.3: strength mostly resets, 0.65: the chapter's example value, 0.9: strength mostly persists; default 0.9).

## Measured results

| Measure | 0.3: strength mostly resets | 0.65: the chapter's example value | 0.9: strength mostly persists |
|---|---|---|---|
| Mean decay chosen | 0.33 | 0.67 | 0.92 |
| Tuned Brier | 0.2321 | 0.2210 | 0.2077 |
| Chapter 0.65 Brier | 0.2347 | 0.2209 | 0.2104 |
| Best in hindsight | 0.2315 | 0.2205 | 0.2069 |
| Leagues where tuning wins | 85% | 50% | 85% |

## What the result says (default, true persistence of team strength from one season to the next = 0.9)

True persistence 0.9: leagues choosing on seasons 2 to 6 pick a decay of 0.92 on average. On seasons 7 to 10 the tuned decay scores 0.2077 Brier and the chapter's 0.65 scores 0.2104: 0.2104 - 0.2077 = 0.0027 in the tuned decay's favour. Tuning beats 0.65 in 85% of the 20 leagues. Tuning on past seasons recovers a real gain over the example value.

- Mean decay chosen on seasons 2 to 6: 0.92 (true persistence 0.9).
- Chapter value minus tuned, seasons 7 to 10: 0.2104 - 0.2077 = +0.0027.
- Tuned minus best in hindsight: 0.2077 - 0.2069 = +0.0008.
- Tuned decay beats 0.65 in 85% of leagues; spread of the gain across leagues 0.0030.

## Apply it to a competition

- Choose offseason decay, K and the like on earlier seasons only, then score the chosen setting on later seasons you did not tune on.
- Plot the score against the candidate value: a flat valley means the exact choice hardly matters, a steep one means it does.
- Compare against the example value on the later seasons; adopt the tuned one only when its gain exceeds the spread across leagues or seasons.
- Re-tune when the population changes (a new rule, a league merger), because the right carryover is a property of the competition.

## Assumptions and limits

Constructed leagues: random pairings, neutral sites, no home advantage, strength following an AR(1) process, and ratings that all start at 1500. Real tournament teams, schedules and transfer behaviour differ, so 0.65 or any other value is not a recommendation. The Brier floor is the score of the true win probabilities, which no rating can reach. The first third of each season is scored because the carried rating matters most there; the chapter's matchup feature is read at tournament start, after a full season of updates. On the last third of each season the same tuned decay gained at most 0.0012 Brier over 0.65, under half of its first-third gain at persistence 0.3 and 0.9.

Constructed leagues with a known AR(1) strength process. The sizes are properties of this generator; at the chapter's own value tuning did not help, and no claim is made about real basketball.

## Reproduce it

The chapter notebook `notebooks/06-march-mania-case-study.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch06` (`run`, `explain`, `draw`).

```python
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
```

Book location: Chapter 6, The Feature Set: Domain Knowledge Encoded as Numbers. Constructed example: seeded synthetic leagues with known persistence and the chapter's Elo update, measured by the chapter activity.
