# Chapter 63: Lessons from the Expanded Winning-Solution Corpus

**When winners use a technique more often than the field does, how much does that say about the technique's effect?**

The chapter's evidence boundary is that presence in a winning system documents use, and only an attributed comparison can support a narrower contribution claim. A write-up count mixes the technique's effect with who chose to use it.

## The experiment

2,000 constructed competitions of 40 teams each. A team's score is its skill plus noise, plus the true effect of technique T if it uses T. The field adopts T at 70%. In the skill-linked world stronger teams adopt T more often (the odds rise with skill); in the unrelated world adoption ignores skill. The first-place team of each competition is the one with the highest score. The control is the true effect of T in skill standard deviations. The curve repeats this for effects from -1.5 to +1.0. A paired ablation then runs 30 teams with and without T, 60 times over.

Control: True effect of technique T on a team's score (skill standard deviations) (-0.5: T hurts, 0: no effect, +0.25, +0.75: T helps a lot; default 0.0).

## Measured results

| Measure | -0.5: T hurts | 0: no effect | +0.25 | +0.75: T helps a lot |
|---|---|---|---|---|
| Winners using T, skill-linked adoption | 83% | 93% | 96% | 99% |
| Winners using T, unrelated adoption | 48% | 70% | 78% | 90% |
| Field adoption | 70% | 70% | 70% | 70% |
| Paired ablation, mean estimate | -0.517 | -0.017 | +0.233 | +0.733 |
| Paired ablation, spread | 0.203 | 0.203 | 0.203 | 0.203 |

## What the result says (default, true effect of technique t on a team's score (skill standard deviations) = 0.0)

With a true effect of +0.00, 93% of first-place teams used T when stronger teams adopt it more, against 70% of the whole field: 0.932 - 0.700 = 0.232 of apparent lift. When adoption ignores skill the share is 70%. The skill-linked share only falls back to the field's rate when T lowers scores by about 0.9. A paired on-off comparison averages -0.017 across 60 experiments (spread 0.203).

- Apparent lift in the skill-linked world: 0.932 - 0.700 = 0.232.
- Apparent lift when adoption ignores skill: 0.696 - 0.700 = -0.004.
- Paired ablation error: -0.017 - 0.000 = -0.017 (average estimate minus true effect).
- Effect at which the skill-linked share meets the field rate: -0.88.

## Apply it to a competition

- Treat a technique's presence in winning write-ups as evidence that it was used, never as evidence that it helped.
- Ask for the with-and-without comparison on the same setup; if the write-up has none, plan one on your own validation.
- Compare against how common the technique is in the field before reading a high share as a lift.
- Keep each borrowed idea with its conditions and the evidence type, and reproduce it before adding it to your pipeline.

## Assumptions and limits

Constructed data and one assumed link between skill and adoption (odds multiply by 2.7 per skill standard deviation); nothing here measures how real teams adopt techniques. The size of the inflation depends on that link, which is why the unrelated world is shown beside it. The paired ablation has no confounding by construction, and its estimate still varies with a standard deviation of about 0.2 across 30-team experiments.

Constructed data. The inflation of the winners' share depends on the assumed link between skill and adoption; the activity shows the mechanism, not how strong it is in real competitions.

## Reproduce it

The chapter notebook `notebooks/63-patterns-from-226-solutions.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch63` (`run`, `explain`, `draw`).

```python
import numpy as np

from kaggle_companion.activities._common import clean

SEED = 63
COMPETITIONS, TEAMS = 2000, 40
ADOPTION = 0.70                 # share of all teams that use technique T
SKILL_LINK = 1.0                # how strongly team skill raises the odds of adopting T (0 = unrelated)
NOISE = 0.7                     # run-to-run noise in a team's score
SWEEP = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]   # true effects shown on the curve
EXPERIMENTS, PAIRS = 60, 30     # ablation: 60 repeated experiments, each comparing T on and off for 30 teams


def adoption_odds(skill, link):
    """Each team's chance of using T: logistic in skill, with the intercept set so the field adopts T at ADOPTION."""
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (1 / (1 + np.exp(-(mid + link * skill)))).mean() < ADOPTION else (lo, mid)
    return 1 / (1 + np.exp(-((lo + hi) / 2 + link * skill)))


def winners_share(effect, skill, uniform, noise, link):
    """Share of first-place teams that used T, when T shifts every user's score by `effect` skill standard deviations."""
    used = uniform < adoption_odds(skill, link)
    score = skill + effect * used + noise                       # same skills and noise for every effect: a smooth curve
    winner = score.argmax(axis=1)
    return float(used[np.arange(len(skill)), winner].mean()), float(used.mean())


def ablation(effect, rng):
    """Paired experiments: each team runs once with T and once without. Skill cancels in the difference."""
    skill = rng.normal(size=(EXPERIMENTS, PAIRS))
    with_t = skill + effect + rng.normal(0, NOISE, skill.shape)
    without_t = skill + rng.normal(0, NOISE, skill.shape)
    return (with_t - without_t).mean(axis=1)                    # one effect estimate per experiment


def run(effect):
    rng = np.random.default_rng(SEED)
    skill = rng.normal(size=(COMPETITIONS, TEAMS))
    uniform = rng.random((COMPETITIONS, TEAMS))
    noise = rng.normal(0, NOISE, (COMPETITIONS, TEAMS))
    confounded = [winners_share(e, skill, uniform, noise, SKILL_LINK)[0] for e in SWEEP]
    unrelated = [winners_share(e, skill, uniform, noise, 0.0)[0] for e in SWEEP]
    here_c, field = winners_share(effect, skill, uniform, noise, SKILL_LINK)
    here_u, _ = winners_share(effect, skill, uniform, noise, 0.0)
    # effect at which the confounded winners' share falls back to the field's adoption rate (linear interpolation)
    below = [i for i in range(len(SWEEP) - 1) if confounded[i] <= field < confounded[i + 1]]
    crossing = SWEEP[below[0]] + (field - confounded[below[0]]) / (confounded[below[0] + 1] - confounded[below[0]]) * (SWEEP[1] - SWEEP[0])
    estimates = ablation(effect, rng)
    return clean({
        "effect": effect, "field_adoption": field,
        "winners_share_skill_linked": here_c, "winners_share_unrelated": here_u,
        "sweep": {"effect": SWEEP, "skill_linked": confounded, "unrelated": unrelated},
        "crossing_effect": crossing,
        "ablation": {"estimates": estimates, "mean": float(estimates.mean()), "sd": float(estimates.std(ddof=1)),
                     "right_sign_share": float(np.mean(np.sign(estimates) == np.sign(effect))) if effect != 0 else None},
        "competitions": COMPETITIONS, "teams": TEAMS, "experiments": EXPERIMENTS, "pairs": PAIRS,
    })
```

Book location: Chapter 63, Follow the Evidence Boundary. Constructed example: 2,000 simulated competitions of 40 teams, seeded, measured by the chapter activity; no real competition data.
