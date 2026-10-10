"""Chapter 63: Lessons from the Expanded Winning-Solution Corpus. How often winners use a technique, against what it does."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
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
# notebook-end


SPEC = {
    "chapter": 63,
    "chapter_title": "Lessons from the Expanded Winning-Solution Corpus",
    "subtitle": "Borrow a decision with its conditions: a method's presence in winning systems documents use, not benefit.",
    "summary": ("How often a technique appears among winners is not its effect. One demonstration simulates thousands of constructed "
                "competitions and compares the share of winners using a technique with what a paired on-off comparison measures."),
    "title": "How often winners use a technique, against what the technique does",
    "question": "When winners use a technique more often than the field does, how much does that say about the technique's effect?",
    "why": ("The chapter's evidence boundary is that presence in a winning system documents use, and only an attributed comparison can "
            "support a narrower contribution claim. A write-up count mixes the technique's effect with who chose to use it."),
    "method": ("2,000 constructed competitions of 40 teams each. A team's score is its skill plus noise, plus the true effect of technique T "
               "if it uses T. The field adopts T at 70%. In the skill-linked world stronger teams adopt T more often (the odds rise with skill); in "
               "the unrelated world adoption ignores skill. The first-place team of each competition is the one with the highest score. "
               "The control is the true effect of T in skill standard deviations. The curve repeats this for effects from -1.5 to +1.0. "
               "A paired ablation then runs 30 teams with and without T, 60 times over."),
    "control": {"key": "effect", "label": "True effect of technique T on a team's score (skill standard deviations)",
                "values": [-0.5, 0.0, 0.25, 0.75], "default": 0.0,
                "value_labels": ["-0.5: T hurts", "0: no effect", "+0.25", "+0.75: T helps a lot"]},
    "source_section": "Follow the Evidence Boundary",
    "symbols": ("T is the technique, p_win the share of first-place teams that used T, p_field the share of all teams that used T, and "
                "the effect is the average change in a team's score caused by using T, in skill standard deviations."),
    "explanation": ("Winning is a selection on score. If stronger teams are likelier to try T, winners contain extra T users whatever T "
                    "does, so the share of winners using T stays well above the field's 70% even when T has no effect, or hurts. "
                    "Only when adoption ignores skill does the share track the effect, and even then a high share is reached "
                    "by modest effects. A paired comparison runs the same team with and without T, which cancels skill."),
    "application": ("Read a mention count as a record of use. Before borrowing a technique, look for a with-and-without comparison on the "
                    "same setup, and reproduce it on your own validation."),
    "assumptions": ("Constructed data and one assumed link between skill and adoption (odds multiply by 2.7 per skill standard deviation); "
                    "nothing here measures how real teams adopt techniques. The size of the inflation depends on that link, which is why the "
                    "unrelated world is shown beside it. The paired ablation has no confounding by construction, and its estimate still "
                    "varies with a standard deviation of about 0.2 across 30-team experiments."),
    "prediction": ("Technique T lowers every user's score by 0.5 skill standard deviations. The field adopts it at 70%, and stronger teams "
                   "are somewhat likelier to adopt. What share of first-place teams used T?"),
    "prediction_options": ["Well below 70%: a harmful technique is filtered out", "About 70%", "Above 70%"],
    "prediction_answer": 2,
    "prediction_feedback": {
        "correct": "83% of winners used T although it lowers scores by 0.5, because the stronger teams that win adopted it more often.",
        "incorrect": "83% of winners used T, above the field's 70%, although T lowers scores by 0.5: the stronger teams that win adopted it more often.",
    },
    "check": "With no effect at all, 93% of winners use T. How would a reader tell this from a technique that works?",
    "answer": ("A share of 93% is what 70% adoption and skill-linked adoption produce when T does nothing, and it differs little from "
               "the 99% seen when T helps by 0.75. When adoption is unrelated to skill, the same no-effect case gives 70%. The count alone "
               "cannot separate these cases. A paired on-off comparison can: its estimate at no effect averages "
               "-0.017 with a spread of 0.203 across 60 experiments, while at an effect of 0.75 it averages 0.733."),
    "provenance": "Constructed example: 2,000 simulated competitions of 40 teams, seeded, measured by the chapter activity; no real competition data.",
    "apply": [
        "Treat a technique's presence in winning write-ups as evidence that it was used, never as evidence that it helped.",
        "Ask for the with-and-without comparison on the same setup; if the write-up has none, plan one on your own validation.",
        "Compare against how common the technique is in the field before reading a high share as a lift.",
        "Keep each borrowed idea with its conditions and the evidence type, and reproduce it before adding it to your pipeline.",
    ],
    "honesty": ("Constructed data. The inflation of the winners' share depends on the assumed link between skill and adoption; the activity shows "
                "the mechanism, not how strong it is in real competitions."),
}

EQUATIONS = [{"tex": r"\text{lift}=p_{\mathrm{win}}-p_{\mathrm{field}}",
              "alt": "The apparent lift equals the share of winners using the technique minus the share of all teams using it",
              "basis": "The activity's own measure of the apparent lift in a mention count; the chapter states the evidence boundary in words, without a display equation."}]
NCOLS = 2
HEIGHT = 4.4


def draw(axes, result, parameter):
    left, right = axes
    sweep = result["sweep"]
    left.plot(sweep["effect"], sweep["skill_linked"], color=COLORS["terracotta"], lw=2, marker="o", ms=3, label="Stronger teams adopt more")
    left.plot(sweep["effect"], sweep["unrelated"], color=COLORS["navy"], lw=2, marker="o", ms=3, label="Adoption unrelated to skill")
    left.axhline(result["field_adoption"], color=COLORS["grey"], ls=(0, (4, 3)), lw=1.4, label=f"Field adoption: {fmt(result['field_adoption'], 2)}")
    left.scatter([parameter, parameter], [result["winners_share_skill_linked"], result["winners_share_unrelated"]], s=90, facecolor="none",
                 edgecolor=COLORS["ink"], lw=2, zorder=4)
    left.set_ylim(0, 1.0)
    left.set_xlim(-1.55, 1.05)
    left.set_xlabel("True effect of technique T (skill standard deviations)")
    left.set_ylabel("Share of first-place teams using T")
    left.legend(loc="lower right", frameon=False, fontsize=10)

    est = result["ablation"]["estimates"]
    rng = np.random.default_rng(1)
    right.scatter(rng.uniform(-0.18, 0.18, len(est)), est, s=14, color=COLORS["navy"], zorder=3)
    right.errorbar([0.45], [result["ablation"]["mean"]], yerr=[2 * result["ablation"]["sd"]], marker="D", ms=7, color=COLORS["terracotta"],
                   lw=2, capsize=6, zorder=4)
    right.axhline(parameter, color=COLORS["gold"], ls=(0, (4, 3)), lw=1.6, label=f"True effect: {signed(parameter, 2)}")
    right.axhline(0, color=COLORS["grey"], lw=0.8)
    right.set_xlim(-0.5, 0.95)
    right.set_xticks([0, 0.45], ["Each dot: one\n30-team experiment", "Mean and\n2 spreads"])
    low, high = min(min(est), parameter, -0.1), max(max(est), parameter, 0.1)
    right.set_ylim(low - 0.25, high + 0.25)
    right.set_ylabel("Paired estimate of the effect (with minus without T)")
    right.set_xlabel("Paired ablation, 60 experiments")
    right.legend(loc="upper left", frameon=False, fontsize=10)


def explain(result, parameter):
    pw, pu, pf = result["winners_share_skill_linked"], result["winners_share_unrelated"], result["field_adoption"]
    ab = result["ablation"]
    lift = float(fmt(pw, 3)) - float(fmt(pf, 3))
    interpretation = (
        f"With a true effect of {signed(parameter, 2)}, {round(100 * pw)}% of first-place teams used T when stronger teams adopt it more, against {round(100 * pf)}% "
        f"of the whole field: {fmt(pw)} - {fmt(pf)} = {fmt(lift)} of apparent lift. When adoption ignores skill the share is {round(100 * pu)}%. "
        f"The skill-linked share only falls back to the field's rate when T lowers scores by about {fmt(abs(result['crossing_effect']), 1)}. "
        f"A paired on-off comparison averages {signed(ab['mean'], 3)} across {result['experiments']} experiments (spread {fmt(ab['sd'])}).")
    steps = [
        f"Apparent lift in the skill-linked world: {fmt(pw)} - {fmt(pf)} = {fmt(lift)}.",
        f"Apparent lift when adoption ignores skill: {fmt(pu)} - {fmt(pf)} = {fmt(float(fmt(pu, 3)) - float(fmt(pf, 3)))}.",
        f"Paired ablation error: {fmt(ab['mean'])} - {fmt(parameter) if parameter >= 0 else '(' + fmt(parameter) + ')'} = {fmt(float(fmt(ab['mean'], 3)) - float(fmt(parameter, 3)))} (average estimate minus true effect).",
        f"Effect at which the skill-linked share meets the field rate: {fmt(result['crossing_effect'], 2)}.",
    ]
    metrics = {"Winners using T, skill-linked adoption": f"{round(100 * pw)}%", "Winners using T, unrelated adoption": f"{round(100 * pu)}%",
               "Field adoption": f"{round(100 * pf)}%", "Paired ablation, mean estimate": signed(ab["mean"], 3),
               "Paired ablation, spread": fmt(ab["sd"])}
    alt = (f"Left: share of first-place teams using technique T against its true effect, for skill-linked adoption and for adoption unrelated to skill, "
           f"with the field's {round(100 * pf)}% as a dashed line; at an effect of {signed(parameter, 2)} the shares are {round(100 * pw)}% and {round(100 * pu)}%. "
           f"Right: sixty paired on-off estimates of the effect, averaging {signed(ab['mean'], 3)}, against the true effect.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for e, res in results.items():
        assert abs(res["field_adoption"] - 0.70) < 0.01, "field adoption is 70%"
        assert res["winners_share_skill_linked"] > res["winners_share_unrelated"], f"skill link inflates the share at {e}"
        assert res["winners_share_skill_linked"] > 0.7, f"skill-linked share above the field rate even for e={e}"
        assert abs(res["ablation"]["mean"] - e) < 0.08, f"paired ablation unbiased at {e}"
    sweep = results[0.0]["sweep"]
    assert all(a < b for a, b in zip(sweep["skill_linked"], sweep["skill_linked"][1:])), "share rises with effect"
    assert all(a < b for a, b in zip(sweep["unrelated"], sweep["unrelated"][1:])), "share rises with effect (unrelated)"
    zero = results[0.0]
    assert abs(zero["winners_share_unrelated"] - zero["field_adoption"]) < 0.03, "no effect and no link: share equals adoption"
    assert -1.0 < zero["crossing_effect"] < -0.7, "text: falls to the field rate near -0.9"
    assert round(100 * results[-0.5]["winners_share_skill_linked"]) == 83, "prediction feedback: 83%"
    assert round(100 * zero["winners_share_skill_linked"]) == 93 and round(100 * results[0.75]["winners_share_skill_linked"]) == 99, "check answer shares"
    assert round(100 * zero["winners_share_unrelated"]) == 70, "check answer: 70% when adoption is unrelated"
    assert fmt(zero["ablation"]["mean"]) == "-0.017" and fmt(zero["ablation"]["sd"]) == "0.203" and fmt(results[0.75]["ablation"]["mean"]) == "0.733", "check answer ablation"
    assert 0.15 < zero["ablation"]["sd"] < 0.25, "text: spread about 0.2"
