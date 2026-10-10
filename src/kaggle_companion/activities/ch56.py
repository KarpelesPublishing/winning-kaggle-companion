"""Chapter 56: Reinforcement Learning for Game Competitions. A Q-learner on a corridor game with a farmable shaping bonus: shaped return against the real win rate."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import random
from functools import lru_cache

import numpy as np

from kaggle_companion.activities._common import clean

SEED = 56
CELLS, MINE, STEPS = 14, 3, 30    # corridor of 14 cells, a resource cell at 3, the goal at the far end, 30 steps to win
SLIP, GAMMA = 0.1, 0.97           # a move slips (reverses) 10% of the time; discount for the learner
EPISODES, LEARNERS, EVAL_GAMES = 800, 8, 200
ALPHA, EPSILON = 0.2, 0.1
OPTIMISM = 1.0                    # optimistic initial action values make the learner try every action before settling


def move(state, action, rng):
    """One step of the corridor game: action 1 is right, 0 is left; the move slips with probability SLIP."""
    if rng.random() < SLIP:
        action = 1 - action
    return min(CELLS - 1, max(0, state + (1 if action else -1)))


def reward(state, nxt, bonus, kind):
    """True reward (+1 for winning) plus shaping. 'bonus' pays for each fresh arrival at the resource cell;
    'potential' pays the change in a progress potential (gamma * phi(next) - phi(state)), which cannot change the best policy."""
    done = nxt == CELLS - 1
    r = 1.0 if done else 0.0
    if kind == "bonus" and nxt == MINE and state != MINE:
        r += bonus
    if kind == "potential":
        phi = lambda s: bonus * s / (CELLS - 1)
        r += GAMMA * (0.0 if done else phi(nxt)) - phi(state)
    return r


def train(bonus, kind, seed):
    """Tabular Q-learning with epsilon-greedy exploration; returns the greedy policy (one action per cell)."""
    rng = random.Random(seed)
    q = [[OPTIMISM, OPTIMISM] for _ in range(CELLS)]
    for _ in range(EPISODES):
        state = 0
        for _ in range(STEPS):
            if rng.random() < EPSILON:
                action = rng.randrange(2)
            else:
                action = 0 if q[state][0] > q[state][1] else 1 if q[state][1] > q[state][0] else rng.randrange(2)
            nxt = move(state, action, rng)
            done = nxt == CELLS - 1
            target = reward(state, nxt, bonus, kind) + (0.0 if done else GAMMA * max(q[nxt]))
            q[state][action] += ALPHA * (target - q[state][action])
            state = nxt
            if done:
                break
    return [0 if row[0] > row[1] else 1 for row in q]


def optimal_policy(bonus):
    """Value iteration on the same game with the bonus reward: the best any learner could settle on."""
    q = np.zeros((CELLS, 2))
    for _ in range(600):
        for s in range(CELLS - 1):
            for a in (0, 1):
                total = 0.0
                for actual, p in ((a, 1 - SLIP), (1 - a, SLIP)):
                    nxt = min(CELLS - 1, max(0, s + (1 if actual else -1)))
                    done = nxt == CELLS - 1
                    total += p * (reward(s, nxt, bonus, "bonus") + (0.0 if done else GAMMA * q[nxt].max()))
                q[s, a] = total
    return [int(q[s, 1] >= q[s, 0]) for s in range(CELLS)]


def play(policy, bonus, rng):
    """Mean win rate and mean shaped return (win plus bonus per resource arrival) of a policy over fresh games."""
    wins, shaped = 0, 0.0
    for _ in range(EVAL_GAMES):
        state = 0
        for _ in range(STEPS):
            nxt = move(state, policy[state], rng)
            shaped += reward(state, nxt, bonus, "bonus")
            state = nxt
            if state == CELLS - 1:
                wins += 1
                break
    return wins / EVAL_GAMES, shaped / EVAL_GAMES


@lru_cache(maxsize=None)
def sparse_policies(seed):
    """The learners trained on the true reward only; they do not depend on the bonus, so they are trained once."""
    return [train(0.0, "bonus", seed * 100 + i) for i in range(LEARNERS)]


def run(bonus):
    rng = random.Random(SEED)
    sparse = [play(p, bonus, rng) for p in sparse_policies(SEED)]
    shaped = [play(train(bonus, "bonus", SEED * 100 + 50 + i), bonus, rng) for i in range(LEARNERS)]
    potential = [play(train(bonus, "potential", SEED * 100 + 80 + i), bonus, rng) for i in range(LEARNERS)]
    best = play(optimal_policy(bonus), bonus, rng)
    column = lambda rows, k: [row[k] for row in rows]
    return clean({
        "bonus": bonus,
        "win": {"sparse": float(np.mean(column(sparse, 0))), "bonus": float(np.mean(column(shaped, 0))),
                "potential": float(np.mean(column(potential, 0))), "optimal": best[0]},
        "shaped_return": {"sparse": float(np.mean(column(sparse, 1))), "bonus": float(np.mean(column(shaped, 1))),
                          "optimal": best[1]},
        "per_learner_win": {"sparse": column(sparse, 0), "bonus": column(shaped, 0), "potential": column(potential, 0)},
        "learners": LEARNERS, "games": EVAL_GAMES,
    })
# notebook-end


SPEC = {
    "chapter": 56,
    "chapter_title": "Reinforcement Learning for Game Competitions",
    "subtitle": "Shape a reward toward winning, then check the policy against the real win rate.",
    "summary": ("An intermediate reward gives the learner feedback, and a learner optimizes the reward it is given. One demonstration "
                "trains Q-learners on a corridor game with a farmable resource bonus and compares the shaped return and the real win rate "
                "of the policies they learn."),
    "title": "Shaped return against real win rate, as the resource bonus grows",
    "question": "At what size of intermediate reward does a learner stop trying to win, and does the shaped return reveal it?",
    "why": ("Game rewards are sparse, so competitors add intermediate rewards for resources, territory or survival. If an intermediate "
            "reward can be collected repeatedly, a larger one pays more than winning, and a policy chosen by its shaped return stops winning."),
    "method": ("A constructed corridor game: 14 cells, the agent starts at the left end and wins (+1) by reaching the right end within "
               "30 steps; each move slips with probability 0.1. A resource cell near the start pays a bonus for every fresh arrival, so it "
               "can be farmed by stepping off and on. Tabular Q-learning with optimistic initial values runs for 800 games per learner, 8 "
               "learners per setting. Four policies are compared at the control's bonus: learned on the true reward only, learned with the "
               "bonus, learned with a potential-based progress reward of the same weight, and the best policy for the bonus reward found "
               "by value iteration. Each is scored on 200 fresh games for win rate and for shaped return (win plus bonuses)."),
    "control": {"key": "bonus", "label": "Reward per fresh arrival at the resource cell (a win pays 1)",
                "values": [0.02, 0.05, 0.1, 0.5], "default": 0.1,
                "value_labels": ["0.02", "0.05", "0.1", "0.5"]},
    "source_section": "The Fundamental Challenge: Sparse Rewards, Long Horizons",
    "symbols": ("w is the bonus per resource arrival, a win pays 1, and the shaped return of a game is the win indicator plus w times "
                "the number of fresh arrivals at the resource cell. The win rate is the share of games that reach the goal within 30 steps. "
                "gamma is the learner's discount, 0.97."),
    "explanation": ("A small bonus does no harm: the policy still heads for the goal and picks the resource up on the way. Once the bonus "
                    "is large enough, circling the resource pays more than the discounted value of winning, the learner settles on circling, "
                    "and the win rate falls to zero while the shaped return rises. The best policy for the bonus reward does the same, so this "
                    "is a property of the reward and not a failure to explore. In between, at 0.05, the learner already loses most games "
                    "although farming lowers even the shaped return per game: it maximizes a discounted reward with no clock, which matches "
                    "neither score, so at that weight the shaped return still flags the loss and at 0.1 it no longer does. A potential-based "
                    "reward of the same weight leaves the best policy unchanged."),
    "application": ("Pick checkpoints and reward weights by the real win rate against the declared opponents, not by the shaped return the "
                    "learner was trained on, and make intermediate rewards pay once and point toward winning."),
    "assumptions": ("A constructed one-dimensional game, tabular Q-learning instead of PPO, and a bonus that can be farmed by design; "
                    "real games have richer shaped rewards, and whether a given one can be farmed depends on the game. Optimistic initial "
                    "values make the learner explore enough to find the goal: without them, even a tiny bonus stops this learner "
                    "from ever finding it, which is a different failure (no exploration) from the one measured here. On this short corridor "
                    "the sparse reward alone is learned quickly, so the benefit of shaping for very long horizons is not shown."),
    "prediction": "With a resource bonus of 0.1 per arrival (a win pays 1), what win rate does the learner trained with the bonus reach?",
    "prediction_options": ["About the same as the sparse learner, near 100%", "About half", "Close to zero"],
    "prediction_answer": 2,
    "prediction_feedback": {
        "correct": "It wins 0% of games, against 100% for the sparse learner, while its shaped return is 1.23 against 1.13.",
        "incorrect": "It wins 0% of games, against 100% for the sparse learner, while its shaped return is 1.23 against 1.13. It circles the resource, because that pays more than winning.",
    },
    "check": "At a bonus of 0.1 the shaped return of the farming policy is only 0.10 above that of the winning policy. Why does the learner prefer farming, and what does the shaped return hide?",
    "answer": ("The learner discounts: a bonus collected now is worth more than a win 13 steps away, and a farming loop pays again every two "
               "steps. The shaped return per game of the farming policy is already higher at a bonus of 0.1 (1.23 against 1.13), so "
               "ranking policies by it picks one that never wins. Only the win rate shows the loss."),
    "provenance": "Constructed example: a seeded corridor game, tabular Q-learners and value iteration, measured by the chapter activity.",
    "apply": [
        "Evaluate every checkpoint on the real win rate against the declared opponents and seeds, and keep the shaped return as a training diagnostic only.",
        "Make intermediate rewards pay once per event, or as a potential-based difference, so they cannot be farmed.",
        "Keep the weight of a shaping term small enough that the best policy for the shaped reward is still a winning policy.",
        "Check that exploration is not the problem before blaming the reward: a learner that never reaches the goal looks the same from outside.",
    ],
    "honesty": ("Constructed game; the bonus size at which winning stops depends on the corridor length, the discount and the slip rate. "
                "The learned and the value-iteration policies agree here because the learner explores enough, which holds for this small game only."),
}

EQUATIONS = [{"tex": r"R_{\mathrm{shaped}} = \mathbf{1}[\text{win}] + w\,N_{\mathrm{arrivals}}, \qquad F(s,s') = \gamma\,\phi(s') - \phi(s)",
              "alt": "the shaped return equals the win indicator plus w times the number of arrivals at the resource cell; a potential-based shaping reward F of a move from s to s prime is gamma times phi of s prime minus phi of s",
              "basis": "The activity's shaped reward and the potential-based form used as the safe comparison; the chapter's shaped_reward function is a code example, not a display equation."}]
NCOLS = 2
HEIGHT = 4.4
WIN_BARS = [("sparse", "True reward\nonly", COLORS["light"]), ("bonus", "With resource\nbonus", COLORS["terracotta"]),
            ("potential", "Potential-\nbased", COLORS["teal"]), ("optimal", "Best policy\n(value iteration)", COLORS["gold"])]


def draw(axes, result, parameter):
    left, right = axes
    xs = list(range(len(WIN_BARS)))
    for x, (key, _, color) in zip(xs, WIN_BARS):
        value = result["win"][key]
        left.bar(x, value, width=0.6, color=color, edgecolor=COLORS["ink"], lw=0.6)
        left.text(x, value + 0.02, f"{round(100 * value)}%", ha="center", va="bottom", fontsize=10)
        if key in result["per_learner_win"]:
            dots = result["per_learner_win"][key]
            left.scatter([x + (i - (len(dots) - 1) / 2) * 0.05 for i in range(len(dots))], dots, s=10, color=COLORS["ink"], zorder=3,
                         label="One learner" if x == 0 else None)
    left.set_xticks(xs, [name for _, name, _ in WIN_BARS])
    left.set_xlim(-0.6, len(WIN_BARS) - 0.4)
    left.set_ylim(0, 1.18)
    left.set_ylabel("Real win rate (200 fresh games)")
    left.set_xlabel(f"Policy, resource bonus {fmt(parameter, 2)}")
    left.legend(loc="upper right", frameon=False, fontsize=10)

    shaped = [("sparse", "True reward\nonly", COLORS["light"]), ("bonus", "With resource\nbonus", COLORS["terracotta"])]
    for x, (key, _, color) in enumerate(shaped):
        value = result["shaped_return"][key]
        right.bar(x, value, width=0.5, color=color, edgecolor=COLORS["ink"], lw=0.6)
        right.text(x, value + 0.03, fmt(value, 2), ha="center", va="bottom", fontsize=10)
    right.set_xticks(range(len(shaped)), [name for _, name, _ in shaped])
    right.set_xlim(-0.6, len(shaped) - 0.4)
    right.set_ylim(0, max(result["shaped_return"][k] for k, _, _ in shaped) * 1.25)
    right.set_ylabel("Shaped return per game (win + bonuses)")
    right.set_xlabel("Policy, scored by the shaped reward")


def diff(a, b, digits=2):
    """Difference of the two numbers as displayed, so the hand calculation on the page adds up."""
    return float(fmt(a, digits)) - float(fmt(b, digits))


def explain(result, parameter):
    w, s = result["win"], result["shaped_return"]
    lost = diff(w["sparse"], w["bonus"])
    gain = diff(s["bonus"], s["sparse"])
    interpretation = (
        f"With a bonus of {fmt(parameter, 2)} per arrival, the learner trained on the true reward wins {round(100 * w['sparse'])}% of games and the "
        f"one trained with the bonus {round(100 * w['bonus'])}%, so {fmt(w['sparse'], 2)} - {fmt(w['bonus'], 2)} = {fmt(lost, 2)} of the wins "
        f"are lost to the bonus. The potential-based reward of the same weight wins {round(100 * w['potential'])}% and the best policy for the "
        f"bonus reward {round(100 * w['optimal'])}%. In shaped return the bonus-trained policy scores {fmt(s['bonus'], 2)} and the true-reward "
        f"policy {fmt(s['sparse'], 2)}, a difference of {fmt(gain, 2)}"
        + (" in favour of the policy that stopped winning." if gain > 0 and w["bonus"] < w["sparse"] - 0.1 else "."))
    steps = [
        f"Wins lost to the bonus: {fmt(w['sparse'], 2)} - {fmt(w['bonus'], 2)} = {fmt(lost, 2)}.",
        f"Shaped return, bonus-trained minus true-reward policy: {fmt(s['bonus'], 2)} - {fmt(s['sparse'], 2)} = {signed(gain, 2)}.",
        f"Potential-based learner win rate: {round(100 * w['potential'])}%.",
        f"Best policy for the bonus reward: wins {round(100 * w['optimal'])}% with shaped return {fmt(s['optimal'], 2)}.",
    ]
    metrics = {"Win rate, true reward only": f"{round(100 * w['sparse'])}%", "Win rate, resource bonus": f"{round(100 * w['bonus'])}%",
               "Win rate, potential-based": f"{round(100 * w['potential'])}%", "Shaped return, true reward only": fmt(s["sparse"], 2),
               "Shaped return, resource bonus": fmt(s["bonus"], 2)}
    alt = (f"Left: real win rate at a bonus of {fmt(parameter, 2)}: true reward only {round(100 * w['sparse'])}%, with the bonus "
           f"{round(100 * w['bonus'])}%, potential-based {round(100 * w['potential'])}%, best policy {round(100 * w['optimal'])}%. "
           f"Right: shaped return {fmt(s['sparse'], 2)} for the true-reward policy and {fmt(s['bonus'], 2)} with the bonus.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def _qualitative(results):
    """Directional claims; checked again under a different seed base."""
    for b, res in results.items():
        w = res["win"]
        assert w["sparse"] > 0.95, f"sparse learners should win at bonus {b}"
        assert w["potential"] > 0.95, f"potential-based shaping should keep winning at bonus {b}"
    small, big = results[0.02], results[0.1]
    assert small["win"]["bonus"] > 0.8 and small["win"]["optimal"] > 0.95, "a small bonus does no harm"
    for b in (0.1, 0.5):
        assert results[b]["win"]["bonus"] < 0.1 and results[b]["win"]["optimal"] < 0.1, f"large bonus: farming at {b}"
    assert big["shaped_return"]["bonus"] > big["shaped_return"]["sparse"], "shaped return prefers the farmer at 0.1"
    mid = results[0.05]
    assert 0.1 < mid["win"]["bonus"] < 0.5, "at 0.05 the learner loses most games"
    assert mid["shaped_return"]["bonus"] < mid["shaped_return"]["sparse"], "at 0.05 the shaped return still flags the loss"
    assert results[0.5]["shaped_return"]["bonus"] > 2 * results[0.5]["shaped_return"]["sparse"], "at 0.5 shaped return strongly prefers farming"


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    _qualitative(results)
    r = results[0.1]
    assert round(100 * r["win"]["bonus"]) == 0 and round(100 * r["win"]["sparse"]) == 100, "feedback win rates"
    assert fmt(r["shaped_return"]["bonus"], 2) == "1.23" and fmt(r["shaped_return"]["sparse"], 2) == "1.13", "feedback shaped returns"
    assert fmt(diff(r["shaped_return"]["bonus"], r["shaped_return"]["sparse"]), 2) == "0.10", "check text: 0.10 above"
