"""Chapter 55: Pancake Sorting as a Constructed Search Example. Beam search with three rankings under one beam-width budget, against optimal flip counts."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import itertools
from functools import lru_cache

import numpy as np
from scipy.stats import spearmanr

from kaggle_companion.activities._common import clean

SEED = 55
N = 8                    # pancakes: 8! = 40,320 states, small enough to know the true answer for every state
INSTANCES = 200          # random starting orders, the same for every policy and width
MAX_FLIPS = 12           # a beam that has not reached the goal after 12 flips reports failure (the longest optimum is 9)


def flip(state, k):
    """Reverse the first k entries; a legal move needs k >= 2."""
    return state[:k][::-1] + state[k:]


@lru_cache(maxsize=None)
def state_space():
    """Every state, its neighbours, the optimal flip count of every state (breadth-first from the goal) and two rankings."""
    states = list(itertools.permutations(range(N)))
    index = {s: i for i, s in enumerate(states)}
    neighbours = np.array([[index[flip(s, k)] for k in range(2, N + 1)] for s in states])
    goal = index[tuple(range(N))]
    optimal = np.full(len(states), -1)
    optimal[goal] = 0
    frontier = [goal]
    while frontier:                                   # each flip costs 1, so breadth-first order gives optimal distances
        nxt = []
        for u in frontier:
            for v in neighbours[u]:
                if optimal[v] < 0:
                    optimal[v] = optimal[u] + 1
                    nxt.append(v)
        frontier = nxt
    P = np.array(states)
    misplaced = (P != np.arange(N)).sum(axis=1)       # the chapter's ranking: positions holding the wrong value
    # Gap count: adjacent pairs, with a final plate of size N, that differ by more than 1.
    gaps = (np.abs(np.diff(np.concatenate([P, np.full((len(P), 1), N)], axis=1), axis=1)) != 1).sum(axis=1)
    return states, neighbours, goal, optimal, misplaced, gaps


def beam_search(start, width, score, rng):
    """Beam search that returns a verified flip list on success and None on failure, plus the states generated."""
    states, neighbours, goal, _, _, _ = state_space()
    parent = np.full(len(states), -1)
    move = np.zeros(len(states), dtype=int)
    seen = np.zeros(len(states), dtype=bool)
    seen[start] = True
    frontier, generated = np.array([start]), 0
    for _ in range(MAX_FLIPS):
        if start == goal:
            break
        children = neighbours[frontier].ravel()
        generated += len(children)
        sources = np.repeat(frontier, N - 1)
        flips = np.tile(np.arange(2, N + 1), len(frontier))
        children, first = np.unique(children, return_index=True)
        sources, flips = sources[first], flips[first]
        fresh = ~seen[children]
        children, sources, flips = children[fresh], sources[fresh], flips[fresh]
        seen[children] = True
        parent[children], move[children] = sources, flips
        if seen[goal] and goal in children:
            break
        if len(children) == 0:
            return None, generated, frontier
        order = np.argsort(score[children] + 0.5 * rng.random(len(children)))   # random tie-break
        frontier = children[order[:width]]
    else:
        return None, generated, frontier
    path, node = [], goal
    while node != start:
        path.append(int(move[node]))
        node = parent[node]
    path.reverse()
    reached = states[start]                           # replay every returned move before trusting it
    for k in path:
        reached = flip(reached, k)
    assert reached == tuple(range(N)), "replay did not reach the goal"
    return path, generated, frontier


def run(beam_width):
    states, _, _, optimal, misplaced, gaps = state_space()
    starts = np.random.default_rng(SEED).integers(0, len(states), INSTANCES)
    rankings = {"misplaced": misplaced.astype(float), "gap": gaps.astype(float), "random": np.zeros(len(states))}
    out = {}
    for name, score in rankings.items():
        rng = np.random.default_rng(SEED + 1)
        solved, excess, generated, near_miss = 0, [], [], 0
        for s in starts:
            path, gen, frontier = beam_search(s, beam_width, score, rng)
            generated.append(gen)
            if path is None:
                near_miss += int(misplaced[frontier].min() <= 2)   # an unfinished path that looks almost sorted
            else:
                solved += 1
                excess.append(len(path) - optimal[s])
        out[name] = {"solve_rate": solved / INSTANCES,
                     "excess_flips": float(np.mean(excess)) if excess else None,
                     "states_generated": float(np.mean(generated)),
                     "failures": INSTANCES - solved, "near_miss_failures": near_miss}
    return clean({
        "beam_width": beam_width, "policies": out,
        "misplaced_overestimates": float((misplaced > optimal).mean()),   # share of all 40,320 states
        "gap_overestimates": float((gaps > optimal).mean()),
        # what a beam actually uses is the order: how well does each ranking track the true distance over all states?
        "misplaced_rank_corr": float(spearmanr(misplaced, optimal)[0]),
        "gap_rank_corr": float(spearmanr(gaps, optimal)[0]),
        "mean_optimal_flips": float(optimal.mean()), "max_optimal_flips": int(optimal.max()),
        "instances": INSTANCES, "states": len(states),
    })
# notebook-end


SPEC = {
    "chapter": 55,
    "chapter_title": "Pancake Sorting as a Constructed Search Example",
    "subtitle": "An unfinished path is not a solution, however promising its heuristic score.",
    "summary": ("A search is judged by completed, replayed solutions under one budget. One demonstration runs beam search on "
                "8-pancake instances with three rankings and measures how many it solves, how many flips it wastes against the "
                "true optimum, and how often an unfinished path looks almost sorted."),
    "title": "Beam search with three rankings, by beam width, against the true optimum",
    "question": "Under the same beam width, how many instances does each ranking actually solve, and how far from optimal are its solutions?",
    "why": ("A beam search is only as good as the ranking that decides what to keep. A ranking that looks sensible, like the number of "
            "misplaced positions, can fail to finish, and a failed run with an almost sorted state is still a failure."),
    "method": ("Eight pancakes, so 40,320 states. A breadth-first search from the sorted order gives the optimal flip count of "
               "every state. On 200 random starting orders, beam search expands all flips of every state in the beam, drops repeated "
               "states, and keeps the best W by a ranking: the number of misplaced positions, the gap count (adjacent pancakes that "
               "differ by more than one, a published ranking for this puzzle that never overestimates the true flip count), or no ranking at "
               "all. A run that has not reached the sorted order within 12 flips fails. Every returned flip list is replayed to confirm that it "
               "sorts the start."),
    "control": {"key": "beam_width", "label": "Beam width (states kept after each flip)",
                "values": [1, 4, 16, 64], "default": 4,
                "value_labels": ["1: greedy", "4", "16", "64"]},
    "source_section": "A Heuristic Is Not Automatically a Lower Bound",
    "symbols": ("s_i is the size of the i-th pancake from the top (sizes 1 to n), s_{n+1} = n + 1 stands for the plate, and "
                "d*(s) is the optimal flip count of state s. W is the beam width. A ranking h assigns each state a number, and the beam keeps the W lowest. A ranking is a "
                "lower bound when it never exceeds the true number of flips left. The excess of a solution is its flip count minus "
                "the optimal flip count of the same start."),
    "explanation": ("The number of misplaced positions is not a lower bound for prefix flips: it overestimates the remaining flips for a "
                    "large share of states, so the search loses any optimality guarantee. Worse for a beam, which keeps states by order "
                    "alone, it barely tracks the true distance, so a narrow beam keeps the wrong states and loses the path. The gap count "
                    "counts the adjacent pairs that must still be separated, so it never overestimates, tracks the true distance closely, "
                    "and a narrow beam finds short solutions. A beam with no ranking only gets lucky. More width buys a weaker ranking more completions, "
                    "at the price of more states generated."),
    "application": ("Measure completed, replayed solutions against a trusted baseline on small instances before scaling a search. "
                    "Choose a ranking that tracks the real remaining cost, and treat an unfinished state with a good score as a failure."),
    "assumptions": ("A constructed 8-pancake puzzle, not the Santa 2024 competition; random tie-breaks make single-instance results "
                    "noisy, so 200 instances give the rates. The gap ranking is a known strong ranking for pancake flipping; "
                    "its quality here says nothing about other puzzles. Beam search is one policy among those the chapter lists; "
                    "nested Monte Carlo search is not run. A 12-flip limit counts as the budget end."),
    "prediction": "With a beam width of 4, what share of the 200 instances does the misplaced-position ranking solve?",
    "prediction_options": ["Almost all of them", "About 2 in 10", "Hardly any"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "The misplaced-position ranking solves 21% at width 4, while the gap ranking solves all of them.",
        "incorrect": "The misplaced-position ranking solves 21% at width 4, while the gap ranking solves all of them. It needs a width of 64 to solve nearly everything.",
    },
    "check": "Why can the misplaced-position ranking mislead the beam, and why does a failed run with an almost sorted state not count as a result?",
    "answer": ("Misplaced positions are not a lower bound: they overestimate the flips left in 45% of the states, so nothing guarantees "
               "short solutions. For a beam, which uses only the order, the bigger problem is that they barely track the true distance: "
               "their rank correlation with the optimal flip count over all states is 0.06, against 0.85 for the gap count. A narrow "
               "beam ordered by them keeps states that are no closer to sorted and throws away useful ones. A path that ends one or two positions from sorted "
               "has not reached the goal, so it is not a legal completed solution; the search returns failure and the replay confirms it."),
    "provenance": "Constructed example: 200 seeded random 8-pancake instances, optimal flip counts from breadth-first search, measured by the chapter activity.",
    "apply": [
        "Return a verified completed solution or an explicit failure, and replay every returned move list with the official validator.",
        "Get optimal or best-known answers on small instances by exhaustive search before trusting a heuristic on large ones.",
        "Test whether a ranking tracks the real remaining cost on small solved instances, for example by its rank correlation with the exact distance; a loosely tracking ranking loses the path unless the beam is wide.",
        "Compare policies at the same budget (states generated, width, time) and report solve rate and excess over the baseline, not only the best score reached.",
    ],
    "honesty": ("Constructed puzzle and instances; the gap ranking is unusually well matched to pancake flipping, so the gulf between "
                "the two rankings overstates what a generic ranking would give. Widening the beam closes much of it."),
}

EQUATIONS = [{"tex": r"h_{\mathrm{gap}}(s) = \#\{\,i \le n : |s_i - s_{i+1}| \neq 1\,\},\ s_{n+1} = n+1, \qquad \text{excess} = \text{flips} - d^{*}(s)",
              "alt": "the gap ranking of a state s counts the positions i up to n where the absolute difference between s i and s i plus 1 is not 1, with s n plus 1 equal to n plus 1 for the plate; the excess of a solution is its flip count minus the optimal flip count d star of s",
              "basis": "The activity's own rankings and excess measure; the chapter gives no display equation, and its text names the misplaced-position ranking as not a lower bound."}]
NCOLS = 2
HEIGHT = 4.4
POLICIES = [("misplaced", "Misplaced", COLORS["terracotta"]), ("gap", "Gap count", COLORS["teal"]), ("random", "No ranking", COLORS["light"])]


def draw(axes, result, parameter):
    left, right = axes
    xs = list(range(len(POLICIES)))
    for ax in axes:
        ax.set_xticks(xs, [name for _, name, _ in POLICIES])
        ax.set_xlim(-0.6, len(POLICIES) - 0.4)
        ax.set_xlabel(f"Ranking, beam width {parameter}")
    for x, (key, _, color) in zip(xs, POLICIES):
        p = result["policies"][key]
        left.bar(x, p["solve_rate"], width=0.55, color=color, edgecolor=COLORS["ink"], lw=0.6)
        left.text(x, p["solve_rate"] + 0.02, pct(p["solve_rate"]), ha="center", va="bottom", fontsize=10)
        if p["excess_flips"] is None:
            right.text(x, 0.1, "no\nsolutions", ha="center", va="bottom", fontsize=10, color=COLORS["grey"])
        else:
            right.bar(x, p["excess_flips"], width=0.55, color=color, edgecolor=COLORS["ink"], lw=0.6)
            few = round(p["solve_rate"] * result["instances"]) < 10     # a mean of a few runs is flagged
            right.text(x, p["excess_flips"] + 0.06, fmt(p["excess_flips"], 2) + (f"\n(n = {round(p['solve_rate'] * result['instances'])})" if few else ""),
                       ha="center", va="bottom", fontsize=10)
    left.set_ylim(0, 1.15)
    left.set_ylabel("Share of 200 instances solved")
    top = max([p["excess_flips"] for p in result["policies"].values() if p["excess_flips"] is not None] + [1.0])
    right.set_ylim(0, top * 1.4)
    right.set_ylabel("Extra flips over the optimum (solved only)")


def pct(x):
    """Percent text; a rate under 1% keeps a decimal so a handful of solved runs does not read as zero."""
    return f"{100 * x:.1f}%" if 0 < x < 0.01 or 0.99 < x < 1 else f"{round(100 * x)}%"


def diff(a, b, digits=3):
    """Difference of the two numbers as displayed, so the hand calculation on the page adds up."""
    return float(fmt(a, digits)) - float(fmt(b, digits))


def explain(result, parameter):
    m, g, n = (result["policies"][k] for k in ("misplaced", "gap", "random"))
    gap_rate = diff(g["solve_rate"], m["solve_rate"])
    m_solved = round(m["solve_rate"] * result["instances"])
    few = f" (from only {m_solved} solved run{'s' if m_solved != 1 else ''})" if m_solved < 10 else ""
    exc = (f"{fmt(m['excess_flips'], 2)} extra flips for the misplaced ranking{few} and {fmt(g['excess_flips'], 2)} for the gap count"
           if m["excess_flips"] is not None and g["excess_flips"] is not None else "too few solved runs for the misplaced ranking to compare flips")
    interpretation = (
        f"With a beam width of {parameter}, the misplaced-position ranking solves {pct(m['solve_rate'])} of {result['instances']} "
        f"instances and the gap count {pct(g['solve_rate'])}, so {fmt(g['solve_rate'])} - {fmt(m['solve_rate'])} = {fmt(gap_rate)} "
        f"of the instances separate them; with no ranking {pct(n['solve_rate'])} are solved. Solved runs use {exc}. "
        f"The misplaced count overestimates the true flips left in {round(100 * result['misplaced_overestimates'])}% of all {result['states']:,} states "
        f"and the gap count in {round(100 * result['gap_overestimates'])}%, and their rank correlations with the true flips left are "
        f"{fmt(result['misplaced_rank_corr'], 2)} and {fmt(result['gap_rank_corr'], 2)}. The gap beam generates {round(g['states_generated']):,} states per instance and "
        f"the misplaced beam {round(m['states_generated']):,}.")
    steps = [
        f"Solve rate gap minus misplaced: {fmt(g['solve_rate'])} - {fmt(m['solve_rate'])} = {signed(gap_rate)}.",
        (f"Failed misplaced runs that ended within 2 positions of sorted: {m['near_miss_failures']} of {m['failures']} (still failures)."
         if m["failures"] else "The misplaced ranking has no failed runs at this width."),
        f"Share of states where the misplaced count exceeds the true flips left: {round(100 * result['misplaced_overestimates'])}%.",
        f"Mean optimal flips over all {result['states']:,} states: {fmt(result['mean_optimal_flips'], 2)}, longest {result['max_optimal_flips']}.",
    ]
    metrics = {"Misplaced ranking, solved": f"{pct(m['solve_rate'])}", "Gap ranking, solved": f"{pct(g['solve_rate'])}",
               "No ranking, solved": f"{pct(n['solve_rate'])}", "Gap minus misplaced, solve rate": fmt(gap_rate),
               "Misplaced overestimates": f"{round(100 * result['misplaced_overestimates'])}% of states"}
    alt = (f"Left: share of {result['instances']} 8-pancake instances solved at beam width {parameter}: misplaced "
           f"{pct(m['solve_rate'])}, gap count {pct(g['solve_rate'])}, no ranking {pct(n['solve_rate'])}. "
           f"Right: extra flips over the optimum among the solved runs.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def _qualitative(results):
    """Directional claims; checked again under a different seed base."""
    for w, res in results.items():
        p = res["policies"]
        assert p["gap"]["solve_rate"] >= p["misplaced"]["solve_rate"], f"gap ranking should solve at least as many at width {w}"
        assert w == 1 or p["misplaced"]["solve_rate"] > p["random"]["solve_rate"], f"misplaced should beat no ranking at width {w}"
        assert res["gap_overestimates"] == 0.0, "gap count must never exceed the true flips left"
        assert 0.3 < res["misplaced_overestimates"] < 0.6, "misplaced overestimates in a large share of states"
    rates = [results[w]["policies"]["misplaced"]["solve_rate"] for w in (1, 4, 16, 64)]
    assert rates == sorted(rates) and rates[0] < 0.1 and rates[-1] > 0.9, "misplaced: width buys completions"
    assert results[4]["policies"]["gap"]["solve_rate"] > 0.97, "gap ranking solves nearly all at width 4"
    assert results[4]["policies"]["misplaced"]["solve_rate"] < 0.4, "misplaced fails mostly at width 4"
    assert results[64]["policies"]["random"]["solve_rate"] < 0.2, "a beam without a ranking stays far behind"
    assert results[64]["policies"]["gap"]["excess_flips"] < 0.05, "gap beam at width 64 is optimal"
    gens = [results[w]["policies"]["gap"]["states_generated"] for w in (1, 4, 16, 64)]
    assert gens == sorted(gens), "more width, more states generated"


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    _qualitative(results)
    r4 = results[4]
    assert round(100 * r4["policies"]["misplaced"]["solve_rate"]) == 21 and r4["policies"]["gap"]["solve_rate"] == 1.0, "feedback numbers"
    assert round(100 * results[64]["policies"]["misplaced"]["solve_rate"]) >= 95, "needs width 64 to solve nearly everything"
    assert round(100 * r4["misplaced_overestimates"]) == 45, "answer: 45% overestimates"
    assert fmt(r4["misplaced_rank_corr"], 2) == "0.06" and fmt(r4["gap_rank_corr"], 2) == "0.85", "answer: rank correlations"
    assert r4["max_optimal_flips"] == 9 and r4["states"] == 40320, "method text: 40,320 states, longest optimum 9"
