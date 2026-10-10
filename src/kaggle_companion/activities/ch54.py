"""Chapter 54: Combinatorial Optimization. Simulated annealing on a tour under a fixed move budget: starting temperature from sampled worsening moves, random against nearest-neighbour start."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import math
import random

import numpy as np

from kaggle_companion.activities._common import clean

SEED = 54
CITIES, BUDGET, INSTANCES = 150, 60000, 8   # every run gets the same 60,000 evaluated 2-opt moves
END_TEMPERATURE = 0.01                      # final temperature, as a fraction of the mean edge of the nearest-neighbour tour


def make_instance(seed):
    """Random Euclidean cities as a distance table (nested lists: fastest for scalar lookups)."""
    points = np.random.default_rng(seed).random((CITIES, 2))
    return np.sqrt(((points[:, None] - points[None]) ** 2).sum(-1)).tolist()


def tour_length(D, tour):
    return sum(D[tour[i]][tour[(i + 1) % CITIES]] for i in range(CITIES))


def nearest_neighbour_tour(D):
    """Cheap legal initial tour: always walk to the closest unvisited city."""
    left, tour = set(range(1, CITIES)), [0]
    while left:
        nxt = min(left, key=lambda j: D[tour[-1]][j])
        tour.append(nxt)
        left.remove(nxt)
    return tour


def two_opt_delta(D, tour, i, j):
    """Change in length if tour[i..j] is reversed (swap two edges)."""
    a, b, c, d = tour[i - 1], tour[i], tour[j], tour[(j + 1) % CITIES]
    return D[a][c] + D[b][d] - D[a][b] - D[c][d]


def sample_worsening(D, tour, rng, count=300):
    """Worsening 2-opt deltas sampled from the current tour."""
    out = []
    while len(out) < count:
        i, j = sorted(rng.sample(range(1, CITIES), 2))
        delta = two_opt_delta(D, tour, i, j)
        if delta > 0:
            out.append(delta)
    return out


def acceptance(deltas, temperature):
    """Mean probability that a sampled worsening move is accepted: mean of exp(-delta / T)."""
    return sum(math.exp(-d / temperature) for d in deltas) / len(deltas)


def solve_temperature(deltas, target):
    """The chapter's recipe: the acceptance is monotone in T, so bisect for the T that gives the target."""
    lo, hi = 1e-9, 1e3
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if acceptance(deltas, mid) < target else (lo, mid)
    return hi


def anneal(D, tour, target, rng, end_temperature):
    """Geometric cooling from the solved T0 to end_temperature over exactly BUDGET evaluated moves."""
    tour = list(tour)
    length = tour_length(D, tour)
    t0 = solve_temperature(sample_worsening(D, tour, rng), target)
    temperature, cool = t0, (end_temperature / t0) ** (1 / BUDGET) if t0 > end_temperature else 1.0
    for _ in range(BUDGET):
        i, j = sorted(rng.sample(range(1, CITIES), 2))
        delta = two_opt_delta(D, tour, i, j)
        if delta < 0 or rng.random() < math.exp(-delta / temperature):
            tour[i:j + 1] = tour[i:j + 1][::-1]
            length += delta
        temperature *= cool
    assert sorted(tour) == list(range(CITIES)), "invalid tour"          # report only valid completed tours
    assert abs(length - tour_length(D, tour)) < 1e-6, "tracked length drifted"
    return length, t0


def run(target_acceptance):
    ratio = {"random": [], "nn": []}
    t0_scaled = {"random": [], "nn": []}
    naive_acceptance = []
    for k in range(INSTANCES):
        D = make_instance(SEED * 100 + k)
        rng = random.Random(SEED * 100 + k)
        nn = nearest_neighbour_tour(D)
        base = tour_length(D, nn)                                    # reference: the nearest-neighbour tour, no search
        edge = base / CITIES
        shuffled = list(range(CITIES))
        rng.shuffle(shuffled)
        for name, start in (("random", shuffled), ("nn", nn)):
            length, t0 = anneal(D, start, target_acceptance, rng, END_TEMPERATURE * edge)
            ratio[name].append(length / base)
            t0_scaled[name].append(t0 / edge)
        # the shortcut the chapter warns about: set T from the mean worsening delta so that exp(-mean/T) equals the target
        deltas = sample_worsening(D, nn, rng)
        naive_t = -(sum(deltas) / len(deltas)) / math.log(target_acceptance)
        naive_acceptance.append(acceptance(deltas, naive_t))
    return clean({
        "target_acceptance": target_acceptance,
        "ratio": {k: float(np.mean(v)) for k, v in ratio.items()},
        "per_instance": ratio,
        "t0_over_edge": {k: float(np.mean(v)) for k, v in t0_scaled.items()},
        "naive_acceptance": float(np.mean(naive_acceptance)),
        "cities": CITIES, "budget": BUDGET, "instances": INSTANCES,
    })
# notebook-end


SPEC = {
    "chapter": 54,
    "chapter_title": "Combinatorial Optimization",
    "subtitle": "Calibrate the starting temperature from sampled moves, and compare schedules and starts under one budget.",
    "summary": ("A fixed tour is scored directly, so the work is search. One demonstration anneals 150-city tours under a fixed "
                "budget of evaluated moves and measures how the starting temperature, solved from a target acceptance, and the "
                "starting tour change the final length."),
    "title": "Simulated annealing under a fixed budget: starting temperature and starting tour",
    "question": "With a fixed number of evaluated moves, does a hotter starting temperature or a better starting tour give the shorter final tour?",
    "why": ("A temperature that is too hot spends the budget accepting moves that undo progress, and one that is too cold "
            "may stop improving early. Choosing it from sampled moves, then comparing starts under the same move count, is "
            "how to tell which wastes the budget."),
    "method": ("Eight constructed instances of 150 random cities. Simulated annealing uses 2-opt moves (reverse a segment) and "
               "geometric cooling over exactly 60,000 evaluated moves. The starting temperature is solved by bisection so that the mean "
               "probability of accepting a sampled worsening move, mean(exp(-delta/T)), equals the control; the final "
               "temperature is 1% of the mean edge length of the nearest-neighbour tour. Each instance is annealed from a random "
               "tour and from the nearest-neighbour tour. The score is the final (current) tour length divided by the "
               "length of the nearest-neighbour tour, and every returned tour is checked to be a valid permutation with a recomputed length."),
    "control": {"key": "target_acceptance", "label": "Target starting acceptance of a worsening move",
                "values": [0.01, 0.1, 0.5, 0.9], "default": 0.5,
                "value_labels": ["0.01: nearly greedy", "0.1", "0.5", "0.9: very hot"]},
    "source_section": "Simulated Annealing: The Most Robust Starting Point",
    "symbols": ("delta is the increase in tour length of a worsening move, T the temperature and a the mean acceptance "
                "probability of sampled worsening moves. The score divides the final length by the nearest-neighbour tour length, "
                "so values below 1 mean the search improved on the constructive tour."),
    "explanation": ("Annealing accepts a worsening move with probability exp(-delta/T). From a nearest-neighbour tour, which is "
                    "already decent, a hot start throws away that quality and the fixed budget is too short to rebuild it, so "
                    "colder is better. From a random tour the search must build structure itself: the two low settings (0.01 and 0.1) end "
                    "within noise of each other, and every hotter start wastes more of the budget. Setting T from the mean worsening delta, "
                    "rather than solving for the acceptance, misses the target, most of all at cold settings."),
    "application": ("Sample worsening moves from the starting tour, solve the temperature for a chosen acceptance, then compare a few "
                    "acceptances and both starts under the same evaluation budget before spending more compute on one."),
    "assumptions": ("Constructed random Euclidean instances of 150 cities, 2-opt moves only and one cooling shape (geometric to a fixed "
                    "final temperature); a longer budget, Or-opt moves or a different final temperature would move every number. "
                    "Lengths are relative to the nearest-neighbour tour, a named reproducible baseline, since no proved optimum is "
                    "computed here. Only the coldest start is close to plain greedy 2-opt descent. The nearest-neighbour construction "
                    "itself (about 11,000 distance lookups, roughly the cost of 2,800 move evaluations) is not charged to the budget."),
    "prediction": "Starting simulated annealing from the nearest-neighbour tour, which target acceptance gives the shortest final tour under 60,000 moves?",
    "prediction_options": ["0.01, almost greedy", "0.5, a moderate temperature", "0.9, very hot"],
    "prediction_answer": 0,
    "prediction_feedback": {
        "correct": "The nearly greedy start (0.01) ends at 0.856 of the nearest-neighbour length, against 0.925 at 0.5 and 0.940 at 0.9.",
        "incorrect": "The nearly greedy start (0.01) ends at 0.856 of the nearest-neighbour length, against 0.925 at 0.5 and 0.940 at 0.9. A hot start throws away the quality the nearest-neighbour tour already has.",
    },
    "check": "What does solving the temperature for a target acceptance buy, and why does the nearest-neighbour start still need a lower target?",
    "answer": ("A temperature is in units of tour length, so a copied constant means different things on different instances; "
               "an acceptance is scale-free. It is not neutral across starts, though: random moves sampled from a good tour break "
               "short edges and are large, so the same acceptance of 0.5 solves to 14.8 mean edge lengths from the "
               "nearest-neighbour tour against 4.2 from a random one. The nearest-neighbour start therefore needs a much lower "
               "target acceptance to keep the quality it starts with."),
    "provenance": "Constructed example: eight seeded random 150-city instances, annealed with 2-opt moves under a fixed budget, measured by the chapter activity.",
    "apply": [
        "Sample worsening moves from the starting solution and solve for the temperature that gives a chosen acceptance, instead of copying a constant.",
        "Compare a few starting temperatures and both a cheap constructive start and a random start under the same number of evaluated moves.",
        "From a good constructive start, begin cool: a hot start discards the quality you paid for.",
        "Recompute the objective of the returned solution with the official scorer and keep the legal starting solution as the fallback.",
    ],
    "honesty": ("Constructed instances and a single budget: with a much longer budget the random start would catch up. The difference "
                "between a random start's low acceptance settings is within noise here, so no sharp optimum is claimed for it."),
}

EQUATIONS = [{"tex": r"\Pr(\text{accept}) = \min\!\bigl(1, e^{-\Delta/T}\bigr), \qquad a(T) = \frac{1}{m}\sum_{s=1}^{m} e^{-\Delta_s/T}",
              "alt": "the probability of accepting a move is the minimum of 1 and e to the power of minus delta over T; the mean acceptance a of T is the average over m sampled worsening deltas of e to the power of minus delta s over T",
              "basis": "The acceptance rule and the sampled mean acceptance the chapter tells the reader to solve for T (Simulated Annealing section); not a display equation in the manuscript."}]
NCOLS = 2
HEIGHT = 4.4
STARTS = [("random", "Random tour", COLORS["light"]), ("nn", "Nearest-neighbour tour", COLORS["teal"])]


def draw(axes, result, parameter):
    left, right = axes
    xs = list(range(len(STARTS)))
    for x, (key, _, color) in zip(xs, STARTS):
        left.hlines(result["ratio"][key], x - 0.3, x + 0.3, color=color if key != "random" else COLORS["navy"], lw=4, zorder=2)
        dots = result["per_instance"][key]
        left.scatter([x + (i - (len(dots) - 1) / 2) * 0.05 for i in range(len(dots))], dots, s=12, color=COLORS["ink"], zorder=3,
                     label="One instance" if x == 0 else None)
        left.text(x + 0.33, result["ratio"][key], fmt(result["ratio"][key]), ha="left", va="center", fontsize=10)
    left.axhline(1.0, color=COLORS["gold"], ls=(0, (4, 3)), lw=1.6, label="Nearest-neighbour tour: 1")
    left.set_xticks(xs, [name for _, name, _ in STARTS])
    left.set_xlim(-0.6, 1.9)
    left.set_ylim(0.7, max(max(max(v) for v in result["per_instance"].values()), 1.0) + 0.12)
    left.set_ylabel("Final length / nearest-neighbour length")
    left.set_xlabel(f"Starting tour, target acceptance {fmt(parameter, 2)}")
    left.legend(loc="upper left", frameon=False, fontsize=10, ncol=2)

    target = result["target_acceptance"]
    bars = [("Target", target, COLORS["gold"]), ("Naive T from\nmean delta", result["naive_acceptance"], COLORS["terracotta"])]
    for x, (name, value, color) in enumerate(bars):
        right.bar(x, value, width=0.55, color=color, edgecolor=COLORS["ink"], lw=0.6)
        right.text(x, value + 0.015, fmt(value), ha="center", va="bottom", fontsize=10)
    right.set_xticks(range(len(bars)), [name for name, _, _ in bars])
    right.set_ylim(0, 1.15)
    right.set_ylabel("Mean acceptance of sampled worsening moves")
    right.set_xlabel("Setting the starting temperature")


def diff(a, b, digits=3):
    """Difference of the two numbers as displayed, so the hand calculation on the page adds up."""
    return float(fmt(a, digits)) - float(fmt(b, digits))


def explain(result, parameter):
    rr, nn = result["ratio"]["random"], result["ratio"]["nn"]
    naive = result["naive_acceptance"]
    better = "nearest-neighbour" if nn < rr else "random"
    gain = diff(1.0, min(rr, nn))
    interpretation = (
        f"With a target acceptance of {fmt(parameter, 2)} and 60,000 moves, annealing from a random tour ends at {fmt(rr)} of the nearest-neighbour "
        f"length and from the nearest-neighbour tour at {fmt(nn)}, so the {better} start is shorter and 1 - {fmt(min(rr, nn))} = {fmt(gain)} "
        f"is the saving over no search. The temperature solved for this acceptance is {fmt(result['t0_over_edge']['random'], 2)} mean edge lengths "
        f"for the random tour and {fmt(result['t0_over_edge']['nn'], 2)} for the nearest-neighbour tour. Setting T from the mean "
        f"worsening delta gives an acceptance of {fmt(naive)} instead of {fmt(parameter, 2)}.")
    steps = [
        f"Random start against nearest-neighbour start: {fmt(rr)} - {fmt(nn)} = {signed(diff(rr, nn))} (negative means random is shorter).",
        f"Saving of the better start over no search: 1 - {fmt(min(rr, nn))} = {fmt(gain)}.",
        f"Naive temperature: acceptance {fmt(naive)} - target {fmt(parameter, 2)} = {signed(float(fmt(naive)) - parameter)}.",
        f"Solved temperature for the random and the nearest-neighbour start: {fmt(result['t0_over_edge']['random'], 2)} and {fmt(result['t0_over_edge']['nn'], 2)} mean edges.",
    ]
    metrics = {"Random start, final / NN length": fmt(rr), "Nearest-neighbour start, final / NN length": fmt(nn),
               "Saving of the better start": fmt(gain), "Naive-T acceptance": fmt(naive),
               "Solved T0, random start": f"{fmt(result['t0_over_edge']['random'], 2)} edges"}
    alt = (f"Left: final tour length relative to the nearest-neighbour tour at a target acceptance of {fmt(parameter, 2)}, "
           f"{fmt(rr)} from a random start and {fmt(nn)} from the nearest-neighbour start. Right: the target acceptance "
           f"{fmt(parameter, 2)} against the acceptance {fmt(naive)} that setting T from the mean worsening delta gives.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def _qualitative(results):
    """Directional claims; checked again under a different seed base."""
    cold, hot = results[0.01], results[0.9]
    ordered = [results[a]["ratio"]["nn"] for a in (0.01, 0.1, 0.5, 0.9)]
    assert ordered == sorted(ordered) and ordered[-1] - ordered[0] > 0.05, "nearest-neighbour start: hotter is worse"
    assert cold["ratio"]["nn"] < 1.0, "cold annealing improves on the constructive tour"
    assert hot["ratio"]["random"] - results[0.1]["ratio"]["random"] > 0.02, "random start: very hot wastes the budget"
    assert abs(cold["ratio"]["random"] - results[0.1]["ratio"]["random"]) < 0.03, "random start flat at low acceptance"
    assert results[0.9]["ratio"]["random"] > results[0.5]["ratio"]["random"] > max(cold["ratio"]["random"], results[0.1]["ratio"]["random"]), \
        "random start: every hotter start wastes more of the budget"
    best = min(r["ratio"][s] for r in results.values() for s in ("random", "nn"))
    assert best == cold["ratio"]["nn"], "cold nearest-neighbour start is the best cell"
    for a in (0.01, 0.1):
        assert results[a]["naive_acceptance"] - a > 0.05, f"naive temperature should overshoot a cold target at {a}"
    assert abs(results[0.9]["naive_acceptance"] - 0.9) < 0.02, "naive temperature is close only at hot settings"
    for a, res in results.items():
        assert res["t0_over_edge"]["nn"] > 2 * res["t0_over_edge"]["random"], f"solved T0 differs a lot between starts at {a}"


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    _qualitative(results)
    nn = {a: results[a]["ratio"]["nn"] for a in results}
    assert fmt(nn[0.01]) == "0.856" and fmt(nn[0.5]) == "0.925" and fmt(nn[0.9]) == "0.940", "prediction feedback numbers"
    half = results[0.5]["t0_over_edge"]
    assert fmt(half["random"], 1) == "4.2" and fmt(half["nn"], 1) == "14.8", "answer numbers"
