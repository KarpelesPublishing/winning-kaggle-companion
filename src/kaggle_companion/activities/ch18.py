"""Chapter 18: Final Submission Selection. Public-only, local-only and pooled selection among close candidates, by public share."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from kaggle_companion.activities._common import clean

SEED = 18
WORLDS = 5          # independent training sets, each with its own twelve fitted candidates
DRAWS = 300         # random hidden splits per training set
N_TRAIN, N_POOL = 2500, 12000
N_TEST, N_LOCAL = 2000, 1000   # hidden test rows (public + private) and a separate local validation set
# Twelve gradient-boosting variants: learning rate x leaf count x L2 penalty. All are reasonable, so they sit close together.
CONFIGS = [(lr, leaves, l2) for lr in (0.04, 0.08) for leaves in (4, 6, 10) for l2 in (0.0, 8.0)]
RULES = ("public", "local", "pooled")


def candidate_losses(seed):
    """Fit every candidate once and return each pool row's log loss under each candidate (rows x candidates)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.45, 8)

    def rows(n):
        X = rng.normal(size=(n, 8))
        logit = X @ w + 0.6 * X[:, 0] * X[:, 1] + 0.5 * np.maximum(X[:, 2], 0) - 0.2
        return X, (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    X_train, y_train = rows(N_TRAIN)
    X_pool, y_pool = rows(N_POOL)       # a large pool: its mean loss is the "true" loss of each candidate
    columns = []
    for lr, leaves, l2 in CONFIGS:
        model = HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=leaves, l2_regularization=l2,
                                               max_iter=50, early_stopping=False, random_state=0)
        columns.append(np.clip(model.fit(X_train, y_train).predict_proba(X_pool)[:, 1], 1e-3, 1 - 1e-3))
    p = np.column_stack(columns)
    return -(y_pool[:, None] * np.log(p) + (1 - y_pool[:, None]) * np.log(1 - p))


def run(public_fraction):
    n_public = int(public_fraction * N_TEST)
    stats = {rule: {"regret": [], "best": [], "rank": [], "optimism": []} for rule in RULES}
    gaps = []
    for w in range(WORLDS):
        loss = candidate_losses(SEED * 100 + w)
        truth = loss.mean(0)                                   # true log loss of each candidate
        best = int(np.argmin(truth))
        gaps.append(np.sort(truth)[1] - truth[best])
        rng = np.random.default_rng(SEED * 1000 + w)
        for _ in range(DRAWS):
            order = rng.permutation(len(loss))
            test, local = order[:N_TEST], order[N_TEST:N_TEST + N_LOCAL]
            public, private = test[:n_public], test[n_public:]
            private_loss = loss[private].mean(0)               # what the final standings will use
            scores = {"public": loss[public].mean(0), "local": loss[local].mean(0),
                      "pooled": (loss[public].sum(0) + loss[local].sum(0)) / (n_public + N_LOCAL)}
            for rule, score in scores.items():
                pick = int(np.argmin(score))                   # the candidate each evidence source would submit
                stats[rule]["regret"].append(truth[pick] - truth[best])
                stats[rule]["best"].append(pick == best)
                stats[rule]["rank"].append(1 + int((private_loss < private_loss[pick]).sum()))
                stats[rule]["optimism"].append(truth[pick] - score[pick])   # how much better the evidence said it was
    return clean({
        "public_fraction": public_fraction, "public_rows": n_public, "private_rows": N_TEST - n_public,
        "local_rows": N_LOCAL, "candidates": len(CONFIGS), "worlds": WORLDS, "draws": WORLDS * DRAWS,
        "gap_to_second": float(np.mean(gaps)),
        "rules": {rule: {"regret": float(np.mean(s["regret"])), "p_best": float(np.mean(s["best"])),
                         "private_rank": float(np.mean(s["rank"])), "optimism": float(np.mean(s["optimism"])),
                         "regret_se": float(np.std(s["regret"]) / np.sqrt(len(s["regret"])))}
                  for rule, s in stats.items()},
    }, digits=6)   # regrets are a few ten-thousandths of log loss: keep enough digits to tell the rules apart
# notebook-end


SPEC = {
    "chapter": 18,
    "chapter_title": "Final Submission Selection",
    "subtitle": "Choose a recoverable champion from credible evidence, and know how much a small public split can tell you.",
    "summary": ("Among close candidates, a small public leaderboard is noisy evidence. One demonstration measures how often "
                "public-only, local-only and pooled evidence each pick the truly best candidate as the public share changes."),
    "title": "Choosing among twelve close candidates: public score, local score or both",
    "question": "How much public evidence does it take before the public leaderboard picks as well as a local validation set?",
    "why": ("Late in a competition the candidates are close, and the public leaderboard is the loudest evidence available. "
            "Measuring how often it picks the best candidate tells you when to lean on it and when to lean on local, paired evidence."),
    "method": ("Five constructed binary tasks. For each, twelve gradient-boosting variants (learning rate, leaf count, L2 penalty) are "
               "fitted on 2,500 rows and scored by log loss on a pool of 12,000 further rows; the pool mean is each candidate's true "
               "loss. In each of 300 random splits per task, 2,000 pool rows form the hidden test set, divided into public rows "
               "(the control sets their share) and private rows, and 1,000 other rows form a local validation set. Three rules "
               "submit the candidate with the lowest loss on the public rows, on the local rows, or on both pooled. Regret is the true "
               "loss of the submitted candidate minus the true loss of the best candidate."),
    "control": {"key": "public_fraction", "label": "Share of the 2,000 hidden test rows that are public",
                "values": [0.05, 0.1, 0.25, 0.5], "default": 0.1,
                "value_labels": ["5%: 100 public rows", "10%: 200 public rows", "25%: 500 public rows",
                                 "50%: 1,000 public rows"]},
    "source_section": "Simulating Public/Private Split Stability",
    "symbols": ("S is the set of rows an evidence source uses, loss_c,i the log loss of candidate c on row i, s_c the mean of loss_c,i "
                "over S, c* the submitted candidate and L_c the true loss of candidate c on the large pool."),
    "explanation": ("Every rule scores candidates on the same rows, so each comparison is paired, but a small set of rows still "
                    "leaves differences of a few thousandths of log loss within noise. The public rule is the most exposed, because the "
                    "candidate that wins a noisy contest is partly the luckiest one, and its public score flatters it. More rows, "
                    "from any source, shrink both effects. Pooling the public and local rows uses all the evidence and picks at least "
                    "as well as either alone."),
    "application": ("Keep a local validation set that no candidate was fitted or tuned on, compare candidates on identical rows, "
                    "and treat the public leaderboard as one more block of rows whose weight grows with its size."),
    "assumptions": ("Constructed data and one model family. The local rows are assumed to be fresh rows from the same population as the "
                    "hidden test, which real local validation only approximates; a local score that guided tuning would be more optimistic "
                    "than here. The public rows are read once, at selection; a public score that has already guided many submissions is no longer "
                    "fresh evidence, which is why the chapter asks you to record that influence. This is a local stress test, not a reproduction of any hidden split. The crossover near 50% is where the "
                    "public set has as many rows as the local set (1,000), so it moves with the local set's size."),
    "prediction": "With 10% of the hidden rows public (200 rows), how often does the best public score belong to the truly best of the twelve candidates?",
    "prediction_options": ["Almost always", "A little over half the time", "Rarely"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "It picks the best candidate in 0.594 of splits, against 0.796 for the local set's 1,000 rows.",
        "incorrect": "It picks the best candidate in 0.594 of splits, against 0.796 for the local set's 1,000 rows: 200 rows are too few to separate close candidates.",
    },
    "check": "Why does the pooled rule beat both single sources, and why does the public rule only match the local one near a 50% public share?",
    "answer": ("Pooling uses every row once, so it has the most evidence at every share. The public and local rules each use only their "
               "own rows, and they tie when the public set (1,000 rows at 50%) is as large as the local set. Even the best rule does not "
               "pick the true best candidate every time, so the private ranking of the pick still moves."),
    "provenance": "Constructed example: five seeded synthetic tasks with twelve fitted gradient-boosting variants, measured by the chapter activity.",
    "apply": [
        "Freeze a champion from paired, local evidence before reading the public leaderboard, and write the evidence down.",
        "Treat the public leaderboard as a block of rows: weight it by its size, and pool it with local evidence rather than replacing it.",
        "When candidates differ by less than the noise of the rows that scored them, prefer the simpler or the more recoverable one.",
        "Do not read the winner's public score as its expected private score: the selected candidate's score is flattered by the selection.",
    ],
    "honesty": ("Constructed data. The sizes of the effects depend on this generator, on twelve candidates and on a 1,000-row local set; "
                "the direction (more rows pick better, pooling beats either source) is the claim, not the numbers."),
}

EQUATIONS = [{"tex": r"s_c = \frac{1}{|S|}\sum_{i \in S} \ell_{c,i}, \qquad c^{*} = \arg\min_c s_c, \qquad \text{regret} = L_{c^{*}} - \min_c L_c",
              "alt": "s c is the mean of the per-row loss of candidate c over the evidence rows S; the submitted candidate c star minimizes s c; regret is the true loss of c star minus the smallest true loss",
              "basis": "The activity's own selection rule and regret, a local stress test of the evidence discussed in Simulating Public/Private Split Stability."}]
NCOLS = 2
HEIGHT = 4.4
LABELS = {"public": "Public rows only", "local": "Local rows only", "pooled": "Public + local"}
RULE_COLORS = {"public": COLORS["terracotta"], "local": COLORS["light"], "pooled": COLORS["teal"]}


def draw(axes, result, parameter):
    left, right = axes
    xs = range(len(RULES))
    for x, rule in zip(xs, RULES):
        left.bar(x, result["rules"][rule]["regret"] * 1000, width=0.6, color=RULE_COLORS[rule], edgecolor=COLORS["ink"], lw=0.6)
        left.text(x, result["rules"][rule]["regret"] * 1000 + 0.05, f"{result['rules'][rule]['regret'] * 1000:.2f}\nbest in {fmt(result['rules'][rule]['p_best'], 2)}",
                  ha="center", va="bottom", fontsize=10)
        right.bar(x, result["rules"][rule]["optimism"] * 1000, width=0.6, color=RULE_COLORS[rule], edgecolor=COLORS["ink"], lw=0.6)
        right.text(x, result["rules"][rule]["optimism"] * 1000 + 0.1, f"{result['rules'][rule]['optimism'] * 1000:.2f}", ha="center", va="bottom", fontsize=10)
    left.set_ylim(0, max(result["rules"][r]["regret"] for r in RULES) * 1000 * 1.45 + 0.3)
    left.set_xticks(list(xs), [LABELS[r] for r in RULES], fontsize=10)
    left.set_ylabel("Regret (true loss above best, x 0.001)")
    left.set_xlabel(f"Evidence used, {result['public_rows']} public rows")
    right.set_ylim(0, max(result["rules"][r]["optimism"] for r in RULES) * 1000 * 1.25 + 0.3)
    right.set_xticks(list(xs), [LABELS[r] for r in RULES], fontsize=10)
    right.set_ylabel("Optimism of the winner's score (x 0.001)")
    right.set_xlabel("Evidence used")


def explain(result, parameter):
    p, l, c = (result["rules"][k] for k in RULES)
    steps = [
        f"Public rows only: best candidate picked in {fmt(p['p_best'])} of splits, regret {fmt(p['regret'], 5)}.",
        f"Local rows only: best candidate picked in {fmt(l['p_best'])} of splits, regret {fmt(l['regret'], 5)}.",
        f"Pooled rows: best candidate picked in {fmt(c['p_best'])} of splits, regret {fmt(c['regret'], 5)}.",
        f"Public minus pooled regret: {fmt(p['regret'], 5)} - {fmt(c['regret'], 5)} = {fmt(round(p['regret'], 5) - round(c['regret'], 5), 5)}.",
    ]
    more = "public" if p["regret"] < l["regret"] - 0.0001 else "local" if l["regret"] < p["regret"] - 0.0001 else None
    verdict = (f"Public rows now pick better than the {result['local_rows']} local rows." if more == "public" else
               f"Local rows pick better than the {result['public_rows']} public rows." if more == "local" else
               "Public and local evidence pick about equally well.")
    interpretation = (
        f"With {result['public_rows']} public rows out of {result['public_rows'] + result['private_rows']}, the public rule submits the "
        f"best of {result['candidates']} candidates in {fmt(p['p_best'])} of splits and the {result['local_rows']}-row local rule in {fmt(l['p_best'])}. "
        f"Mean regret is {fmt(p['regret'], 5)} against {fmt(l['regret'], 5)}, so {fmt(p['regret'], 5)} - {fmt(l['regret'], 5)} = "
        f"{fmt(round(p['regret'], 5) - round(l['regret'], 5), 5)}. {verdict} Pooling gives {fmt(c['regret'], 5)}. The public winner's score is "
        f"{fmt(p['optimism'], 5)} better than its true loss, so a winning public score overstates it.")
    metrics = {"Public rule picks the best": fmt(p["p_best"]), "Local rule picks the best": fmt(l["p_best"]),
               "Pooled rule picks the best": fmt(c["p_best"]), "Public rule regret": fmt(p["regret"], 5),
               "Public winner's optimism": fmt(p["optimism"], 5)}
    alt = (f"Two panels with three bars each for public-only, local-only and pooled selection at a {result['public_rows']}-row public set; "
           f"left shows regret, right shows the optimism of the selected candidate's score. The public rule has regret {fmt(p['regret'], 5)}.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for share, res in results.items():
        r = res["rules"]
        assert r["pooled"]["regret"] <= min(r["public"]["regret"], r["local"]["regret"]) + 0.0001, f"pooled should not lose at {share}"
        assert r["pooled"]["p_best"] >= max(r["public"]["p_best"], r["local"]["p_best"]) - 0.01, f"pooled p_best at {share}"
        assert r["public"]["optimism"] > 0 and r["public"]["optimism"] >= r["local"]["optimism"] - 0.0005, f"optimism at {share}"
        assert res["gap_to_second"] < 0.005, "candidates should be close"
    # More public rows pick better; the public rule is much worse than the local rule at small shares.
    shares = sorted(results)
    for a, b in zip(shares, shares[1:]):
        assert results[b]["rules"]["public"]["regret"] < results[a]["rules"]["public"]["regret"], "public regret falls with share"
    assert results[0.05]["rules"]["public"]["regret"] > 3 * results[0.05]["rules"]["local"]["regret"], "public much worse at 5%"
    assert abs(results[0.5]["rules"]["public"]["regret"] - results[0.5]["rules"]["local"]["regret"]) < 0.0003, "tie near 50%"
    low = results[0.1]["rules"]
    assert fmt(low["public"]["p_best"]) == "0.594" and fmt(low["local"]["p_best"]) == "0.796", "prediction feedback numbers"
    assert 0.5 < low["public"]["p_best"] < 0.7, "prediction option says a little over half"
    assert all(res["rules"]["pooled"]["p_best"] < 0.99 for res in results.values()), "even pooled does not always pick the best"
