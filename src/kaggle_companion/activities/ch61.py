"""Chapter 61: Playground Series Strategy. A public-leaderboard winner's curse among near-equal candidates."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np

from kaggle_companion.activities._common import clean

SEED = 61
DATASETS = 12                        # independent constructed competitions; every estimate is their mean
SPLITS = 300                         # random public/private splits of the test rows per competition
F, N_TRAIN, N_TEST, N_FRESH = 30, 1500, 2000, 10000
PUBLIC_SHARE = 0.2                   # 400 public rows, 1,600 private rows
NOISE = 3.0


def make_data(rng, beta, n):
    X = rng.normal(size=(n, F))
    return X, X @ beta + rng.normal(0, NOISE, n)


def ridge(X, y, alpha):
    """Closed-form ridge without an intercept (the constructed target has mean zero)."""
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)


def rmse(y, pred):
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def one_competition(n_candidates, seed):
    """Candidates are near-duplicate ridge recipes: each drops two features, trains on a random 75% of the rows and
    picks its own penalty. Returns each candidate's squared errors on the test rows, its 5-fold CV RMSE on the
    training rows and its RMSE on 10,000 fresh rows (the 'truth' that nobody gets to see in a real competition)."""
    rng = np.random.default_rng(seed)
    beta = rng.normal(0, 0.35, F)
    Xtr, ytr = make_data(rng, beta, N_TRAIN)
    Xte, yte = make_data(rng, beta, N_TEST)
    Xfr, yfr = make_data(rng, beta, N_FRESH)
    folds = rng.permutation(N_TRAIN) % 5
    err_test, cv, truth = [], [], []
    for _ in range(n_candidates):
        cols = np.sort(rng.choice(F, F - 2, replace=False))
        alpha = 10 ** rng.uniform(-1, 2)
        rows = rng.random(N_TRAIN) < 0.75
        w = ridge(Xtr[rows][:, cols], ytr[rows], alpha)
        err_test.append((yte - Xte[:, cols] @ w) ** 2)
        truth.append(rmse(yfr, Xfr[:, cols] @ w))
        oof = np.zeros(N_TRAIN)
        for k in range(5):
            fit = rows & (folds != k)
            oof[folds == k] = Xtr[folds == k][:, cols] @ ridge(Xtr[fit][:, cols], ytr[fit], alpha)
        cv.append(rmse(ytr, oof))
    return np.array(err_test), np.array(cv), np.array(truth), rng


def run(n_candidates):
    n_pub = int(PUBLIC_SHARE * N_TEST)
    per = {k: [] for k in ("public_shown", "private_earned", "truth_earned", "regret_public", "regret_cv", "regret_random",
                           "leader_stays", "leader_is_best")}
    for d in range(DATASETS):
        err, cv, truth, rng = one_competition(n_candidates, SEED * 100 + d)
        mask = np.zeros((SPLITS, N_TEST))
        for s in range(SPLITS):
            mask[s, rng.permutation(N_TEST)[:n_pub]] = 1
        sp = np.sqrt(mask @ err.T / n_pub)                              # public RMSE of every candidate, per split
        sv = np.sqrt((1 - mask) @ err.T / (N_TEST - n_pub))             # private RMSE of every candidate, per split
        pick = sp.argmin(1)                                             # the public leader in each split
        rows = np.arange(SPLITS)
        per["public_shown"].append(float(np.mean(np.median(sp, 1) - sp[rows, pick])))
        per["private_earned"].append(float(np.mean(np.median(sv, 1) - sv[rows, pick])))
        per["truth_earned"].append(float(np.mean(np.median(truth) - truth[pick])))
        per["regret_public"].append(float(np.mean(truth[pick] - truth.min())))
        per["regret_cv"].append(float(truth[cv.argmin()] - truth.min()))
        per["regret_random"].append(float(truth.mean() - truth.min()))
        per["leader_stays"].append(float(np.mean(sv.argmin(1) == pick)))
        per["leader_is_best"].append(float(np.mean(truth.argmin() == pick)))
    mean = {k: float(np.mean(v)) for k, v in per.items()}
    return clean({"candidates": n_candidates, **mean, "per_dataset": per, "datasets": DATASETS, "splits": SPLITS,
                  "public_rows": n_pub, "private_rows": N_TEST - n_pub, "train_rows": N_TRAIN})
# notebook-end


SPEC = {
    "chapter": 61,
    "chapter_title": "Playground Series Strategy",
    "subtitle": "Small leaderboard differences are noise plus selection: keep a supported candidate, not the public leader.",
    "summary": ("The public leaderboard is a small sample, and picking its leader among many near-equal candidates selects luck. "
                "One demonstration measures how much of the leader's visible advantage survives on private rows, and how a local "
                "cross-validation pick compares, as the number of screened candidates grows."),
    "title": "The public leader's advantage, shown and earned, by number of candidates screened",
    "question": "How much of the public leader's visible advantage is still there on private rows as more near-equal candidates are screened?",
    "why": ("A dense Playground leaderboard rewards many small experiments, and every experiment is one more draw against the same "
            "400 public rows. The chapter asks for paired local evidence and a preserved supported candidate rather than the public leader."),
    "method": ("Twelve constructed competitions. Each has 1,500 training rows, 2,000 test rows (400 public, 1,600 private) and a "
               "regression target with noise standard deviation 3. Candidates are near-duplicate ridge recipes: each drops two of 30 features, trains on a "
               "random 75% of the rows and picks its own penalty, and the number of candidates screened is the control. For every "
               "competition, 300 random public/private splits are drawn. The public leader is picked in each, and its RMSE is compared "
               "with the median candidate on public rows, on private rows and on 10,000 fresh rows. A second rule picks the "
               "candidate with the best 5-fold cross-validation RMSE on the training rows, and a third picks at random."),
    "control": {"key": "candidates", "label": "Candidates screened against the public leaderboard",
                "values": [3, 10, 30, 100], "default": 30,
                "value_labels": ["3", "10", "30", "100"]},
    "source_section": "Reading the Leaderboard: Signal vs. Noise",
    "symbols": ("s_c is candidate c's RMSE (lower is better), the superscripts pub and priv mark public and private rows, "
                "median over c is taken across the screened candidates and pick is the candidate with the lowest public RMSE."),
    "explanation": ("Each candidate's public RMSE is its true quality plus sampling noise from only 400 rows. Taking the minimum over more "
                    "candidates picks more noise: the advantage the leaderboard shows keeps growing with the candidate count, while "
                    "the advantage that exists on private rows or fresh rows changes far less (up or down by about 0.01 across seeds), so the noise in the visible lead keeps growing. Cross-validation uses 1,500 rows, so "
                    "it is usually a quieter judge here, but it is also one more selection, so its pick loses ground too."),
    "application": ("Treat a small public lead as weak evidence, judge each change on paired local folds, and fill the final submission "
                    "slots with candidates that have supported local evidence rather than only the public leader."),
    "assumptions": ("Constructed data with ridge variants standing in for the many near-equal models of a Playground competition, and one fixed "
                    "public share (20%). Cross-validation had less regret than the public pick in every state here, by 0.002 to 0.01, but under another seed the two were tied (cross-validation very slightly worse) in "
                    "one state; it is a property of the training set (1,500 rows) being much larger "
                    "than the public set (400 rows). With a relatively larger public set or a leaking cross-validation the order could "
                    "change. Costs here are RMSE units on a target with noise standard deviation 3."),
    "prediction": ("With 100 candidates screened, the public leader looks 0.062 RMSE better than the median candidate on the public rows. "
                   "How much of that advantage shows up on private rows?"),
    "prediction_options": ["Nearly all of it", "About two thirds of it", "About one third of it"],
    "prediction_answer": 2,
    "prediction_feedback": {
        "correct": "The leader earns 0.021 on private rows against 0.062 shown: about one third, so about two thirds of the visible lead was selected noise.",
        "incorrect": "The leader earns only 0.021 on private rows against 0.062 shown, about one third. The other two thirds was selected noise.",
    },
    "check": "If the public lead overstates so much, why not ignore the public score and pick at random?",
    "answer": ("The public score still carries signal: with 100 candidates, picking the public leader is 0.019 RMSE worse than the best candidate on "
               "fresh rows, against 0.048 for a random pick. The leader is the best candidate only 13% of the time, though, and the "
               "leader on the public rows is also the leader on private rows only 8% of the time. Use the public score as one weak vote, "
               "not as the decision."),
    "provenance": "Constructed example: seeded synthetic regression data and 3 to 100 ridge candidates per competition, measured by the chapter activity.",
    "apply": [
        "Record which candidates were chosen using public feedback, and count how many were screened against it.",
        "Rank candidates on paired local folds that use far more rows than the public sample; treat a small public lead as a tie-breaker.",
        "Use the final submission slots for candidates with different supported assumptions, not only the top public scores.",
        "Preserve the best locally supported candidate and its artifacts before any late move made for a public gain.",
    ],
    "honesty": ("Constructed data. The visible lead, the earned lead and the regrets are properties of this generator and of the 20% public "
                "share; they illustrate selection noise, not a claim about any actual competition."),
}

EQUATIONS = [{"tex": r"\text{lead}^{\mathrm{pub}} = \operatorname{median}_c\, s^{\mathrm{pub}}_c - s^{\mathrm{pub}}_{\mathrm{pick}}",
              "alt": "The public lead is the median over candidates of the public RMSE minus the public RMSE of the pick",
              "basis": "The activity's own measure of the lead the leaderboard shows; the same difference on private or fresh rows is the lead earned. Not a display equation in the manuscript."}]
NCOLS = 2
HEIGHT = 4.4


def sub(a, b):
    """Difference of two values as the page shows them, so a hand calculation always adds up."""
    return float(fmt(a)) - float(fmt(b))


def draw(axes, result, parameter):
    left, right = axes
    per = result["per_dataset"]
    groups = [("public_shown", "Shown on\npublic rows", COLORS["terracotta"]), ("private_earned", "Earned on\nprivate rows", COLORS["navy"]),
              ("truth_earned", "Earned on\nfresh rows", COLORS["teal"])]
    top = max(max(per[k]) for k, _, _ in groups)
    for x, (key, name, color) in enumerate(groups):
        left.bar([x], [result[key]], width=0.6, color=color, edgecolor=COLORS["ink"], lw=0.6)
        dots = per[key]
        left.scatter([x + (i - (len(dots) - 1) / 2) * 0.04 for i in range(len(dots))], dots, s=10, color=COLORS["ink"], zorder=3)
        left.text(x, max(dots) + top * 0.02, fmt(result[key]), ha="center", va="bottom", fontsize=10)
    left.set_xticks(range(3), [g[1] for g in groups])
    left.axhline(0, color=COLORS["grey"], lw=0.8)
    left.set_ylim(min(0, min(min(per[k]) for k, _, _ in groups) - top * 0.03), top * 1.15)
    left.set_ylabel("Lead over median candidate (RMSE)")
    left.set_xlabel(f"Public leader of {parameter} candidates (dots: competitions)")

    rules = [("regret_public", "Public\nleader", COLORS["terracotta"]), ("regret_cv", "Best 5-fold\nCV score", COLORS["teal"]),
             ("regret_random", "Random\ncandidate", COLORS["light"])]
    top = max(max(per[k]) for k, _, _ in rules)
    for x, (key, name, color) in enumerate(rules):
        right.bar([x], [result[key]], width=0.6, color=color, edgecolor=COLORS["ink"], lw=0.6)
        dots = per[key]
        right.scatter([x + (i - (len(dots) - 1) / 2) * 0.04 for i in range(len(dots))], dots, s=10, color=COLORS["ink"], zorder=3)
        right.text(x, max(dots) + top * 0.02, fmt(result[key]), ha="center", va="bottom", fontsize=10)
    right.set_xticks(range(3), [r[1] for r in rules])
    right.set_ylim(0, top * 1.15)
    right.set_ylabel("Regret on fresh rows (RMSE)")
    right.set_xlabel("How the candidate is chosen (regret: RMSE above the best)")


def explain(result, parameter):
    shown, earned, truth = result["public_shown"], result["private_earned"], result["truth_earned"]
    curse = sub(shown, earned)
    kept = earned / shown
    interpretation = (
        f"With {parameter} candidates the public leader looks {fmt(shown)} RMSE better than the median candidate on the public rows, "
        f"but earns {fmt(earned)} on private rows and {fmt(truth)} on fresh rows: {fmt(shown)} - {fmt(earned)} = {fmt(curse)} of the visible "
        f"lead was selection noise, and {round(100 * kept)}% of it is real. The public leader is also the private leader in "
        f"{round(100 * result['leader_stays'])}% of splits. Against the best candidate, picking by public score costs {fmt(result['regret_public'])} "
        f"RMSE on fresh rows, picking by cross-validation {fmt(result['regret_cv'])} and picking at random {fmt(result['regret_random'])}.")
    steps = [
        f"Noise in the visible lead: {fmt(shown)} - {fmt(earned)} = {fmt(curse)} (shown minus earned on private rows).",
        f"Share of the visible lead that is real: {fmt(earned)} / {fmt(shown)} = {fmt(kept)}.",
        f"Cross-validation against the public leader: {fmt(result['regret_public'])} - {fmt(result['regret_cv'])} = {fmt(sub(result['regret_public'], result['regret_cv']))} less regret.",
        f"Public leader against a random pick: {fmt(result['regret_random'])} - {fmt(result['regret_public'])} = {fmt(sub(result['regret_random'], result['regret_public']))} less regret.",
    ]
    metrics = {"Lead shown on public rows": fmt(shown), "Lead earned on private rows": fmt(earned),
               "Lead earned on fresh rows": fmt(truth), "Public leader also private leader": f"{round(100 * result['leader_stays'])}% of splits",
               "Regret: public / CV / random": f"{fmt(result['regret_public'])} / {fmt(result['regret_cv'])} / {fmt(result['regret_random'])}"}
    alt = (f"Left: bars of the public leader's RMSE lead over the median candidate with {parameter} candidates: {fmt(shown)} shown on public rows, "
           f"{fmt(earned)} earned on private rows and {fmt(truth)} on fresh rows. Right: regret on fresh rows for picking by public score "
           f"({fmt(result['regret_public'])}), by cross-validation ({fmt(result['regret_cv'])}) and at random ({fmt(result['regret_random'])}).")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    keys = sorted(results)
    shown = [results[k]["public_shown"] for k in keys]
    assert all(a < b for a, b in zip(shown, shown[1:])), f"visible lead should grow with candidates: {shown}"
    for k in keys:
        res = results[k]
        assert res["public_shown"] > res["private_earned"], f"visible lead should exceed earned lead at {k}"
        assert res["regret_public"] < res["regret_random"] * 0.6, f"public pick should beat random at {k}"
        assert res["regret_cv"] < res["regret_public"], f"CV pick should have less regret than public pick at {k}"
        assert res["regret_cv"] < res["regret_random"] * 0.6, f"CV pick should beat random at {k}"
    reg = [results[k]["regret_public"] for k in keys]
    assert all(a < b for a, b in zip(reg, reg[1:])), "public-pick regret should grow with candidates"
    assert max(results[k]["private_earned"] for k in keys[1:]) - min(results[k]["private_earned"] for k in keys[1:]) < 0.015, "earned lead about flat"
    noise = [results[k]["public_shown"] - results[k]["private_earned"] for k in keys]
    assert all(a < b for a, b in zip(noise, noise[1:])), f"explanation: noise in the visible lead keeps growing: {noise}"
    big = results[100]
    assert fmt(big["public_shown"]) == "0.062" and fmt(big["private_earned"]) == "0.021", "prediction feedback numbers"
    assert 0.55 < 1 - big["private_earned"] / big["public_shown"] < 0.8, "roughly two thirds of the lead is noise"
    assert fmt(big["regret_public"]) == "0.019" and fmt(big["regret_random"]) == "0.048", "check answer regrets"
    assert round(100 * big["leader_is_best"]) == 13 and round(100 * big["leader_stays"]) == 8, "check answer percentages"
