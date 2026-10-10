"""Chapter 25: The Continuous Learning System. A logged CV gain against the fold-seed noise floor and the noise that seeds do not touch."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from scipy.stats import rankdata

from kaggle_companion.activities._common import clean

SEED = 25
CAMPAIGNS = 32        # independent constructed competitions; each logs notes for 16 candidate features
N_ROWS, BASE_FEATURES, CANDIDATES, REAL = 1200, 10, 16, 5   # 5 candidates carry a small true effect, 11 are pure noise
EFFECT = 0.15         # true coefficient (log odds per standard deviation) of a real candidate
SEEDS = 10            # fold seeds tried for every candidate; a note with s seeds averages the first s
FRESH_ROWS = 20000    # new rows from the same generator: each candidate's true gain
RIDGE = 10.0


def auc(y, score):
    ranks = rankdata(score)
    positives = y.sum()
    return (ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * (len(y) - positives))


def fit_ridge(X, target):
    """Closed-form ridge regression on -1/+1 labels: a fast linear stand-in for a gradient-boosted baseline."""
    A = np.column_stack([np.ones(len(X)), X])
    penalty = RIDGE * np.eye(A.shape[1])
    penalty[0, 0] = 0
    return np.linalg.solve(A.T @ A + penalty, A.T @ target)


def predict(w, X):
    return w[0] + X @ w[1:]


def cv_auc(X, y, fold_seed):
    """Pooled out-of-fold AUC of 5-fold cross-validation with this fold seed."""
    folds = np.array_split(np.random.default_rng(1000 + fold_seed).permutation(len(y)), 5)
    target, score = 2.0 * y - 1, np.zeros(len(y))
    for val in folds:
        fit_rows = np.ones(len(y), bool)
        fit_rows[val] = False
        score[val] = predict(fit_ridge(X[fit_rows], target[fit_rows]), X[val])
    return auc(y, score)


def one_campaign(seed):
    """One competition: a fixed set of rows, a baseline, and 16 candidate features. Returns each candidate's per-seed gains and true gain."""
    rng = np.random.default_rng(seed)
    base_weights = rng.normal(0, 0.45, BASE_FEATURES)
    beta = np.r_[np.full(REAL, EFFECT), np.zeros(CANDIDATES - REAL)]

    def rows(n):
        X, F = rng.normal(size=(n, BASE_FEATURES)), rng.normal(size=(n, CANDIDATES))
        logit = X @ base_weights + F @ beta - 0.1
        return X, F, (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    (X, F, y), (X_new, F_new, y_new) = rows(N_ROWS), rows(FRESH_ROWS)
    base = np.array([cv_auc(X, y, s) for s in range(SEEDS)])
    base_new = auc(y_new, predict(fit_ridge(X, 2.0 * y - 1), X_new))
    out = []
    for j in range(CANDIDATES):
        augmented = np.column_stack([X, F[:, j]])
        gains = np.array([cv_auc(augmented, y, s) for s in range(SEEDS)]) - base         # paired: same folds, same rows
        true = auc(y_new, predict(fit_ridge(augmented, 2.0 * y - 1), np.column_stack([X_new, F_new[:, j]]))) - base_new
        out.append((j < REAL, gains, true))
    return out


def run(seeds_averaged):
    items = [c for i in range(CAMPAIGNS) for c in one_campaign(SEED * 1000 + i)]
    seed_sd = float(np.mean([gains.std(ddof=1) for _, gains, _ in items]))           # spread of one candidate's gain across fold seeds
    logged = np.array([gains[:seeds_averaged].mean() for _, gains, _ in items])      # the number written in the note
    real = np.array([r for r, _, _ in items])
    true = np.array([t for _, _, t in items])
    seed_se = seed_sd / np.sqrt(seeds_averaged)
    rules = {}
    for name, bar in (("any_positive", 0.0), ("seed_floor", 2 * seed_se)):
        kept = logged > bar
        rules[name] = {"bar": bar, "keeps_real": float(kept[real].mean()), "keeps_noise": float(kept[~real].mean()),
                       "precision": float(real[kept].mean()), "logged_of_kept": float(logged[kept].mean()), "true_of_kept": float(true[kept].mean())}
    return clean({
        "seeds_averaged": seeds_averaged, "candidates": len(items), "noise_candidates": int((~real).sum()), "campaigns": CAMPAIGNS,
        "seed_sd": seed_sd, "seed_se": seed_se, "noise_spread": float(logged[~real].std()),
        "true_gain_real": float(true[real].mean()), "true_gain_noise": float(true[~real].mean()),
        "logged_gain_real": float(logged[real].mean()), "rules": rules,
    }, digits=6)   # the gains are a few ten-thousandths of AUC: keep enough digits for the ratios shown
# notebook-end


SPEC = {
    "chapter": 25,
    "chapter_title": "The Continuous Learning System",
    "subtitle": "A logged gain is evidence only when it is larger than the noise that produced it.",
    "summary": ("A learning record keeps a gain such as +0.002 AUC. One demonstration measures the noise behind such a note: the fold-seed spread, "
                "the larger spread that seeds cannot remove, and what a seed-based threshold lets through."),
    "title": "A logged CV gain against the seed noise floor",
    "question": "How much of the noise behind a logged gain does averaging more fold seeds remove, and what does a seed-based threshold then keep?",
    "why": ("The chapter says a negative result is local evidence, that a retrieval note must be checked against a real decision, and that counting "
            "notes is not evidence the system helped. A note's gain has noise from the fold seed and from the finite sample; measuring both shows "
            "how far a seed check goes."),
    "method": ("Thirty-two constructed competitions of 1,200 rows with 10 baseline features and 16 candidate features: 5 carry a small true effect "
               "and 11 are pure noise. Each candidate's gain is the paired difference in 5-fold cross-validated AUC between baseline plus candidate and the "
               "baseline, with a ridge-regularized linear model as the learner, repeated over 10 fold seeds. A note averages the first s seeds (the control). "
               "Truth is each candidate's gain on 20,000 new rows. Two rules decide whether a note is kept: any positive gain, and a gain above twice the "
               "fold-seed standard error."),
    "control": {"key": "seeds_averaged", "label": "Fold seeds averaged in each logged note",
                "values": [1, 3, 10], "default": 10,
                "value_labels": ["1: a single run", "3 seeds", "10 seeds"]},
    "source_section": "Preserve Failures and Reproduction Conditions",
    "symbols": ("g_s is the paired AUC gain of one candidate under fold seed s, sd_seed the standard deviation of g_s across seeds, "
                "SE = sd_seed / sqrt(k) the standard error of a note that averages k seeds and the seed floor is 2 SE."),
    "explanation": ("Averaging k seeds divides the fold-seed spread by sqrt(k), so a threshold built from seed noise keeps shrinking. But the logged gains "
                    "of pure-noise features also vary because the sample is finite, and re-seeding the folds does not touch that part. The floor falls "
                    "faster than the real noise, so more seeds let more noise features through. They also let more real features through, but the share of "
                    "kept notes that are real falls, and at every setting the gains of kept notes overstate their true gains."),
    "application": ("Treat a logged gain as a measurement with a noise level you estimate from pure-noise candidates and fresh rows, not from re-seeding "
                    "alone; store the seeds, fold IDs and the true holdout result with the note."),
    "assumptions": ("Constructed data with a linear model and a deliberately small true effect (about +0.0016 AUC). Candidates are independent of each other. "
                    "With a larger true effect every rule keeps more real features and the noise features matter less; with correlated candidates the noise is worse. "
                    "Averaging seeds still helps: it removes the seed part of the noise and makes a note reproducible."),
    "prediction": "Moving from a single run to 10 averaged fold seeds shrinks the seed standard error about threefold. What happens to the share of pure-noise features that a gain-above-twice-the-seed-error rule keeps?",
    "prediction_options": ["It falls toward zero", "It stays about the same", "It rises"],
    "prediction_answer": 2,
    "prediction_feedback": {
        "correct": "It rises from 0.043 with one run to 0.091 with 10 seeds, because the seed floor shrinks while the sample noise does not.",
        "incorrect": "It rises from 0.043 with one run to 0.091 with 10 seeds: the seed floor shrinks about threefold, but the noise from the finite sample stays.",
    },
    "check": "A note logs +0.002 averaged over 10 seeds, about ten times the seed standard error. Why is it still weak evidence?",
    "answer": ("Averaging seeds removes only the seed noise. Pure-noise features still log gains that spread by 0.0008 across candidates, "
               "and the 10-seed floor is only 0.0004. The kept notes' logged gains average 0.0024 while their true gains average 0.0009, so a note should be "
               "confirmed on rows the seeds never touched, and recorded with its noise estimate."),
    "provenance": "Constructed example: thirty-two seeded synthetic competitions with ridge-regularized linear models, measured by the chapter activity.",
    "apply": [
        "Fix and store the fold seeds and fold IDs with each note, so the number can be regenerated, as the chapter's reproduction checks require.",
        "Estimate the noise floor from pure-noise candidates (shuffled or random columns) run through the same pipeline, not only from re-seeding the folds.",
        "Expect a kept note's gain to overstate the true gain; confirm any note you will build on with fresh rows or a second, independent campaign.",
        "Record negative and null results with the same care: a near-zero gain inside the noise is information about the noise, not a permanent ban.",
    ],
    "honesty": ("Constructed data. The effect, the number of candidates and the learner set the sizes; the direction (seeds shrink their own noise, not "
                "the sample's, and kept gains overstate) is the claim."),
}

EQUATIONS = [{"tex": r"\mathrm{SE}_k = \frac{\mathrm{sd}_{\mathrm{seed}}}{\sqrt{k}}, \qquad \text{keep a note if } \bar g_k > 2\,\mathrm{SE}_k",
              "alt": "the standard error for a note averaging k seeds is the seed standard deviation divided by the square root of k; keep a note if its average gain exceeds twice that standard error",
              "basis": "The activity's seed-floor rule; the chapter states no formula for it (Preserve Failures and Reproduction Conditions)."}]
NCOLS = 2
HEIGHT = 4.4
RULE_NAMES = [("any_positive", "Any positive\ngain"), ("seed_floor", "Gain above\n2 seed SE")]


def draw(axes, result, parameter):
    left, right = axes
    vals = [1000 * result["seed_se"], 1000 * result["noise_spread"]]
    left.bar([0, 1], vals, width=0.55, color=[COLORS["light"], COLORS["terracotta"]], edgecolor=COLORS["ink"], lw=0.6)
    for x, v in enumerate(vals):
        left.text(x, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    left.axhline(1000 * result["true_gain_real"], color=COLORS["teal"], ls=(0, (4, 3)), lw=1.6,
                 label=f"True gain of a real feature: {1000 * result['true_gain_real']:.2f}")
    left.axhline(1000 * result["rules"]["seed_floor"]["bar"], color=COLORS["gold"], ls=(0, (1, 2)), lw=1.8,
                 label=f"Seed floor (2 SE): {1000 * result['rules']['seed_floor']['bar']:.2f}")
    left.set_xticks([0, 1], ["Seed noise\n(SE of the note)", "Spread of noise\nfeatures' gains"], fontsize=10)
    left.set_ylim(0, 2.4)
    left.set_ylabel("AUC (x 0.001)")
    left.set_xlabel(f"Noise behind a note averaging {parameter} seed(s)")
    left.legend(loc="upper right", frameon=False, fontsize=10)
    for x, (key, _) in enumerate(RULE_NAMES):
        rule = result["rules"][key]
        right.bar(x - 0.2, rule["keeps_real"], width=0.4, color=COLORS["teal"], edgecolor=COLORS["ink"], lw=0.6, label="Real features kept" if x == 0 else None)
        right.bar(x + 0.2, rule["keeps_noise"], width=0.4, color=COLORS["terracotta"], edgecolor=COLORS["ink"], lw=0.6, label="Noise features kept" if x == 0 else None)
        right.text(x - 0.2, rule["keeps_real"] + 0.015, fmt(rule["keeps_real"], 2), ha="center", va="bottom", fontsize=10)
        right.text(x + 0.2, rule["keeps_noise"] + 0.015, fmt(rule["keeps_noise"], 2), ha="center", va="bottom", fontsize=10)
    right.set_xticks(range(2), [label for _, label in RULE_NAMES], fontsize=10)
    right.set_ylim(0, 1.0)
    right.set_ylabel("Share of candidates kept")
    right.set_xlabel("Rule for keeping a note")
    right.legend(loc="upper right", frameon=False, fontsize=10)


def explain(result, parameter):
    seed, floor = result["rules"]["seed_floor"], result["rules"]["any_positive"]
    se, spread = result["seed_se"], result["noise_spread"]
    interpretation = (
        f"A note averaging {parameter} seed(s) has a seed standard error of {fmt(se, 5)}, so the seed floor is 2 x {fmt(se, 5)} = {fmt(2 * round(se, 5), 5)}. "
        f"Across {result['noise_candidates']} pure-noise features the logged gains still spread by {fmt(spread, 5)}, {fmt(spread / se, 1)} times the seed error. "
        f"The seed-floor rule keeps {fmt(seed['keeps_noise'])} of noise features and {fmt(seed['keeps_real'])} of real ones; any positive gain keeps "
        f"{fmt(floor['keeps_noise'])} of noise features. Kept notes log {fmt(seed['logged_of_kept'], 5)} on average against a true {fmt(seed['true_of_kept'], 5)}: "
        f"{fmt(seed['logged_of_kept'], 5)} - {fmt(seed['true_of_kept'], 5)} = {fmt(round(seed['logged_of_kept'], 5) - round(seed['true_of_kept'], 5), 5)} of overstatement.")
    steps = [
        f"Seed standard deviation of one run: {fmt(result['seed_sd'], 5)}; with {parameter} seed(s) the standard error is {fmt(se, 5)}.",
        f"Spread of the noise features' logged gains: {fmt(spread, 5)} (seed error {fmt(se, 5)}).",
        f"Seed-floor rule: keeps {fmt(seed['keeps_real'])} of real and {fmt(seed['keeps_noise'])} of noise features; precision {fmt(seed['precision'], 2)}.",
        f"Kept notes: logged {fmt(seed['logged_of_kept'], 5)} - true {fmt(seed['true_of_kept'], 5)} = {fmt(round(seed['logged_of_kept'], 5) - round(seed['true_of_kept'], 5), 5)}.",
    ]
    metrics = {"Seed standard error of a note": fmt(se, 5), "Spread of noise features' gains": fmt(spread, 5),
               "Seed floor (2 SE)": fmt(2 * round(se, 5), 5), "Noise features kept by the seed floor": fmt(seed["keeps_noise"]),
               "Real features kept by the seed floor": fmt(seed["keeps_real"])}
    alt = (f"Left, bars of the seed standard error ({fmt(se, 5)}) and the spread of pure-noise features' gains ({fmt(spread, 5)}) for notes averaging {parameter} seed(s), "
           f"with lines for the true gain and the seed floor; right, the share of real and noise features kept by two rules.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    seeds = sorted(results)
    for s, res in results.items():
        assert res["noise_spread"] > res["seed_se"], f"noise spread should exceed the seed error at s={s}"
        assert res["true_gain_real"] > 0.001 and res["true_gain_noise"] < 0, "real features gain, noise features cost"
        assert res["rules"]["any_positive"]["keeps_noise"] > 0.1, f"any positive gain should keep many noise features at s={s}"
        seed = res["rules"]["seed_floor"]
        assert seed["logged_of_kept"] > 1.4 * seed["true_of_kept"], f"kept notes should overstate at s={s}"
        assert seed["keeps_real"] > seed["keeps_noise"] + 0.2, f"rule should still separate at s={s}"
    ses = [results[s]["seed_se"] for s in seeds]
    assert all(b < a for a, b in zip(ses, ses[1:])), "seed SE falls with s"
    assert 2.5 < results[1]["seed_se"] / results[10]["seed_se"] < 4, "about threefold (the square root of 10)"
    spreads = [results[s]["noise_spread"] for s in seeds]
    assert spreads[0] < 2 * spreads[-1] and spreads[-1] > 2 * ses[-1], "noise spread plateaus well above the seed error"
    assert results[10]["rules"]["seed_floor"]["keeps_noise"] > results[1]["rules"]["seed_floor"]["keeps_noise"], "more seeds let more noise through"
    floor = [results[s]["rules"]["seed_floor"] for s in seeds]
    assert all(b["keeps_real"] > a["keeps_real"] for a, b in zip(floor, floor[1:])), "explanation: more seeds keep more real features"
    assert all(b["precision"] < a["precision"] for a, b in zip(floor, floor[1:])), "explanation: precision of kept notes falls"
    assert fmt(results[1]["rules"]["seed_floor"]["keeps_noise"], 3) == "0.043" and fmt(results[10]["rules"]["seed_floor"]["keeps_noise"], 3) == "0.091", "prediction feedback numbers"
    top = results[10]
    assert fmt(top["noise_spread"], 4) == "0.0008" and fmt(top["rules"]["seed_floor"]["bar"], 4) == "0.0004", "check answer numbers"
    assert fmt(top["rules"]["seed_floor"]["logged_of_kept"], 4) == "0.0024" and fmt(top["rules"]["seed_floor"]["true_of_kept"], 4) == "0.0009", "check answer numbers"
    assert fmt(top["seed_se"], 4) == "0.0002", "check question: +0.002 is about ten times the seed standard error"
