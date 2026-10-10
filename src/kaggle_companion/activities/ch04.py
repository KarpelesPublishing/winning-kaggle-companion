"""Chapter 4: Tracking the CV-LB Gap. Sampling-only gaps and paired versus unpaired reading of a small true gain."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from kaggle_companion.activities._common import clean

SEED = 4
REPLICATES = 20          # independent constructed competitions (training set, local CV and large pool of "hidden" rows)
DRAWS = 300              # public samples drawn from the hidden pool in each replicate
N_TRAIN, POOL = 3000, 60000
WEIGHTS = np.array([0.9, -0.7, 0.5, 0.4, -0.3, 0.15])   # the sixth feature carries a small true signal
Z95 = 1.96


def generate(rng, n):
    X = rng.normal(size=(n, 6))
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ WEIGHTS - 0.2)))).astype(int)
    return X, y


def row_loss(y, p):
    """Log loss of each row; the leaderboard score is the mean of these."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def one_competition(public_rows, seed):
    rng = np.random.default_rng(seed)
    X, y = generate(rng, N_TRAIN)
    X_hidden, y_hidden = generate(rng, POOL)
    base, cand = slice(0, 5), slice(0, 6)          # baseline: five features; candidate: all six

    # Local CV of the unchanged baseline: 5-fold out-of-fold loss on the training rows.
    oof = np.zeros(N_TRAIN)
    for a, b in KFold(5, shuffle=True, random_state=0).split(X):
        oof[b] = LogisticRegression().fit(X[a][:, base], y[a]).predict_proba(X[b][:, base])[:, 1]
    cv_base = row_loss(y, oof).mean()

    # Per-row loss of both final models on the hidden pool. The same rows are scored by both.
    loss_b = row_loss(y_hidden, LogisticRegression().fit(X[:, base], y).predict_proba(X_hidden[:, base])[:, 1])
    loss_c = row_loss(y_hidden, LogisticRegression().fit(X[:, cand], y).predict_proba(X_hidden[:, cand])[:, 1])

    # Many public samples of `public_rows` hidden rows each.
    idx = rng.integers(0, POOL, (DRAWS, public_rows))
    B, C = loss_b[idx], loss_c[idx]
    D = B - C                                           # per-row paired difference, positive when the candidate is better
    gain = D.mean(axis=1)
    se_paired = D.std(axis=1, ddof=1) / np.sqrt(public_rows)
    se_unpaired = np.sqrt((B.var(axis=1, ddof=1) + C.var(axis=1, ddof=1)) / public_rows)   # two independent scores
    gap = B.mean(axis=1) - cv_base                      # chapter's gap: public loss minus local loss
    return {"pool_base": loss_b.mean(), "pool_cand": loss_c.mean(), "gap": gap, "gain": gain,
            "worse": (gain < 0).mean(), "paired_hit": (gain / se_paired > Z95).mean(),
            "unpaired_hit": (gain / se_unpaired > Z95).mean(),
            "se_paired": se_paired.mean(), "se_unpaired": se_unpaired.mean()}


def run(public_rows):
    runs = [one_competition(public_rows, SEED * 1000 + i) for i in range(REPLICATES)]
    mean = lambda key: float(np.mean([r[key] for r in runs]))
    pool_base, pool_cand = round(mean("pool_base"), 4), round(mean("pool_cand"), 4)
    return clean({
        "public_rows": public_rows,
        "pool_base": pool_base, "pool_cand": pool_cand, "true_gain": round(pool_base - pool_cand, 4),
        # spread of the gap of an UNCHANGED model: public sampling within a competition, and the development side across them
        "gap_public_sd": float(np.mean([r["gap"].std() for r in runs])),
        "gap_dev_sd": float(np.std([r["gap"].mean() for r in runs])),
        "looks_worse": mean("worse"), "paired_detects": mean("paired_hit"), "unpaired_detects": mean("unpaired_hit"),
        "se_paired": mean("se_paired"), "se_unpaired": mean("se_unpaired"),
        "public_gains": np.concatenate([r["gain"][:25] for r in runs]).tolist(),
        "replicates": REPLICATES, "draws": DRAWS, "training_rows": N_TRAIN, "hidden_rows": POOL,
    })
# notebook-end


SPEC = {
    "chapter": 4,
    "chapter_title": "Tracking the CV-LB Gap",
    "subtitle": "A gap is a discrepancy to investigate; judge a change with paired differences, not with two raw scores.",
    "summary": ("A public score is a sample. One demonstration measures how far an unchanged model's gap wanders from sampling alone, "
                "how often a truly better candidate looks worse on the public rows, and how paired and unpaired reading of the same rows differ."),
    "title": "Sampling-only gaps and a small true gain, by public sample size",
    "question": ("With no shift and no leak, how far does the CV-LB gap move from sampling alone, and can a public sample of a given "
                 "size reveal a candidate that is truly better by a small amount?"),
    "why": ("A competitor sees a gap move and a public score go up or down after a change. Before blaming the feature, you need to "
            "know how much movement sampling alone produces and which comparison can separate a real gain from it."),
    "method": ("A constructed binary task with six features, 3,000 training rows and a hidden pool of 60,000 rows from the same "
               "generator, so there is no shift. The baseline logistic model uses five features; the candidate adds a sixth that "
               "carries a small real signal. The gap is public log loss minus local 5-fold log loss. Public samples of m rows are "
               "drawn from the pool, 300 per competition, over 20 independent competitions. The candidate's gain is judged two "
               "ways: paired (the standard error of the per-row loss difference) and unpaired (treating the two public scores as "
               "independent, each with its own standard error). Detection means the gain exceeds 1.96 standard errors."),
    "control": {"key": "public_rows", "label": "Rows in the public sample",
                "values": [250, 1000, 4000, 16000], "default": 1000,
                "value_labels": ["250 rows", "1,000 rows", "4,000 rows", "16,000 rows"]},
    "source_section": "The Three Gap Patterns",
    "symbols": ("gap = public loss - local loss (the chapter's definition for a loss). d_i = l_i(baseline) - l_i(candidate) is the "
                "per-row paired loss difference, m the public rows, and SE = sd(d_i) / sqrt(m) its standard error; a positive d "
                "means the candidate is better."),
    "explanation": ("An unchanged model has no gap pattern to find, yet its gap moves by sampling alone, by a lot at small m. The "
                    "candidate's real gain is much smaller than that movement, so a raw reading (did the public score go up or "
                    "down?) fails often. The two models make nearly the same errors on each row, so their per-row difference has a "
                    "far smaller spread than either score. Pairing uses that; reading the two scores as independent does not."),
    "application": ("Do not read a gap change of the size shown here as a diagnosis. Compare candidate and baseline on the same rows "
                    "with a paired difference and its standard error, and ask for more rows or blocks before deciding."),
    "assumptions": ("Constructed data with no shift, no leak and no selection, one logistic model pair, 20 competitions of 300 public "
                    "draws each. Real gaps add population shift and selection on top of the sampling spread measured here. The "
                    "unpaired rule is a deliberately naive reading of two public numbers, shown to contrast with the paired one. "
                    "The local-side spread comes from a 3,000 row development set and is measured across competitions."),
    "prediction": ("With 1,000 public rows, how often does a candidate that is truly better by about 0.002 log loss look worse "
                   "than the baseline on the public sample?"),
    "prediction_options": ["Almost never", "About 1 time in 6", "About half the time"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "It looks worse in 16% of public samples, while an unchanged model's gap wanders by a standard deviation of 0.014.",
        "incorrect": ("It looks worse in 16% of public samples at 1,000 rows (32% at 250 rows). An unchanged model's gap alone "
                      "wanders by a standard deviation of 0.014, about seven times the real gain."),
    },
    "check": "Why can the paired comparison detect a real gain that the two raw public scores cannot, using the same public rows?",
    "answer": ("Both models make nearly the same error on each row, so the per-row difference between them varies far less than "
               "either score does: the paired standard error at 1,000 rows is 0.002 against 0.020 when the two scores are treated as "
               "independent. A paired reading found the gain in 19% of public samples at 1,000 rows and 94% at 16,000; the "
               "unpaired reading found it in almost none. Even paired, 1,000 rows is too few to settle a 0.002 gain."),
    "provenance": "Constructed example: seeded synthetic rows and logistic models, measured by the chapter activity. The numbers describe this generator only.",
    "apply": [
        "Expect a gap to move by sampling alone: estimate that spread from the size of the public sample before reading a change as shift or leakage.",
        "Compare baseline and candidate row by row on the same rows, and report the paired difference with its standard error.",
        "If the paired interval includes zero, the evidence cannot separate the candidate from the baseline; collect more rows or blocks, or stop.",
        "Never reject or accept a feature because the public score moved by less than the sampling spread; log the paired result instead.",
    ],
    "honesty": ("Constructed data with no shift, leak or selection; a real gap also contains those. The unpaired rule is a naive "
                "baseline for contrast, not a recommended practice."),
}

EQUATIONS = [{"tex": r"\text{gap} = \text{public loss} - \text{local loss}",
              "alt": "gap equals public loss minus local loss",
              "basis": "The chapter's gap definition for a loss (The Experiment Log in Practice)."},
             {"tex": r"d_i = \ell_i(\text{baseline}) - \ell_i(\text{candidate}), \qquad \mathrm{SE} = \frac{\mathrm{sd}(d_i)}{\sqrt{m}}",
              "alt": "d i is the baseline row loss minus the candidate row loss; the standard error is the standard deviation of d i over the square root of m",
              "basis": "The activity's paired difference for the chapter's paired block differences (The Three Gap Patterns); not a display equation."}]
NCOLS = 2
HEIGHT = 4.4


def draw(axes, result, parameter):
    left, right = axes
    gains = np.array(result["public_gains"])
    edges = np.linspace(min(gains.min(), -0.01), max(gains.max(), 0.012), 31)
    counts, bins, patches = left.hist(gains, bins=edges, color=COLORS["teal"], edgecolor="white", lw=0.5)
    for left_edge, patch in zip(bins[:-1], patches):
        if left_edge + (bins[1] - bins[0]) / 2 < 0:
            patch.set_facecolor(COLORS["terracotta"])
    left.axvline(0, color=COLORS["ink"], lw=0.9)
    left.axvline(result["true_gain"], color=COLORS["gold"], ls=(0, (4, 3)), lw=1.6,
                 label=f"True gain: {fmt(result['true_gain'])}")
    left.set_xlabel("Public improvement of the candidate (loss)")
    left.set_ylabel("Public samples (500)")
    left.legend(loc="upper left", frameon=False, fontsize=10)

    names = ["Looks worse\non public", "Unpaired finds\nthe gain", "Paired finds\nthe gain"]
    vals = [result["looks_worse"], result["unpaired_detects"], result["paired_detects"]]
    colors = [COLORS["terracotta"], COLORS["light"], COLORS["teal"]]
    right.bar(range(3), [100 * v for v in vals], width=0.55, color=colors, edgecolor=COLORS["ink"], lw=0.6)
    for x, v in enumerate(vals):
        right.text(x, 100 * v + 2, f"{round(100 * v)}%", ha="center", va="bottom", fontsize=10)
    right.set_xticks(range(3), names)
    right.set_ylim(0, 112)
    right.set_ylabel("Share of public samples (%)")
    right.set_xlabel(f"Reading of the same {parameter:,} public rows")


def explain(result, parameter):
    sd, dev, g = result["gap_public_sd"], result["gap_dev_sd"], result["true_gain"]
    interpretation = (
        f"With {parameter:,} public rows an unchanged model's gap wanders by a standard deviation of {fmt(sd)} from public "
        f"sampling alone, and by {fmt(dev)} from the development side, against a true gain of {fmt(result['pool_base'], 4)} - "
        f"{fmt(result['pool_cand'], 4)} = {fmt(g, 4)}. The truly better candidate looks worse on {round(100 * result['looks_worse'])}% "
        f"of public samples. Read as two independent scores the gain is found in {round(100 * result['unpaired_detects'])}% of "
        f"samples (standard error {fmt(result['se_unpaired'])}); read as a paired difference it is found in "
        f"{round(100 * result['paired_detects'])}% (standard error {fmt(result['se_paired'])}).")
    steps = [
        f"True gain: {fmt(result['pool_base'], 4)} - {fmt(result['pool_cand'], 4)} = {fmt(g, 4)} (baseline minus candidate loss on 60,000 hidden rows).",
        f"Sampling-only gap spread, public side: {fmt(sd)}; development side: {fmt(dev)}.",
        f"Share of public samples where the better candidate looks worse: {round(100 * result['looks_worse'])}%.",
        f"Share detected at 1.96 standard errors: paired {round(100 * result['paired_detects'])}%, unpaired {round(100 * result['unpaired_detects'])}%.",
    ]
    metrics = {"Gap spread (public sampling)": fmt(sd), "Gap spread (development side)": fmt(dev), "True gain": fmt(g, 4),
               "Better candidate looks worse": f"{round(100 * result['looks_worse'])}%",
               "Paired detects": f"{round(100 * result['paired_detects'])}%",
               "Unpaired detects": f"{round(100 * result['unpaired_detects'])}%"}
    alt = (f"Left: histogram of the candidate's public improvement over 500 public samples of {parameter:,} rows, with a dashed line "
           f"at the true gain {fmt(g, 4)}. Right: bars for how often the candidate looks worse ({round(100 * result['looks_worse'])}%) "
           f"and how often the unpaired and paired readings find the gain.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    sizes = sorted(results)
    for a, b in zip(sizes, sizes[1:]):
        assert results[a]["gap_public_sd"] > results[b]["gap_public_sd"], "gap spread should shrink with public rows"
        assert results[a]["looks_worse"] > results[b]["looks_worse"], "looks-worse share should shrink with public rows"
        assert results[a]["paired_detects"] < results[b]["paired_detects"], "paired detection should grow with public rows"
    for m, res in results.items():
        assert res["unpaired_detects"] < 0.02, f"unpaired reading should almost never detect at {m}"
        assert res["se_unpaired"] > 4 * res["se_paired"], f"pairing should shrink the standard error at least fourfold at {m}"
        assert 0.001 < res["true_gain"] < 0.003, "true gain is about 0.002"
    mid = results[1000]
    assert round(100 * mid["looks_worse"]) == 16 and round(100 * results[250]["looks_worse"]) == 32, "prediction feedback"
    assert fmt(mid["gap_public_sd"]) == "0.014", "prediction feedback gap spread"
    assert 5 < mid["gap_public_sd"] / mid["true_gain"] < 9, "text says about seven times the real gain"
    assert fmt(mid["se_paired"]) == "0.002" and fmt(mid["se_unpaired"]) == "0.020", "check answer standard errors"
    assert round(100 * mid["paired_detects"]) == 19 and round(100 * results[16000]["paired_detects"]) == 94, "check answer rates"
    assert mid["paired_detects"] < 0.5, "1,000 rows is too few to settle the gain"
