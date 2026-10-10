"""Chapter 2: The Process Mindset. Choosing among logged versions by a public sample, against private and fresh rows."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.linear_model import LogisticRegression

from kaggle_companion.activities._common import clean

SEED = 2
REPLICATES = 16                      # independent constructed competitions; every estimate is their mean
SPLITS = 10                          # random public/private splits of each competition's test rows
N_TRAIN, N_TEST, N_FRESH = 800, 2000, 20000
PUBLIC_SHARE = 0.20                  # public leaderboard rows: 400 of the 2,000 test rows
N_BASE, N_CANDIDATES, N_USEFUL, USEFUL_WEIGHT = 6, 100, 2, 0.25
LOGGED = [3, 10, 30, 100]            # numbers of logged versions the control can take


def generate(n, rng, w_base, w_cand):
    """Six base columns and 100 candidate columns. Only two candidates truly help, and only a little."""
    base = rng.normal(size=(n, N_BASE))
    cand = rng.normal(size=(n, N_CANDIDATES))
    p = 1 / (1 + np.exp(-(base @ w_base + cand @ w_cand)))
    return np.column_stack([base, cand]), (rng.random(n) < p).astype(int)


def row_loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def one_competition(seed):
    """Version 0 is the baseline. Version k adds candidate column k and refits. Returns every version's
    per-row test loss and its mean loss on a large fresh sample (the truth)."""
    rng = np.random.default_rng(seed)
    w_base = rng.normal(0, 0.6, N_BASE)
    w_cand = np.zeros(N_CANDIDATES)
    w_cand[rng.choice(N_CANDIDATES, N_USEFUL, replace=False)] = USEFUL_WEIGHT
    X, y = generate(N_TRAIN, rng, w_base, w_cand)
    X_test, y_test = generate(N_TEST, rng, w_base, w_cand)
    X_fresh, y_fresh = generate(N_FRESH, rng, w_base, w_cand)
    test_loss = np.zeros((max(LOGGED), N_TEST))
    fresh_loss = np.zeros(max(LOGGED))
    for k in range(max(LOGGED)):
        cols = list(range(N_BASE)) + ([] if k == 0 else [N_BASE + k - 1])
        model = LogisticRegression(max_iter=200).fit(X[:, cols], y)
        test_loss[k] = row_loss(model.predict_proba(X_test[:, cols])[:, 1], y_test)
        fresh_loss[k] = row_loss(model.predict_proba(X_fresh[:, cols])[:, 1], y_fresh).mean()
    # Each split draws a public sample; the winner is the version with the lowest public loss among the first N.
    gains = {n: [] for n in LOGGED}
    for _ in range(SPLITS):
        public = np.zeros(N_TEST, bool)
        public[rng.choice(N_TEST, int(PUBLIC_SHARE * N_TEST), replace=False)] = True
        pub_loss, priv_loss = test_loss[:, public].mean(1), test_loss[:, ~public].mean(1)
        for n in LOGGED:
            k = int(np.argmin(pub_loss[:n]))
            # gain = baseline loss minus the winner's loss, so positive means the winner looks better
            gains[n].append([pub_loss[0] - pub_loss[k], priv_loss[0] - priv_loss[k], fresh_loss[0] - fresh_loss[k]])
    return {n: np.mean(g, axis=0) for n, g in gains.items()}, {n: np.mean(np.array(g)[:, 2] > 0) for n, g in gains.items()}


def run(versions):
    studies = [one_competition(SEED * 1000 + i) for i in range(REPLICATES)]
    curve = {}
    for n in LOGGED:
        g = np.array([s[0][n] for s in studies])           # replicates x (public, private, fresh)
        curve[n] = {"public": g[:, 0].mean(), "private": g[:, 1].mean(), "fresh": g[:, 2].mean(),
                    "public_sd": g[:, 0].std(), "private_sd": g[:, 1].std(),
                    "truly_better": float(np.mean([s[1][n] for s in studies])),
                    "per_study_public": g[:, 0], "per_study_private": g[:, 1]}
    return clean({"versions": versions, "selected": curve[versions],
                  "curve": [{"versions": n, **curve[n]} for n in LOGGED],
                  "replicates": REPLICATES, "splits": SPLITS, "public_rows": int(PUBLIC_SHARE * N_TEST),
                  "private_rows": N_TEST - int(PUBLIC_SHARE * N_TEST), "fresh_rows": N_FRESH})
# notebook-end


SPEC = {
    "chapter": 2,
    "chapter_title": "The Process Mindset",
    "subtitle": "Preserve the comparison, and treat repeatedly inspected feedback as development evidence.",
    "summary": ("Every logged version that is chosen by the public score uses that score for development. One demonstration measures how much "
                "of the public gain of the winning version survives on private rows and on fresh rows, as the log grows."),
    "title": "Picking the best of N logged versions by the public sample",
    "question": "How much of the winning version's public gain is still there on rows the choice never used, as the number of logged versions grows?",
    "why": ("A log of many versions invites the question of which one to keep. If the public score picks it, the winner's public gain "
            "contains a share of luck that grows with the number of versions, and the log has created no new independent labels."),
    "method": ("Sixteen constructed competitions. Each has 800 training rows, 2,000 test rows (400 public and 1,600 private, drawn at random "
               "ten times) and 20,000 fresh rows as the truth. A baseline logistic regression uses 6 columns; version k adds one of 100 "
               "candidate columns and refits, and only 2 of the 100 candidates truly help, by a small amount. The control is how many versions are "
               "logged (the baseline plus candidates). The winner is the version with the lowest public log loss. Gain is the baseline's loss "
               "minus the winner's loss, so a positive gain looks like improvement."),
    "control": {"key": "versions", "label": "Versions logged (including the baseline)",
                "values": LOGGED, "default": 30,
                "value_labels": ["3", "10", "30", "100"]},
    "source_section": "Preserve the Comparison",
    "symbols": ("g_public, g_private and g_fresh are the baseline's log loss minus the winning version's log loss on public rows, on the "
                "remaining private rows and on 20,000 fresh rows; the winner is chosen on the public rows only."),
    "explanation": ("Each extra version is another noisy comparison against the baseline on the same 400 public rows. The best of many such "
                    "comparisons is high even when no version helps, so the public gain keeps rising with N. The private rows played no "
                    "part in the choice, so they show what is left, and the fresh rows show the truth the private rows estimate. Move the "
                    "control to find where the public and private gains separate."),
    "application": ("Log every version, but decide with data the choice has not touched. When the log is long, discount the public gain "
                    "of the best version, and keep a private comparison or a reserved assessment that did not take part in picking it."),
    "assumptions": ("Constructed competitions with a logistic regression, one added column per version, and 2 truly useful candidates among "
                    "100. With every candidate useless, the private gain is negative at every N (checked by the build with the "
                    "two useful candidates removed). The gain sizes are properties of this generator."),
    "prediction": ("Thirty versions are logged and the best one on the 400 public rows is kept. How does its private gain compare with its public gain?"),
    "prediction_options": ["About the same", "A fraction of it: the private gain is under 0.002",
                           "Larger, because the private set is four times bigger"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "The public gain is 0.0045 and the private gain is 0.0008, with 0.0010 on the fresh rows.",
        "incorrect": "The public gain is 0.0045, but the private gain is only 0.0008 and the fresh rows say 0.0010. The extra private rows reduce noise; they cannot restore the luck in the pick.",
    },
    "check": "At 10 logged versions the winner improved the public loss by 0.0030 yet gained 0.0001 on private rows. What did the public gain measure?",
    "answer": ("Mostly the best of ten noisy comparisons: the winner's public gain rose from 0.0007 at 3 versions to 0.0083 at 100, "
               "while the private gain stayed near zero until the log was long enough to contain a truly useful column (0.0032 at 100 "
               "against 0.0036 on fresh rows). The public gain overstates by 0.0051 at 100 versions."),
    "provenance": "Constructed example: sixteen seeded synthetic competitions and logistic regressions, measured by the chapter activity.",
    "apply": [
        "Record which feedback chose each version; once the public score picks a version it is development evidence.",
        "Do not read a public gain from the best of many versions as the gain you will see on hidden rows; expect it to shrink.",
        "Keep a comparison that played no part in the pick (a reserved block or a private score) and compare the winner with the baseline there.",
        "Adding runs adds no independent labels: before logging more versions, ask what new information each one carries.",
    ],
    "honesty": ("Constructed data. Whether the pick helps on hidden rows depends on how many versions truly help; here 2 of 100 do, by "
                "design, and the sizes are properties of this generator."),
}

EQUATIONS = [{"tex": r"g = L_{\text{baseline}} - L_{\text{winner}}, \qquad \text{winner} = \arg\min_{k \le N} L^{\text{public}}_k",
              "alt": "gain g equals the baseline loss minus the winner's loss, where the winner is the version with the lowest public loss among N",
              "basis": "The activity's own gain and selection rule; not a display equation in the manuscript."}]
NCOLS = 1
HEIGHT = 4.4


def draw(ax, result, parameter):
    curve = result["curve"]
    xs = [c["versions"] for c in curve]
    for key, sd, color, label, marker in (("public", "public_sd", COLORS["terracotta"], "Public gain (used to pick)", "o"),
                                          ("private", "private_sd", COLORS["teal"], "Private gain (not used)", "s")):
        mid = [c[key] for c in curve]
        ax.fill_between(xs, [m - c[sd] for m, c in zip(mid, curve)], [m + c[sd] for m, c in zip(mid, curve)], color=color, alpha=0.14, lw=0)
        ax.plot(xs, mid, color=color, lw=1.8, marker=marker, ms=5, label=label)
    ax.plot(xs, [c["fresh"] for c in curve], color=COLORS["gold"], lw=1.6, ls=(0, (4, 3)), marker="D", ms=4,
            label="Fresh 20,000 rows (truth)")
    ax.axhline(0, color=COLORS["grey"], lw=0.7)
    ax.axvline(parameter, color=COLORS["ink"], lw=1.0, ls=":")
    sel = result["selected"]
    ax.scatter([parameter] * 2, [sel["public"], sel["private"]], s=70, facecolor="none", edgecolor=COLORS["ink"], lw=1.4, zorder=5)
    ax.set_xscale("log")
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.minorticks_off()
    ax.set_xlabel("Versions logged (log scale; band is 1 SD across competitions)")
    ax.set_ylabel("Gain over baseline (log-loss decrease)")
    ax.set_ylim(-0.0045, 0.0145)
    ax.legend(loc="upper left", frameon=False, fontsize=10)


def sub(value, digits=4):
    """A subtrahend for display: negatives in parentheses, so a step never reads 'a - -b'."""
    text = fmt(value, digits)
    return f"({text})" if text.startswith("-") else text


def explain(result, parameter):
    s, n = result["selected"], parameter
    over = s["public"] - s["private"]
    if s["private"] < 0.002 and over > 0.002:
        reading = "Most of the public gain is the luck of picking the best of many comparisons."
    elif over > 0.002:
        reading = "Part of the public gain survives on private rows; the rest is the luck of the pick."
    elif s["fresh"] <= 0:
        reading = ("The overstatement is small only because the public gain is small: the pick has no real gain on "
                   "fresh rows, so even this public gain is luck.")
    else:
        reading = "At this log length the public gain is a fair guide to the private gain."
    interpretation = (
        f"With {n} versions logged, the winner's public gain is {fmt(s['public'], 4)} and its private gain is {fmt(s['private'], 4)}, "
        f"so {fmt(s['public'], 4)} - {sub(s['private'])} = {fmt(over, 4)} of the public gain is not there on private rows. "
        f"On {result['fresh_rows']:,} fresh rows the winner gains {signed(s['fresh'], 4)}. {reading} The winner truly beats the baseline in "
        f"{fmt(100 * s['truly_better'], 0)}% of the picks.")
    steps = [
        f"Overstatement: {fmt(s['public'], 4)} - {sub(s['private'])} = {signed(over, 4)} (public gain minus private gain).",
        f"Private gain against the truth: {fmt(s['private'], 4)} - {sub(s['fresh'])} = {signed(s['private'] - s['fresh'], 4)}.",
        f"Public gain against the truth: {fmt(s['public'], 4)} - {sub(s['fresh'])} = {signed(s['public'] - s['fresh'], 4)}.",
        f"Picks that truly beat the baseline: {fmt(100 * s['truly_better'], 0)}% of {result['replicates'] * result['splits']} public-split picks.",
    ]
    metrics = {"Public gain": fmt(s["public"], 4), "Private gain": fmt(s["private"], 4), "Fresh-row gain": fmt(s["fresh"], 4),
               "Public minus private": signed(over, 4), "Picks truly better": f"{fmt(100 * s['truly_better'], 0)}%"}
    alt = (f"Lines of public, private and fresh-row gain against the number of logged versions, with the {n}-version point marked: public "
           f"{fmt(s['public'], 4)}, private {fmt(s['private'], 4)}, fresh {fmt(s['fresh'], 4)}.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    pub = [results[n]["selected"]["public"] for n in LOGGED]
    assert all(b > a for a, b in zip(pub, pub[1:])), "public gain should rise with the number of versions"
    for n in LOGGED:
        s = results[n]["selected"]
        assert abs(s["private"] - s["fresh"]) < 0.0015, f"private gain should track the fresh rows at {n}"
        if n >= 10:
            assert s["public"] - s["private"] > 0.002, f"public gain should overstate at {n}"
        if n <= 30:
            assert s["truly_better"] < 0.5, f"most picks should not truly beat the baseline at {n}"
    assert results[100]["selected"]["truly_better"] > 0.5, "at 100 versions most picks help"
    assert results[3]["selected"]["private"] < 0.001 and results[10]["selected"]["private"] < 0.002
    assert (results[100]["selected"]["public"] - results[100]["selected"]["private"]
            > results[3]["selected"]["public"] - results[3]["selected"]["private"] + 0.002), "overstatement should grow"
    thirty = results[30]["selected"]
    assert (fmt(thirty["public"], 4), fmt(thirty["private"], 4), fmt(thirty["fresh"], 4)) == ("0.0045", "0.0008", "0.0010")
    assert thirty["private"] < 0.002, "prediction option says under 0.002"
    ten = results[10]["selected"]
    assert (fmt(ten["public"], 4), fmt(ten["private"], 4)) == ("0.0030", "0.0001"), "check question numbers"
    assert fmt(results[3]["selected"]["public"], 4) == "0.0007" and fmt(results[100]["selected"]["public"], 4) == "0.0083"
    big = results[100]["selected"]
    assert (fmt(big["private"], 4), fmt(big["fresh"], 4)) == ("0.0032", "0.0036")
    assert fmt(big["public"] - big["private"], 4) == "0.0051", "answer number"
    # The assumptions text says that with no useful candidate the private gain is negative at every N.
    global USEFUL_WEIGHT
    saved, USEFUL_WEIGHT = USEFUL_WEIGHT, 0.0
    try:
        useless = run(100)
    finally:
        USEFUL_WEIGHT = saved
    assert all(c["private"] < 0 and c["fresh"] < 0 for c in useless["curve"]), "useless candidates should lose on private rows"
