"""Chapter 7: Domain Features as Testable Hypotheses. Entity history against pseudo-ID collisions and future rows."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from kaggle_companion.activities._common import clean

SEED = 7
REPLICATES = 8        # independent constructed transaction logs; every score is their mean
N_CUSTOMERS = 400
CUTOFF = 0.7          # transactions before this time are development rows, the rest are the forward holdout
FRAUD_RATE, FRAUD_MULTIPLE = 0.06, 3.0


def generate(per_key, seed):
    """One transaction log sorted by time. Customers differ a lot in spend level; fraud is 3x the customer's own level.
    The pipeline never sees the customer, only a pseudo-ID key shared by `per_key` customers chosen at random."""
    rng = np.random.default_rng(seed)
    level = np.exp(rng.normal(3.5, 1.2, N_CUSTOMERS))           # each customer's typical amount
    customer = np.repeat(np.arange(N_CUSTOMERS), rng.integers(6, 15, N_CUSTOMERS))
    time = rng.random(len(customer))
    fraud = (rng.random(len(customer)) < FRAUD_RATE).astype(int)
    amount = level[customer] * np.exp(rng.normal(0, 0.5, len(customer)) + fraud * np.log(FRAUD_MULTIPLE))
    key = (rng.permutation(N_CUSTOMERS) // per_key)[customer]   # per_key = 1 is a perfect entity ID
    order = np.argsort(time, kind="stable")
    return time[order], key[order], amount[order], fraud[order]


def history_means(time, key, amount):
    """Two entity means for every row. Past-only: earlier rows of the same key. Whole-period: every development
    row of the key (past, present and future) for development rows; rows seen up to now for holdout rows."""
    n = len(amount)
    dev = time < CUTOFF
    past, to_date = np.full(n, np.nan), np.empty(n)
    total, count = {}, {}
    for i in range(n):                                          # rows are in time order
        s, c = total.get(key[i], 0.0), count.get(key[i], 0)
        past[i] = s / c if c else np.nan                        # the first row of a key has no history
        total[key[i]], count[key[i]] = s + amount[i], c + 1
        to_date[i] = total[key[i]] / count[key[i]]              # includes the current row, no future
    dev_total, dev_count = {}, {}
    for i in np.where(dev)[0]:
        dev_total[key[i]] = dev_total.get(key[i], 0.0) + amount[i]
        dev_count[key[i]] = dev_count.get(key[i], 0) + 1
    whole = np.array([dev_total[key[i]] / dev_count[key[i]] if dev[i] else to_date[i] for i in range(n)])
    return dev, past, whole


def columns(amount, mean):
    """log(amount / mean) plus a flag for rows with no history (ratio set to 0 there)."""
    missing = np.isnan(mean)
    ratio = np.where(missing, 0.0, np.log(amount) - np.log(np.where(missing, 1.0, mean)))
    return [ratio, missing.astype(float)]


def model():
    return HistGradientBoostingClassifier(max_iter=80, learning_rate=0.1, max_depth=3, random_state=0)


def one_log(per_key, seed):
    time, key, amount, y = generate(per_key, seed)
    dev, past, whole = history_means(time, key, amount)
    sets = {"raw": [np.log(amount)],
            "past_only": [np.log(amount)] + columns(amount, past),
            "whole_period": [np.log(amount)] + columns(amount, whole)}
    out = {}
    for name, cols in sets.items():
        X = np.column_stack(cols)
        Xd, yd, Xh, yh = X[dev], y[dev], X[~dev], y[~dev]
        oof = np.zeros(len(yd))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=1).split(Xd, yd):
            oof[b] = model().fit(Xd[a], yd[a]).predict_proba(Xd[b])[:, 1]
        out[name] = {"cv": roc_auc_score(yd, oof),               # development estimate: 5-fold on development rows
                     "holdout": roc_auc_score(yh, model().fit(Xd, yd).predict_proba(Xh)[:, 1])}
    out["rows"] = [int(dev.sum()), int((~dev).sum())]
    return out


def run(customers_per_key):
    logs = [one_log(customers_per_key, SEED * 100 + i) for i in range(REPLICATES)]
    names = ["raw", "past_only", "whole_period"]
    result = {"customers_per_key": customers_per_key, "replicates": REPLICATES, "customers": N_CUSTOMERS,
              "development_rows": float(np.mean([g["rows"][0] for g in logs])),
              "holdout_rows": float(np.mean([g["rows"][1] for g in logs]))}
    for name in names:
        result[name] = {m: float(np.mean([g[name][m] for g in logs])) for m in ("cv", "holdout")}
        result[name]["holdout_per_log"] = [g[name]["holdout"] for g in logs]
    return clean(result)
# notebook-end


SPEC = {
    "chapter": 7,
    "chapter_title": "Domain Features as Testable Hypotheses",
    "subtitle": "An entity history is a hypothesis about identity, and it needs an availability check and a held-out comparison.",
    "summary": ("A purchase is unusual only relative to its customer, so a past-only entity mean can add a lot. One demonstration "
                "measures how much it adds as the pseudo-ID merges more customers into one key, and what a mean that sees later rows "
                "appears to add in development."),
    "title": "Entity history against pseudo-ID collisions, scored on later transactions",
    "question": ("How much does an entity-history feature add over the raw amount when the key merges several customers, and "
                 "how much of its apparent value comes from rows the prediction could not see?"),
    "why": ("The chapter's fraud example turns on an entity the data does not name. A pseudo-ID built from descriptors can merge "
            "unrelated customers, and a whole-period mean can read future behavior. Measuring both shows when the history is worth "
            "building and when its development score stops describing the deployed model."),
    "method": ("A constructed log of 400 customers with 6 to 14 transactions each (about 4,000 rows, 6% fraud). Customers differ widely in "
               "typical spend, and a fraudulent amount is 3 times the customer's own level, so the raw amount is a weak signal and the "
               "amount relative to the customer's history is a strong one. The model sees only a pseudo-ID key shared by the number of "
               "customers set by the control. Three feature sets feed a histogram gradient boosting classifier: the raw amount, the raw "
               "amount plus the log ratio to the past-only mean of the key, and the raw amount plus the log ratio to a whole-period mean "
               "(every development row of the key, including later ones). Each is scored by AUC two ways: 5-fold cross-validation on the "
               "first 70% of time, and a forward holdout on the last 30%, where only history up to the current row exists. "
               "Every score is the mean over 8 constructed logs."),
    "control": {"key": "customers_per_key", "label": "Customers sharing one pseudo-ID key",
                "values": [1, 2, 5, 20], "default": 1,
                "value_labels": ["1: the key is the customer", "2", "5", "20: keys mix many customers"]},
    "source_section": "The Pattern: Find the Entity the Problem Is Really About",
    "symbols": ("x_i is the amount of transaction i, h_i the mean amount of earlier transactions with the same key, and r_i the "
                "log ratio fed to the model. Rows with no history get r_i = 0 and a missing-history flag. A whole-period mean replaces "
                "h_i by the mean of every development row of the key."),
    "explanation": ("With a perfect key the past-only ratio removes each customer's spend level and the fraud shows clearly. As the "
                    "key merges customers, its mean describes a blend of spend levels, so the ratio stops describing the row's own "
                    "customer and the lift over the raw amount fades. A whole-period mean also looks better in development, because "
                    "early rows borrow later rows of the same key. At deployment those later rows do not exist, and the extra "
                    "development score does not carry over."),
    "application": ("Name the entity, check how many customers each key holds, build every history from rows before the prediction "
                    "time, and score the history feature against the raw baseline on a later period."),
    "assumptions": ("Constructed data and one model family. The collision here merges customers at random, and the lift stays positive "
                    "while a key holds a few customers; real collisions may be milder or worse. Holdout rows come later and so have longer "
                    "histories than development rows, which is why history scores there exceed the cross-validation scores. "
                    "The whole-period arm is scored on the holdout with the mean of rows up to and including the present, the most "
                    "generous deployable version."),
    "prediction": "With 20 customers sharing each key, what does the past-only ratio add over the raw amount on the forward holdout?",
    "prediction_options": ["About +0.20 AUC, the same as with an exact ID", "About half of the exact-ID lift", "Nothing measurable"],
    "prediction_answer": 2,
    "prediction_feedback": {
        "correct": "It adds +0.001 (0.707 against 0.706), compared with +0.202 when the key is the customer.",
        "incorrect": "It adds +0.001 (0.707 against 0.706), compared with +0.202 when the key is the customer. A key that mixes many customers has a mean that describes nobody.",
    },
    "check": "With an exact ID, why does the whole-period mean score higher than the past-only mean in development but not on the holdout?",
    "answer": ("In development the whole-period mean gives each row a history that includes later rows, so even early rows get a "
               "well-estimated customer level (0.907 against 0.864 for the past-only mean). On the holdout only history to date exists, "
               "and the two arms score alike (0.906 against 0.908). The development gap measures access to the future, not a better feature."),
    "provenance": "Constructed example: seeded synthetic transactions with a built-in spend level per customer, measured by the chapter activity.",
    "apply": [
        "Write down the prediction unit, the contextual entity and the history cutoff before building the feature.",
        "Count how many records each pseudo-ID key holds; a key that mixes many entities has a mean that describes none of them.",
        "Build histories from rows before the prediction time only, and test a whole-period mean only to see what it would falsely add.",
        "Keep the raw-amount baseline and compare the history feature against it on a later period, not on cross-validation alone.",
    ],
    "honesty": ("Constructed data; the spend-level spread, the 3 times fraud multiple and the random merging are properties of this "
                "generator, not measurements from a fraud competition."),
}

EQUATIONS = [{"tex": r"r_i = \log x_i - \log h_i, \qquad h_i = \frac{1}{n_i}\sum_{j:\, \mathrm{key}_j=\mathrm{key}_i,\ t_j<t_i} x_j",
              "alt": "r i equals log x i minus log h i, where h i is the mean amount of the earlier transactions with the same key",
              "basis": "The activity's own past-only ratio feature; the chapter describes the idea in words (The Pattern: Find the Entity the Problem Is Really About) and has no display equation."}]
NCOLS = 1
HEIGHT = 4.4
SETS = [("raw", "Raw amount"), ("past_only", "Raw + past-only\nratio"), ("whole_period", "Raw + whole-period\nratio")]


def draw(ax, result, parameter):
    xs = list(range(len(SETS)))
    cv = [result[k]["cv"] for k, _ in SETS]
    hold = [result[k]["holdout"] for k, _ in SETS]
    ax.bar([x - 0.2 for x in xs], cv, width=0.4, color=COLORS["light"], edgecolor=COLORS["ink"], lw=0.6,
           label="Development CV")
    ax.bar([x + 0.2 for x in xs], hold, width=0.4, color=COLORS["teal"], edgecolor=COLORS["ink"], lw=0.6,
           label="Forward holdout")
    for x, (k, _) in zip(xs, SETS):
        dots = result[k]["holdout_per_log"]
        ax.scatter([x + 0.2 + (i - (len(dots) - 1) / 2) * 0.035 for i in range(len(dots))], dots, s=12, color=COLORS["ink"],
                   zorder=3, label="One log" if x == 0 else None)
        top = max(dots)
        ax.text(x - 0.2, cv[x] + 0.008, fmt(cv[x]), ha="center", va="bottom", fontsize=10)
        ax.text(x + 0.2, max(top, hold[x]) + 0.008, fmt(hold[x]), ha="center", va="bottom", fontsize=10)
    ax.axhline(result["raw"]["holdout"], color=COLORS["gold"], ls=(0, (4, 3)), lw=1.4,
               label=f"Raw, holdout: {fmt(result['raw']['holdout'])}")
    ax.set_xticks(xs, [name for _, name in SETS])
    ax.set_ylim(0.5, 1.18)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])                 # AUC cannot exceed 1; the space above holds the legend
    ax.set_ylabel("AUC")
    ax.set_xlabel(f"Feature set, {parameter} customer{'s' if parameter != 1 else ''} per key")
    ax.legend(loc="upper left", frameon=False, fontsize=10, ncol=2)


def explain(result, parameter):
    raw, p, w = result["raw"], result["past_only"], result["whole_period"]
    lift = p["holdout"] - raw["holdout"]
    dev_gap = w["cv"] - p["cv"]
    hold_gap = w["holdout"] - p["holdout"]
    if lift > 0.05:
        verdict = "the past-only history is worth building at this key quality"
    elif lift > 0.01:
        verdict = "the past-only history adds only a little, and a cleaner key would matter more than a better model"
    else:
        verdict = "the key mixes too many customers: the history adds nothing over the raw amount"
    interpretation = (
        f"With {parameter} customer{'s' if parameter != 1 else ''} per key, the raw amount scores {fmt(raw['holdout'])} on the forward "
        f"holdout and the past-only ratio {fmt(p['holdout'])}: {fmt(p['holdout'])} - {fmt(raw['holdout'])} = {fmt(lift)} of lift, so {verdict}. "
        f"In cross-validation the whole-period mean scores {fmt(abs(dev_gap))} {'above' if dev_gap >= 0 else 'below'} the past-only mean "
        f"({fmt(w['cv'])} against {fmt(p['cv'])}); "
        f"on the holdout, where later rows do not exist, the difference is {signed(hold_gap)}.")
    steps = [
        f"Lift of the past-only ratio on the holdout: {fmt(p['holdout'])} - {fmt(raw['holdout'])} = {signed(lift)}.",
        f"Development advantage of the whole-period mean: {fmt(w['cv'])} - {fmt(p['cv'])} = {signed(dev_gap)}.",
        f"Holdout advantage of the whole-period mean: {fmt(w['holdout'])} - {fmt(p['holdout'])} = {signed(hold_gap)}.",
        f"Past-only ratio, cross-validation minus holdout: {fmt(p['cv'])} - {fmt(p['holdout'])} = {signed(p['cv'] - p['holdout'])} "
        f"(holdout rows have longer histories).",
    ]
    metrics = {"Raw amount, holdout": fmt(raw["holdout"]), "Past-only ratio, holdout": fmt(p["holdout"]),
               "Past-only lift": signed(lift), "Whole-period, cross-validation": fmt(w["cv"]),
               "Whole-period, holdout": fmt(w["holdout"])}
    alt = (f"Paired bars of development cross-validation AUC and forward-holdout AUC for three feature sets with {parameter} "
           f"customers per key; the past-only ratio lifts the holdout AUC by {fmt(lift)} over the raw amount.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    lifts = {k: res["past_only"]["holdout"] - res["raw"]["holdout"] for k, res in results.items()}
    assert lifts[1] > 0.15, "exact key should give a large lift"
    assert lifts[1] > lifts[2] > lifts[5] > lifts[20], f"lift should fade as keys merge customers: {lifts}"
    assert abs(lifts[20]) < 0.01, "20 customers per key should add nothing measurable"
    for k, res in results.items():
        assert abs(res["whole_period"]["holdout"] - res["past_only"]["holdout"]) < 0.02, f"holdout arms should match at {k}"
        assert abs(res["raw"]["holdout"] - results[1]["raw"]["holdout"]) < 1e-9, "raw arm is the same at every key size"
    one = results[1]
    assert one["whole_period"]["cv"] - one["past_only"]["cv"] > 0.02, "whole-period should look better in development"
    assert fmt(one["whole_period"]["cv"]) == "0.907" and fmt(one["past_only"]["cv"]) == "0.864", "answer numbers"
    assert fmt(one["whole_period"]["holdout"]) == "0.906" and fmt(one["past_only"]["holdout"]) == "0.908", "answer numbers"
    assert fmt(lifts[1]) == "0.202" and fmt(lifts[20]) == "0.001", "prediction feedback numbers"
    twenty = results[20]
    assert fmt(twenty["past_only"]["holdout"]) == "0.707" and fmt(twenty["raw"]["holdout"]) == "0.706", "prediction feedback numbers"
    assert one["past_only"]["holdout"] > one["past_only"]["cv"], "holdout rows have longer histories"
