"""Chapter 36: Categorical Embeddings. Label codes, frequency and target encoding against test rows from unseen categories."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from kaggle_companion.activities._common import clean

SEED = 36
REPLICATES = 6                       # independent worlds per control value; results are means over them
SEEN, NEW = 1500, 1500               # categories that can appear in training, and a pool of new ones for test
N_TRAIN, N_TEST = 10_000, 6_000
M = 10.0                             # smoothing strength of the chapter's smoothed mean


def make_world(rng):
    """Zipf popularity for the training categories; a category effect plus a rarity effect (rare categories run hotter)."""
    p = 1 / np.arange(1, SEEN + 1) ** 0.9
    p /= p.sum()
    log_p = np.concatenate([np.log(p), np.full(NEW, np.log(p[-1]))])        # new categories are tail-rare
    rarity = -0.45 * (log_p - np.log(p).mean()) / np.log(p).std()
    return p, rng.normal(0, 0.7, SEEN + NEW) + rarity


def draw_rows(categories, effect, rng):
    x = rng.normal(size=(len(categories), 2))
    logit = -1.2 + 0.7 * x[:, 0] - 0.3 * x[:, 1] + effect[categories]
    return x, (rng.random(len(categories)) < 1 / (1 + np.exp(-logit))).astype(int)


def smoothed_means(cat, y, prior):
    """e_c = (s_c + m*mu) / (n_c + m). A category with n_c = 0 gets exactly the prior mu."""
    s = np.bincount(cat, weights=y, minlength=SEEN + NEW)
    n = np.bincount(cat, minlength=SEEN + NEW)
    return (s + M * prior) / (n + M)


def one_world(unseen_share, seed):
    rng = np.random.default_rng(seed)
    p, effect = make_world(rng)
    cat_tr = rng.choice(SEEN, N_TRAIN, p=p)
    x_tr, y_tr = draw_rows(cat_tr, effect, rng)
    n_new = int(round(unseen_share * N_TEST))
    cat_te = np.concatenate([rng.choice(SEEN, N_TEST - n_new, p=p), SEEN + rng.integers(0, NEW, n_new)])
    x_te, y_te = draw_rows(cat_te, effect, rng)

    known = np.zeros(SEEN + NEW, bool)
    known[cat_tr] = True
    unseen = ~known[cat_te]                                  # unseen means: absent from the training dictionary
    counts = np.bincount(cat_tr, minlength=SEEN + NEW)       # frequency encoding from training rows only; unseen -> 0
    codes = -np.ones(SEEN + NEW, int)                        # label codes in order of first appearance; unseen -> -1
    first = np.unique(cat_tr, return_index=True)
    order = first[0][np.argsort(first[1])]                   # in a shuffled table, first appearance follows popularity
    codes[order] = np.arange(known.sum())
    random_codes = -np.ones(SEEN + NEW, int)                 # the same categories numbered in random order; unseen -> -1
    random_codes[np.random.default_rng([seed, 1]).permutation(order)] = np.arange(known.sum())

    prior = y_tr.mean()
    te_train = np.zeros(N_TRAIN)                             # nested: each training row is encoded from the other folds
    for fit, out in KFold(5, shuffle=True, random_state=0).split(cat_tr):
        te_train[out] = smoothed_means(cat_tr[fit], y_tr[fit], y_tr[fit].mean())[cat_tr[out]]
    full_map = smoothed_means(cat_tr, y_tr, prior)
    te_known = np.where(known[cat_te], full_map[cat_te], prior)    # the documented rule: unknown -> training prior
    te_zero = np.where(known[cat_te], full_map[cat_te], 0.0)       # the shortcut: unknown -> 0

    arms = {                                                 # (training features, test features)
        "none": (x_tr, x_te),
        "label": (np.c_[x_tr, codes[cat_tr]], np.c_[x_te, codes[cat_te]]),
        "label_random": (np.c_[x_tr, random_codes[cat_tr]], np.c_[x_te, random_codes[cat_te]]),
        "frequency": (np.c_[x_tr, counts[cat_tr]], np.c_[x_te, counts[cat_te]]),
        "target_prior": (np.c_[x_tr, te_train, counts[cat_tr]], np.c_[x_te, te_known, counts[cat_te]]),
        "target_zero": (np.c_[x_tr, te_train, counts[cat_tr]], np.c_[x_te, te_zero, counts[cat_te]]),
    }
    scores = {}
    for name, (a, b) in arms.items():
        model = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, random_state=0).fit(a, y_tr)
        pred = model.predict_proba(b)[:, 1]
        scores[name] = [roc_auc_score(y_te, pred), roc_auc_score(y_te[~unseen], pred[~unseen]),
                        roc_auc_score(y_te[unseen], pred[unseen])]
    return scores, float(unseen.mean())


def run(unseen_share):
    worlds = [one_world(unseen_share, SEED * 100 + r) for r in range(REPLICATES)]
    names = list(worlds[0][0])
    mean = {k: np.mean([w[0][k] for w in worlds], axis=0) for k in names}
    below = {k: float(np.mean([w[0][k][0] < w[0]["none"][0] for w in worlds])) for k in names}
    return clean({
        "unseen_share": unseen_share,
        "measured_unseen_share": float(np.mean([w[1] for w in worlds])),
        "auc": {k: mean[k][0] for k in names}, "auc_seen": {k: mean[k][1] for k in names},
        "auc_unseen": {k: mean[k][2] for k in names}, "below_no_feature": below,
        "replicates": REPLICATES, "test_rows": N_TEST,
    })
# notebook-end


SPEC = {
    "chapter": 36,
    "chapter_title": "Categorical Embeddings",
    "subtitle": "Every categorical encoding needs a rule for categories it has never seen.",
    "summary": ("Test rows often carry categories the training dictionary lacks. One demonstration measures how a label code, a "
                "frequency encoding and a nested target encoding with two different unknown rules hold up as the share of "
                "unseen-category rows grows, against a model with no categorical feature at all."),
    "title": "Encodings against test rows from unseen categories",
    "question": "How large can the share of unseen categories get before an integer label code is worse than dropping the column, and what does the unknown rule cost a target encoding?",
    "why": ("Seen-category rows reward every encoding. The leaderboard may contain new products, users or zip codes, and then the "
            "unknown rule decides the score. The chapter asks for a documented dictionary and an explicit unknown code for this reason."),
    "method": ("A constructed binary task with 1,500 Zipf-popular training categories, 10,000 training rows, two numeric features "
               "and a category effect plus a rarity effect (rare categories have higher event rates). 6,000 test rows are "
               "drawn from the training categories, except for the control's share, which comes from 1,500 new categories. Five "
               "gradient-boosted models are compared by pooled AUC: no category feature, an integer label code numbered in order of "
               "first appearance, as pandas factorize numbers them (unknown = -1), "
               "frequency counts (unknown = 0), nested smoothed target means plus frequency with unknown = the training prior, "
               "and the same with unknown = 0. A sixth model numbers the same categories in random order, to show what the "
               "label code's result depends on. Results are means over 6 independent worlds."),
    "control": {"key": "unseen_share", "label": "Share of test rows from categories absent in training",
                "values": [0, 0.1, 0.25, 0.5], "default": 0.25,
                "value_labels": ["0%", "10%", "25%", "50%"]},
    "source_section": "Handling Unseen Categories at Inference",
    "symbols": ("e_c is the encoding of category c, s_c the sum and n_c the count of permitted training targets in c, mu their "
                "mean (the prior) and m the smoothing strength (10 here). For an unseen category n_c = 0."),
    "explanation": ("Numbered in order of first appearance in a shuffled table, label codes follow popularity, so the model's "
                    "splits on them partly track how common a category is. The unknown code -1 sits beside the codes of the most "
                    "popular categories, so every row from a new category receives their low prediction, although new categories "
                    "are rare and run hotter here. Once unseen rows are common, that misplaced unknown costs more than the seen "
                    "rows gain. Numbered in random order, the same code stays above the no-feature baseline. A frequency count has a "
                    "natural value for a new category (zero occurrences), which is informative when rarity matters. The smoothed "
                    "target mean returns exactly the prior when n_c = 0, which is the documented fallback; writing 0 instead "
                    "tells the model the category almost never has events. The seen rows are unaffected by either unknown rule."),
    "application": ("Persist the training dictionary, choose the unknown value on purpose (the training prior for a target mean, "
                    "zero for counts), and score the encoding on a validation slice that contains unseen categories in the share the test set will."),
    "assumptions": ("Constructed data and HistGradientBoostingClassifier standing in for a GBM library. The unseen categories are "
                    "rare by construction, so frequency carries signal about them; with no rarity effect the gap between frequency "
                    "and target encoding would change. New categories run hotter than average here, which enlarges the cost of any rule "
                    "that places them at the low end (unknown = 0 for a target mean, or -1 beside the popular codes). The label "
                    "code's unknown value is -1, a documented code with no training rows; whether the label code falls below the "
                    "no-feature baseline depends on the code order, and the crossover share moves with the random draw."),
    "prediction": ("Half the test rows come from categories absent in training. How does the integer label code (unknown = -1) compare "
                   "with a model that has no categorical feature at all?"),
    "prediction_options": ["Still better: it keeps the information of the seen categories",
                           "About the same", "Worse than dropping the column"],
    "prediction_answer": 2,
    "prediction_feedback": {
        "correct": "Pooled AUC is 0.573 with the label code against 0.646 with no categorical feature: the column hurts more than it helps.",
        "incorrect": "Pooled AUC is 0.573 with the label code against 0.646 with no categorical feature. The seen rows gain, but -1 sits beside the codes of the most popular categories, so every unseen row receives their low prediction although new categories run hotter here. Numbered in random order, the same code scores 0.698.",
    },
    "check": "Filling unseen categories with 0 instead of the training prior leaves the seen-row score unchanged. Why does the pooled score still fall?",
    "answer": ("Only unseen rows receive the unknown value. With 0 they look like categories that almost never have events, "
               "so they are ranked below the seen rows although their true event rate is higher; the prior places them at an "
               "average value. At 50% unseen the pooled AUC is 0.742 with the prior rule and 0.665 with 0, while the seen-row AUC "
               "is identical (0.776)."),
    "provenance": "Constructed example: seeded synthetic categories and gradient-boosted models, measured by the chapter activity.",
    "apply": [
        "Save the training dictionary, prior and counts with the model and map new categories through them; never recode at inference.",
        "Decide the unknown value for each encoding in writing: prior for target means, zero for frequency counts, a learned or neutral vector for embeddings.",
        "Build a validation slice with the unseen share you expect on the leaderboard (new users or items in a time split) and score encodings there.",
        "Keep frequency encoding as a cheap baseline: it needs no labels and handles new categories by construction.",
    ],
    "honesty": ("Constructed data; frequency is strong here because unseen categories are rare by construction. The label code "
                "falls below the no-feature baseline because of where -1 lands in a popularity-ordered code; in random order it "
                "stays above it, and the crossover share moves with the random draw."),
}

EQUATIONS = [{"tex": r"e_c = \frac{s_c + m\mu}{n_c+m}, \qquad e_c = \mu \text{ when } n_c = 0",
              "alt": "e c equals s c plus m times mu, divided by n c plus m; when n c is zero e c equals mu",
              "basis": "The smoothed mean of Chapter 9, which Chapter 36 (Target Encoding: The Workhorse) uses; the n_c = 0 case is the unknown-category rule."}]
NCOLS = 2
HEIGHT = 4.6
ARMS = [("label", "Label\ncode"), ("frequency", "Frequency\ncount"), ("target_prior", "Target,\nunk. =\nprior"),
        ("target_zero", "Target,\nunk. = 0")]
ARM_NAMES = {"label": "label code", "frequency": "frequency", "target_prior": "target mean with the prior",
             "target_zero": "target mean with 0"}


def draw(axes, result, parameter):
    left, right = axes
    xs = list(range(len(ARMS)))
    pooled = [("label_random", "Label,\nrandom\norder")] + ARMS
    px = list(range(len(pooled)))
    colors = [COLORS["light"], COLORS["gold"], COLORS["olive"], COLORS["teal"], COLORS["terracotta"]]
    auc = [result["auc"][k] for k, _ in pooled]
    left.bar(px, auc, color=colors, edgecolor=COLORS["ink"], lw=0.6, width=0.6)
    for x, v in zip(px, auc):
        left.text(x, v + 0.006, fmt(v), ha="center", va="bottom", fontsize=10)
    left.axhline(result["auc"]["none"], color=COLORS["ink"], ls=(0, (4, 3)), lw=1.4,
                 label=f"No categorical feature: {fmt(result['auc']['none'])}")
    left.set_xticks(px, [name for _, name in pooled], fontsize=10)
    left.set_ylim(0.5, 0.86)
    left.set_ylabel("Pooled AUC on test rows")
    left.set_xlabel(f"Encoding, {round(100 * parameter)}% of test rows unseen")
    left.legend(loc="upper left", frameon=False, fontsize=10)

    seen = [result["auc_seen"][k] for k, _ in ARMS]
    new = [result["auc_unseen"][k] for k, _ in ARMS]
    right.bar([x - 0.2 for x in xs], seen, width=0.4, color=COLORS["light"], edgecolor=COLORS["ink"], lw=0.6, label="Seen-category rows")
    right.bar([x + 0.2 for x in xs], new, width=0.4, color=COLORS["navy"], edgecolor=COLORS["ink"], lw=0.6, label="Unseen-category rows")
    for x, a, b in zip(xs, seen, new):
        right.text(x - 0.2, a + 0.006, fmt(a)[1:], ha="center", va="bottom", fontsize=10)
        right.text(x + 0.2, b + 0.006, fmt(b)[1:], ha="center", va="bottom", fontsize=10)
    right.set_xticks(xs, [name for _, name in ARMS], fontsize=10)
    right.set_ylim(0.5, 0.86)
    right.set_ylabel("AUC within each group of rows")
    right.set_xlabel("Encoding")
    right.legend(loc="upper left", frameon=False, fontsize=10)


def explain(result, parameter):
    a, seen, new = result["auc"], result["auc_seen"], result["auc_unseen"]
    gap = a["target_prior"] - a["target_zero"]
    label_vs = a["label"] - a["none"]
    verdict = ("worse than dropping the column" if label_vs < -0.005 else
               "better than dropping the column" if label_vs > 0.005 else "no better than dropping the column")
    best = max(("label", "frequency", "target_prior", "target_zero"), key=lambda k: a[k])
    interpretation = (
        f"With {round(100 * parameter)}% of test rows from unseen categories (measured {round(100 * result['measured_unseen_share'])}%), "
        f"the label code scores {fmt(a['label'])} against {fmt(a['none'])} with no categorical feature, "
        f"{fmt(a['label'])} - {fmt(a['none'])} = {signed(label_vs)}: {verdict}. The target mean with the prior scores {fmt(a['target_prior'])}; "
        f"with unknown = 0 it scores {fmt(a['target_zero'])}, so the unknown rule alone costs "
        f"{fmt(a['target_prior'])} - {fmt(a['target_zero'])} = {fmt(gap)} AUC. Seen rows score {fmt(seen['target_prior'])} under both "
        f"target rules. Best overall here: {ARM_NAMES[best]}, with frequency alone {fmt(a['target_prior'] - a['frequency'])} behind the best target rule. "
        f"Numbered in random order, the label code scores {fmt(a['label_random'])}, so its result depends on where -1 lands.")
    steps = [
        f"Label code against no feature: {fmt(a['label'])} - {fmt(a['none'])} = {signed(label_vs)}.",
        f"Cost of unknown = 0: {fmt(a['target_prior'])} - {fmt(a['target_zero'])} = {fmt(gap)}.",
        f"Frequency against the best target rule: {fmt(a['target_prior'])} - {fmt(a['frequency'])} = {fmt(a['target_prior'] - a['frequency'])}.",
        f"Label code fell below the no-feature model in {round(100 * result['below_no_feature']['label'])}% of {result['replicates']} worlds.",
    ]
    metrics = {"No categorical feature": fmt(a["none"]), "Label code": fmt(a["label"]), "Frequency count": fmt(a["frequency"]),
               "Target mean, unknown = prior": fmt(a["target_prior"]), "Target mean, unknown = 0": fmt(a["target_zero"]),
               "Unseen rows, label code": fmt(new["label"]), "Label code, random order": fmt(a["label_random"])}
    alt = (f"Left: pooled AUC of five encodings when {round(100 * parameter)}% of test rows are unseen; the label code scores "
           f"{fmt(a['label'])} and the random-order label code {fmt(a['label_random'])} against a no-feature line at {fmt(a['none'])}. Right: AUC within seen and unseen rows for each encoding.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for u, res in results.items():
        a = res["auc"]
        assert a["target_prior"] > a["frequency"] > a["none"] + 0.08, f"target+frequency best, frequency close at {u}"
        assert a["target_prior"] - a["frequency"] < 0.012, f"frequency within about 0.01 of target at {u}"
        assert a["target_prior"] > a["label"] + 0.04, f"label code worse than target at {u}"
        assert abs(res["auc_seen"]["target_prior"] - res["auc_seen"]["target_zero"]) < 1e-9, f"seen rows unaffected at {u}"
        if u > 0:
            assert a["target_prior"] - a["target_zero"] > 0.015, f"unknown rule should cost at {u}"
    assert results[0]["auc"]["label"] > results[0]["auc"]["none"] + 0.05, "with few unseen rows the label code helps"
    assert results[0.1]["auc"]["label"] > results[0.1]["auc"]["none"], "label code still helps at 10% unseen"
    assert results[0.5]["auc"]["label"] < results[0.5]["auc"]["none"] - 0.03, "label code worse than no feature at 50% unseen"
    assert results[0.25]["auc"]["label"] < results[0.25]["auc"]["none"], "default state shows the crossover"
    half = results[0.5]
    assert fmt(half["auc"]["label"]) == "0.573" and fmt(half["auc"]["none"]) == "0.646", "prediction feedback numbers"
    assert fmt(half["auc"]["target_prior"]) == "0.742" and fmt(half["auc"]["target_zero"]) == "0.665"
    assert fmt(half["auc_seen"]["target_prior"]) == "0.776" and fmt(half["auc_seen"]["target_zero"]) == "0.776"
    assert half["auc_seen"]["label"] > half["auc"]["label"] + 0.1, "label code still gains on seen rows"
    assert results[0]["measured_unseen_share"] < 0.05 and abs(half["measured_unseen_share"] - 0.5) < 0.05
    for u, res in results.items():
        assert res["auc"]["label_random"] > res["auc"]["none"], f"random-order label code stays above the baseline at {u}"
    assert fmt(half["auc"]["label_random"]) == "0.698", "prediction feedback: random-order label code"
