"""Chapter 5: Adversarial Validation. Domain AUC against target-model behaviour under four kinds of shift."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from kaggle_companion.activities._common import clean

SEED = 5
REPLICATES = 16      # independent train/test draws per shift type; every number is their mean
N = 2000             # training rows and test rows
CAP = 10.0           # the chapter's weight cap
SHIFTS = {0: "none", 1: "covariate", 2: "timestamp", 3: "sign_flip"}


def generate(shift, seed):
    """Train and test rows. Columns: x1, x2, x3 (noise), t (a time-like field). Test differs in one way only."""
    rng = np.random.default_rng(seed)

    def rows(n, test):
        x = rng.normal(size=(n, 3))
        t = rng.uniform(0, 1, n)
        beta2 = 0.8
        if test and shift == "covariate":
            x[:, 0] += 1.0                      # x1 moves right; P(y | x) is untouched
        if test and shift == "timestamp":
            t = t + 1.0                         # the test period is later; the label rule is untouched
        if test and shift == "sign_flip":
            beta2 = -0.8                        # same x, but x2 now pushes the label the other way
        logit = 1.2 * x[:, 0] + beta2 * x[:, 1] + 0.7 * (x[:, 0] ** 2 - 1)
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        return np.column_stack([x, t]), y

    return rows(N, False), rows(N, True)


def domain_density_weights(q_oof, n_train, n_test, cap=CAP):
    """The chapter's weights: capped odds q/(1-q) with the sampling-prior correction, scaled to mean 1."""
    q = np.clip(np.asarray(q_oof, dtype=float), 1e-6, 1 - 1e-6)
    weights = np.minimum(q / (1 - q) * n_train / n_test, cap)
    return weights / weights.mean()


def domain_auc(features, domain, seed=0):
    """Out-of-fold AUC of a classifier that tries to tell the two populations apart."""
    oof = np.zeros(len(domain))
    for fit, out in StratifiedKFold(5, shuffle=True, random_state=seed).split(features, domain):
        model = HistGradientBoostingClassifier(max_iter=60, max_depth=3, random_state=0).fit(features[fit], domain[fit])
        oof[out] = model.predict_proba(features[out])[:, 1]
    return roc_auc_score(domain, oof), oof


def one_replicate(shift, seed):
    (x_tr, y_tr), (x_te, y_te) = generate(shift, seed)
    features = np.vstack([x_tr, x_te])
    domain = np.r_[np.zeros(N), np.ones(N)]
    auc, oof = domain_auc(features, domain)
    shuffled_auc, _ = domain_auc(features, np.random.default_rng(seed + 1).permutation(domain))
    w = domain_density_weights(oof[:N], N, N)               # weights for training rows from their out-of-fold q
    ess = w.sum() ** 2 / (w ** 2).sum() / N                 # effective sample size as a fraction of N

    cv = []                                                 # the target model: logistic regression on x1, x2, x3 (not t)
    for fit, out in StratifiedKFold(5, shuffle=True, random_state=1).split(x_tr, y_tr):
        m = LogisticRegression(max_iter=500).fit(x_tr[fit, :3], y_tr[fit])
        cv.append(roc_auc_score(y_tr[out], m.predict_proba(x_tr[out, :3])[:, 1]))
    plain = LogisticRegression(max_iter=500).fit(x_tr[:, :3], y_tr)
    weighted = LogisticRegression(max_iter=500).fit(x_tr[:, :3], y_tr, sample_weight=w)
    return {"domain_auc": auc, "shuffled_domain_auc": shuffled_auc, "cv_auc": np.mean(cv),
            "test_auc": roc_auc_score(y_te, plain.predict_proba(x_te[:, :3])[:, 1]),
            "weighted_test_auc": roc_auc_score(y_te, weighted.predict_proba(x_te[:, :3])[:, 1]),
            "weight_ess": ess}


def run(shift_type):
    shift = SHIFTS[int(shift_type)]
    reps = [one_replicate(shift, SEED * 1000 + i) for i in range(REPLICATES)]
    keys = list(reps[0])
    out = {k: float(np.mean([r[k] for r in reps])) for k in keys}
    out["per_replicate"] = {k: [r[k] for r in reps] for k in ("domain_auc", "test_auc", "weighted_test_auc")}
    out["weighting_gain"] = float(np.mean([r["weighted_test_auc"] - r["test_auc"] for r in reps]))
    return clean({"shift_type": int(shift_type), "shift": shift, **out,
                  "train_rows": N, "test_rows": N, "replicates": REPLICATES})
# notebook-end


SPEC = {
    "chapter": 5,
    "chapter_title": "Adversarial Validation",
    "subtitle": "Locate visible separation between populations, then test the target model before acting.",
    "summary": ("A domain classifier asks whether inputs can tell train from test; it does not ask whether the label rule changed. "
                "One demonstration trains the same target model against four test populations and measures domain AUC beside what "
                "the target model actually scores, with and without the chapter's density-ratio weights."),
    "title": "Domain AUC beside the target model's score, for four kinds of shift",
    "question": "When train and test differ, does the domain AUC tell you whether the target model's score has changed?",
    "why": ("A high domain AUC invites a fix and a near-0.5 one invites relief. Measuring both beside the target score shows when "
            "each reaction is wrong, and what reweighting does when it is applied anyway."),
    "method": ("Constructed binary task with 2,000 training rows and 2,000 test rows, three numeric features and a time-like field. "
               "The same training set is used in all four states; only the test population changes: nothing, x1 shifted right "
               "(covariate shift, label rule unchanged), the time field moved later (label rule unchanged), or the effect of x2 on the "
               "label reversed with x unchanged. A gradient-boosting domain classifier (HistGradientBoostingClassifier standing in for "
               "the chapter's LightGBM) gives out-of-fold AUC from all four input columns (three numeric features and the time-like field). The target model is a logistic regression on the "
               "three numeric features, scored by 5-fold cross-validation, on the test rows, and on the test rows again after "
               "training with the chapter's capped, mean-scaled density-ratio weights. Each number is a mean over 16 independent draws."),
    "control": {"key": "shift_type", "label": "How the test population differs from training",
                "values": [0, 1, 2, 3], "default": 3,
                "value_labels": ["0: no shift", "1: covariate shift (x1 moves right)", "2: test-only timestamp",
                                 "3: x2's effect on the label reverses"]},
    "source_section": "Interpreting the AUC",
    "symbols": ("q_i is the out-of-fold probability that training row i looks like a test row, n_train and n_test the row counts, "
                "c the weight cap (10) and w_i the weight handed to the target model. Domain AUC is the out-of-fold AUC of the "
                "domain classifier; 0.5 means it found no separation."),
    "explanation": ("Domain AUC reads only the inputs. A shifted input distribution and a later timestamp both raise it, yet the "
                    "label rule can be stable in both; a reversed label rule leaves it at chance. The target model's score on test rows is "
                    "the only column that answers the question that matters. Weighting by q/(1-q) can help when the shift is overlap-"
                    "preserving, and can wreck the fit when one feature separates the populations perfectly."),
    "application": ("Use domain AUC to find what separates the populations, name the separating feature, and then compare the target "
                    "model on a held-out set that mimics the test conditions before deciding whether to drop, keep or reweight."),
    "assumptions": ("Constructed data. The logistic target model is deliberately a little misspecified (the true log odds contain an x1 "
                    "squared term it cannot represent), which is the situation where covariate-shift weighting has something to fix; a "
                    "flexible model would gain less. The sign-flip state reverses one of two effects, so the test AUC falls to about "
                    "0.57, not below 0.5. The weights use the chapter's cap and are not calibrated further."),
    "prediction": ("Training and test rows have the same input distribution (domain AUC about 0.50), and cross-validation on training "
                   "gives AUC 0.771. What does the model score on the test rows when the effect of x2 on the label has reversed?"),
    "prediction_options": ["About 0.77, the same as cross-validation", "About 0.58, far below cross-validation",
                           "Above 0.77, because the domain AUC is at chance"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "It scores 0.576 against 0.771 in cross-validation, while the domain AUC stays at 0.499: the inputs look identical.",
        "incorrect": "It scores 0.576 against 0.771 in cross-validation, while the domain AUC stays at 0.499. Domain AUC sees only the inputs, and here only the label rule changed.",
    },
    "check": "A timestamp alone gives domain AUC 1.000. Why does the target model still score about the same on test, and why does weighting hurt?",
    "answer": ("The target model never uses the timestamp, and the label rule did not change, so it scores 0.776 on test against 0.771 in "
               "cross-validation. The domain classifier separates the groups perfectly on a field the target does not need; weighting by "
               "that separation gives almost every training row a near-zero weight (effective sample 0.5% of rows), and the weighted "
               "model scores 0.703. Reweighting cannot recover behaviour where training has no support."),
    "provenance": "Constructed example: seeded synthetic rows, a gradient-boosting domain classifier and a logistic target model, measured by the chapter activity.",
    "apply": [
        "Run the domain classifier, then name the separating feature and why it separates before touching anything.",
        "Compare the target model on a held-out set built to resemble the test (new subjects, later dates) before and after any change.",
        "When the separating field is a date or ID the test will always differ on, keep it out of the target model rather than reweighting on it.",
        "Check the effective sample size of any weights; if it is a tiny fraction of the rows, the weights are a symptom, not a fix.",
    ],
    "honesty": ("Constructed data; the sizes of these effects are properties of this generator. Weighting helped one state a little, did "
                "nothing in two and hurt in one; none of that is a competition result."),
}

EQUATIONS = [{"tex": r"w_i \propto \min\!\left(\frac{q_i}{1-q_i}\cdot\frac{n_{\mathrm{train}}}{n_{\mathrm{test}}},\; c\right)",
              "alt": "w i is proportional to the smaller of q i over one minus q i times n train over n test, and c",
              "basis": "The density ratio in Chapter 5, The Three Fixes (domain_density_weights); the chapter states it in prose and code, not as a display equation."}]
NCOLS = 2
HEIGHT = 4.4
SHORT = ["No shift", "Covariate", "Timestamp", "x2 reverses"]


def draw(axes, result, parameter):
    left, right = axes
    dots = result["per_replicate"]
    left.bar([0], [result["domain_auc"]], width=0.5, color=COLORS["navy"], edgecolor=COLORS["ink"], lw=0.6)
    left.scatter([(i - 7.5) * 0.02 for i in range(len(dots["domain_auc"]))], dots["domain_auc"], s=14,
                 color=COLORS["ink"], zorder=3, label="One draw")
    left.axhline(result["shuffled_domain_auc"], color=COLORS["gold"], ls=(0, (4, 3)), lw=1.6,
                 label=f"Shuffled domain labels: {fmt(result['shuffled_domain_auc'])}")
    left.text(0, max(dots["domain_auc"]) + 0.02, fmt(result["domain_auc"]), ha="center", va="bottom", fontsize=10)
    left.set_xticks([0], [SHORT[result["shift_type"]]])
    left.set_xlim(-0.6, 0.6)
    left.set_ylim(0.4, 1.2)
    left.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    left.set_ylabel("Domain AUC (out of fold)")
    left.set_xlabel("Test population")
    left.legend(loc="upper left", frameon=False, fontsize=10)

    xs = [0, 1, 2]
    vals = [result["cv_auc"], result["test_auc"], result["weighted_test_auc"]]
    colors = [COLORS["light"], COLORS["teal"], COLORS["gold"]]
    right.bar(xs, vals, width=0.6, color=colors, edgecolor=COLORS["ink"], lw=0.6)
    tops = [vals[0], max(dots["test_auc"]), max(dots["weighted_test_auc"])]
    for x, v, top in zip(xs, vals, tops):
        right.text(x, max(v, top) + 0.015, fmt(v), ha="center", va="bottom", fontsize=10)
    for x, key in ((1, "test_auc"), (2, "weighted_test_auc")):
        d = dots[key]
        right.scatter([x + (i - 7.5) * 0.025 for i in range(len(d))], d, s=12, color=COLORS["ink"], zorder=3)
    right.axhline(0.5, color=COLORS["grey"], lw=0.8, ls=":")
    right.set_xticks(xs, ["Cross-\nvalidation", "Test,\nunweighted", "Test,\nweighted"])
    right.set_ylim(0.4, 1.0)
    right.set_ylabel("Target model AUC")
    right.set_xlabel("Where the target model is scored (dotted line: chance)")


def explain(result, parameter):
    d, cv, te, tw = result["domain_auc"], result["cv_auc"], result["test_auc"], result["weighted_test_auc"]
    ess = result["weight_ess"]
    if te < cv:
        calc = f"{fmt(cv)} - {fmt(te)} = {fmt(cv - te)} below"
    else:
        calc = f"{fmt(te)} - {fmt(cv)} = {fmt(te - cv)} above"
    gain = tw - te
    if gain > 0.01:
        weighting = f"Weighting raises the test AUC by {fmt(gain)}."
    elif gain < -0.01:
        weighting = f"Weighting lowers the test AUC by {fmt(-gain)}, with an effective sample of {fmt(100 * ess, 1)}% of the rows."
    else:
        weighting = "Weighting changes the test AUC by less than 0.01."
    if d < 0.55 and abs(te - cv) < 0.03:
        reading = "Nothing separates the populations and the target model carries over."
    elif d < 0.55:
        reading = "The domain classifier sees nothing, yet the target model's score changed: the label rule moved, not the inputs."
    elif abs(te - cv) < 0.03:
        reading = "The populations separate easily, yet the target model's score carries over."
    else:
        reading = "The populations separate and the target model's score moved with them."
    interpretation = (f"Domain AUC is {fmt(d)} (shuffled labels give {fmt(result['shuffled_domain_auc'])}). The target model scores {fmt(te)} "
                      f"on test rows against {fmt(cv)} in cross-validation: {calc} cross-validation. {reading} {weighting}")
    steps = [
        f"Domain AUC: {fmt(d)} against {fmt(result['shuffled_domain_auc'])} with shuffled labels.",
        f"Test minus cross-validation: {fmt(te)} - {fmt(cv)} = {signed(te - cv)}.",
        f"Weighting gain on test: {fmt(tw)} - {fmt(te)} = {signed(gain)}.",
        f"Effective sample size of the weights: {fmt(100 * ess, 1)}% of the {result['train_rows']:,} training rows.",
    ]
    metrics = {"Domain AUC": fmt(d), "Cross-validation AUC": fmt(cv), "Test AUC, unweighted": fmt(te),
               "Test AUC, weighted": fmt(tw), "Weight effective sample": f"{fmt(100 * ess, 1)}%"}
    alt = (f"Left: bar of domain AUC {fmt(d)} with a dashed line at the shuffled-label value. Right: bars of the target model's "
           f"cross-validation AUC {fmt(cv)}, test AUC {fmt(te)} and weighted test AUC {fmt(tw)}, for the {SHORT[result['shift_type']].lower()} state.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for v, res in results.items():
        assert abs(res["shuffled_domain_auc"] - 0.5) < 0.02, f"shuffled-label domain AUC not near 0.5 at {v}"
        assert abs(res["cv_auc"] - results[0]["cv_auc"]) < 1e-9, "the training set must be the same in every state"
    none, cov, ts, flip = results[0], results[1], results[2], results[3]
    assert none["domain_auc"] < 0.55 and abs(none["test_auc"] - none["cv_auc"]) < 0.02, "no-shift state should carry over"
    assert flip["domain_auc"] < 0.55 and flip["cv_auc"] - flip["test_auc"] > 0.15, "sign flip should collapse test AUC unseen by domain AUC"
    assert fmt(flip["domain_auc"]) == "0.499" and fmt(flip["cv_auc"]) == "0.771" and fmt(flip["test_auc"]) == "0.576", "prediction text"
    assert 0.55 < flip["test_auc"] < 0.60 and flip["test_auc"] > 0.5, "'about 0.58' option and 'not below 0.5'"
    assert 0.7 < cov["domain_auc"] < 0.8 and cov["test_auc"] > cov["cv_auc"], "covariate shift: separable, test not worse"
    assert cov["weighting_gain"] > 0.01, "weighting should help under covariate shift here"
    assert abs(none["weighting_gain"]) < 0.01 and abs(flip["weighting_gain"]) < 0.01, "weighting does nothing in the two no-input-shift states"
    assert ts["domain_auc"] > 0.99 and abs(ts["test_auc"] - ts["cv_auc"]) < 0.02, "timestamp: separable, target unaffected"
    assert fmt(ts["domain_auc"]) == "1.000" and fmt(ts["test_auc"]) == "0.776" and fmt(ts["cv_auc"]) == "0.771", "answer text numbers"
    assert ts["weighting_gain"] < -0.03 and fmt(ts["weighted_test_auc"]) == "0.703", "weighting hurts under timestamp separation"
    assert ts["weight_ess"] < 0.01 and fmt(100 * ts["weight_ess"], 1) == "0.5", "effective sample 0.5% of rows"
