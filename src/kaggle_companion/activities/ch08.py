"""Chapter 8: The Five-Stage Feature Engineering Process. Raw, log and smearing-corrected target policies under two metrics."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from kaggle_companion.activities._common import clean

SEED = 8
REPLICATES = 8        # independent constructed datasets; every score is their mean
N_TRAIN, N_NEW = 6000, 8000


def generate(n, sigma, rng):
    """Five features; the target is multiplicative: y = exp(f(x) + sigma * noise). exp(f) is its median."""
    X = rng.normal(size=(n, 5))
    f = 3.0 + 0.8 * X[:, 0] + 0.5 * X[:, 1] - 0.4 * X[:, 2] + 0.3 * X[:, 0] * X[:, 1]
    y = np.exp(f + sigma * rng.normal(size=n))
    return X, y, f


def model():
    return HistGradientBoostingRegressor(max_iter=100, learning_rate=0.08, max_depth=3, min_samples_leaf=30, random_state=0)


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def rmsle(y, p):
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(np.maximum(p, 0))) ** 2)))


def one_dataset(sigma, seed):
    rng = np.random.default_rng(seed)
    X, y, _ = generate(N_TRAIN, sigma, rng)
    X_new, y_new, f_new = generate(N_NEW, sigma, rng)

    raw = model().fit(X, y).predict(X_new)                     # policy 1: fit y directly, squared error on the raw scale

    log_y = np.log1p(y)                                         # policy 2: fit log1p(y), then undo the transform
    m_new = model().fit(X, log_y).predict(X_new)                # predicted mean of log1p(y)
    naive = np.expm1(m_new)                                     # exponentiate the mean log prediction
    oof = np.zeros(N_TRAIN)                                     # policy 3: smearing. Cross-fitted log residuals e_i give
    for a, b in KFold(5, shuffle=True, random_state=0).split(X):  # mean(expm1(m + e_i)) = exp(m) * mean(exp(e_i)) - 1
        oof[b] = model().fit(X[a], log_y[a]).predict(X[b])
    smear = np.exp(m_new) * np.mean(np.exp(log_y - oof)) - 1.0

    policies = {"raw_fit": raw, "log_naive": naive, "log_smeared": smear, "constant": np.full(N_NEW, y.mean())}
    out = {k: {"rmse": rmse(y_new, p), "rmsle": rmsle(y_new, p), "mean_prediction": float(np.mean(p))}
           for k, p in policies.items()}
    out["known_mean"] = {"rmse": rmse(y_new, np.exp(f_new + sigma ** 2 / 2)), "rmsle": rmsle(y_new, np.exp(f_new + sigma ** 2 / 2))}
    out["known_median"] = {"rmse": rmse(y_new, np.exp(f_new)), "rmsle": rmsle(y_new, np.exp(f_new))}
    out["mean_target"] = float(np.mean(y_new))
    return out


def run(sigma):
    sets = [one_dataset(sigma, SEED * 100 + i) for i in range(REPLICATES)]
    result = {"sigma": sigma, "replicates": REPLICATES, "training_rows": N_TRAIN, "new_rows": N_NEW,
              "mean_target": float(np.mean([s["mean_target"] for s in sets]))}
    for name in ("raw_fit", "log_naive", "log_smeared", "constant", "known_mean", "known_median"):
        result[name] = {m: float(np.mean([s[name][m] for s in sets])) for m in sets[0][name]}
        result[name]["rmse_per_set"] = [s[name]["rmse"] for s in sets]
        result[name]["rmsle_per_set"] = [s[name]["rmsle"] for s in sets]
    return clean(result)
# notebook-end


SPEC = {
    "chapter": 8,
    "chapter_title": "The Five-Stage Feature Engineering Process",
    "subtitle": "A target transform changes the quantity a model optimizes, so the official metric decides how to undo it.",
    "summary": ("Squared error on log(y) optimizes a different quantity from raw-scale RMSE. One demonstration measures a raw-target fit, "
                "a log fit exponentiated back, and a smearing-corrected log fit under both metrics as the multiplicative noise grows."),
    "title": "Three target policies under raw-scale RMSE and log-scale RMSLE",
    "question": ("When the target is multiplicative and skewed, does fitting log1p(y) help, and does the answer depend on the metric "
                 "and on how the prediction is brought back to the raw scale?"),
    "why": ("A skewed target tempts a log transform, but the chapter warns that exponentiating a mean log prediction need not recover the "
            "raw-scale conditional mean. Measuring both metrics on the same new rows shows which policy each metric rewards."),
    "method": ("A constructed regression with 5 features and a log-normal target, y = exp(f(x) + s * noise), where the noise spread s is the "
               "control. A histogram gradient boosting regressor is trained on 6,000 rows under three policies: fit y directly; fit log1p(y) "
               "and take expm1 of the prediction; and fit log1p(y) with a smearing correction that averages expm1(m + e) over cross-fitted "
               "log residuals e. A constant (the training mean) is the simple predictor. Each is scored on 8,000 new rows from the same generator by "
               "raw-scale RMSE and by RMSLE, the square root of the mean squared difference of log1p values. Every score is the mean over "
               "8 constructed datasets."),
    "control": {"key": "sigma", "label": "Multiplicative noise spread s (standard deviation of the log-scale noise)",
                "values": [0.3, 0.8, 1.2, 1.6], "default": 0.8,
                "value_labels": ["0.3: mild skew", "0.8", "1.2", "1.6: heavy tail"]},
    "source_section": "Stage 4: Target Transforms",
    "symbols": ("y is the target, m(x) the model's predicted mean of log1p(y), e_i the cross-fitted residuals on the training rows and "
                "s the log-scale noise spread. RMSE is computed on y, RMSLE on log1p(y)."),
    "explanation": ("Squared error on log1p(y) targets a typical value, close to the median, while raw-scale RMSE targets the mean. With "
                    "log-normal noise the mean exceeds the median by about exp(s squared over 2), so the plain back-transform predicts "
                    "low on average, and more so as s grows; the fitted model falls a little further short than that factor alone. "
                    "The smearing correction brings the average prediction most of the way back (above 80% of the target mean "
                    "here) and wins on RMSE, but "
                    "it moves the prediction away from the log-scale median and loses on RMSLE. Fitting the raw target needs no inverse, but "
                    "it chases the extreme values."),
    "application": ("Read the official metric, fit the target on a scale you can justify, choose the inverse policy that suits that "
                    "metric, and compare against a raw-target fit and a simple predictor on the same assessment rows."),
    "assumptions": ("Constructed data with log-normal noise, one boosting model and 6,000 training rows. Raw-scale RMSE here is dominated "
                    "by noise no model can remove (the known-mean line), so the raw fit and the plain log fit are within about 3% of each other "
                    "for s of 0.8 and above, and their order is not stable; no claim is made about which of those two wins on RMSE. "
                    "A histogram gradient boosting regressor stands in for LightGBM."),
    "prediction": "At noise spread 0.8, which target policy has the lowest RMSLE, and which has the lowest raw-scale RMSE?",
    "prediction_options": ["The plain log fit wins both metrics", "The plain log fit wins RMSLE; the smearing-corrected log fit wins RMSE",
                           "The raw-target fit wins both metrics"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "The plain log fit has RMSLE 0.749 against 0.804 for smearing, while smearing has RMSE 203.4 against 213.1 for the plain log fit.",
        "incorrect": "The plain log fit has RMSLE 0.749 against 0.804 for smearing, while smearing has RMSE 203.4 against 213.1. The metric decides the inverse policy.",
    },
    "check": "Why does the plain log fit win on RMSLE yet predict an average well below the average target?",
    "answer": ("It targets the log-scale typical value, which is below the raw-scale mean when noise is multiplicative. At s = 0.8 its "
               "predictions average 39.2 against a target mean of 58.6, and the raw-target fit's RMSLE is 1.034, against 0.749 for the log fit. "
               "RMSLE rewards the typical value; raw-scale RMSE rewards the mean, which is why smearing helps there."),
    "provenance": "Constructed example: seeded synthetic log-normal targets and a boosted regressor, measured by the chapter activity.",
    "apply": [
        "Read which metric scores the competition before deciding whether to transform the target at all.",
        "If the metric is RMSLE or another log-scale loss, fit log1p(y) and invert it plainly; if it is raw-scale RMSE, test a smearing-style correction.",
        "Check the average of your predictions against the average of the training target; a large shortfall means the inverse policy is off.",
        "Keep a raw-target fit and a constant predictor as comparators, and choose the transform inside development, not on the assessment rows.",
    ],
    "honesty": ("Constructed data with log-normal noise; the sizes of these gaps are properties of this generator. No policy wins every metric, "
                "and with this multiplicative noise the raw-target fit is never a clear winner."),
}

EQUATIONS = [{"tex": r"\mathrm{E}[y\mid x]=e^{\mu(x)+s^2/2},\qquad \hat y_{\mathrm{smear}}=e^{m(x)}\cdot\frac{1}{n}\sum_i e^{e_i}-1",
              "alt": "The conditional mean of y is e to the mu of x plus s squared over 2. The smeared prediction is e to the m of x times the average of e to the e i, minus 1",
              "basis": "The activity's own log-normal mean and smearing correction; the chapter's Stage 4 makes the same point in words and has no display equation."}]
NCOLS = 2
HEIGHT = 4.4
POLICIES = [("raw_fit", "Raw\nfit", COLORS["terracotta"]), ("log_naive", "Log fit,\nplain", COLORS["teal"]),
            ("log_smeared", "Log fit,\nsmeared", COLORS["gold"]), ("constant", "Constant\nmean", COLORS["light"])]


def draw(axes, result, parameter):
    for ax, metric, known, label, digits in ((axes[0], "rmse", "known_mean", "RMSE on new rows (raw scale)", 0),
                                             (axes[1], "rmsle", "known_median", "RMSLE on new rows (log scale)", 3)):
        xs = list(range(len(POLICIES)))
        ax.bar(xs, [result[k][metric] for k, _, _ in POLICIES], width=0.6, color=[c for _, _, c in POLICIES],
               edgecolor=COLORS["ink"], lw=0.6)
        top = 0
        for x, (k, _, _) in zip(xs, POLICIES):
            dots = result[k][metric + "_per_set"]
            ax.scatter([x + (i - (len(dots) - 1) / 2) * 0.05 for i in range(len(dots))], dots, s=12, color=COLORS["ink"],
                       zorder=3, label="One dataset" if (x == 0 and ax is axes[0]) else None)
            high = max(max(dots), result[k][metric])
            top = max(top, high)
            ax.text(x, high + 0.015 * top, fmt(result[k][metric], digits), ha="center", va="bottom", fontsize=10)
        ref_name = "Known conditional mean" if metric == "rmse" else "Known conditional median"
        ax.axhline(result[known][metric], color=COLORS["navy"], ls=(0, (4, 3)), lw=1.4,
                   label=f"{ref_name}: {fmt(result[known][metric], digits)}")
        ax.set_xticks(xs, [name for _, name, _ in POLICIES])
        ax.set_ylim(0, top * 1.38)
        ax.set_ylabel(label)
        ax.set_xlabel(f"Target policy, noise spread {parameter}")
        ax.legend(loc="upper left", frameon=False, fontsize=10)


def explain(result, parameter):
    raw, nv, sm, ct = result["raw_fit"], result["log_naive"], result["log_smeared"], result["constant"]
    share = nv["mean_prediction"] / result["mean_target"]
    rmse_gain = nv["rmse"] - sm["rmse"]
    rmsle_gain = sm["rmsle"] - nv["rmsle"]
    interpretation = (
        f"At noise spread {parameter}, the plain log fit scores RMSLE {fmt(nv['rmsle'])} (raw-target fit {fmt(raw['rmsle'])}) but its "
        f"predictions average {fmt(nv['mean_prediction'], 1)} against a target mean of {fmt(result['mean_target'], 1)}, so "
        f"{fmt(nv['mean_prediction'], 1)} / {fmt(result['mean_target'], 1)} = {fmt(share)} of the mean is recovered. The smearing correction "
        f"lowers RMSE from {fmt(nv['rmse'], 1)} to {fmt(sm['rmse'], 1)} and raises RMSLE from {fmt(nv['rmsle'])} to {fmt(sm['rmsle'])}. "
        f"The raw-target fit has RMSE {fmt(raw['rmse'], 1)} and the constant {fmt(ct['rmse'], 1)}.")
    steps = [
        f"Mean recovered by the plain back-transform: {fmt(nv['mean_prediction'], 1)} / {fmt(result['mean_target'], 1)} = {fmt(share)}.",
        f"Gain of smearing on RMSE: {fmt(nv['rmse'], 1)} - {fmt(sm['rmse'], 1)} = {signed(rmse_gain, 1)} (positive means smearing is better).",
        f"Cost of smearing on RMSLE: {fmt(sm['rmsle'])} - {fmt(nv['rmsle'])} = {signed(rmsle_gain)} (positive means smearing is worse).",
        f"Raw-target fit against the plain log fit on RMSLE: {fmt(raw['rmsle'])} - {fmt(nv['rmsle'])} = {signed(raw['rmsle'] - nv['rmsle'])}.",
    ]
    metrics = {"RMSLE, plain log fit": fmt(nv["rmsle"]), "RMSLE, raw-target fit": fmt(raw["rmsle"]),
               "RMSE, plain log fit": fmt(nv["rmse"], 1), "RMSE, smeared log fit": fmt(sm["rmse"], 1),
               "RMSE, raw-target fit": fmt(raw["rmse"], 1), "Mean prediction, plain log fit": fmt(nv["mean_prediction"], 1)}
    alt = (f"Two bar charts at noise spread {parameter} comparing a raw-target fit, a plain log fit, a smeared log fit and a constant: raw-scale RMSE "
           f"on the left with a known-mean reference line, RMSLE on the right with a known-median reference line.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for s, res in results.items():
        nv, sm, raw, ct = res["log_naive"], res["log_smeared"], res["raw_fit"], res["constant"]
        assert sm["rmse"] < nv["rmse"] * 0.995, f"smearing should lower RMSE at s={s}"
        assert raw["rmsle"] > nv["rmsle"] + 0.25, f"raw fit should lose clearly on RMSLE at s={s}"
        assert nv["rmsle"] < ct["rmsle"] and sm["rmse"] < ct["rmse"], f"fitted policies should beat the constant at s={s}"
        assert nv["mean_prediction"] < res["mean_target"], f"plain back-transform should predict low at s={s}"
        if s >= 0.8:
            assert sm["rmsle"] > nv["rmsle"] + 0.03, f"smearing should cost RMSLE at s={s}"
            assert abs(raw["rmse"] / nv["rmse"] - 1) < 0.03, f"raw and plain log RMSE should be within 3% at s={s}"
    for s, res in results.items():
        share, smear_share = (res[k]["mean_prediction"] / res["mean_target"] for k in ("log_naive", "log_smeared"))
        assert share < np.exp(-s ** 2 / 2), f"plain fit should fall short of the median-mean factor at s={s}"
        assert 0.8 < smear_share < 1.0 and smear_share > share, f"smearing should recover most, not all, of the mean at s={s}"
    shares = [results[s]["log_naive"]["mean_prediction"] / results[s]["mean_target"] for s in sorted(results)]
    assert all(a > b for a, b in zip(shares, shares[1:])), "recovered share of the mean should shrink as noise grows"
    r = results[0.8]
    assert fmt(r["log_naive"]["rmsle"]) == "0.749" and fmt(r["log_smeared"]["rmsle"]) == "0.804", "prediction feedback numbers"
    assert fmt(r["log_smeared"]["rmse"], 1) == "203.4" and fmt(r["log_naive"]["rmse"], 1) == "213.1", "prediction feedback numbers"
    assert fmt(r["log_naive"]["mean_prediction"], 1) == "39.2" and fmt(r["mean_target"], 1) == "58.6", "answer numbers"
    assert fmt(r["raw_fit"]["rmsle"]) == "1.034", "answer numbers"
    assert min(r["log_naive"]["rmsle"], r["log_smeared"]["rmsle"], r["raw_fit"]["rmsle"]) == r["log_naive"]["rmsle"]
    assert min(r["log_naive"]["rmse"], r["log_smeared"]["rmse"], r["raw_fit"]["rmse"]) == r["log_smeared"]["rmse"]
