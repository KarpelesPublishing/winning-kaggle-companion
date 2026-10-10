"""Chapter 30: External Data and Leakage. An event-date join and a point-in-time (as-of) join of a revised indicator, by revision size."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from kaggle_companion.activities._common import clean

SEED = 30
REPLICATES = 100        # independent constructed series; every estimate is their mean
MONTHS, DAYS = 300, 30  # 300 months of 30 daily rows each
TRAIN_END, VALID_END = 100, 140   # months 0-99 train, 100-139 validate, 140-299 are the deployment period
LAG, REVISION_DELAY = 10, 15      # first release 10 days after the period ends, revised 15 days later
REVISED_SD = 0.1                  # the revised value is close to the truth, whatever the first release was


def generate(first_release_sd, seed):
    """A monthly indicator with a first release (noisy) and a later revision (close to the truth), and daily rows.
    Each daily row concerns the month that just ended, so its label depends on that month's true indicator."""
    rng = np.random.default_rng(seed)
    truth = np.zeros(MONTHS)
    for m in range(1, MONTHS):
        truth[m] = 0.6 * truth[m - 1] + rng.normal(0, 0.8)
    first = truth + rng.normal(0, first_release_sd, MONTHS)
    revised = truth + rng.normal(0, REVISED_SD, MONTHS)

    t = np.arange(MONTHS * DAYS)                    # absolute day of each row: the row's prediction cutoff
    month = t // DAYS
    period = np.maximum(month - 1, 0)               # the indicator period the row is about
    own = rng.normal(size=len(t))                   # a feature known inside the table
    y = truth[period] + 0.5 * own + rng.normal(0, 0.7, len(t))

    # Event-date join: the revised value of the row's period, which exists only weeks after the cutoff.
    event_date = revised[period]
    # As-of join: for each row, the latest period whose first release is out by the cutoff, in its latest published version.
    as_of = np.zeros(len(t))
    for age in (0, 1):                              # try the row's own period, then the one before it
        p = np.maximum(period - age, 0)
        released = DAYS * (p + 1) + LAG <= t
        revised_out = DAYS * (p + 1) + LAG + REVISION_DELAY <= t
        value = np.where(revised_out, revised[p], first[p])
        as_of = np.where(released & (as_of == 0), value, as_of)   # 0 means nothing eligible found yet
    return month, np.column_stack([own, event_date]), np.column_stack([own, as_of]), y


def one_series(first_release_sd, seed):
    month, X_event, X_asof, y = generate(first_release_sd, seed)
    train, valid, deploy = month < TRAIN_END, (month >= TRAIN_END) & (month < VALID_END), month >= VALID_END
    out = {}
    # Event-date pipeline: train and validate on revised values, then deploy where only as-of values exist.
    model = Ridge(1.0).fit(X_event[train], y[train])
    out["event_valid"] = r2_score(y[valid], model.predict(X_event[valid]))
    out["event_deploy"] = r2_score(y[deploy], model.predict(X_asof[deploy]))
    # As-of pipeline: the same point-in-time join in training, validation and deployment.
    model = Ridge(1.0).fit(X_asof[train], y[train])
    out["asof_valid"] = r2_score(y[valid], model.predict(X_asof[valid]))
    out["asof_deploy"] = r2_score(y[deploy], model.predict(X_asof[deploy]))
    return out


def run(first_release_sd):
    series = [one_series(first_release_sd, SEED * 1000 + i) for i in range(REPLICATES)]
    keys = ["event_valid", "event_deploy", "asof_valid", "asof_deploy"]
    return clean({
        "first_release_sd": first_release_sd,
        **{k: float(np.mean([s[k] for s in series])) for k in keys},
        "sd": {k: float(np.std([s[k] for s in series])) for k in keys},
        "replicates": REPLICATES, "lag_days": LAG, "revision_days": REVISION_DELAY,
        "deployment_rows": (MONTHS - VALID_END) * DAYS,
    })
# notebook-end


SPEC = {
    "chapter": 30,
    "chapter_title": "External Data and Leakage",
    "subtitle": "Join each row to the version of an external source that existed at that row's cutoff, in validation and in training.",
    "summary": ("An external indicator that is revised after its first release can be joined by event date or as of the prediction "
                "time. One demonstration measures how far each join's validation score sits from the score the model earns once "
                "only the versions available at the time exist, as the revisions grow."),
    "title": "Event-date join against point-in-time join, by revision size",
    "question": "How far does a validation score built on revised data drift from deployment, and when does the training join start to cost accuracy too?",
    "why": ("Adjusted and revised sources are common, and joining them by event date is the default of most merge code. The "
            "validation score then describes data that did not exist at prediction time, and the model learns to trust it too much."),
    "method": ("One hundred constructed series, each with 300 months of 30 daily rows. A monthly indicator follows a persistent process; "
               "its first release comes 10 days after the month ends with noise of the control size (standard deviation, "
               "the true indicator has standard deviation 1.0), and a revision 15 days later is close to the truth. Each daily row is about the month that just "
               "ended, and its label depends on that month's true value plus an internal feature. A ridge regression is trained on "
               "months 0 to 99 and validated on months 100 to 139. The event-date join uses the revised value for every row; the as-of "
               "join uses the latest version of the latest period published at the row's cutoff. Both models are then scored on "
               "months 140 to 299 (4,800 rows) with the as-of join, the only data that exist at prediction time."),
    "control": {"key": "first_release_sd", "label": "Noise in the first release (standard deviation of its error)",
                "values": [0, 0.3, 0.6, 1.0], "default": 0.6,
                "value_labels": ["0: first release exact", "0.3", "0.6", "1.0: first release mostly noise"]},
    "source_section": "A Version-Aware Join",
    "symbols": ("c is a row's prediction cutoff, v ranges over the published versions of an indicator, p_v is the time version v became "
                "public, and v*(c) is the version the row may use: the latest one published at or before c."),
    "explanation": ("The event-date join hands every row the revised value, including rows whose cutoff came before even the first "
                    "release. Its validation score is high because the indicator is nearly the truth. At deployment only the first "
                    "release (or the previous month, before it is out) exists, so the same model scores far lower, and it was fitted "
                    "to trust a column that is cleaner than the one it receives. The as-of join reproduces deployment conditions in "
                    "training and in validation, so its estimate lands near the deployment score."),
    "application": ("Store event period and publication timestamp for every external source, join each row to the latest version "
                    "published at or before its cutoff in training, validation and the final pipeline, and count rows with nothing "
                    "eligible. Compare against the model without the source."),
    "assumptions": ("Constructed series; the indicator, the lags and the revision process are declared in the method and change the "
                    "sizes below. Even when the first release is exact, rows whose cutoff precedes it have only the previous "
                    "month, so the event-date estimate is still too high. Single validation windows are noisy (the standard deviation "
                    "over series is shown), which is why 100 series are averaged. A 40-month validation window also reads a little below "
                    "the 160-month deployment window for any model, because a slow indicator spans less of its range in a short window; "
                    "that is why the as-of estimate is close to, not exactly at, its deployment score."),
    "prediction": "With a first release of noise 0.6, how does the validation score of the event-date join compare with its score at deployment?",
    "prediction_options": ["About the same", "About 0.24 higher than at deployment", "Lower than at deployment"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "The event-date join validates at 0.691 but scores 0.450 at deployment, 0.241 lower; the as-of join validates at 0.468 and scores 0.494.",
        "incorrect": "The event-date join validates at 0.691 but scores 0.450 at deployment, 0.241 lower, because validation used revised values. The as-of join validates at 0.468 and scores 0.494.",
    },
    "check": "When the first release has no noise at all, the event-date join still misleads. Why, and does it cost accuracy at deployment?",
    "answer": ("Rows whose cutoff falls before the first release have no value for their own period, so the event-date join gives them "
               "information they could not have had: validation 0.691 against 0.553 at deployment. The model fitted on it still "
               "deploys within 0.010 of the as-of model, so at zero noise the join misstates the estimate more than it harms the model; "
               "the harm to the model grows with the revision size (0.146 at noise 1.0)."),
    "provenance": "Constructed example: one hundred seeded synthetic series with a declared release and revision schedule and a ridge regression, measured by the chapter activity.",
    "apply": [
        "Record both the event period and the first-publication timestamp of every external source, and keep each vintage you download.",
        "Join every row to the latest version published at or before its cutoff, in training and in validation, never by event date alone.",
        "Count rows with no eligible version and decide the fallback before modelling; do not fill them from a later revision.",
        "Compare the new source against the unchanged baseline on the same point-in-time folds, and keep failed joins in the log.",
    ],
    "honesty": "Constructed series with declared release and revision schedules; the sizes of these effects are properties of this generator, not a competition result.",
}

EQUATIONS = [{"tex": r"v^{*}(c) = \operatorname*{arg\,max}_{v:\; p_v \le c} \; p_v",
              "alt": "v star of c equals the version v with the latest publication time p v among versions published at or before the cutoff c",
              "basis": "The as-of rule the chapter states in words (The Leakage Risk with External Data, A Version-Aware Join); written as an equation by the activity."}]
NCOLS = 1
HEIGHT = 4.4
JOINS = [("event", "Event-date join\n(revised value)"), ("asof", "As-of join\n(point in time)")]


def draw(ax, result, parameter):
    xs = np.arange(len(JOINS))
    valid = [result[k + "_valid"] for k, _ in JOINS]
    deploy = [result[k + "_deploy"] for k, _ in JOINS]
    ax.bar(xs - 0.2, valid, width=0.4, color=COLORS["light"], edgecolor=COLORS["ink"], lw=0.6, label="Validation estimate",
           yerr=[result["sd"][k + "_valid"] for k, _ in JOINS], error_kw={"ecolor": COLORS["ink"], "lw": 1.0, "capsize": 3})
    ax.bar(xs + 0.2, deploy, width=0.4, color=COLORS["teal"], edgecolor=COLORS["ink"], lw=0.6, label="Score at deployment",
           yerr=[result["sd"][k + "_deploy"] for k, _ in JOINS], error_kw={"ecolor": COLORS["ink"], "lw": 1.0, "capsize": 3})
    for x, a, b, (k, _) in zip(xs, valid, deploy, JOINS):
        ax.text(x - 0.2, a + result["sd"][k + "_valid"] + 0.01, fmt(a), ha="center", va="bottom", fontsize=10)
        ax.text(x + 0.2, b + result["sd"][k + "_deploy"] + 0.01, fmt(b), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(xs, [name for _, name in JOINS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R-squared")
    ax.set_xlabel(f"How the indicator is joined, first-release noise {parameter}\nError bars: one standard deviation over {result['replicates']} series")
    ax.legend(loc="upper right", frameon=False, fontsize=10)


def explain(result, parameter):
    ev, ed, av, ad = result["event_valid"], result["event_deploy"], result["asof_valid"], result["asof_deploy"]
    inflation = float(fmt(ev)) - float(fmt(ed))
    asof_gap = float(fmt(av)) - float(fmt(ad))
    harm = float(fmt(ad)) - float(fmt(ed))
    if harm >= 0.0005:
        cost = f"Training on the event-date join also costs {fmt(harm)} of deployment R-squared"
    elif harm <= -0.0005:
        cost = f"Here the event-date model deploys {fmt(-harm)} better than the as-of model"
    else:
        cost = "Training on the event-date join costs nothing measurable at deployment"
    interpretation = (
        f"With first-release noise {parameter}, the event-date join validates at {fmt(ev)} but scores {fmt(ed)} at deployment, "
        f"{fmt(ev)} - {fmt(ed)} = {fmt(inflation)} of inflation. The as-of join validates at {fmt(av)} and scores {fmt(ad)}, "
        f"a difference of {signed(asof_gap)}. {cost} ({fmt(ad)} for the as-of model against {fmt(ed)}).")
    steps = [
        f"Event-date join: {fmt(ev)} - {fmt(ed)} = {fmt(inflation)} (validation minus deployment).",
        f"As-of join: {fmt(av)} - {fmt(ad)} = {signed(asof_gap)}.",
        f"Cost of the event-date training join at deployment: {fmt(ad)} - {fmt(ed)} = {signed(harm)}.",
        f"Spread of the as-of validation estimate over {result['replicates']} series: one standard deviation is {fmt(result['sd']['asof_valid'])}.",
    ]
    metrics = {"Event-date validation": fmt(ev), "Event-date at deployment": fmt(ed),
               "As-of validation": fmt(av), "As-of at deployment": fmt(ad), "Validation inflation": fmt(inflation)}
    alt = (f"Paired bars of validation R-squared and deployment R-squared for the event-date and as-of joins at first-release noise {parameter}, "
           f"with error bars; the event-date join validates {fmt(inflation)} above its deployment score.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for s, res in results.items():
        assert res["event_valid"] - res["event_deploy"] > 0.08, f"event-date join should be inflated at {s}"
        assert abs(res["asof_valid"] - res["asof_deploy"]) < 0.06, f"as-of estimate should be near deployment at {s}"
        assert res["asof_deploy"] >= res["event_deploy"] - 0.005, f"as-of model should not deploy worse at {s}"
    infl = [results[s]["event_valid"] - results[s]["event_deploy"] for s in (0, 0.3, 0.6, 1.0)]
    assert infl[0] < infl[1] < infl[2] < infl[3], "inflation should grow with revision size"
    harm = [results[s]["asof_deploy"] - results[s]["event_deploy"] for s in (0, 0.3, 0.6, 1.0)]
    assert harm[0] < 0.03 and harm[3] > 0.08 and harm[0] < harm[1] < harm[2] < harm[3], "harm to the model grows with revision size"
    r = results[0.6]
    assert fmt(r["event_valid"]) == "0.691" and fmt(r["event_deploy"]) == "0.450", "prediction feedback numbers"
    assert fmt(r["asof_valid"]) == "0.468" and fmt(r["asof_deploy"]) == "0.494", "prediction feedback numbers"
    assert fmt(r["event_valid"] - r["event_deploy"]) == "0.241", "prediction feedback gap"
    z = results[0]
    assert fmt(z["event_valid"]) == "0.691" and fmt(z["event_deploy"]) == "0.553", "check answer numbers"
    assert fmt(z["asof_deploy"] - z["event_deploy"]) == "0.010", "check answer harm at zero noise"
    assert fmt(results[1.0]["asof_deploy"] - results[1.0]["event_deploy"]) == "0.146", "check answer harm at 1.0"
