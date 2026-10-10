"""Chapter 41: Online Learning. Frozen, rolling-refit and incremental policies replayed on a drifting stream, by label delay."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np

from kaggle_companion.activities._common import clean

SEED = 41
REPLICATES = 20         # independent streams per delay; results are means over them
STEPS, WARM, SELECT = 2000, 300, 500   # prediction origins run from WARM; the first SELECT origins choose each policy's setting
FEATURES, RIDGE = 3, 1.0
DRIFT_NOISE, DRIFT_PULL = 0.12, 0.02   # the drifting part wanders and is pulled back toward zero
WINDOWS = (30, 60, 120, 240)           # candidate rolling-window lengths, in released rows
DISCOUNTS = (0.95, 0.98, 0.99, 0.995)  # candidate forgetting factors for the incremental policy
POINTS = 40                            # resolution of the cumulative error curves


def make_stream(rng):
    """Row t has features x_t and label y_t = w_t . x_t + noise, where w_t = a stable part + a drifting part."""
    X = rng.normal(size=(STEPS, FEATURES))
    stable = rng.normal(0, 1, FEATURES)                    # the part of the signal a frozen model can keep
    spread = DRIFT_NOISE / np.sqrt(1 - (1 - DRIFT_PULL) ** 2)   # long-run spread of the drifting part
    drift = np.zeros((STEPS, FEATURES))
    drift[0] = rng.normal(0, spread, FEATURES)
    for t in range(1, STEPS):
        drift[t] = drift[t - 1] * (1 - DRIFT_PULL) + DRIFT_NOISE * rng.normal(size=FEATURES)
    w = stable + drift
    return X, (w * X).sum(axis=1) + rng.normal(0, 1, STEPS)


def ridge(X, y):
    return np.linalg.solve(X.T @ X + RIDGE * np.eye(FEATURES), X.T @ y)


def replay(X, y, delay):
    """Squared error at every origin t >= WARM for each policy. Row i is released at i + delay and usable only at origins t > i + delay."""
    origins = range(WARM, STEPS)
    errors = {"frozen": None}
    frozen = ridge(X[:WARM - delay], y[:WARM - delay])           # fitted once, on rows released before the first origin
    errors["frozen"] = (y[WARM:] - X[WARM:] @ frozen) ** 2
    for window in WINDOWS:
        eligible, ignoring_delay = [], []
        for t in origins:
            hi = t - delay                                         # rows 0 .. hi-1 are released before origin t
            w = ridge(X[max(0, hi - window):hi], y[max(0, hi - window):hi])
            eligible.append((y[t] - X[t] @ w) ** 2)
            w = ridge(X[max(0, t - window):t], y[max(0, t - window):t])   # INVALID: also uses labels not yet released
            ignoring_delay.append((y[t] - X[t] @ w) ** 2)
        errors[("rolling", window)], errors[("invalid", window)] = np.array(eligible), np.array(ignoring_delay)
    for discount in DISCOUNTS:                                     # incremental: discounted ridge, one released row at a time
        A, b = np.zeros((FEATURES, FEATURES)), np.zeros(FEATURES)
        for i in range(WARM - delay):
            A, b = discount * A + np.outer(X[i], X[i]), discount * b + X[i] * y[i]
        errs = []
        for t in origins:
            i = t - delay - 1                                      # the row released since the previous origin
            if i >= WARM - delay:
                A, b = discount * A + np.outer(X[i], X[i]), discount * b + X[i] * y[i]
            errs.append((y[t] - X[t] @ np.linalg.solve(A + RIDGE * np.eye(FEATURES), b)) ** 2)
        errors[("incremental", discount)] = np.array(errs)
    return errors


def pick(errors, kind, options):
    """Choose the setting with the lowest error on the first SELECT origins; report it on the later origins only."""
    best = min(options, key=lambda o: errors[(kind, o)][:SELECT].mean())
    return best, errors[(kind, best)][SELECT:]


def run(delay):
    scores = {k: [] for k in ("frozen", "rolling", "incremental", "invalid")}
    settings = {"rolling": [], "incremental": []}
    curves = {k: [] for k in scores}
    for r in range(REPLICATES):
        X, y = make_stream(np.random.default_rng(SEED * 100 + r))
        errors = replay(X, y, delay)
        later = {"frozen": errors["frozen"][SELECT:]}
        for kind, options in (("rolling", WINDOWS), ("incremental", DISCOUNTS), ("invalid", WINDOWS)):
            chosen, later[kind] = pick(errors, kind, options)
            if kind in settings:
                settings[kind].append(chosen)
        for k, e in later.items():
            scores[k].append(e.mean())
            running = np.cumsum(e) / np.arange(1, len(e) + 1)
            curves[k].append(running[np.linspace(0, len(e) - 1, POINTS).astype(int)])
    mean = {k: float(np.mean(v)) for k, v in scores.items()}
    return clean({
        "delay": delay, "mse": mean,
        "gain_rolling": mean["frozen"] - mean["rolling"], "gain_incremental": mean["frozen"] - mean["incremental"],
        "gain_invalid": mean["frozen"] - mean["invalid"],
        "selected_window": float(np.mean(settings["rolling"])), "selected_discount": float(np.mean(settings["incremental"])),
        "curve_steps": np.linspace(1, STEPS - WARM - SELECT, POINTS), "curves": {k: np.mean(v, axis=0) for k, v in curves.items()},
        "noise_floor": 1.0, "replicates": REPLICATES, "scored_origins": STEPS - WARM - SELECT,
    })
# notebook-end


SPEC = {
    "chapter": 41,
    "chapter_title": "Online Learning",
    "subtitle": "Replay the timeline: a label can only update the model after it is released.",
    "summary": ("An adaptive model is only as good as the labels it is allowed to see. One demonstration replays a drifting stream "
                "with delayed labels and compares a frozen model, a rolling refit and an incremental update, all using released labels "
                "only, against a replay that ignores the delay."),
    "title": "What label delay does to an adaptive model",
    "question": "How much of an adaptive model's advantage over a frozen model survives as labels arrive later, and how wrong is a replay that ignores the delay?",
    "why": ("In a live stream the newest labels are not yet known. A replay that fits on the most recent rows by event time "
            "reports a gain the system cannot deliver; the chapter asks for release time to decide eligibility."),
    "method": ("A constructed stream of 2,000 rows with three features and a linear signal whose coefficients are a fixed stable part "
               "plus a drifting part that is pulled back toward zero; label noise has variance 1. The control is the delay between a row and the release of its label. "
               "At each origin from step 300 the model predicts, scored by squared error. A frozen ridge is fitted once on the labels "
               "released before step 300; the rolling policy refits ridge on the latest released rows; the incremental policy "
               "updates a discounted ridge one released row at a time. The window length and discount are chosen on the first 500 origins "
               "and every score is reported on the later 1,200. A fourth policy fits the rolling window on the latest rows by event time, "
               "ignoring the delay; it is invalid. Results are means over 20 streams."),
    "control": {"key": "delay", "label": "Label delay (steps between a row and its label's release)",
                "values": [0, 10, 40, 100], "default": 40,
                "value_labels": ["0", "10", "40", "100"]},
    "source_section": "A Delayed-Label Contrast",
    "symbols": ("t is the prediction origin, i a training row, d the label delay, w-hat_t the coefficients fitted for origin t and x_t its "
                "features; row i is eligible at origin t only if i + d < t."),
    "explanation": ("With no delay the newest rows describe the current coefficients and a short window tracks the drift. With a "
                    "delay every released row is already d steps old, so the model lags the drift, and the best window is "
                    "longer to average out noise. A frozen model still holds the stable part of the signal, so at long delays the "
                    "adaptive gain nearly vanishes. The invalid replay does not lag, so it reports the zero-delay result at "
                    "every delay: an estimate that stops describing what a live system would score."),
    "application": ("Store release time beside event time, fit each origin only on rows with release time before it, choose window "
                    "and update schedule on earlier origins, and compare against a frozen model under the same replay."),
    "assumptions": ("Constructed linear stream with stable coefficients plus mean-reverting drift, ridge models and fixed settings grids. The frozen model is fitted "
                    "once on the first 300 steps' released labels. Delays are in steps; the shape of the curves depends on the drift speed relative to the delay."),
    "prediction": ("With labels released instantly, a rolling refit beats a frozen model by 1.05 in squared error. If labels arrive 40 steps late "
                   "(window re-selected), how much of that gain remains?"),
    "prediction_options": ["Nearly all of it", "About two thirds of it", "Less than a third of it"],
    "prediction_answer": 2,
    "prediction_feedback": {
        "correct": "A gain of 0.30 remains out of 1.05: the delay makes every released row stale by 40 steps, and the window can only average noise, not recover recency.",
        "incorrect": "Only 0.30 of the 1.05 gain remains, less than a third: the delay makes every released row stale by 40 steps, and a longer window averages noise but cannot recover recency.",
    },
    "check": "A replay that fits on the latest rows by event time scores 1.52 at every delay. Why is that invalid, and what does the selected window length tell you?",
    "answer": ("It uses labels that a live system would not have received yet, so it never lags the drift and reports the no-delay score "
               "(1.52) even at a delay of 100, where the eligible rolling policy scores 2.48 and a frozen model 2.70. The selected window grows from "
               "30 rows with no delay to about 213 at delay 100, because stale rows are only useful averaged over more of them."),
    "provenance": "Constructed example: a seeded synthetic drifting stream and ridge models, measured by the chapter activity.",
    "apply": [
        "Write the timeline first: observation time, prediction origin and label-release time, with eligibility defined as release before origin.",
        "Replay frozen, incremental and rolling policies on the same stream; if a policy only wins when it can see unreleased labels, it has not won.",
        "Select window length and update schedule on earlier origins and report on later ones; expect longer windows as delay grows.",
        "Report the gain over a frozen model at the delay of the real system, not at zero delay.",
    ],
    "honesty": ("Constructed stream; the incremental policy beat the rolling refit at every delay here, a property of this drift and these "
                "settings grids. At a delay of 100 the rolling refit's edge over the frozen model is about as large as the variation between "
                "seed bases (under one other seed base it reversed); the incremental policy kept its edge."),
}

EQUATIONS = [{"tex": r"\hat{y}_t = x_t^{\top}\hat{w}_t,\qquad \hat{w}_t \text{ fitted on rows } \{\, i : i + d < t \,\}",
              "alt": "The prediction at origin t is x t transposed times w hat t, where w hat t is fitted only on rows i with i plus d less than t",
              "basis": "The chapter's strict convention that label release time is before the prediction origin (Write the Timeline First); the chapter has no display equation."}]
NCOLS = 2
HEIGHT = 4.4
POLICIES = [("frozen", "Frozen", COLORS["grey"]), ("rolling", "Rolling refit", COLORS["teal"]),
            ("incremental", "Incremental", COLORS["navy"]), ("invalid", "Invalid replay\n(ignores delay)", COLORS["terracotta"])]


def draw(axes, result, parameter):
    left, right = axes
    for key, name, color in POLICIES:
        style = (0, (4, 3)) if key == "invalid" else "-"
        left.plot(result["curve_steps"], result["curves"][key], color=color, lw=1.8, ls=style, label=name.replace("\n", " "))
    left.axhline(result["noise_floor"], color=COLORS["ink"], ls=":", lw=1.2, label="Noise floor: 1.00")
    left.set_xlabel("Prediction origins scored so far")
    left.set_ylabel("Running mean squared error")
    peak = max(max(c) for c in result["curves"].values())
    left.set_ylim(0.8, peak + 1.1)                          # headroom so the legend sits above every curve
    left.legend(loc="upper right", frameon=False, fontsize=10, ncol=2)

    xs = list(range(len(POLICIES)))
    values = [result["mse"][k] for k, _, _ in POLICIES]
    right.bar(xs, values, color=[c for _, _, c in POLICIES], edgecolor=COLORS["ink"], lw=0.6, width=0.6)
    for x, v in zip(xs, values):
        right.text(x, v + 0.04, fmt(v), ha="center", va="bottom", fontsize=10)
    right.axhline(result["noise_floor"], color=COLORS["ink"], ls=":", lw=1.2)
    right.set_xticks(xs, [name for _, name, _ in POLICIES], fontsize=10)
    right.set_ylim(0, max(values) * 1.2)
    right.set_ylabel("Mean squared error on later origins")
    right.set_xlabel(f"Policy, labels released {parameter} steps late")


def explain(result, parameter):
    m = result["mse"]
    interpretation = (
        f"With labels released {parameter} steps after their rows, the frozen model scores {fmt(m['frozen'])}, the rolling refit "
        f"{fmt(m['rolling'])} (window {round(result['selected_window'])} rows on average) and the incremental policy {fmt(m['incremental'])}. "
        f"The rolling gain over frozen is {fmt(m['frozen'])} - {fmt(m['rolling'])} = {fmt(result['gain_rolling'])}. "
        f"The invalid replay scores {fmt(m['invalid'])}, suggesting a gain of {fmt(result['gain_invalid'])} over frozen; its optimism "
        f"against the eligible rolling refit is {fmt(m['rolling'])} - {fmt(m['invalid'])} = {fmt(m['rolling'] - m['invalid'])}. "
        f"The noise floor is {fmt(result['noise_floor'])}.")
    steps = [
        f"Rolling gain over frozen: {fmt(m['frozen'])} - {fmt(m['rolling'])} = {fmt(result['gain_rolling'])}.",
        f"Incremental gain over frozen: {fmt(m['frozen'])} - {fmt(m['incremental'])} = {fmt(result['gain_incremental'])}.",
        f"Optimism of the invalid replay: {fmt(m['rolling'])} - {fmt(m['invalid'])} = {fmt(m['rolling'] - m['invalid'])}.",
        f"Rolling error above the noise floor: {fmt(m['rolling'])} - {fmt(result['noise_floor'])} = {fmt(m['rolling'] - result['noise_floor'])}.",
    ]
    metrics = {"Frozen": fmt(m["frozen"]), "Rolling refit": fmt(m["rolling"]), "Incremental": fmt(m["incremental"]),
               "Invalid replay": fmt(m["invalid"]), "Selected window": f"{round(result['selected_window'])} rows",
               "Rolling gain over frozen": fmt(result["gain_rolling"])}
    alt = (f"Left: running mean squared error over the scored origins for the frozen, rolling, incremental and invalid-replay policies with labels "
           f"released {parameter} steps late. Right: final errors, frozen {fmt(m['frozen'])}, rolling {fmt(m['rolling'])}, "
           f"incremental {fmt(m['incremental'])}, invalid replay {fmt(m['invalid'])}.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    order = [0, 10, 40, 100]
    gains = [results[d]["gain_rolling"] for d in order]
    assert gains[0] > gains[1] > gains[2] > gains[3], "the eligible rolling gain shrinks with delay"
    assert gains[3] < 0.25 * gains[0], "at long delays the adaptive gain nearly vanishes"
    windows = [results[d]["selected_window"] for d in order]
    assert windows[0] < windows[1] < windows[2] < windows[3], "longer windows are selected as delay grows"
    for d in order:
        m = results[d]["mse"]
        assert abs(m["invalid"] - results[0]["mse"]["rolling"]) < 1e-9, f"the invalid replay ignores the delay at {d}"
        assert m["incremental"] < m["rolling"], f"incremental beats rolling at {d}"
        assert m["incremental"] < m["frozen"], f"incremental keeps an edge over frozen at {d}"
        assert m["invalid"] > results[d]["noise_floor"]
    assert fmt(results[0]["gain_rolling"], 2) == "1.05" and fmt(results[40]["gain_rolling"], 2) == "0.30", "prediction feedback gains"
    assert results[40]["gain_rolling"] / results[0]["gain_rolling"] < 1 / 3, "less than a third"
    assert fmt(results[0]["mse"]["invalid"], 2) == "1.52" and fmt(results[100]["mse"]["invalid"], 2) == "1.52"
    assert fmt(results[100]["mse"]["rolling"], 2) == "2.48" and fmt(results[100]["mse"]["frozen"], 2) == "2.70"
    assert round(results[0]["selected_window"]) == 30 and round(results[100]["selected_window"]) == 213
    assert results[100]["mse"]["rolling"] - results[100]["mse"]["invalid"] > 0.8, "invalid replay is very optimistic at 100"
