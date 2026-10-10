"""Chapter 50: LLM Fine-Tuning for Kaggle. A rank sweep for a low-rank (LoRA-style) update against the original and a full update."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np

from kaggle_companion.activities._common import clean

SEED = 50
DRAWS = 20                       # independent tasks; every estimate is a mean over draws
D_IN, D_OUT = 40, 20             # one frozen "pretrained" linear layer W0 of 800 weights
TRUE_RANK = 2                    # the target task differs from the pretrained task by a rank-2 change
N_TARGET, STEPS, LR = 100, 300, 0.01
CURVE_RANKS = [1, 2, 3, 4, 6, 8, 12, 16]


def make_task(draw):
    """Pretrained layer W0, a rank-2 shift to the target task, 100 noisy target rows and 4,000 clean held-out rows."""
    rng = np.random.default_rng([SEED, draw])
    W0 = rng.normal(size=(D_IN, D_OUT)) / np.sqrt(D_IN)
    shift = rng.normal(size=(D_IN, TRUE_RANK)) @ rng.normal(size=(TRUE_RANK, D_OUT)) / np.sqrt(D_IN * TRUE_RANK)
    X, X_held = rng.normal(size=(N_TARGET, D_IN)), rng.normal(size=(4000, D_IN))
    Y = X @ (W0 + shift) + rng.normal(size=(N_TARGET, D_OUT))
    return W0, X, Y, X_held, X_held @ (W0 + shift)


def adam(params, gradients, steps=STEPS, lr=LR):
    """Plain Adam on a list of arrays; every method below gets the same optimiser, steps and learning rate."""
    m1 = [np.zeros_like(p) for p in params]
    m2 = [np.zeros_like(p) for p in params]
    for t in range(1, steps + 1):
        for i, g in enumerate(gradients(params)):
            m1[i] = 0.9 * m1[i] + 0.1 * g
            m2[i] = 0.999 * m2[i] + 0.001 * g * g
            params[i] -= lr * (m1[i] / (1 - 0.9 ** t)) / (np.sqrt(m2[i] / (1 - 0.999 ** t)) + 1e-8)
    return params


def full_update(W0, X, Y):
    """Fine-tune every weight: W = W0 + Delta, with Delta trained from zero."""
    grad = lambda p: [X.T @ (X @ (W0 + p[0]) - Y) * 2 / Y.size]
    return W0 + adam([np.zeros_like(W0)], grad)[0]


def lora_update(W0, X, Y, rank, seed):
    """LoRA: W = W0 + B A with B (D_IN x r) starting at zero, A (r x D_OUT) random, scaling alpha / r = 1. Only A and B train."""
    A = np.random.default_rng(seed).normal(size=(rank, D_OUT)) / np.sqrt(D_OUT)

    def grad(p):
        G = X.T @ (X @ (W0 + p[1] @ p[0]) - Y) * 2 / Y.size       # gradient with respect to the whole update
        return [p[1].T @ G, G @ p[0].T]
    A, B = adam([A, np.zeros((D_IN, rank))], grad)
    return W0 + B @ A


def run(rank):
    ranks = sorted(set(CURVE_RANKS) | {rank})
    original, full, curve = [], [], {r: [] for r in ranks}
    for draw in range(DRAWS):
        W0, X, Y, X_held, Y_held = make_task(draw)
        mse = lambda W: float(np.mean((X_held @ W - Y_held) ** 2))      # held-out error against the noiseless target
        original.append(mse(W0))
        full.append(mse(full_update(W0, X, Y)))
        for r in ranks:
            curve[r].append(mse(lora_update(W0, X, Y, r, seed=1000 * draw + r)))
    chosen = np.array(curve[rank])
    saved = np.array(full) - chosen
    return clean({
        "rank": rank, "original": np.mean(original), "full": np.mean(full), "lora": chosen.mean(),
        "gain_over_full": saved.mean(), "gain_over_full_se": saved.std(ddof=1) / np.sqrt(DRAWS),
        "curve_ranks": ranks, "curve": [np.mean(curve[r]) for r in ranks],
        "per_draw_lora": chosen, "per_draw_full": full, "per_draw_original": original,
        "lora_parameters": rank * (D_IN + D_OUT), "full_parameters": D_IN * D_OUT, "true_rank": TRUE_RANK,
        "target_rows": N_TARGET, "draws": DRAWS,
    })
# notebook-end


SPEC = {
    "chapter": 50,
    "chapter_title": "LLM Fine-Tuning for Kaggle",
    "subtitle": "Compare the adapted model with the original checkpoint and a full update under matched data, and record rank and trainable parameters.",
    "summary": ("A low-rank update trains far fewer parameters than a full one. One demonstration adapts a frozen linear layer with "
                "a LoRA-style update of growing rank on 100 target rows and measures held-out error against the original layer and a "
                "full update."),
    "title": "Low-rank updates of growing rank against the original layer and a full update",
    "question": "How does the rank of a LoRA-style update change held-out error against the original checkpoint and a full update when target data is small?",
    "why": ("The chapter says parameter efficiency does not guarantee a quality gain and asks for the adapted model to be compared with "
            "the original checkpoint under matched data. A rank sweep with both comparators shows where a low-rank update helps, "
            "where it stops helping, and how much a full update gains over leaving the model alone."),
    "method": ("A constructed linear layer W0 (40 inputs, 20 outputs, 800 weights) stands in for a frozen pretrained layer. "
               "The target task is W0 plus a hidden rank-2 change, observed through 100 noisy target rows. Three adaptations are "
               "trained with the same Adam optimiser, 300 steps and learning rate: a full update of all 800 weights, and a "
               "LoRA update W0 + B A with B starting at zero, A random, scaling alpha / r = 1 and the control's rank r. Error is the mean squared "
               "difference from the noiseless target on 4,000 held-out rows, averaged over 20 tasks. The original layer, with no "
               "adaptation, is the comparator."),
    "control": {"key": "rank", "label": "LoRA rank r (trainable parameters: 60 r)",
                "values": [1, 2, 4, 16], "default": 4,
                "value_labels": ["1: 60 parameters", "2: 120 parameters (the true rank)", "4: 240 parameters", "16: 960 parameters"]},
    "source_section": "Why LoRA Changes the Game",
    "symbols": ("W0 is the frozen layer, B A the low-rank update with B of size 40 by r and A of size r by 20, r the rank, and the "
                "held-out error is the mean squared difference between the adapted layer's output and the noiseless target."),
    "explanation": ("A low-rank update can only express changes of rank r, so it needs r at least as large as the true change, and "
                    "every extra rank adds parameters that fit noise in 100 rows. The error therefore falls until r matches the hidden "
                    "rank and then rises, approaching the full update (800 parameters) as r grows. The original layer shows what "
                    "adaptation has to beat."),
    "application": ("Report rank, scaling and trainable parameters, compare every adapter with the original checkpoint and with a full "
                    "update at matched data, and choose the rank by a sweep on development rows, not by habit."),
    "assumptions": ("A single linear layer trained on squared error stands in for a transformer, and the target change has an exact "
                    "low rank by construction, which real tasks do not guarantee: the sweep, not rank 2, is the transferable part. "
                    "Scaling is fixed at alpha / r = 1; other scalings or learning rates move the curve. The full update has no weight decay "
                    "or early stopping; in a side check, stopping it early helped only a little. With much more target data "
                    "the full update catches up, and with a smaller task shift it can lose to the original layer."),
    "prediction": "The true change has rank 2. Which LoRA rank gives the lowest held-out error, and how does rank 16 compare with a full update?",
    "prediction_options": ["Rank 16: more capacity always helps",
                           "Rank 2, and rank 16 ends up about as poor as the full update",
                           "Rank 1: the fewest parameters regularise the most"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "Rank 2 reaches 0.095 held-out error; rank 16 reaches 0.651, close to the full update's 0.682 (the original layer: 1.015).",
        "incorrect": "Rank 2 reaches 0.095 held-out error; rank 1 cannot express the change (0.364) and rank 16 reaches 0.651, close to the full update's 0.682 (the original layer: 1.015).",
    },
    "check": "Rank 2 is only best because the true rank is known here. What do you do in a competition, and what does the original-layer line add?",
    "answer": ("Run the sweep on development rows and pick the smallest rank after which the error stops falling; the curve here has a clear "
               "minimum, and rank 4 already costs 0.245 against 0.095 at rank 2. The original-layer line (1.015) shows the gain from "
               "adapting at all, and the full update (0.682) shows that touching every weight gives most of that gain back as noise."),
    "provenance": "Constructed example: seeded synthetic linear layers and numpy gradient training, measured by the chapter activity.",
    "apply": [
        "Record the rank, scaling, target modules and trainable parameter count with every adapter you compare.",
        "Compare each adapter with the original checkpoint and a full update at the same data, steps and learning rate.",
        "Sweep the rank on development rows and prefer the smallest rank whose error is within noise of the best.",
        "Expect full updates to overfit when target data is small, and re-check the comparison when the data grows.",
    ],
    "honesty": "A linear stand-in with an exactly low-rank change; the sizes of these effects are properties of this generator, not a competition result.",
}

EQUATIONS = [{"tex": r"W = W_0 + \frac{\alpha}{r}\, B A, \qquad B \in \mathbb{R}^{d_{\mathrm{in}}\times r},\; A \in \mathbb{R}^{r\times d_{\mathrm{out}}}",
              "alt": "The adapted weight W equals the frozen weight W zero plus alpha over r times the product B A, with B of size d in by r and A of size r by d out",
              "basis": "The low-rank update with alpha / r scaling described in Why LoRA Changes the Game; the manuscript describes it in prose, so this is the activity's own notation."}]
NCOLS = 2
HEIGHT = 4.4


def draw(axes, result, parameter):
    ax, bx = axes
    ranks, curve = result["curve_ranks"], result["curve"]
    ax.plot(ranks, curve, color=COLORS["teal"], lw=2, marker="o", ms=5, label="LoRA update")
    ax.axhline(result["original"], color=COLORS["grey"], ls=(0, (4, 3)), lw=1.6, label=f"Original layer: {fmt(result['original'])}")
    ax.axhline(result["full"], color=COLORS["gold"], ls=(0, (1, 2)), lw=1.8, label=f"Full update: {fmt(result['full'])}")
    ax.axvline(result["true_rank"], color=COLORS["light"], lw=1.5, zorder=0)
    ax.scatter([parameter], [result["lora"]], s=170, color=COLORS["terracotta"], zorder=4, edgecolor="white", linewidth=1.2,
               label=f"Selected rank {parameter}: {fmt(result['lora'])}")
    ax.text(result["true_rank"] + 0.2, 0.5, "true rank", fontsize=10, va="center")
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("Held-out mean squared error")
    ax.set_xticks([1, 2, 4, 8, 12, 16])
    ax.set_ylim(0, 1.75)
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    names = ["Original\n0 trained", f"LoRA r = {parameter}\n{result['lora_parameters']} trained", f"Full update\n{result['full_parameters']} trained"]
    values = [result["original"], result["lora"], result["full"]]
    draws = [result["per_draw_original"], result["per_draw_lora"], result["per_draw_full"]]
    bx.bar(range(3), values, width=0.55, color=[COLORS["light"], COLORS["teal"], COLORS["gold"]], edgecolor=COLORS["ink"], lw=0.6)
    for x, (v, dots) in enumerate(zip(values, draws)):
        bx.scatter([x + (i - (len(dots) - 1) / 2) * 0.05 for i in range(len(dots))], dots, s=14, color="white", edgecolor=COLORS["ink"],
                   linewidth=0.8, zorder=3, label="One task" if x == 0 else None)
        bx.text(x, max(v, max(dots)) + 0.03, fmt(v), ha="center", va="bottom", fontsize=10)
    bx.set_xticks(range(3), names)
    bx.set_ylim(0, 1.9)
    bx.set_ylabel("Held-out mean squared error")
    bx.set_xlabel("Adaptation (trainable parameters)")
    bx.legend(loc="upper right", frameon=False, fontsize=10)


def explain(result, parameter):
    o, f, l = result["original"], result["full"], result["lora"]
    best = min(zip(result["curve"], result["curve_ranks"]))
    gain, se = result["gain_over_full"], result["gain_over_full_se"]
    shown = round(f, 3) - round(l, 3)          # differences of the numbers as displayed, so each sum checks by hand
    shown_o = round(o, 3) - round(l, 3)
    hand = f"{fmt(f)} - {fmt(l)} = {fmt(shown)} better than" if shown >= 0 else f"{fmt(l)} - {fmt(f)} = {fmt(-shown)} worse than"
    interpretation = (
        f"At rank {parameter} ({result['lora_parameters']} trainable parameters against {result['full_parameters']} for a full update) the "
        f"adapted layer has held-out error {fmt(l)}. The original layer scores {fmt(o)} and the full update {fmt(f)}, so the LoRA update is "
        f"{hand} the full update (paired standard error {fmt(se)}) and {fmt(o)} - {fmt(l)} = {fmt(shown_o)} better than doing nothing. "
        f"Across the ranks tried the lowest error is {fmt(best[0])} at rank {best[1]}; the hidden change has rank {result['true_rank']}.")
    steps = [f"Gain over the original layer: {fmt(o)} - {fmt(l)} = {signed(shown_o)}.",
             f"Gain over the full update: {fmt(f)} - {fmt(l)} = {signed(shown)} (standard error {fmt(se)}).",
             f"Trainable parameters: {result['lora_parameters']} of {result['full_parameters']} ({100 * result['lora_parameters'] / result['full_parameters']:.1f}%).",
             f"Best rank in the sweep: {best[1]} with held-out error {fmt(best[0])}."]
    metrics = {"Original layer error": fmt(o), "Full update error": fmt(f), f"LoRA error, rank {parameter}": fmt(l),
               "Gain over full (SE)": f"{signed(gain)} ({fmt(se)})", "Trainable parameters": str(result["lora_parameters"]),
               "Best rank in the sweep": str(best[1])}
    alt = (f"Two panels. Left, held-out error against LoRA rank, with a falling then rising curve whose lowest value {fmt(best[0])} is at rank "
           f"{best[1]}, dashed lines for the original layer ({fmt(o)}) and the full update ({fmt(f)}), and a marker at rank {parameter} ({fmt(l)}). "
           f"Right, bars of held-out error for the original layer, the selected LoRA rank and the full update, with a dot per task.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    # The LoRA gradient must match finite differences (shapes, transposes and the zero-initialised B).
    rng = np.random.default_rng(7)
    X, Y, W0 = rng.normal(size=(10, D_IN)), rng.normal(size=(10, D_OUT)), rng.normal(size=(D_IN, D_OUT))
    A, B = rng.normal(size=(3, D_OUT)), rng.normal(size=(D_IN, 3))
    loss = lambda A, B: np.mean((X @ (W0 + B @ A) - Y) ** 2)
    G = X.T @ (X @ (W0 + B @ A) - Y) * 2 / Y.size
    for idx in [(0, 0), (2, 5)]:
        e = np.zeros_like(A); e[idx] = 1e-6
        assert abs((loss(A + e, B) - loss(A - e, B)) / 2e-6 - (B.T @ G)[idx]) < 1e-6
        e = np.zeros_like(B); e[idx[0], idx[1] % 3] = 1e-6
        assert abs((loss(A, B + e) - loss(A, B - e)) / 2e-6 - (G @ A.T)[idx[0], idx[1] % 3]) < 1e-6
    r1, r2, r4, r16 = results[1], results[2], results[4], results[16]
    curve = dict(zip(r2["curve_ranks"], r2["curve"]))
    assert min(curve, key=curve.get) == 2, "the lowest error is at the true rank"
    assert curve[1] > 2 * curve[2] and all(curve[a] < curve[b] for a, b in zip([2, 3, 4, 6, 8, 12], [3, 4, 6, 8, 12, 16])), "error rises above the true rank"
    assert abs(r2["lora"] - curve[2]) < 1e-9 and abs(r16["lora"] - curve[16]) < 1e-9, "curve agrees across control values"
    assert r2["original"] - r2["lora"] > 0.8 and r2["gain_over_full"] > 4 * r2["gain_over_full_se"], "rank 2 beats both comparators clearly"
    assert r4["lora"] < r4["full"] - 0.3, "rank 4 still beats the full update"
    assert abs(r16["lora"] - r16["full"]) < 0.1, "rank 16 is about as poor as the full update"
    assert r2["full"] < r2["original"], "the full update still improves on the original here"
    assert fmt(curve[2]) == "0.095" and fmt(curve[16]) == "0.651" and fmt(curve[1]) == "0.364" and fmt(curve[4]) == "0.245"
    assert fmt(r2["full"]) == "0.682" and fmt(r2["original"]) == "1.015"
    assert curve[1] > curve[2], "incorrect feedback: rank 1 cannot express the change"
