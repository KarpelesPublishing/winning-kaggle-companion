# Chapter 50: LLM Fine-Tuning for Kaggle

**How does the rank of a LoRA-style update change held-out error against the original checkpoint and a full update when target data is small?**

The chapter says parameter efficiency does not guarantee a quality gain and asks for the adapted model to be compared with the original checkpoint under matched data. A rank sweep with both comparators shows where a low-rank update helps, where it stops helping, and how much a full update gains over leaving the model alone.

## The experiment

A constructed linear layer W0 (40 inputs, 20 outputs, 800 weights) stands in for a frozen pretrained layer. The target task is W0 plus a hidden rank-2 change, observed through 100 noisy target rows. Three adaptations are trained with the same Adam optimiser, 300 steps and learning rate: a full update of all 800 weights, and a LoRA update W0 + B A with B starting at zero, A random, scaling alpha / r = 1 and the control's rank r. Error is the mean squared difference from the noiseless target on 4,000 held-out rows, averaged over 20 tasks. The original layer, with no adaptation, is the comparator.

Control: LoRA rank r (trainable parameters: 60 r) (1: 60 parameters, 2: 120 parameters (the true rank), 4: 240 parameters, 16: 960 parameters; default 4).

## Measured results

| Measure | 1: 60 parameters | 2: 120 parameters (the true rank) | 4: 240 parameters | 16: 960 parameters |
|---|---|---|---|---|
| Original layer error | 1.015 | 1.015 | 1.015 | 1.015 |
| Full update error | 0.682 | 0.682 | 0.682 | 0.682 |
| LoRA error, rank 4 |  |  | 0.245 |  |
| Gain over full (SE) | +0.319 (0.035) | +0.588 (0.015) | +0.438 (0.010) | +0.032 (0.001) |
| Trainable parameters | 60 | 120 | 240 | 960 |
| Best rank in the sweep | 2 | 2 | 2 | 2 |

## What the result says (default, lora rank r (trainable parameters: 60 r) = 4)

At rank 4 (240 trainable parameters against 800 for a full update) the adapted layer has held-out error 0.245. The original layer scores 1.015 and the full update 0.682, so the LoRA update is 0.682 - 0.245 = 0.437 better than the full update (paired standard error 0.010) and 1.015 - 0.245 = 0.770 better than doing nothing. Across the ranks tried the lowest error is 0.095 at rank 2; the hidden change has rank 2.

- Gain over the original layer: 1.015 - 0.245 = +0.770.
- Gain over the full update: 0.682 - 0.245 = +0.437 (standard error 0.010).
- Trainable parameters: 240 of 800 (30.0%).
- Best rank in the sweep: 2 with held-out error 0.095.

## Apply it to a competition

- Record the rank, scaling, target modules and trainable parameter count with every adapter you compare.
- Compare each adapter with the original checkpoint and a full update at the same data, steps and learning rate.
- Sweep the rank on development rows and prefer the smallest rank whose error is within noise of the best.
- Expect full updates to overfit when target data is small, and re-check the comparison when the data grows.

## Assumptions and limits

A single linear layer trained on squared error stands in for a transformer, and the target change has an exact low rank by construction, which real tasks do not guarantee: the sweep, not rank 2, is the transferable part. Scaling is fixed at alpha / r = 1; other scalings or learning rates move the curve. The full update has no weight decay or early stopping; in a side check, stopping it early helped only a little. With much more target data the full update catches up, and with a smaller task shift it can lose to the original layer.

A linear stand-in with an exactly low-rank change; the sizes of these effects are properties of this generator, not a competition result.

## Reproduce it

The chapter notebook `notebooks/50-llm-fine-tuning-for-kaggle.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch50` (`run`, `explain`, `draw`).

```python
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
```

Book location: Chapter 50, Why LoRA Changes the Game. Constructed example: seeded synthetic linear layers and numpy gradient training, measured by the chapter activity.
