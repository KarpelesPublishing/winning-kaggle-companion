# Chapter 16: The Fourth-Root Blend

A blend's weights can be constructed from each model's improvement above a baseline. Raising those positive margins to a fractional exponent compresses differences. This is a heuristic for avoiding extreme weighting, not a solution for the best covariance-aware ensemble.

Move the exponent from one fourth toward one. Inspect the normalized weights and their best-to-worst ratio. The three illustrative margins are already close, so both proportional and fourth-root weights remain near equal. Calculate one weight manually to check that the displayed fractions follow the formula rather than a plausible-looking story.

For your models, define the baseline and score direction before forming margins. A loss needs baseline minus model loss; a reward metric uses model score minus baseline. Examine actual paired errors and validate the blend, because a weaker model can contribute useful complementary predictions or merely add noise. Apply rank transforms only when their effect matches the metric and output shape. Ranking probability matrices without a class policy can change decision boundaries, while rank-only metrics and calibration-sensitive metrics ask different questions. Preserve raw predictions so both alternatives can be evaluated honestly.

## Worked example

Compare margin weighting exponents.

Higher-is-better micro-F1; majority baseline .50; positive margins.

```python
parameter = 0.25
import json
assert 0.25 <= parameter <= 1
scores=[.742,.738,.729];baseline=.5;margins=[s-baseline for s in scores];raw=[m**parameter for m in margins];weights=[v/sum(raw) for v in raw]
assert abs(sum(weights)-1)<1e-12
result={'scores':scores,'baseline':baseline,'margins':margins,'weights':weights,'weight_ratio_best_to_worst':weights[0]/weights[-1]}
print(json.dumps(result))

```

Compressed weights often near equal; no covariance-aware optimality claimed.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
