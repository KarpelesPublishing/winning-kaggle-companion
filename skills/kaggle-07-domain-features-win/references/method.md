# Chapter 7: Domain Features as Testable Hypotheses

A transaction amount gains meaning when compared with an entity's prior behavior. A large purchase can be normal for one customer and unusual for another. Ask whether it differs from information available about the entity before the current prediction.

Change the current amount. The four prior purchases remain fixed, with mean 25. Inspect the amount-to-history ratio and absolute deviation. Compare the self-included deviation, whose baseline includes the current purchase. That baseline moves toward the event being described, diluting its contrast with history.

For your task, decide whether the grouping key reliably identifies an entity. An anonymized descriptor tuple is an identity hypothesis, not proof every match belongs to one physical card. Check frequencies, changes and collisions. Define which history was available for each prediction, then construct and validate the feature under that boundary. This activity assumes identity is reliable and teaches only the historical contrast. A high ratio neither labels fraud nor promises a ranking gain. Sparse histories need explicit counts and fallbacks so uncertainty is visible to the model.

## Worked example

Compare a current amount to past entity history.

Reliable entity key; current amount available; prior history only.

```python
parameter = 120
import json
assert 10 <= parameter <= 500
past=[20.,25.,30.,25.];mean=sum(past)/4;contaminated=(sum(past)+parameter)/5
assert mean==25
result={'historical_mean':mean,'current_amount':parameter,'past_only_ratio':parameter/mean,'past_only_deviation':parameter-mean,'self_included_deviation':parameter-contaminated}
print(json.dumps(result))

```

Amount contrast is not fraud probability.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
