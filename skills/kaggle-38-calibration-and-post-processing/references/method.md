# Chapter 38: Calibration and Post-Processing

Probability magnitude matters even when ranking remains unchanged. Four constructed logits here order the rows consistently, but two middle rows have outcomes that disagree with that confidence pattern. Change the temperature and inspect the resulting probabilities, log loss and Brier score. Positive temperature leaves the order intact while moving probabilities toward or away from one half. Explain why this can change a probability-sensitive score without changing which row ranks highest. If you select a temperature from these labels, those labels have become tuning data. Reserve another evaluation set before claiming an improvement, and report reliability-bin counts when diagnosing calibration on your own predictions. Change temperature, convert each logit with the sigmoid and calculate log loss and Brier score. A better score on these rows is a tuning result, not independent calibration evidence. Evaluate the chosen temperature on separate rows. If temperature is tuned until these rows look ideal, the displayed minimum becomes optimistic evidence. Keep a separate prediction set for assessment, and check whether the probability adjustment helps the metric your competition actually uses.

This constructed activity illustrates the chapter topic. The revised chapter develops the full fitting and assessment boundaries.

## Worked example

Can temperature scaling improve log loss while leaving ranking unchanged?

Four binary logits and labels are constructed. Positive temperature preserves score order.

```python
parameter = 1.5
import json, math
assert 0.5<=parameter<=4
logits=[-4.,-1.,1.,4.]; y=[0,1,0,1]
p=[1/(1+math.exp(-z/parameter)) for z in logits]
nll=-sum(t*math.log(q)+(1-t)*math.log1p(-q) for t,q in zip(y,p))/4
brier=sum((t-q)**2 for t,q in zip(y,p))/4
assert p==sorted(p)
print(json.dumps({'temperature':parameter,'probabilities':p,'log_loss':nll,'brier_score':brier,'rank_order_preserved':1}))

```

A better score on these rows is a tuning result, not independent calibration evidence. Evaluate the chosen temperature on separate rows.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
