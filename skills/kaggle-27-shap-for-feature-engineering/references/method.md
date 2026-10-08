# Chapter 27: SHAP as a Feature Engineering Signal

Before reading an explanation plot, identify what its vertical axis measures. In this constructed binary model, the baseline logit is minus one and another feature contributes 0.3. The control changes a second contribution in log-odds units. Add the contributions first, then use the sigmoid to obtain a probability. Compare the probability change with the controlled logit contribution: they are different quantities, and the conversion depends on the starting point. This is why an explanation can be additive without its numbers being additive probability changes. For your next feature-engineering experiment, name the output unit and formulate one falsifiable hypothesis from the plot. Vary one contribution, sum both with the baseline logit, and transform the total through the sigmoid. Logit contributions add exactly; individual probability changes generally do not. A dependency pattern generates an ablation hypothesis. Ask whether the model already captures the shape it explains. A new log feature or interaction deserves a fresh comparison; the explanation’s visual pattern does not tell you how much extra signal the feature adds.

## Worked example

Can you reconcile an additive explanation in log-odds and probability units?

Two invented feature contributions add to a binary logit; they are not computed SHAP values.

```python
parameter = 0.8
import json, math
assert -2<=parameter<=2
base,other=-1.0,0.3
sigmoid=lambda z:1/(1+math.exp(-z))
total=base+other+parameter
probability=sigmoid(total)
assert abs(total-base-other-parameter)<1e-12
print(json.dumps({'baseline_logit':base,'other_contribution':other,'controlled_contribution':parameter,'prediction_logit':total,'prediction_probability':probability,'probability_change_from_baseline':probability-sigmoid(base)}))

```

Logit contributions add exactly; individual probability changes generally do not. A dependency pattern generates an ablation hypothesis.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
