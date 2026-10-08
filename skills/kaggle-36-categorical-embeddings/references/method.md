# Chapter 36: Categorical Embeddings

A target encoder has more than one route by which labels can enter a feature. Its category mean may be fold-local while its fallback or smoothing prior still uses every label. Here the training labels are [0, 1] and the validation labels are [1, 1]. A training category has one observed positive outcome. Change the smoothing count and compare the encoding obtained from the training-only mean with the encoding using the all-row mean. Explain why stronger smoothing increases the influence of whichever prior you selected. In your own pipeline, trace every target-derived quantity, including rare-category fallbacks, learned embeddings and later feature-selection decisions. Vary smoothing strength and compare a training-only prior with an all-row prior. The gap exposes a target dependency even though the category statistic itself uses training rows only. Unseen categories can expose the same problem through the fallback value. Ask which labels supplied that fallback and whether its computation respected the prediction’s training boundary before interpreting the encoded feature.

## Worked example

Does a held-out target influence an encoding through its global prior?

Four binary labels are split into two training and two validation rows. One training category observation has label one.

```python
parameter = 10
import json, math
assert parameter in range(21)
train=[0,1]; validation=[1,1]
prior_train=sum(train)/len(train); prior_all=sum(train+validation)/4
proper=(1+parameter*prior_train)/(1+parameter)
leaked=(1+parameter*prior_all)/(1+parameter)
assert 0<=proper<=1 and 0<=leaked<=1
print(json.dumps({'smoothing_count':int(parameter),'training_prior':prior_train,'all_row_prior':prior_all,'proper_encoding':proper,'leaked_prior_encoding':leaked,'encoding_difference':leaked-proper}))

```

The gap exposes a target dependency even though the category statistic itself uses training rows only.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
