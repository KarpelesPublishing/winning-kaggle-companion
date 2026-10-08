# Chapter 26: Advanced Target Encoding

Two categories can have the same sample mean but very different reasons to trust that mean. This activity isolates one reason: noisy observations. The constructed category has five observations with mean 0.8; the shared prior is 0.4 and the between-category variance is 0.2. Move the within-variance control and watch the reliability weight change. At zero estimated noise, the activity trusts the category mean completely. Increasing noise gives the shared prior more influence. Now imagine changing the sample count instead: more observations would reduce the variance of the mean. Explain which quantity measures observation variability and which measures uncertainty in the estimated category mean. Increase within-category noise while holding count and between-category variance fixed. The displayed reliability weight is tau²/(tau²+sigma²/n). The estimate moves toward the prior as noise grows. This demonstrates uncertainty weighting without endorsing the mismatched manuscript formula. Use one rare category from your own data to distinguish observation count from estimated noise. Decide which estimate you would trust after either quantity changes, then test the chosen estimator within your validation setup.

## Worked example

How does estimated category noise change shrinkage toward the prior?

Normal-means-style illustration uses a fixed prior mean and between-category variance; it is not certified James-Stein.

```python
parameter = 1
import json, math
assert 0<=parameter<=4
prior,mean,n,tau2=0.4,0.8,5,0.2
reliability=tau2/(tau2+parameter/n)
estimate=prior+reliability*(mean-prior)
assert prior<=estimate<=mean
print(json.dumps({'within_variance':parameter,'reliability_weight':reliability,'shrinkage_weight':1-reliability,'estimate':estimate}))

```

The estimate moves toward the prior as noise grows. This demonstrates uncertainty weighting without endorsing the mismatched manuscript formula.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
