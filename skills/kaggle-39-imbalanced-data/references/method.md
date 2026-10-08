# Chapter 39: Imbalanced Data

A model trained after negative downsampling sees a different class prior. This illustration starts with three probabilities assumed calibrated to that sampled distribution. Keep every positive and vary the fraction of negatives retained. The correction multiplies the sampled odds by that retention fraction, then converts back to probability. Compare the corrected value of one half with the raw value; an apparently ambiguous sampled prediction can correspond to a rare event in the original population. Name the sampling assumptions before using this formula. If retention varies with features, or if a separate calibrator already corrects the distribution, this simple second correction is inappropriate. Vary negative retention and apply the odds correction p*r/(1-p+p*r). This corrects the sampling prior under stated assumptions. Do not apply it again after a calibrator already restores the original distribution. The correction changes magnitude while preserving order under these assumptions. Therefore a ranking score can remain similar while log loss changes considerably. Compare the probability scale before deciding which imbalance remedy to keep.

## Worked example

How does negative retention change a sampled model probability?

All positives are retained; negatives are sampled independently with probability r; the raw probability is calibrated under that sampled distribution.

```python
parameter = 0.1
import json, math
assert 0.01<=parameter<=1
raw=[0.1,0.5,0.9]
corrected=[p*parameter/(1-p+p*parameter) for p in raw]
assert all(0<p<1 for p in corrected)
assert all(a<=b for a,b in zip(corrected,raw))
print(json.dumps({'negative_retention':parameter,'raw_probabilities':raw,'corrected_probabilities':corrected,'half_probability_corrected':corrected[1]}))

```

This corrects the sampling prior under stated assumptions. Do not apply it again after a calibrator already restores the original distribution.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
