# Chapter 45: Metric Learning and ArcFace

I understand the chapter as a shift from predicting a fixed class to learning a useful comparison space. The strongest explanation is the separation between training prototypes and a labeled inference gallery. I would reread the phrase about optimizing angular distance instead of softmax: the code still computes cross-entropy on modified logits. The activity makes that distinction visible. Use one unit query at 30 degrees to its true prototype and a competing prototype with fixed cosine 0.5. Add the selected margin only to the true-class angle, multiply both logits by 16, and compute stable two-class softmax and cross-entropy. Return the original and penalized cosine, the training probability, and the inference cosine. A larger margin lowers the true-class training logit and raises its loss at the same embedding. The unmodified retrieval cosine stays constant. This shows what training demands; it does not show tighter learned clusters or improved retrieval. Predict the zero-margin case, then compare it with 0.5 radians. Explain why changing a training logit alone supplies no evidence about the nearest neighbor returned by an untrained embedding.

## Worked example

How does an angular margin change the target logit without changing inference cosine similarity?

Angles are radians; the true angle plus margin remains below pi. This is one forward calculation, not optimization or embedding training.

```python
parameter = 0.5
import math, json
assert 0 <= parameter <= 0.8
angle=math.pi/6
plain=math.cos(angle)
penalized=math.cos(angle+parameter)
a,b=16*penalized,16*0.5
z=max(a,b)
p=math.exp(a-z)/(math.exp(a-z)+math.exp(b-z))
assert penalized <= plain+1e-12 and 0<p<1
print(json.dumps({'margin':parameter,'target_cosine':plain,'training_cosine':penalized,'training_probability':p,'cross_entropy':-math.log(p),'inference_cosine':plain}))

```

A larger margin lowers the true-class training logit and raises its loss at the same embedding. The unmodified retrieval cosine stays constant. This shows what training demands; it does not show tighter learned clusters or improved retrieval.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
