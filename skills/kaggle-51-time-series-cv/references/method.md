# Chapter 51: Time Series Cross-Validation

The chapter’s useful change in my understanding is that a label can extend past its row timestamp. I can understand why fitting a scaler globally is unsafe in a forecast experiment. I lose the thread when week-based GroupKFold is recommended after insisting all validation must follow training. The activity tests the label boundary directly without relying on a splitter name. Set a validation origin at day 20 and a five-day forward-label horizon. Exclude the selected number of rows immediately before the origin, then inspect every retained training row’s label end time. Return the number of labels ending at or after the origin and the latest safe training endpoint. A gap is meaningful through label availability, not through autocorrelation alone. Under the strict endpoint convention a gap of five removes overlap. A different data latency or interval convention would change the sufficient gap. Move the gap from zero to five. Explain the exact timestamp inequality behind the first safe result, then describe how a reporting delay would change that boundary.

This constructed activity illustrates the chapter topic. The revised chapter develops the full fitting and assessment boundaries.

## Worked example

Which training labels are still unavailable at a forecast origin after choosing a row gap?

One equally spaced row per day. Labels must end strictly before day 20 under this activity’s information convention. This is a forward split, not a general purged K-fold implementation.

```python
parameter = 5
import json
assert int(parameter)==parameter and 0<=parameter<=10
gap=int(parameter); origin=20; horizon=5
train=list(range(origin-gap))
rows=[{'day':t,'label_end':t+horizon,'available':int(t+horizon<origin)} for t in train]
overlap=sum(1-r['available'] for r in rows)
assert all(t<origin for t in train)
if gap>=horizon: assert overlap==0
print(json.dumps({'gap':gap,'train_rows':len(train),'unavailable_labels':overlap,'last_train_day':max(train),'latest_safe_train_day':max(r['day'] for r in rows if r['available']),'table':rows[-8:]}))

```

A gap is meaningful through label availability, not through autocorrelation alone. Under the strict endpoint convention a gap of five removes overlap. A different data latency or interval convention would change the sufficient gap.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
