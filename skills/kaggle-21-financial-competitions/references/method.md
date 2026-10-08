# Chapter 21: Financial Competitions: Time, Regime, and Leakage

A backward-looking feature window and a forward-looking label interval have different leakage risks. A correctly constructed feature can use past training history for the first validation prediction. A training label that extends into the validation period can reveal outcomes the model would not yet know at its fitting cutoff.

Change the forward label horizon. Validation begins at time eight. Inspect retained and purged training timestamps. Under the stated closed-interval convention, a label ending exactly at validation start is purged. Longer horizons remove more boundary rows. The rule follows label availability, not the length of a feature's past lookback.

For real financial panels, write prediction time, feature availability and label end time as separate fields. Split by distinct timestamps so different assets at one time do not straddle an inappropriate boundary. Fit regime thresholds and normalizers using permitted past or training information. Decide whether same-time cross-sectional values are available together during inference. A row-count gap is meaningful only when it represents the actual time horizon and ordering. Validate the complete pipeline with its future-use structure; a time split alone cannot reveal a target-derived predictor already embedded in validation features.

## Worked example

Purge overlapping forward-label intervals.

Integer timestamps; closed forward-label intervals; validation starts8.

```python
parameter = 3
import json
assert 1 <= parameter <= 5
assert int(parameter) == parameter
validation_start=8;training=list(range(8));h=int(parameter);kept=[t for t in training if t+h<validation_start];purged=[t for t in training if t+h>=validation_start]
assert all(t+h<validation_start for t in kept)
result={'validation_start':validation_start,'label_horizon':h,'kept_times':kept,'purged_times':purged,'purged_count':len(purged),'rule':'closed label interval ending at validation start is purged'}
print(json.dumps(result))

```

Purge follows label horizon, not backward feature lookback.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
