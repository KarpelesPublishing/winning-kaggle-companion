# Chapter 41: Online Learning

Online updates require an availability clock as well as an event clock. Six observations in this constructed stream occur before prediction time eight. Their outcomes become known only after the delay you choose. Compare the naive count based on observation time with the count of labels actually released before prediction. A financial return measured over a future horizon can be unavailable even though its starting observation is old. Apply the same reasoning to delayed diagnoses, purchases and sensor annotations. Write the prediction time, feature availability time and target availability time for one row in your own task. Keep simultaneous predictions together so none receives a peer’s outcome early. Vary the outcome delay and compare event-time eligibility with release-time eligibility. An earlier observation is not necessarily a labeled training example yet. Predict first, then update only when the permitted outcome is revealed. A return measured over several future days cannot be used immediately after its starting trade. Batch events by prediction time and update only after the outcome horizon and official release conditions are satisfied.

## Worked example

Which labels are available before the next prediction?

Six events have outcomes released after a fixed delay. Predict at integer time eight, using only released outcomes.

```python
parameter = 3
import json, math
assert parameter in range(7)
prediction=8; events=list(range(2,8))
naive=[t for t in events if t<prediction]
eligible=[t for t in events if t+parameter<prediction]
assert all(t+parameter<prediction for t in eligible)
print(json.dumps({'prediction_time':prediction,'label_delay':int(parameter),'event_time_eligible_count':len(naive),'label_available_count':len(eligible),'unavailable_labels_excluded':len(naive)-len(eligible),'eligible_event_times':eligible}))

```

An earlier observation is not necessarily a labeled training example yet. Predict first, then update only when the permitted outcome is revealed.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
