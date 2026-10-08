# Chapter 3: Building a CV That Matches the Test Set

Decide what a validation fold must imitate before choosing a splitter. If the task predicts outcomes after a deadline, an earlier historical season should be predicted using information available before that season. Holding an entire season together protects internal structure, but grouping alone does not enforce chronology.

Select a validation year. One training list contains every other year; the other contains only earlier years. Compare the future-training count and ask which list matches the claim you want to make. All-other-season validation can diagnose differences among seasons under stationarity. It answers a different question from forecasting the next season.

For your task, record prediction time, label availability and entities requiring isolation. Construct one fold and assert those boundaries before fitting. Remove training rows whose forward-label intervals overlap validation. If a warmup period has no OOF predictions, keep those values missing. The next experiment should inherit this documented fold definition. When test construction is uncertain, compare plausible splits and record what each estimates instead of claiming any one is an exact copy of the future.

## Worked example

Separate held-out seasons from future prediction.

Six ordered seasons; no fitted model.

```python
parameter = 3
import json
assert 1 <= parameter <= 5
assert int(parameter) == parameter
years=list(range(2020,2026));k=int(parameter);group=[v for i,v in enumerate(years) if i!=k];past=years[:k]
assert all(v<years[k] for v in past)
result={'validation_year':years[k],'group_training_years':group,'past_only_training_years':past,'future_training_count':sum(v>years[k] for v in group)}
print(json.dumps(result))

```

Future-training count diagnoses chronology, not bias magnitude.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
