# Chapter 30: External Data and Leakage

An external feature has at least two relevant dates: when the event happened and when its value became available. The constructed prediction occurs at time ten. One measurement happened at time four and was released at five; a later measurement happened at eight, with a publication lag controlled here. Compare the event-only lookup with the available-value lookup as you increase that lag. The event-only result stays fixed even when its information arrives too late. Write down which timestamp your own source actually provides, whether values are revised, and which version a competitor could obtain at prediction time. A clean join needs this history as well as matching entity keys. Vary the allowed publication lag and compare an event-only lookup with a release-aware lookup. An old event date does not make a later revision available in the past. Point-in-time correctness requires source version and rules checks too. For a revised economic indicator, the release history may matter more than the event date. Decide how you would recover the first available value and whether that vintage is accessible under the competition rules.

## Worked example

Which external value was actually available when this prediction was made?

Each invented record has an event time and a separate release time. Prediction time is 10.

```python
parameter = 3
import json, math
assert parameter in range(7)
prediction=10
records=[{'event':4,'release':5,'value':2},{'event':8,'release':8+parameter,'value':7}]
eligible=[r for r in records if r['event']<prediction and r['release']<prediction]
safe=max(eligible,key=lambda r:r['event'])
naive=max(records,key=lambda r:r['event'])
assert safe['release']<prediction
print(json.dumps({'prediction_time':prediction,'event_only_value':naive['value'],'available_value':safe['value'],'excluded_records':len(records)-len(eligible),'publication_lag':parameter}))

```

An old event date does not make a later revision available in the past. Point-in-time correctness requires source version and rules checks too.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
