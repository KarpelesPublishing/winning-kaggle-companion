# Chapter 52: Time Series Feature Engineering

This chapter gives me a broad feature checklist, but the checklist is usable only after fixing the forecast origin. I can follow the grouped shift and EMA examples. I would reread the claim that every block respects time alignment because the difference helper includes current sales. The companion exposes a second issue: even shifted lags can be unavailable for a fixed multi-day forecast. Set the forecast origin after day 9. For the chosen horizon h, forecast day 9+h and enumerate lags 1, 2, 7, 14. Report each lag’s source day, availability at the origin, and value only when observed. Show the horizon-safe first difference from days 9 and 8 separately. A feature that is safe for one-step evaluation can require a future target at a longer horizon. A recursive forecast must supply predictions or choose larger lags; an online forecast with new observations is a different information contract. Current-target differences are unavailable by construction. Compare horizons one and seven with the same lag list. Identify the lags that switch from observed to unavailable, and decide whether your own evaluation assumes new truth arrives between forecasts.

## Worked example

Which lagged target values are actually available for a multi-step forecast at a fixed origin?

Target history is observed only through day 9. Future targets are not revealed while producing this multi-step forecast. Lag 14 may refer to nonexistent history and must remain missing.

```python
parameter = 3
import json
assert int(parameter)==parameter and 1<=parameter<=7
h=int(parameter); origin=9; target=origin+h
history={t:10+t+2*(t%3) for t in range(10)}
rows=[]
for lag in (1,2,7,14):
 src=target-lag; available=int(src in history)
 rows.append({'lag':lag,'source_day':src,'available':available,'value':history.get(src),'missing_history':int(src<0),'future_required':int(src>origin)})
assert all(r['source_day']<=origin for r in rows if r['available'])
print(json.dumps({'horizon':h,'forecast_day':target,'safe_previous_difference':history[9]-history[8],'table':rows}))

```

A feature that is safe for one-step evaluation can require a future target at a longer horizon. A recursive forecast must supply predictions or choose larger lags; an online forecast with new observations is a different information contract. Current-target differences are unavailable by construction.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
