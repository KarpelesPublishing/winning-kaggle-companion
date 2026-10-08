# Chapter 6: Running Case: March Mania 2026

An Elo feature is a sequence of updates. Its meaning depends on which games and season boundaries enter the sequence. Accepting a decay parameter without applying it creates a different feature from the one described. Final model scores can hide this implementation difference.

Two equally rated teams have played one neutral game, with Team A winning at K=24. Inspect their resulting ratings. Move carryover between zero and one. Zero resets both teams to 1500; one preserves the entire gap. Intermediate values retain part of the prior information. Read the resulting next-game probability beside the retained gap.

Before using Elo for a tournament, identify the last game allowed to influence each team's tournament-time snapshot. Regress once before the next season begins and exclude that tournament's outcomes from its pre-tournament features. Preserve pregame ratings, expected outcome and updates in a trace. Compare carryover settings under chronological validation rather than choosing from one public result. This activity checks a season-boundary operation; it establishes no optimal basketball decay and does not reproduce the reported ensemble.

## Worked example

Apply an offseason Elo boundary.

Equal1500 ratings, neutral win with K24 before boundary.

```python
parameter = 0.65
import json
assert 0 <= parameter <= 1
ra=1512.;rb=1488.;before=[ra,rb];ra=1500+parameter*(ra-1500);rb=1500+parameter*(rb-1500);p=1/(1+10**((rb-ra)/400))
assert abs(ra+rb-3000)<1e-10
result={'ratings_before_offseason':before,'ratings_after_offseason':[ra,rb],'next_game_probability_a':p,'retained_rating_gap':ra-rb}
print(json.dumps(result))

```

Carryover changes prior strength, not measured optimal decay.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
