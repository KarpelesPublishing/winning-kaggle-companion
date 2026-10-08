# Constructed season forecast

This offline dataset teaches the information boundary and final prediction artifact in Chapters 6 and 19. It is synthetic. Its seasons S01 through S08, teams and outcomes are not NCAA games, a historical Kaggle submission, or a reproduced winning solution.

`fixture.csv` contains 528 games, 12 teams and eight constructed seasons. The fixed generator uses NumPy seed 42. Each pair plays once per season. A season's outcomes become available at the start of the following season. Prior win rates use fixed Beta(1,1) smoothing and only legally revealed games. Teams are known in advance; no new-team generalization is claimed. Game-count differences are zero in this balanced fixture, a deliberately inspectable redundant feature rather than evidence of additional signal.

Train candidates on seasons 1–4, choose using seasons 5–6, then refit the chosen recipe on seasons 1–6 before assessing seasons 7–8. Assessment is sequential: season-7 outcomes may update the input history for season 8 after they arrive. The fitted model and selected recipe remain frozen. The two assessment seasons are not independent replications.

Run from the project root:

```sh
PYTHONPATH=companion/src companion/.venv/bin/python -m kaggle_companion.walkthrough --fixture companion/data/season-walkthrough/fixture.csv --output companion/data/season-walkthrough/output
```

The run writes features, ID-aligned probabilities, the selected fitted model, and a result report. Read `output/results.json` for actual development and assessment scores. Compare by sample ID, never by accidental file order. Only load the locally generated pickle; an arbitrary downloaded pickle is not needed for this activity.

Dependencies are declared in `companion/pyproject.toml`. Use the fresh-kernel Chapter 19 notebook for a guided run. The archived development comparison includes the nonlinear candidate even when it loses. One fixture cannot establish statistical significance, universal superiority, or an expected competition gain.
