# Chapter 19: March Mania: Prediction at Scale

A complete forecast has three jobs: fitting candidates, selecting a recipe, and assessing the frozen result. This activity performs all three on the fixed synthetic fixture from Chapters 6 and 19. It uses earlier-season histories, a constant baseline, logistic regression and a small nonlinear comparator.

Switch between development and assessment. The development comparison chooses logistic regression; the tree model stays visible despite losing. Assessment contains 132 later games and shows the result of the frozen choice, not a new tuning objective. Earlier outcomes update later input histories only after their declared reveal season. The fitted model does not update during assessment.

The linked fixture, results and ID-aligned probability file let you inspect the complete teaching artifact. Reload the saved local model using the documented module command to check reproduction. Changing a later assessment label must not change development selection. This is a constructed calculation, not the historical March Mania campaign.

## Worked example

Run a complete season forecast and distinguish development from assessment.

528 synthetic games, 12 known teams, eight seasons; fixed seed 42; season labels arrive at the next season. Train1 to 4; select on5 to 6; refit before7; assess sequentially on7 to 8 with a frozen model and legally updated history.

```python
parameter = 'assessment'
import json, sys
from pathlib import Path
import pandas as pd
companion_root = Path.cwd() if (Path.cwd()/'src/kaggle_companion').exists() else Path.cwd()/'companion'
sys.path.insert(0, str(companion_root/'src'))
from kaggle_companion.walkthrough import run_walkthrough
assert parameter in ['development','assessment']
fixture = pd.read_csv(companion_root/'data/season-walkthrough/fixture.csv')
report, predictions = run_walkthrough(fixture)
if parameter == 'development':
    result = {'partition':'development seasons 5–6','losses':report['development_brier'],
              'selected_model':report['selected_model'],'rows':132,
              'decision':'Select here, then freeze the recipe before assessment.'}
else:
    result = {'partition':'assessment seasons 7–8','baseline_brier':report['assessment_baseline_brier'],
              'selected_brier':report['assessment_selected_brier'],'by_season':report['assessment_by_season'],
              'prediction_rows':len(predictions),'unique_ids':bool(predictions.sample_id.is_unique),
              'decision':'Report frozen performance here; do not select a new recipe from this score.'}
print(json.dumps(result))

```

The development-selected logistic model beats the constant baseline in this fixture, but later loss is worse than development loss. The nonlinear candidate loses on development and remains reported. One fixture and two related seasons do not establish statistical significance or a historical gain.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
