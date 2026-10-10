# Chapter activity contract

Every chapter of the Winning Kaggle companion has one measured activity: a real experiment on seeded generated data (or a clearly labelled bundled dataset) that measures an effect the chapter teaches. The notebook, the interactive chapter demonstration and the workbook page all read its results. Exemplars that pass everything below: `src/kaggle_companion/activities/ch09.py`, `ch21.py`, `ch38.py`. Read all three before writing one.

## The teaching bar

An activity passes only if all five are true:

1. **Core lesson.** It measures the chapter's main decision, not a side detail. The question is one a competitor actually faces.
2. **Real computation.** `run()` fits, validates, encodes, ensembles, searches or simulates. It never looks up, indexes or does arithmetic on a handful of typed numbers.
3. **Insight.** The control reveals a decision point: a value where the right choice changes, or where an estimate stops describing new rows. The reader learns something they could not have guessed exactly.
4. **Transfer.** The `apply` steps say what to do in a real competition.
5. **Honesty.** Constructed data is labelled. If the effect is small, absent or reversed, the text says so (ch21 reports that the purge gap was within noise). Claims never go beyond the chapter's own text.

Non-technical chapters (strategy, landscape, portfolio, learning systems) still get a measured activity: a simulation of the decision the chapter discusses (for example a public-leaderboard winner's curse, a seed-noise floor, OOF alignment by ID). Use `AUDIT-PROPOSALS.md` in this folder as design input; it is advice, and the chapter text decides.

## Files per chapter NN

- `src/kaggle_companion/activities/chNN.py`: the activity module (contract below).
- `readers/modules/chNN.py`: replace with the two-line adapter (copy the old file to `readers/modules-v1/chNN.py` first):

  ```python
  """Chapter NN reader: the measured activity in kaggle_companion.activities.chNN."""
  from _activity_demo import make

  make(NN, globals())
  ```
- `activities-cache/chNN.json`: written by `scripts/build_activities.py`, never by hand.

Do not edit shared files (`_common.py`, `_activity_demo.py`, `_kg.py`, `build_*.py`, `chapter-map.json`), other chapters' files, the notebooks or the manuscript. Notebooks are generated later by `scripts/build_notebooks.py` in one serial run.

## Module contract

```text
"""Chapter NN: <title>. <one line on the experiment>."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
<imports, constants, generate(...), helpers, run(parameter) -> dict>
# notebook-end

SPEC = {...}          # all keys listed below
EQUATIONS = [...]     # at least one {"tex", "alt", "basis"}
NCOLS = 1 or 2
HEIGHT = 4.2 to 4.6
def draw(ax_or_axes, result, parameter): ...
def explain(result, parameter) -> {"metrics", "interpretation", "steps", "alt"}
def verify(results): ...  # asserts every claim the fixed text makes
```

The code between `# notebook-begin` and `# notebook-end` is copied verbatim into the notebook, so it must be self-contained, readable and commented like teaching code. It may import `clean` from `_common`. `run(parameter)` returns `clean({...})`: plain JSON, floats rounded.

`SPEC` keys (all required): `chapter`, `chapter_title` (exact chapter heading text), `subtitle`, `summary`, `title`, `question`, `why`, `method`, `control` (`key`, `label`, `values` (2 to 4; the engine rejects more), `default`, `value_labels`), `source_section` (an exact `##` heading from `readers/text/NN-*.md`), `symbols` (defines every symbol in `EQUATIONS`), `explanation`, `application`, `assumptions`, `prediction`, `prediction_options` (3), `prediction_answer` (index), `prediction_feedback` (`correct`, `incorrect`), `check`, `answer`, `provenance` (must include the word "Constructed"), `apply` (3 to 4 steps), `honesty`.

`EQUATIONS`: prefer a display equation from the chapter (`basis` names it). If the chapter has none, state the activity's own label, metric or rule and say so in `basis` (ch21).

## Rules

- **Every number shown is measured.** Text with numbers in `explain` is generated from `result`. Numbers in fixed `SPEC` text (prediction feedback, answer) are asserted in `verify()` with `fmt(...) == "0.346"`, so a change in the generator fails the build instead of shipping a stale claim.
- **Seeded and deterministic.** `build_activities.py` reruns the default and compares.
- **Fast.** Each control value runs in about 2 to 3 seconds, hard limit 6 s, single thread (`_common` sets `OMP_NUM_THREADS=1`; pass `n_jobs=1`). Installed: numpy, scipy, scikit-learn, pandas, matplotlib. Not installed: lightgbm, xgboost, catboost, torch. Use `HistGradientBoosting*`, `ExtraTrees*`, `RandomForest*`, `LogisticRegression`, `Ridge`, `MLP*`, `KNeighbors*`, numpy. Say in `assumptions` when a sklearn model stands in for the chapter's library.
- **Robust, not lucky.** A single small sample is noise. Average over replicate series or draws inside `run` (ch21: 8 series; ch38: 30 draws), and before finishing, change the seed base once and confirm every qualitative claim still holds. If it does not, the claim is noise: weaken it or redesign.
- **Score the way practitioners score.** Pool held-out predictions and compute one metric over them (ch21), rather than averaging a metric over tiny blocks, which can bias it. Compare estimates against a large fresh sample from the same generator as the "truth".
- **Text style.** No em dashes or double hyphens anywhere. No first person. ASCII hyphen for minus (`signed()` and `fmt()` do this). British or American spelling consistently within a file. Do not use these words: matplotlib, jinja, jupyter, "pip install", conda, node.js.
- **Engine checks** (run `readers/build.py readers --chapters NN --check`):
  - all figure text at least 10 pt (`fontsize=10`);
  - the interpretation or metrics must contain a hand-sized calculation written `a - b = c` with a plain number after `=` (use `fmt(x)`, not `signed(x)`, right after the equals sign);
  - `source_section` must be a real heading;
  - equations must be non-empty.
- **Figures.** Paired bars, curves, reliability diagrams, staircases: whatever makes the effect visible. Label axes with units, show the "truth" as a reference line, show replicate spread where it matters, no titles inside the axes (the page has them). Legends must not cover data.

## Workflow per chapter

```bash
cd companion
# 1. read the chapter: readers/text/NN-*.md (headings) and ../manuscript/chapters/NN-*.md
# 2. prototype in your scratch folder until the effect is real and stable; time it
# 3. write src/kaggle_companion/activities/chNN.py
.venv/bin/python scripts/build_activities.py --chapters NN      # must print ok
.venv/bin/python readers/build.py readers --chapters NN --check  # must print ok
# 4. replace readers/modules/chNN.py with the adapter (after backing it up), rerun the check
# 5. render the default figure and look at it (draw into a figure and save a PNG in your scratch folder)
```

## Report per chapter

One short block: the question, the measured headline numbers at each control value, whether the effect held under a second seed base, any claim you weakened, and anything you could not make work.
