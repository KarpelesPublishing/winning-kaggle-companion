# Winning Kaggle the Reproducible Way: public companion

*Validation, Feature Engineering and Ensembling for Data Science Competitions*

Companion notebooks, browser workbooks and modeling skills for Jason Karpeles' book. This public repository excludes the manuscript, complete chapters and book editions. The author maintains the full book references in a separate private repository.

## What is here

- 65 executed chapter notebooks. Each runs a measured experiment on seeded generated data (two use scikit-learn's bundled digit images) and measures an effect the chapter teaches, with a figure and the result across a control. The experiments live in `src/kaggle_companion/activities/`; `activities/ACTIVITIES.md` is the contract they meet.
- A browser workbook for every chapter, with prediction prompts, worked results and downloadable answers.
- 561 conditional lessons with source references, application dimensions, audience value grades and limitations.
- 66 companion skills: a master router plus one per chapter. These teach the companion methods and lesson applications; they do not contain the full chapter text.
- Modeling helpers and a complete synthetic season-forecast example.

Open the [interactive companion](https://karpeles.com/companions/winning-kaggle-the-reproducible-way/) on karpeles.com: chapter demonstrations, [workbook](https://karpeles.com/companions/winning-kaggle-the-reproducible-way/workbook/) and [lesson atlas](https://karpeles.com/companions/winning-kaggle-the-reproducible-way/lessons.html). The same pages are mirrored on [GitHub Pages](https://karpelespublishing.github.io/winning-kaggle-companion/).

About the book: https://karpeles.com/publishing/winning-kaggle-the-reproducible-way

## Run the notebooks

Use Python 3.13 or later:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
jupyter lab
```

Open a notebook and select Run All. The browser workbook displays saved results; its supported chapter controls do not execute arbitrary Python. Constructed examples are not historical Kaggle score reproductions. See [technical corrections](docs/errata.md) and [skills](docs/skills-guide.md).

## Checks

```bash
pytest -q
python scripts/verify_package.py
python scripts/check_public_package.py
python skills/winning-kaggle-the-reproducible-way/scripts/check_companion.py
```

The GitHub Actions template is provided in `docs/github-actions-template.yml`; installing it requires a token with workflow permission.

## Copyright

Copyright Jason Karpeles. All rights reserved. This is a companion distribution, not the book. Third-party solution archives and private data are excluded.

Kaggle is a trademark of Google LLC. This book and companion are independent and are not affiliated with, sponsored by or endorsed by Kaggle or Google.
