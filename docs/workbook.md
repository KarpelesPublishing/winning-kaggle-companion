# Rebuild the browser workbook

Run after notebook execution:

```bash
pip install -r requirements-build.txt
python scripts/build_notebook_workbook.py --notebooks notebooks --output site/workbook --title "Winning Kaggle the Reproducible Way" --repo-url https://github.com/KarpelesPublishing/winning-kaggle-companion --book-url https://karpeles.com/publishing/winning-kaggle-the-reproducible-way --chapter-map chapter-map.json --lessons skills/winning-kaggle-the-reproducible-way/references/lessons.json
```

The workbook shows saved results, optional calculation code and conditional lessons. Answer fields export locally and do not send responses to a server. Website hosting and notebook execution are distinct.
