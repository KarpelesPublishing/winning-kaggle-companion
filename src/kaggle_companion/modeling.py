"""Compare a few models on identical folds with fold-fitted preprocessing."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, root_mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits


def better(candidate, incumbent, metric, tolerance=0.0):
    """Respect the competition metric's direction."""
    return candidate > incumbent + tolerance if metric == "accuracy" else candidate < incumbent - tolerance


def make_splits(X, y, task, split, folds=3, seed=42, groups=None, gap=0):
    if not 2 <= folds <= 10:
        raise ValueError("folds must be between 2 and 10")
    if split == "group":
        if groups is None or pd.isna(groups).any():
            raise ValueError("group splitting requires nonmissing group identifiers")
        return list(GroupKFold(folds).split(X, y, groups))
    if split == "time":
        if gap < 0:
            raise ValueError("gap cannot be negative")
        # X must already be in timestamp order; CLI sorts it first.
        return list(TimeSeriesSplit(folds, gap=gap).split(X))
    if split != "random":
        raise ValueError("split must be random, group, or time")
    splitter = StratifiedKFold(folds, shuffle=True, random_state=seed) if task == "classification" else KFold(folds, shuffle=True, random_state=seed)
    return list(splitter.split(X, y))


def preprocessing(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("scale", StandardScaler())]), numeric),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ], remainder="drop")


def score_predictions(y, pred, metric, classes=None):
    if metric == "rmse":
        return float(root_mean_squared_error(y, pred))
    if metric == "accuracy":
        return float(accuracy_score(y, np.asarray(classes)[np.argmax(pred, axis=1)]))
    if metric == "brier":
        if len(classes) != 2:
            raise ValueError("brier in this tool requires a binary target")
        return float(brier_score_loss(np.asarray(y) == classes[1], pred[:, 1]))
    if metric == "log_loss":
        return float(log_loss(y, pred, labels=classes))
    raise ValueError("unsupported metric")


def compare_models(X, y, *, task, metric, split="random", folds=3, seed=42,
                   groups=None, gap=0, candidates=3, max_iter=60):
    if not 1 <= candidates <= 3 or not 1 <= max_iter <= 300:
        raise ValueError("candidates must be 1..3 and max_iter 1..300")
    if task not in {"regression", "classification"}:
        raise ValueError("task must be regression or classification")
    if task == "regression" and metric != "rmse" or task == "classification" and metric not in {"brier", "log_loss", "accuracy"}:
        raise ValueError("metric and task disagree")
    X, y = X.reset_index(drop=True).copy(), pd.Series(y).reset_index(drop=True)
    if len(X) != len(y) or y.isna().any() or not X.shape[1]:
        raise ValueError("supply matching rows, nonmissing labels, and at least one feature")
    # Normalize pandas string/None values for sklearn's categorical imputer.
    for column in X.select_dtypes(exclude="number"):
        X[column] = X[column].astype(object).where(X[column].notna(), np.nan)
    splits = make_splits(X, y, task, split, folds, seed, groups, gap)
    classes = np.unique(y) if task == "classification" else None
    if classes is not None and len(classes) < 2:
        raise ValueError("classification needs at least two classes")
    for tr, va in splits:
        if set(tr) & set(va):
            raise ValueError("training and validation overlap")
        if classes is not None and not np.array_equal(np.unique(y.iloc[tr]), classes):
            raise ValueError("a training fold lacks a class; change the split or gather more labels")
    estimators = ([DummyClassifier(strategy="prior"), LogisticRegression(max_iter=300, random_state=seed), HistGradientBoostingClassifier(max_iter=max_iter, early_stopping=False, random_state=seed)] if classes is not None else
                  [DummyRegressor(), Ridge(alpha=10), HistGradientBoostingRegressor(max_iter=max_iter, early_stopping=False, random_state=seed)])
    names = ["dummy", "linear", "histogram_boosting"]
    coverage = np.zeros(len(X), dtype=bool)
    for _, va in splits:
        coverage[va] = True
    experiments, predictions = [], {}
    with threadpool_limits(limits=1):
        for name, estimator in zip(names[:candidates], estimators[:candidates]):
            oof = np.full((len(X), len(classes)), np.nan) if classes is not None else np.full(len(X), np.nan)
            fold_scores = []
            for tr, va in splits:
                model = Pipeline([("preprocess", preprocessing(X)), ("model", clone(estimator))])
                model.fit(X.iloc[tr], y.iloc[tr])
                pred = model.predict_proba(X.iloc[va]) if classes is not None else model.predict(X.iloc[va])
                oof[va] = pred
                fold_scores.append(score_predictions(y.iloc[va], pred, metric, classes))
            aggregate = score_predictions(y[coverage], oof[coverage], metric, classes)
            experiments.append({"model": name, "oof_score": aggregate, "fold_scores": fold_scores})
            predictions[name] = oof
    winner = experiments[0]
    for experiment in experiments[1:]:
        if better(experiment["oof_score"], winner["oof_score"], metric):
            winner = experiment
    digest = hashlib.sha256(json.dumps([(tr.tolist(), va.tolist()) for tr, va in splits]).encode()).hexdigest()
    report = {"task": task, "metric": metric, "direction": "maximize" if metric == "accuracy" else "minimize", "split": split,
              "seed": seed, "rows": len(X), "evaluated_rows": int(coverage.sum()), "fold_digest": digest,
              "classes": classes.tolist() if classes is not None else None,
              "experiments": experiments, "selected_on_cv": winner["model"],
              "interpretation": "CV selected these candidates. A fresh holdout or future competition result is needed to measure selection optimism."}
    return report, predictions, coverage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--metric", choices=["rmse", "brier", "log_loss", "accuracy"], required=True)
    parser.add_argument("--split", choices=["random", "group", "time"], required=True)
    parser.add_argument("--group")
    parser.add_argument("--time")
    parser.add_argument("--drop", nargs="*", default=[])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--max-rows", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.max_rows <= 100000:
        parser.error("max-rows must be 1..100000")
    frame = pd.read_csv(args.train, nrows=args.max_rows + 1)
    if len(frame) > args.max_rows:
        parser.error("row budget exceeded; choose a deliberate larger budget or prepare a representative dataset")
    frame["_companion_source_row"] = np.arange(len(frame))
    if args.split == "time":
        if not args.time:
            parser.error("time splitting requires --time")
        frame[args.time] = pd.to_datetime(frame[args.time], errors="raise", utc=True)
        if frame[args.time].isna().any() or frame[args.time].duplicated().any():
            parser.error("time values must be nonmissing and unique; repeated timestamps need a grouped temporal splitter")
        frame = frame.sort_values(args.time, kind="stable").reset_index(drop=True)
    excluded = list(dict.fromkeys([args.target, "_companion_source_row"] + args.drop + [c for c in [args.group, args.time] if c]))
    report, predictions, coverage = compare_models(frame.drop(columns=excluded), frame[args.target], task=args.task, metric=args.metric,
        split=args.split, groups=frame[args.group] if args.group else None, folds=args.folds, gap=args.gap,
        seed=args.seed, candidates=args.candidates, max_iter=args.max_iter)
    report["input_sha256"] = hashlib.sha256(args.train.read_bytes()).hexdigest()
    report["excluded_features"] = excluded
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    selected = predictions[report["selected_on_cv"]]
    oof = pd.DataFrame({"source_row": frame["_companion_source_row"], "evaluated": coverage})
    if selected.ndim == 1:
        oof["prediction"] = selected
    else:
        for i, cls in enumerate(report["classes"]):
            oof[f"probability_{cls}"] = selected[:, i]
    oof.to_csv(args.output / "oof.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
