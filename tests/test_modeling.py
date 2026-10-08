import numpy as np
import pandas as pd
import pytest
from kaggle_companion.modeling import better, compare_models, make_splits, preprocessing


def test_metric_direction():
    assert better(.9, 1, "rmse")
    assert not better(1.1, 1, "rmse")
    assert better(.8, .7, "accuracy")


def test_group_boundary():
    X = pd.DataFrame({"x": range(24)})
    groups = np.repeat(range(6), 4)
    for tr, va in make_splits(X, np.arange(24), "regression", "group", groups=groups):
        assert not set(groups[tr]) & set(groups[va])


def test_temporal_boundary_and_uncovered_prefix():
    X = pd.DataFrame({"x": range(60)})
    for tr, va in make_splits(X, np.arange(60), "regression", "time", gap=3):
        assert tr.max() + 3 < va.min()
    report, pred, mask = compare_models(X, np.arange(60), task="regression", metric="rmse", split="time", candidates=2)
    assert 0 < report["evaluated_rows"] < len(X)
    assert np.isnan(pred["dummy"][~mask]).all()


def test_preprocessing_does_not_learn_validation_categories():
    train = pd.DataFrame({"x": [1., 2., 3.], "category": ["a", "b", "a"]})
    prep = preprocessing(train).fit(train)
    prep.transform(pd.DataFrame({"x": [999.], "category": ["validation_only"]}))
    fitted = prep.named_transformers_["category"].named_steps["encode"]
    assert "validation_only" not in fitted.categories_[0]
    assert prep.named_transformers_["numeric"].named_steps["impute"].statistics_[0] == 2


def test_reproducible_classification_and_complete_oof():
    rng = np.random.default_rng(11)
    X = pd.DataFrame({"x": rng.normal(size=90), "c": np.tile(["a", "b", "c"], 30)})
    y = (X.x > 0).astype(int)
    first, pred, mask = compare_models(X, y, task="classification", metric="brier", max_iter=10)
    second, _, _ = compare_models(X, y, task="classification", metric="brier", max_iter=10)
    assert first == second
    assert mask.all()
    assert np.isfinite(pred["histogram_boosting"]).all()
    assert np.allclose(pred["histogram_boosting"].sum(axis=1), 1)


def test_multiclass_probability_and_invalid_metric():
    X = pd.DataFrame({"x": range(90)})
    y = np.tile(["a", "b", "c"], 30)
    report, pred, _ = compare_models(X, y, task="classification", metric="log_loss", candidates=2)
    assert report["classes"] == ["a", "b", "c"]
    assert pred["linear"].shape == (90, 3)
    with pytest.raises(ValueError):
        compare_models(X, y, task="classification", metric="brier", candidates=1)
