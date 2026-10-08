import numpy as np
import pytest
from kaggle_companion.checks import eligible_teachers, expected_calibration_error, stable_softmax


def test_softmax_large_logits_and_translation_invariance():
    p = stable_softmax([[1000, 1001, 999]])
    assert np.allclose(p, [[.244728471, .665240956, .090030573]])
    assert np.allclose(p, stable_softmax([[0, 1, -1]]))
    assert np.allclose(p.sum(axis=1), 1)


def test_ece_weights_not_mean_of_bin_errors():
    labels = [0]*90 + [0]*10
    probabilities = [.05]*90 + [.95]*10
    assert expected_calibration_error(labels, probabilities) == pytest.approx(.14)
    assert expected_calibration_error([0, 1], [0, 1]) == 0


def test_teacher_fold_provenance():
    folds = [set([0,1]), set([2,3]), set([4,5])]
    teachers = [set(range(6))-fold for fold in folds]
    for k, validation in enumerate(folds):
        assert eligible_teachers(validation, teachers) == [k]
        # In the book's 'all teachers except k' variant every teacher saw these labels.
        assert all(validation <= teachers[i] for i in range(3) if i != k)
    with pytest.raises(ValueError):
        eligible_teachers([0], [[0,1],[0,2]])
