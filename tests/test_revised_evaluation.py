import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from kaggle_companion.evaluation import (inner_target_encoding,outer_stack_regression,
    expected_calibration_error,prevalence_weights,forward_date_splits,pancake_search,replay_pancake)

def test_encoding_outer_label_isolation_and_unseen_prior():
    labels=np.array([0.,0.,1.,1.,0.,1.,100.,-100.])
    category=np.array(['a','b','a','b','a','b','new','new'])
    training=np.arange(6);assessment=np.arange(6,8)
    def run(target):
        return inner_target_encoding(category[training],target[training],category[assessment])
    original=run(labels);labels[assessment]=[9999.,9999.];altered=run(labels)
    for before,after in zip(original,altered):np.testing.assert_array_equal(before,after)
    np.testing.assert_allclose(original[1],[0.5,0.5])

def test_inner_training_recipient_labels_do_not_enter_encoding():
    category=np.array(['a']*6);y=np.arange(6,dtype=float)
    splits=[(np.array([2,3,4,5]),np.array([0,1]))]
    before,_=inner_target_encoding(category,y,['a'],splits)
    y[:2]=999
    after,_=inner_target_encoding(category,y,['a'],splits)
    np.testing.assert_array_equal(before,after)
    assert np.isnan(before[2:]).all()

def test_stack_assesses_external_rows_not_fitted_meta_rows():
    X=np.arange(20,dtype=float).reshape(-1,1);y=np.sin(X[:,0]);outer_training=np.arange(15);assessment=np.arange(15,20)
    splits=list(KFold(3,shuffle=True,random_state=7).split(outer_training))
    def run(target):return outer_stack_regression([Ridge(),DecisionTreeRegressor(max_depth=2,random_state=1)],Ridge(),X[outer_training],target[outer_training],X[assessment],splits)
    first=run(y);y[assessment]=99999;second=run(y)
    np.testing.assert_array_equal(first[0],second[0]);assert first[0].shape==(5,);assert first[2].all()

def test_weighted_ece_and_probability_one_boundary():
    # Three correctly calibrated zero predictions and one incorrect p=1.
    assert expected_calibration_error([0,0,0,0],[0,0,0,1],2)==0.25
    assert expected_calibration_error([0,1],[0,1],2)==0
    with pytest.raises(ValueError):expected_calibration_error([0],[np.nan])

@pytest.mark.parametrize('prior',[0.1,0.9])
def test_prior_weights_can_move_prevalence_in_both_directions(prior):
    y=np.array([0,0,1,1]);w=prevalence_weights(y,prior)
    assert np.average(y,weights=w)==pytest.approx(prior)

def test_forward_dates_keep_duplicate_timestamps_and_delayed_labels_out():
    time=['2020-01-03','2020-01-01','2020-01-02','2020-01-02','2020-01-04']
    available=['2020-01-03','2020-01-01','2020-01-02','2020-01-05','2020-01-04']
    training,assessment=next(forward_date_splits(time,[('2020-01-03','2020-01-05')],available))
    np.testing.assert_array_equal(training,[1,2]);np.testing.assert_array_equal(assessment,[0,4])

def test_search_goal_and_budget_failure():
    result=pancake_search([4,3,2,1]);assert result['solved'];assert replay_pancake([4,3,2,1],result['moves'])==(1,2,3,4)
    assert len(result['moves'])==1
    result=pancake_search([3,1,4,2],max_states=1);assert not result['solved'];assert result['moves'] is None
    with pytest.raises(ValueError):replay_pancake([1,2,3],[4])
