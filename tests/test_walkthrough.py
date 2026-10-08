import pickle
import numpy as np
import pandas as pd
import pytest
from kaggle_companion.walkthrough import make_fixture,features_at_origins,run_walkthrough,prediction_artifact,FEATURES


def test_future_labels_cannot_change_earlier_features_or_selection():
    data=make_fixture();changed=data.copy();changed.loc[changed.season>=7,'target']=1-changed.loc[changed.season>=7,'target']
    first=features_at_origins(data);second=features_at_origins(changed)
    np.testing.assert_allclose(first[first.season<=7][FEATURES],second[second.season<=7][FEATURES])
    a,pa=run_walkthrough(data);b,pb=run_walkthrough(changed)
    assert a['selected_model']==b['selected_model'] and a['development_brier']==b['development_brier']
    np.testing.assert_allclose(pa.probability.to_numpy()[:66],pb.probability.to_numpy()[:66])


def test_label_reveal_time_controls_history():
    data=make_fixture();delayed=data.copy();delayed.loc[delayed.season==1,'label_available_at_season']=3
    frame=features_at_origins(delayed);season2=frame[frame.season==2]
    assert (season2[FEATURES]==0).all().all()
    assert (frame[frame.season==3][FEATURES[0]].abs()>0).any()


def test_row_shuffle_keeps_id_aligned_output():
    data=make_fixture();a,pa=run_walkthrough(data);b,pb=run_walkthrough(data.sample(frac=1,random_state=13))
    assert a['development_brier']==b['development_brier']
    pd.testing.assert_frame_equal(pa,pb)


def test_saved_frozen_model_reproduces_prediction_artifact(tmp_path):
    report,pred=run_walkthrough(make_fixture(),tmp_path)
    with (tmp_path/'selected-model.pkl').open('rb') as stream:saved=pickle.load(stream)
    frame=pd.read_csv(tmp_path/'features.csv');assessment=frame[frame.season>=7]
    expected=np.full(len(assessment),.5) if saved['model'] is None else saved['model'].predict_proba(assessment[saved['features']])[:,1]
    np.testing.assert_allclose(expected,pred.probability,rtol=1e-10,atol=1e-12)
    assert report['assessment_rows']==132 and pred.sample_id.is_unique


def test_predictions_reject_duplicates_missing_coverage_and_invalid_probability():
    for ids,rows in [(['a','b'],[('a',.2),('a',.3)]),(['a','b'],[('a',.2)]),(['a'],[('a',np.nan)]),(['a'],[('a',1.2)])]:
        with pytest.raises(ValueError):prediction_artifact(ids,pd.DataFrame(rows,columns=['sample_id','probability']))


def test_initial_season_has_no_future_signal():
    data=make_fixture();frame=features_at_origins(data)
    assert (frame[frame.season==1][FEATURES]==0).all().all()
    broken=data.copy();broken.loc[0,'label_available_at_season']=1
    with pytest.raises(ValueError):features_at_origins(broken)


def test_unreleased_training_labels_do_not_enter_development_models():
    data=make_fixture();data.loc[data.season==4,'label_available_at_season']=6
    changed=data.copy();changed.loc[changed.season==4,'target']=1-changed.loc[changed.season==4,'target']
    # Season-4 outcomes are unavailable to the model fit at season 5.
    # They legitimately affect input history at season 6, so compare only
    # after delaying their history contribution until season 7 as well.
    data.loc[data.season==4,'label_available_at_season']=7
    changed.loc[changed.season==4,'label_available_at_season']=7
    a,_=run_walkthrough(data);b,_=run_walkthrough(changed)
    assert a['development_brier']==b['development_brier']
