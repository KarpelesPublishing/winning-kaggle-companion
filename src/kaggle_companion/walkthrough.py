"""Offline constructed season forecast. This does not reproduce March Mania."""
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss

FEATURES = ['prior_win_rate_difference', 'prior_game_count_difference']

def make_fixture(seed=42):
    rng = np.random.default_rng(seed)
    strength = rng.normal(size=12)
    rows = []
    for season in range(1, 9):
        strength = .85 * strength + rng.normal(0, .25, size=12)
        for a in range(12):
            for b in range(a + 1, 12):
                probability = 1 / (1 + np.exp(-(strength[a] - strength[b])))
                rows.append({'sample_id':f'S{season:02d}-T{a:02d}-T{b:02d}',
                             'season':season, 'team_a':a, 'team_b':b,
                             'label_available_at_season':season + 1,
                             'target':int(rng.random() < probability)})
    return pd.DataFrame(rows)

def validate(data):
    required = {'sample_id','season','team_a','team_b','label_available_at_season','target'}
    if not required.issubset(data.columns):
        raise ValueError('fixture is missing required columns')
    if data.sample_id.isna().any() or data.sample_id.duplicated().any():
        raise ValueError('sample identities must be complete and unique')
    if not data.target.isin([0,1]).all():
        raise ValueError('binary targets required')
    if (data.label_available_at_season <= data.season).any():
        raise ValueError('labels must arrive after their prediction season')

def features_at_origins(data):
    validate(data)
    rows=[]
    for season in sorted(data.season.unique()):
        history=data[(data.season < season) & (data.label_available_at_season <= season)]
        rates={};counts={}
        for team in sorted(set(data.team_a) | set(data.team_b)):
            games=history[(history.team_a==team) | (history.team_b==team)]
            wins=((games.team_a==team) & (games.target==1)).sum()+((games.team_b==team) & (games.target==0)).sum()
            counts[team]=len(games)
            rates[team]=(wins+1)/(len(games)+2)  # fixed Beta(1,1) smoothing
        for row in data[data.season==season].sort_values('sample_id').itertuples():
            rows.append({'sample_id':row.sample_id,'season':season,'target':row.target,
                         'label_available_at_season':row.label_available_at_season,
                         FEATURES[0]:rates[row.team_a]-rates[row.team_b],
                         FEATURES[1]:counts[row.team_a]-counts[row.team_b]})
    return pd.DataFrame(rows)

def models():
    return {'logistic':LogisticRegression(C=1.0,random_state=42,max_iter=1000),
            'small_tree':HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,
                         l2_regularization=1.0,early_stopping=False,random_state=42)}

def prediction_artifact(expected_ids, predictions):
    if predictions.sample_id.duplicated().any() or predictions.sample_id.isna().any():
        raise ValueError('prediction IDs must be complete and unique')
    if len(expected_ids)!=len(set(expected_ids)) or set(expected_ids)!=set(predictions.sample_id):
        raise ValueError('prediction IDs do not match required coverage')
    result=pd.DataFrame({'sample_id':expected_ids}).merge(predictions,on='sample_id',validate='one_to_one',sort=False)
    p=result.probability.to_numpy()
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('finite probabilities between zero and one required')
    return result

def run_walkthrough(data, output_dir=None):
    frame=features_at_origins(data)
    if set(frame.season)!={1,2,3,4,5,6,7,8}:
        raise ValueError('this teaching contract requires seasons 1 through 8')
    train=frame[(frame.season<=4) & (frame.label_available_at_season<=5)]
    development=frame[frame.season.isin([5,6]) & (frame.label_available_at_season<=7)]
    assessment=frame[frame.season.isin([7,8])]
    # The development candidates are fit once, before season 5.
    candidates=models();development_scores={'constant':float(brier_score_loss(development.target,np.full(len(development),.5)))}
    for name,model in candidates.items():
        model.fit(train[FEATURES],train.target)
        development_scores[name]=float(brier_score_loss(development.target,model.predict_proba(development[FEATURES])[:,1]))
    # Selection uses only development labels. Ties prefer constant, then logistic.
    order=['constant','logistic','small_tree']
    selected=min(order,key=lambda name:(development_scores[name],order.index(name)))
    frozen_model=None
    if selected!='constant':
        frozen_model=models()[selected]
        refit=frame[(frame.season<=6) & (frame.label_available_at_season<=7)]
        frozen_model.fit(refit[FEATURES],refit.target)
        p=frozen_model.predict_proba(assessment[FEATURES])[:,1]
    else:p=np.full(len(assessment),.5)
    # A rolling input history can use season-7 outcomes once revealed at season 8.
    # It does not refit the selected model or select against assessment scores.
    predictions=prediction_artifact(assessment.sample_id.tolist(),pd.DataFrame({'sample_id':assessment.sample_id,'probability':p}))
    per_season=[]
    for season in [7,8]:
        mask=assessment.season.to_numpy()==season
        per_season.append({'season':season,'rows':int(mask.sum()),
            'baseline_brier':float(brier_score_loss(assessment.target.to_numpy()[mask],np.full(mask.sum(),.5))),
            'selected_brier':float(brier_score_loss(assessment.target.to_numpy()[mask],p[mask]))})
    report={'example':'constructed synthetic season forecast; not a historical competition reproduction',
            'seed':42,'fixture_rows':len(data),'features':FEATURES,
            'training_seasons':[1,2,3,4],'development_seasons':[5,6],'assessment_seasons':[7,8],
            'prediction_contract':'Each season is forecast at its start. Earlier-season labels are usable only after their declared reveal season. The selected model is frozen before season 7; history inputs update when labels legally arrive.',
            'development_brier':development_scores,'selected_model':selected,
            'assessment_baseline_brier':.25,'assessment_selected_brier':float(brier_score_loss(assessment.target,p)),
            'assessment_by_season':per_season,'assessment_rows':len(predictions),
            'limitation':'One constructed fixture and two correlated later seasons do not establish significance, a medal, or a general model-family advantage.'}
    if output_dir:
        out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
        predictions.to_csv(out/'predictions.csv',index=False)
        frame.to_csv(out/'features.csv',index=False)
        with (out/'selected-model.pkl').open('wb') as stream:pickle.dump({'model':frozen_model,'features':FEATURES,'selected':selected},stream)
        report['artifact_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'predictions.csv',out/'features.csv',out/'selected-model.pkl']}
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    return report,predictions

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--fixture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(run_walkthrough(pd.read_csv(args.fixture),args.output)[0],indent=2))
