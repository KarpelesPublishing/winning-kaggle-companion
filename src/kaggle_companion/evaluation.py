"""Small evaluation-boundary examples used by the revised book.

These are constructed teaching helpers, not historical competition reproductions.
"""
from collections import deque
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold


def target_map(category, target, strength=10.0):
    if strength < 0:
        raise ValueError('strength must be nonnegative')
    rows = pd.DataFrame({'category':np.asarray(category), 'target':np.asarray(target)})
    if len(rows) == 0 or rows['target'].isna().any():
        raise ValueError('finite observed training targets required')
    prior = float(rows.target.mean())
    grouped = rows.groupby('category', dropna=False).target.agg(['sum','count'])
    mapping = (grouped['sum'] + strength*prior)/(grouped['count'] + strength)
    return mapping, prior


def encode_categories(category, mapping, prior):
    return pd.Series(np.asarray(category)).map(mapping).fillna(prior).to_numpy(float)


def inner_target_encoding(category_train, target_train, category_assessment,
                          splits=None, strength=10.0):
    """Cross-fit training encodings; assessment mapping uses training labels only.

    Call separately within each outer fold. Supply time/group-safe inner splits
    when rows are not exchangeable. Uncovered chronological rows stay NaN.
    """
    category_train = np.asarray(category_train)
    target_train = np.asarray(target_train)
    if splits is None:
        splits = KFold(3, shuffle=True, random_state=17).split(category_train)
    encoded = np.full(len(target_train), np.nan)
    seen = np.zeros(len(target_train), dtype=int)
    for training, validation in splits:
        training, validation = np.asarray(training), np.asarray(validation)
        if np.intersect1d(training, validation).size:
            raise ValueError('inner partitions overlap')
        if np.any(seen[validation]):
            raise ValueError('each row may be assessed once')
        mapping, prior = target_map(category_train[training],target_train[training],strength)
        encoded[validation] = encode_categories(category_train[validation],mapping,prior)
        seen[validation] += 1
    mapping, prior = target_map(category_train,target_train,strength)
    return encoded, encode_categories(category_assessment,mapping,prior)


def outer_stack_regression(base_estimators, meta_estimator, X_train, y_train,
                           X_assessment, inner_splits):
    """Fit a stack using inner OOF inputs and predict untouched outer inputs."""
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    X_assessment = np.asarray(X_assessment)
    splits = list(inner_splits)
    inner = np.full((len(y_train),len(base_estimators)),np.nan)
    assessment = np.empty((len(X_assessment),len(base_estimators)))
    counts = np.zeros(len(y_train),dtype=int)
    for training, validation in splits:
        if np.intersect1d(training,validation).size:
            raise ValueError('inner partitions overlap')
        counts[validation] += 1
    if np.any(counts > 1):
        raise ValueError('duplicate inner assessment rows')
    for column, estimator in enumerate(base_estimators):
        for training, validation in splits:
            fitted = clone(estimator).fit(X_train[training],y_train[training])
            inner[validation,column] = fitted.predict(X_train[validation])
        fitted = clone(estimator).fit(X_train,y_train)
        assessment[:,column] = fitted.predict(X_assessment)
    covered = np.isfinite(inner).all(axis=1)
    if not covered.any():
        raise ValueError('no complete inner OOF rows')
    meta = clone(meta_estimator).fit(inner[covered],y_train[covered])
    return meta.predict(assessment), inner, covered


def expected_calibration_error(y, probabilities, bins=10):
    """Binary equal-width ECE, weighted by bin population; includes p=1."""
    y, p = np.asarray(y), np.asarray(probabilities,dtype=float)
    if bins < 1 or not len(y) or y.shape != p.shape:
        raise ValueError('nonempty matching one-dimensional arrays required')
    if y.ndim != 1 or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('binary labels and finite probabilities required')
    membership = np.minimum((p*bins).astype(int),bins-1)
    return sum(float(mask.mean())*abs(float(y[mask].mean()-p[mask].mean()))
               for group in range(bins) if (mask := membership==group).any())


def prevalence_weights(y, target_prior):
    """Expected confusion-count weights under a specified prior-shift assumption."""
    y=np.asarray(y)
    if not np.isin(y,[0,1]).all() or not 0 < target_prior < 1:
        raise ValueError('binary labels and an interior target prior required')
    source=float(y.mean())
    if not 0 < source < 1:
        raise ValueError('both source classes required')
    return np.where(y==1,target_prior/source,(1-target_prior)/(1-source))


def forward_date_splits(timestamps, assessment_blocks, label_available=None):
    """Unique timestamp blocks, in original row order, with strict past training."""
    time=pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
    if time.isna().any():
        raise ValueError('missing timestamp')
    available=time if label_available is None else pd.Series(pd.to_datetime(label_available)).reset_index(drop=True)
    if len(available)!=len(time) or available.isna().any():
        raise ValueError('matching label-availability timestamps required')
    previous_end=None
    for start,end in assessment_blocks:
        start,end=pd.Timestamp(start),pd.Timestamp(end)
        if end<=start or (previous_end is not None and start<previous_end):
            raise ValueError('ordered nonoverlapping assessment blocks required')
        training=np.flatnonzero(((time<start)&(available<start)).to_numpy())
        assessment=np.flatnonzero(((time>=start)&(time<end)).to_numpy())
        if not len(training) or not len(assessment):
            raise ValueError('empty training or assessment partition')
        previous_end=end
        yield training,assessment


def pancake_search(values, max_states=10000):
    """Bounded BFS for a constructed small prefix-flip problem, with replay."""
    start=tuple(values)
    if len(set(start))!=len(start) or max_states<1:
        raise ValueError('distinct values and positive state budget required')
    goal=tuple(sorted(start));queue=deque([(start,())]);seen={start};expanded=0
    while queue and expanded<max_states:
        state,path=queue.popleft();expanded+=1
        if state==goal:
            return {'solved':True,'moves':list(path),'state':state,'expanded':expanded}
        for count in range(2,len(state)+1):
            following=state[:count][::-1]+state[count:]
            if following not in seen:
                seen.add(following);queue.append((following,path+(count,)))
    return {'solved':False,'moves':None,'state':None,'expanded':expanded}


def replay_pancake(values,moves):
    state=tuple(values)
    for count in moves:
        if not isinstance(count,int) or not 2<=count<=len(state):
            raise ValueError('illegal prefix flip')
        state=state[:count][::-1]+state[count:]
    return state
