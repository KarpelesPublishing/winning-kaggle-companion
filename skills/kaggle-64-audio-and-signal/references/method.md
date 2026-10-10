# Chapter 64: Audio and Signal Competitions

**How far does chunk-level cross-validation overstate accuracy on new recordings, and does it distort the comparison between models?**

The chapter says to keep overlapping chunks from the same independent source together when the assessment requires unseen sources. The error is large, grows with the number of chunks per recording, and can manufacture a model lead that new recordings do not show.

## The experiment

Constructed audio: 60 recordings of two species, each with its own background (a random smooth frequency response standing in for site and microphone) and each cut into chunks of 512 samples. Half of a recording's chunks contain its species' call tone. Features are 24 log band powers from an FFT. A flexible model (extremely randomized trees, standing in for a CNN) and a simple one (logistic regression) are scored by pooled AUC under 5-fold cross-validation that splits by chunk, 5-fold cross-validation that splits by recording, and on 100 recordings the models never saw. The control is the number of chunks per recording.

Control: Chunks cut from each recording (1, 3, 8, 16; default 8).

## Measured results

| Measure | 1 | 3 | 8 | 16 |
|---|---|---|---|---|
| Flexible, chunk-level CV | 0.590 | 0.829 | 0.895 | 0.940 |
| Flexible, recording-level CV | 0.613 | 0.656 | 0.628 | 0.651 |
| Flexible, new recordings | 0.667 | 0.641 | 0.684 | 0.723 |
| Simple, new recordings | 0.647 | 0.622 | 0.675 | 0.705 |
| Chunk-level optimism (flexible) | -0.077 | +0.188 | +0.211 | +0.217 |

## What the result says (default, chunks cut from each recording = 8)

With 8 chunks per recording, the flexible model scores AUC 0.684 on new recordings. Chunk-level cross-validation reports 0.895, an optimism of 0.895 - 0.684 = 0.211; recording-level cross-validation reports 0.628. The flexible model's AUC minus the simple model's is +0.154 under chunk-level validation, -0.011 under recording-level validation and +0.009 on new recordings.

- Chunk-level optimism, flexible model: 0.895 - 0.684 = 0.211.
- Recording-level error, flexible model: 0.628 - 0.684 = -0.056.
- Chunk-level lead of the flexible model: 0.895 - 0.741 = 0.154.
- Lead on new recordings: 0.684 - 0.675 = 0.009.

## Apply it to a competition

- Decide the independent unit (recording, site, subject, time window) from the task and split by it in every cross-validation.
- Compare chunk-level and grouped scores once on your data: a large gap means the model can recognise sources.
- Choose between models on the grouped score, and report support per fold for rare classes.
- Hold out whole recordings for the final check, as the host's test set will.

## Assumptions and limits

Constructed audio with a recording-level label, a single kind of background shift and tree and logistic models in place of the chapter's CNN on mel spectrograms. Only 60 recordings, so the recording-level estimate itself varies by a few hundredths between replicates, and the flexible-versus-simple gap on new recordings is small here (under 0.06 across the seeds tried).

Constructed data; the size of the optimism depends on how distinctive each recording's background is and how many chunks it yields. The flexible-model lead on new recordings is small in this generator, so the activity claims no real advantage for either model.

## Reproduce it

The chapter notebook `notebooks/64-audio-and-signal.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch64` (`run`, `explain`, `draw`).

```python
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kaggle_companion.activities._common import clean

SEED = 64
REPLICATES = 5                  # independent constructed collections; every estimate is their mean
RECORDINGS, NEW_RECORDINGS, NEW_CHUNKS = 60, 100, 4
CHUNK, BANDS = 512, 24          # samples per chunk; log-spectrum bands used as features
CALL_AMP = 0.5                  # loudness of the call tone relative to unit-variance background noise
CALL_FREQ = {0: 0.16, 1: 0.24}  # call frequency of species 0 and 1, in cycles per sample
MODELS = ["flexible", "simple"]


def recording_background(rng):
    """Each recording has its own background: a smooth random frequency response (a stand-in for site and microphone)."""
    return rng.normal(0, 3.0, 4) / np.arange(1, 5) ** 0.5, rng.uniform(0, 2 * np.pi, 4)


def chunk_features(background, species, rng, n):
    """n chunks of one recording as log band powers. Background noise is shaped by the recording's response; each chunk
    carries the species' call tone with probability 0.5."""
    amp, phase = background
    freq = np.fft.rfftfreq(CHUNK)
    gain_db = sum(a * np.cos(2 * np.pi * (i + 1) * freq * 2 + p) for i, (a, p) in enumerate(zip(amp, phase)))
    noise = np.fft.irfft(np.fft.rfft(rng.normal(size=(n, CHUNK)), axis=1) * 10 ** (gain_db / 20), CHUNK, axis=1)
    t, window = np.arange(CHUNK), np.hanning(CHUNK)
    call_freq = CALL_FREQ[species] * (1 + rng.normal(0, 0.04, n))
    tone = CALL_AMP * np.sin(2 * np.pi * call_freq[:, None] * t + rng.uniform(0, 2 * np.pi, (n, 1))) * np.sqrt(window)
    signal = noise + (rng.random(n) < 0.5)[:, None] * tone
    power = np.abs(np.fft.rfft(signal * window, axis=1)) ** 2
    width = (CHUNK // 2) // BANDS
    return np.log(power[:, :BANDS * width].reshape(n, BANDS, width).mean(2) + 1e-9)


def collection(recordings, chunks, rng):
    """Recordings alternate between the two species; every chunk inherits its recording's species and id."""
    X, y, group = [], [], []
    for r in range(recordings):
        X.append(chunk_features(recording_background(rng), r % 2, rng, chunks))
        y += [r % 2] * chunks
        group += [r] * chunks
    return np.vstack(X), np.array(y), np.array(group)


def make_model(kind):
    if kind == "flexible":      # a high-capacity learner standing in for a CNN: it can memorize recordings
        return ExtraTreesClassifier(40, max_features=0.5, random_state=0, n_jobs=1)
    return make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=300))


def pooled_auc(model_kind, X, y, splits):
    """Pool every held-out prediction and score one AUC over them."""
    oof = np.zeros(len(y))
    for fit, hold in splits:
        oof[hold] = make_model(model_kind).fit(X[fit], y[fit]).predict_proba(X[hold])[:, 1]
    return roc_auc_score(y, oof)


def run(chunks_per_recording):
    scores = {m: {"chunk_cv": [], "recording_cv": [], "new_recordings": []} for m in MODELS}
    for rep in range(REPLICATES):
        rng = np.random.default_rng(SEED * 100 + rep)
        X, y, group = collection(RECORDINGS, chunks_per_recording, rng)
        X_new, y_new, _ = collection(NEW_RECORDINGS, NEW_CHUNKS, rng)        # recordings the model never saw
        for m in MODELS:
            scores[m]["chunk_cv"].append(pooled_auc(m, X, y, KFold(5, shuffle=True, random_state=rep).split(X)))
            scores[m]["recording_cv"].append(pooled_auc(m, X, y, GroupKFold(5).split(X, y, group)))
            scores[m]["new_recordings"].append(roc_auc_score(y_new, make_model(m).fit(X, y).predict_proba(X_new)[:, 1]))
    return clean({
        "chunks_per_recording": chunks_per_recording,
        **{m: {k: float(np.mean(v)) for k, v in scores[m].items()} for m in MODELS},
        "per_replicate": scores,
        "recordings": RECORDINGS, "training_chunks": RECORDINGS * chunks_per_recording, "new_chunks": NEW_RECORDINGS * NEW_CHUNKS,
        "replicates": REPLICATES,
    })
```

Book location: Chapter 64, Record-Level Cross-Validation Is Not Optional. Constructed example: seeded synthetic audio, FFT features and two classifiers, measured by the chapter activity.
