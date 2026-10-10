"""Chapter 64: Audio and Signal Competitions. Chunk-level against recording-level cross-validation on generated audio."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
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
# notebook-end


SPEC = {
    "chapter": 64,
    "chapter_title": "Audio and Signal Competitions",
    "subtitle": "Keep chunks of one recording together, because the independent unit is the recording.",
    "summary": ("Chunks cut from one recording share its background, so a split by chunk lets a model recognise recordings instead of "
                "calls. One demonstration generates audio and measures chunk-level and recording-level cross-validation against "
                "recordings the model never saw."),
    "title": "Chunk-level and recording-level cross-validation against new recordings",
    "question": "How far does chunk-level cross-validation overstate accuracy on new recordings, and does it distort the comparison between models?",
    "why": ("The chapter says to keep overlapping chunks from the same independent source together when the assessment requires unseen "
            "sources. The error is large, grows with the number of chunks per recording, and can manufacture a model lead that new recordings do not show."),
    "method": ("Constructed audio: 60 recordings of two species, each with its own background (a random smooth frequency response standing "
               "in for site and microphone) and each cut into chunks of 512 samples. Half of a recording's chunks contain its species' "
               "call tone. Features are 24 log band powers from an FFT. A flexible model (extremely randomized trees, standing in for a "
               "CNN) and a simple one (logistic regression) are scored by pooled AUC under 5-fold cross-validation that splits by chunk, "
               "5-fold cross-validation that splits by recording, and on 100 recordings the models never saw. The control is the number "
               "of chunks per recording."),
    "control": {"key": "chunks_per_recording", "label": "Chunks cut from each recording",
                "values": [1, 3, 8, 16], "default": 8,
                "value_labels": ["1", "3", "8", "16"]},
    "source_section": "Record-Level Cross-Validation Is Not Optional",
    "symbols": ("AUC is the area under the ROC curve for the species label, pooled over all held-out chunks. Optimism is a cross-validation AUC "
                "minus the AUC on new recordings."),
    "explanation": ("Every chunk of a recording carries the same background. A flexible model can learn that background as a fingerprint of "
                    "the species, and a chunk-level split puts siblings of every held-out chunk in training, so the fingerprint scores "
                    "well. A recording-level split removes the siblings, so the model is judged on the call itself. More chunks per "
                    "recording gives the fingerprint more to memorize and the optimism grows."),
    "application": ("Group chunks by recording (or site, or subject) in every split, choose models on the grouped score, and check the "
                    "support in each fold when classes are rare."),
    "assumptions": ("Constructed audio with a recording-level label, a single kind of background shift and tree and logistic models in place "
                    "of the chapter's CNN on mel spectrograms. Only 60 recordings, so the recording-level estimate itself varies by a few "
                    "hundredths between replicates, and the flexible-versus-simple gap on new recordings is small here (under 0.06 across the seeds tried)."),
    "prediction": ("With 8 chunks per recording, how does the flexible model's chunk-level cross-validation AUC compare with its AUC on "
                   "recordings it never saw?"),
    "prediction_options": ["About the same (within 0.03)", "About 0.1 higher", "About 0.2 higher"],
    "prediction_answer": 2,
    "prediction_feedback": {
        "correct": "Chunk-level cross-validation reports 0.895 and the model scores 0.684 on new recordings: 0.211 higher.",
        "incorrect": "Chunk-level cross-validation reports 0.895 and the model scores 0.684 on new recordings: 0.211 higher, because siblings of every held-out chunk sit in training.",
    },
    "check": "At 8 chunks per recording, chunk-level validation says the flexible model beats the simple one by 0.154. What do the other two scores say?",
    "answer": ("Recording-level cross-validation puts the flexible model 0.011 below the simple one, and on new recordings it is 0.009 above, "
               "a lead of at most a few hundredths either way. Chunk-level validation would pick the flexible model for its memorized "
               "fingerprints, a gain that is mostly absent on new recordings."),
    "provenance": "Constructed example: seeded synthetic audio, FFT features and two classifiers, measured by the chapter activity.",
    "apply": [
        "Decide the independent unit (recording, site, subject, time window) from the task and split by it in every cross-validation.",
        "Compare chunk-level and grouped scores once on your data: a large gap means the model can recognise sources.",
        "Choose between models on the grouped score, and report support per fold for rare classes.",
        "Hold out whole recordings for the final check, as the host's test set will."
    ],
    "honesty": ("Constructed data; the size of the optimism depends on how distinctive each recording's background is and how many chunks it "
                "yields. The flexible-model lead on new recordings is small in this generator, so the activity claims no real "
                "advantage for either model."),
}

EQUATIONS = [{"tex": r"\text{optimism} = \mathrm{AUC}_{\text{CV}} - \mathrm{AUC}_{\text{new recordings}}",
              "alt": "Optimism equals the cross-validation AUC minus the AUC on new recordings",
              "basis": "The activity's own measure of how far a validation scheme overstates new-recording accuracy; the chapter states the rule in words, without a display equation."}]
NCOLS = 2
HEIGHT = 4.4
SCHEMES = [("chunk_cv", "Chunk-level\nCV"), ("recording_cv", "Recording-level\nCV"), ("new_recordings", "New\nrecordings")]


def sub(a, b):
    """Difference of two values as the page shows them, so a hand calculation always adds up."""
    return float(fmt(a)) - float(fmt(b))


def draw(axes, result, parameter):
    left, right = axes
    for x, (key, name) in enumerate(SCHEMES):
        for offset, model, color in ((-0.2, "flexible", COLORS["terracotta"]), (0.2, "simple", COLORS["teal"])):
            value = result[model][key]
            left.bar([x + offset], [value], width=0.38, color=color, edgecolor=COLORS["ink"], lw=0.6,
                     label={"flexible": "Flexible model (trees)", "simple": "Simple model (logistic)"}[model] if x == 0 else None)
            dots = result["per_replicate"][model][key]
            left.scatter([x + offset + (i - (len(dots) - 1) / 2) * 0.05 for i in range(len(dots))], dots, s=10, color=COLORS["ink"], zorder=3)
            left.text(x + offset, max(dots) + 0.012, fmt(value, 2), ha="center", va="bottom", fontsize=10)
    left.axhline(0.5, color=COLORS["grey"], lw=0.8, ls=":")
    left.set_xticks(range(3), [s[1] for s in SCHEMES])
    left.set_ylim(0.4, 1.12)
    left.set_ylabel("AUC (dots: collections)")
    left.set_xlabel(f"Validation scheme, {parameter} chunk{'s' if parameter != 1 else ''} per recording")
    left.legend(loc="upper right", frameon=False, fontsize=10, ncol=1)

    for x, (key, name) in enumerate(SCHEMES):
        lead = [a - b for a, b in zip(result["per_replicate"]["flexible"][key], result["per_replicate"]["simple"][key])]
        mean = float(np.mean(lead))
        right.bar([x], [mean], width=0.55, color=[COLORS["terracotta"], COLORS["light"], COLORS["teal"]][x], edgecolor=COLORS["ink"], lw=0.6)
        right.scatter([x + (i - (len(lead) - 1) / 2) * 0.05 for i in range(len(lead))], lead, s=10, color=COLORS["ink"], zorder=3)
        right.text(x, max(max(lead), mean) + 0.012, signed(sub(result["flexible"][key], result["simple"][key])), ha="center", va="bottom", fontsize=10)
    right.axhline(0, color=COLORS["grey"], lw=0.8)
    right.set_xticks(range(3), [s[1] for s in SCHEMES])
    right.set_ylim(-0.12, 0.28)
    right.set_ylabel("Flexible minus simple AUC (dots: collections)")
    right.set_xlabel("What each scheme says about the model choice")


def explain(result, parameter):
    f, s = result["flexible"], result["simple"]
    optimism = sub(f["chunk_cv"], f["new_recordings"])
    lead_chunk, lead_group, lead_new = (sub(f[k], s[k]) for k in ("chunk_cv", "recording_cv", "new_recordings"))
    interpretation = (
        f"With {parameter} chunk{'s' if parameter != 1 else ''} per recording, the flexible model scores AUC {fmt(f['new_recordings'])} on new recordings. "
        f"Chunk-level cross-validation reports {fmt(f['chunk_cv'])}, an optimism of {fmt(f['chunk_cv'])} - {fmt(f['new_recordings'])} = {fmt(optimism)}; "
        f"recording-level cross-validation reports {fmt(f['recording_cv'])}. The flexible model's AUC minus the simple model's is {signed(lead_chunk)} "
        f"under chunk-level validation, {signed(lead_group)} under recording-level validation and {signed(lead_new)} on new recordings.")
    steps = [
        f"Chunk-level optimism, flexible model: {fmt(f['chunk_cv'])} - {fmt(f['new_recordings'])} = {fmt(optimism)}.",
        f"Recording-level error, flexible model: {fmt(f['recording_cv'])} - {fmt(f['new_recordings'])} = {fmt(sub(f['recording_cv'], f['new_recordings']))}.",
        f"Chunk-level lead of the flexible model: {fmt(f['chunk_cv'])} - {fmt(s['chunk_cv'])} = {fmt(lead_chunk)}.",
        f"Lead on new recordings: {fmt(f['new_recordings'])} - {fmt(s['new_recordings'])} = {fmt(lead_new)}.",
    ]
    metrics = {"Flexible, chunk-level CV": fmt(f["chunk_cv"]), "Flexible, recording-level CV": fmt(f["recording_cv"]),
               "Flexible, new recordings": fmt(f["new_recordings"]), "Simple, new recordings": fmt(s["new_recordings"]),
               "Chunk-level optimism (flexible)": signed(optimism)}
    alt = (f"Left: AUC of a flexible and a simple model under chunk-level cross-validation, recording-level cross-validation and on new recordings with "
           f"{parameter} chunks per recording; the flexible model scores {fmt(f['chunk_cv'])}, {fmt(f['recording_cv'])} and {fmt(f['new_recordings'])}. "
           f"Right: the flexible model's lead over the simple one under each scheme: {signed(lead_chunk)}, {signed(lead_group)} and {signed(lead_new)}.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    keys = sorted(results)
    opt = [results[k]["flexible"]["chunk_cv"] - results[k]["flexible"]["new_recordings"] for k in keys]
    assert abs(opt[0]) < 0.1, "with one chunk per recording the chunk split is not leaky"
    assert opt[-1] > opt[1] > opt[0], "optimism grows with chunks per recording"
    assert results[3]["flexible"]["chunk_cv"] - results[3]["flexible"]["new_recordings"] > 0.05, "chunk-level CV already optimistic at 3 chunks"
    for k in keys[1:]:
        res = results[k]
        assert abs(res["flexible"]["recording_cv"] - res["flexible"]["new_recordings"]) < 0.12, f"recording-level CV close to new recordings at {k}"
    for k in (8, 16):
        res = results[k]
        assert res["flexible"]["chunk_cv"] - res["flexible"]["new_recordings"] > 0.1, f"chunk-level CV optimistic at {k}"
        assert res["flexible"]["chunk_cv"] - res["simple"]["chunk_cv"] > 0.1, f"chunk-level CV favours the flexible model at {k}"
        assert res["flexible"]["new_recordings"] - res["simple"]["new_recordings"] < 0.06, f"small lead on new recordings at {k}"
        assert res["flexible"]["recording_cv"] - res["simple"]["recording_cv"] < 0.06, f"small lead under recording-level CV at {k}"
    r8 = results[8]
    assert fmt(r8["flexible"]["chunk_cv"]) == "0.895" and fmt(r8["flexible"]["new_recordings"]) == "0.684", "prediction feedback numbers"
    assert fmt(sub(r8["flexible"]["chunk_cv"], r8["flexible"]["new_recordings"])) == "0.211"
    assert fmt(sub(r8["flexible"]["chunk_cv"], r8["simple"]["chunk_cv"])) == "0.154", "check answer: chunk-level lead"
    assert fmt(sub(r8["flexible"]["recording_cv"], r8["simple"]["recording_cv"])) == "-0.011", "check answer: recording-level lead"
    assert fmt(sub(r8["flexible"]["new_recordings"], r8["simple"]["new_recordings"])) == "0.009", "check answer: new recordings lead"
