"""Chapter 47: Medical Imaging Patterns. Image-level against patient-grouped cross-validation, by images per patient."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

from kaggle_companion.activities._common import clean

SEED = 47
DRAWS = 8            # independent cohorts; every estimate is a mean over draws
PATIENTS = 150       # development patients; 600 further patients form the fresh test cohort
FRESH = 600
DIM = 12             # image features


def generate(n_patients, images_each, rng):
    """Patient-level label. Each image = the patient's own anatomy (shared by all their images) + a weak disease signal + image noise."""
    label = (rng.random(n_patients) < 0.5).astype(int)
    anatomy = 1.5 * rng.normal(size=(n_patients, DIM))
    patient = np.repeat(np.arange(n_patients), images_each)
    y = label[patient]
    X = anatomy[patient] + 0.8 * rng.normal(size=(len(patient), DIM))
    X[:, :3] += 0.5 * (2 * y[:, None] - 1)
    return X, y, patient


def model(seed=0):
    return ExtraTreesClassifier(n_estimators=40, max_features=0.5, random_state=seed, n_jobs=1)


def pooled_auc(X, y, splits):
    """Out-of-fold probabilities pooled over all folds, then one AUC."""
    oof = np.zeros(len(y))
    for fit, hold in splits:
        oof[hold] = model().fit(X[fit], y[fit]).predict_proba(X[hold])[:, 1]
    return roc_auc_score(y, oof)


def run(images_per_patient):
    rng = np.random.default_rng(SEED)
    X_new, y_new, _ = generate(FRESH, images_per_patient, rng)       # fresh patients: the truth the CV should predict
    rows = {k: [] for k in ("kfold", "grouped", "fresh", "overlap_kfold", "overlap_grouped")}
    for draw in range(DRAWS):
        X, y, patient = generate(PATIENTS, images_per_patient, rng)
        kfold = list(KFold(5, shuffle=True, random_state=draw).split(X))              # each image on its own
        grouped = list(GroupKFold(5).split(X, y, patient))                              # whole patients together
        rows["kfold"].append(pooled_auc(X, y, kfold))
        rows["grouped"].append(pooled_auc(X, y, grouped))
        rows["fresh"].append(roc_auc_score(y_new, model().fit(X, y).predict_proba(X_new)[:, 1]))
        # The chapter's check: do validation patients also appear in training? (share of validation images)
        rows["overlap_kfold"].append(np.mean([np.isin(patient[b], patient[a]).mean() for a, b in kfold]))
        rows["overlap_grouped"].append(np.mean([np.isin(patient[b], patient[a]).mean() for a, b in grouped]))
    mean = {k: float(np.mean(v)) for k, v in rows.items()}
    return clean({
        "images_per_patient": images_per_patient, **mean,
        "kfold_error": mean["kfold"] - mean["fresh"], "grouped_error": mean["grouped"] - mean["fresh"],
        "per_draw": {k: rows[k] for k in ("kfold", "grouped", "fresh")},
        "patients": PATIENTS, "images": PATIENTS * images_per_patient, "fresh_patients": FRESH, "draws": DRAWS,
    })
# notebook-end


SPEC = {
    "chapter": 47,
    "chapter_title": "Medical Imaging Patterns",
    "subtitle": "Keep every image from a patient in one fold, and check that no patient crosses the boundary.",
    "summary": ("Images from one patient resemble each other, so an image-level split lets a model recognise patients instead of "
                "disease. One demonstration measures how far image-level and patient-grouped cross-validation drift from the "
                "score on fresh patients as the number of images per patient grows."),
    "title": "Image-level and patient-grouped cross-validation against fresh patients, by images per patient",
    "question": "How far does image-level cross-validation overstate the score on new patients, and does grouping by patient remove the error?",
    "why": ("Several images, slides or tiles usually share one patient. The chapter's rule is to keep each patient in one validation "
            "fold; the size of the error it prevents depends on how many images each patient contributes, which this "
            "demonstration varies."),
    "method": ("Constructed cohorts of 150 patients with a patient-level label (half positive). Each image has 12 features: the "
               "patient's own anatomy (shared by all of their images and larger than the within-patient noise), a weak disease "
               "signal on three features, and image noise. An extremely randomized trees classifier is scored by out-of-fold AUC "
               "pooled over five folds, split two ways: ordinary shuffled folds that treat each image alone, and GroupKFold by "
               "patient. Both are compared with the same model's AUC on 600 fresh patients, averaged over 8 cohorts. The control "
               "is the number of images per patient."),
    "control": {"key": "images_per_patient", "label": "Images per patient",
                "values": [1, 2, 4, 8], "default": 4,
                "value_labels": ["1: one image each", "2", "4", "8: many images each"]},
    "source_section": "Keep all images from each patient in one validation fold",
    "symbols": ("AUC is the area under the ROC curve of the pooled out-of-fold probabilities. The error of a validation scheme "
                "is its AUC minus the AUC of the same model on fresh patients."),
    "explanation": ("When a patient's images are split across folds, the model sees the patient's anatomy in training and meets "
                    "it again in validation, so it can read the label from who the patient is. With one image per patient nothing "
                    "can be shared and the two schemes agree. Each added image per patient puts more near-copies across the fold "
                    "boundary and the image-level estimate climbs, while the grouped estimate stays near the fresh-patient score."),
    "application": ("Split by patient (or slide, or study) before anything else, assert that no patient appears in both a training and a "
                    "validation fold, and read an image-level estimate on multi-image data as a measure of memorised patients."),
    "assumptions": ("Constructed features and a patient-level label, so every image of a patient shares the answer; with a label that "
                    "varies within a patient (a lesion on some slices only) the leak is smaller. The tree model memorises easily; "
                    "a regularised model leaks less. Grouped folds run slightly below the fresh-patient score here, by up to "
                    "about 0.04 at eight images. A side check found that training on 120 rather than 150 patients does not explain "
                    "it; with only 150 independent patients per cohort the grouped estimate is noisy, and the small gap changes "
                    "sign under other seeds. It is a property of this setup, not a rule."),
    "prediction": "With 4 images per patient, how does image-level 5-fold AUC compare with the AUC on fresh patients?",
    "prediction_options": ["Within 0.02 of it", "About 0.17 higher", "Lower, because folds train on less data"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "Image-level folds report 0.874 against 0.707 on fresh patients, and 98.8% of validation images belong to a patient already in training.",
        "incorrect": "Image-level folds report 0.874 against 0.707 on fresh patients: 98.8% of validation images belong to a patient already in training, so the model recognises the patient.",
    },
    "check": "With one image per patient the two splits agree, and with eight they differ by about 0.26 AUC. What should you check before trusting any image-level estimate?",
    "answer": ("How many images each patient contributes and whether any patient crosses the boundary. At one image per patient "
               "the image-level AUC is 0.700 against 0.701 grouped, but at eight it is 0.924 against 0.667 grouped while fresh "
               "patients score 0.706: the error grows from nothing to 0.218 purely from repeated patients."),
    "provenance": "Constructed example: seeded synthetic patient cohorts and an extremely randomized trees classifier, measured by the chapter activity.",
    "apply": [
        "Build the fold column from patient, study or slide identifiers, never from image rows, and keep it fixed for every model.",
        "Assert that no patient identifier appears in both a training fold and its validation fold.",
        "Report images per patient next to the score; the more repeats, the larger an image-level error can be.",
        "Check provenance for hidden links such as duplicate scans, shared slides or external data from the same patients.",
    ],
    "honesty": "Constructed data with patient-level labels; the sizes of these effects are properties of this generator, not a competition result.",
}

EQUATIONS = [{"tex": r"\text{error}=\mathrm{AUC}_{\mathrm{CV}}-\mathrm{AUC}_{\mathrm{fresh\ patients}}",
              "alt": "error equals the cross-validation AUC minus the AUC on fresh patients",
              "basis": "The activity's own label for a validation scheme's error; the chapter states the patient-fold rule in prose (Keep all images from each patient in one validation fold)."}]
NCOLS = 2
HEIGHT = 4.4


def draw(axes, result, parameter):
    ax, bx = axes
    names = ["Image-level\n5-fold", "Grouped by\npatient", "Fresh\npatients"]
    keys = ["kfold", "grouped", "fresh"]
    colors = [COLORS["terracotta"], COLORS["teal"], COLORS["light"]]
    ax.bar(range(3), [result[k] for k in keys], width=0.55, color=colors, edgecolor=COLORS["ink"], lw=0.6)
    for x, k in enumerate(keys):
        dots = result["per_draw"][k]
        ax.scatter([x + (i - (len(dots) - 1) / 2) * 0.05 for i in range(len(dots))], dots, s=14, color=COLORS["ink"], zorder=3,
                   label="One cohort" if x == 0 else None)
        ax.text(x, max(result[k], max(dots)) + 0.012, fmt(result[k]), ha="center", va="bottom", fontsize=10)
    ax.axhline(result["fresh"], color=COLORS["gold"], ls=(0, (4, 3)), lw=1.6,
               label=f"Score on fresh patients: {fmt(result['fresh'])}")
    ax.set_xticks(range(3), names)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("AUC (axis starts at chance, 0.5)")
    ax.set_xlabel(f"Validation scheme, {parameter} images per patient")
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    bx.bar([0, 1], [result["overlap_kfold"], result["overlap_grouped"]], width=0.5,
           color=[COLORS["terracotta"], COLORS["teal"]], edgecolor=COLORS["ink"], lw=0.6)
    for x, v in enumerate([result["overlap_kfold"], result["overlap_grouped"]]):
        bx.text(x, v + 0.02, f"{100 * v:.1f}%", ha="center", va="bottom", fontsize=10)
    bx.set_xticks([0, 1], ["Image-level\n5-fold", "Grouped by\npatient"])
    bx.set_ylim(0, 1.15)
    bx.set_ylabel("Validation images whose patient is also in training")
    bx.set_xlabel("Validation scheme")


def explain(result, parameter):
    k, g, f = result["kfold"], result["grouped"], result["fresh"]
    ke, ge = result["kfold_error"], result["grouped_error"]
    shown = round(k, 3) - round(f, 3)           # difference of the displayed numbers, so the sum checks by hand
    ordered = f"{fmt(k)} - {fmt(f)} = {fmt(shown)}" if shown >= 0 else f"{fmt(f)} - {fmt(k)} = {fmt(-shown)} below"
    interpretation = (
        f"With {parameter} image{'s' if parameter != 1 else ''} per patient the same model scores AUC {fmt(f)} on fresh patients. "
        f"Image-level folds report {fmt(k)}: {ordered} (their error). Grouped folds report {fmt(g)}, which is {signed(ge)} from the "
        f"fresh score. {100 * result['overlap_kfold']:.1f}% of image-level validation images belong to a patient that is also in "
        f"training, against {100 * result['overlap_grouped']:.1f}% when grouped."
        + (" With one image per patient nothing can be shared and both splits agree." if parameter == 1 else
           " The image-level number measures how well the model recognises patients it has already seen."))
    steps = [f"Image-level: {fmt(k)} - {fmt(f)} = {signed(ke)} (estimate minus fresh patients).",
             f"Grouped by patient: {fmt(g)} - {fmt(f)} = {signed(ge)}.",
             f"Image-level minus grouped: {fmt(k)} - {fmt(g)} = {signed(k - g)}.",
             f"Patient overlap between training and validation: {100 * result['overlap_kfold']:.1f}% (image-level), {100 * result['overlap_grouped']:.1f}% (grouped)."]
    metrics = {"Image-level 5-fold AUC": fmt(k), "Grouped AUC": fmt(g), "AUC on fresh patients": fmt(f),
               "Image-level error": signed(ke), "Grouped error": signed(ge),
               "Patients shared across folds (image-level)": f"{100 * result['overlap_kfold']:.1f}%"}
    alt = (f"Two panels at {parameter} images per patient. Left, bars of AUC for image-level folds ({fmt(k)}), patient-grouped folds "
           f"({fmt(g)}) and fresh patients ({fmt(f)}), with a dot for each of {result['draws']} cohorts. Right, the share of validation "
           f"images whose patient is also in training: {100 * result['overlap_kfold']:.1f}% image-level and {100 * result['overlap_grouped']:.1f}% grouped.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for v, res in results.items():
        assert res["overlap_grouped"] == 0, "grouped folds must never share a patient"
        assert abs(res["grouped_error"]) < 0.05, f"grouped estimate should stay within 0.05 of fresh patients at {v}"
        assert res["grouped_error"] > -0.045, "assumptions text: grouped below fresh by up to about 0.04"
    assert results[1]["overlap_kfold"] == 0 and abs(results[1]["kfold_error"]) < 0.03, "one image per patient: no leak"
    errors = [results[v]["kfold_error"] for v in (1, 2, 4, 8)]
    assert errors[0] < errors[1] < errors[2] < errors[3], "image-level error grows with images per patient"
    assert errors[2] > 0.1 and results[4]["overlap_kfold"] > 0.95, "four images per patient: large leak"
    r1, r4, r8 = results[1], results[4], results[8]
    assert fmt(r4["kfold"]) == "0.874" and fmt(r4["fresh"]) == "0.707" and f"{100 * r4['overlap_kfold']:.1f}" == "98.8"
    assert 0.16 < r4["kfold_error"] < 0.18, "prediction option says about 0.17"
    assert fmt(r1["kfold"]) == "0.700" and fmt(r1["grouped"]) == "0.701"
    assert fmt(r8["kfold"]) == "0.924" and fmt(r8["grouped"]) == "0.667" and fmt(r8["fresh"]) == "0.706"
    assert fmt(r8["kfold_error"]) == "0.218" and fmt(r8["kfold"] - r8["grouped"]) == "0.258", "check text says about 0.26"
