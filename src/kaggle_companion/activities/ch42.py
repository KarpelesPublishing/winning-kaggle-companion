"""Chapter 42: Image Augmentation. Shift, flip and rotation augmentation on scikit-learn's digits, by training-set size."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import warnings

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

from kaggle_companion.activities._common import clean

warnings.filterwarnings("ignore")      # lbfgs convergence notices on tiny training sets
SEED = 42
REPLICATES = 12          # independent train/test splits per training size; results are means over them
TEST_ROWS = 500
POLICIES = ("none", "duplicate", "shift", "hflip", "rot180")


def shifted(image, rng):
    """Move the 8 x 8 image by -1, 0 or +1 pixel in each direction, padding with zeros."""
    dx, dy = rng.integers(-1, 2, 2)
    out = np.zeros_like(image)
    src = image[max(0, -dy):8 - max(0, dy), max(0, -dx):8 - max(0, dx)]
    out[max(0, dy):max(0, dy) + src.shape[0], max(0, dx):max(0, dx) + src.shape[1]] = src
    return out


TRANSFORMS = {"duplicate": lambda im, rng: im,                 # control: same rows twice, no change to the picture
              "shift": shifted,
              "hflip": lambda im, rng: im[:, ::-1],            # mirror left to right
              "rot180": lambda im, rng: im[::-1, ::-1]}        # turn upside down (a 6 becomes a 9)


def fit_predict(images, labels, test_images, policy, rng):
    """Train on the originals plus one transformed copy of each (or on the originals alone) and predict the test images."""
    if policy != "none":
        copies = np.stack([TRANSFORMS[policy](im, rng) for im in images])
        images, labels = np.concatenate([images, copies]), np.concatenate([labels, labels])
    model = LogisticRegression(max_iter=300).fit(images.reshape(len(images), -1), labels)
    return model.predict(test_images.reshape(len(test_images), -1))


def run(training_images):
    digits = load_digits()
    pictures, labels = digits.images / 16.0, digits.target     # 1,797 real 8 x 8 grayscale scans, scaled to [0, 1]
    accuracy = {v: {p: [] for p in POLICIES} for v in ("scanned", "mirrored")}
    swapped = {p: [] for p in ("none", "rot180")}              # share of true 6s and 9s predicted as the other one
    for r in range(REPLICATES):
        rng = np.random.default_rng(SEED * 100 + r)
        order = rng.permutation(len(labels))
        test, train = order[:TEST_ROWS], order[TEST_ROWS:TEST_ROWS + training_images]
        mirror = rng.random(len(labels)) < 0.5
        variants = {"scanned": pictures,                                                   # digits as scanned: orientation carries the label
                    "mirrored": np.where(mirror[:, None, None], pictures[:, :, ::-1], pictures)}   # constructed: half the images mirrored, labels unchanged
        for variant, images in variants.items():
            for policy in POLICIES:
                pred = fit_predict(images[train], labels[train], images[test], policy, np.random.default_rng(r))
                accuracy[variant][policy].append(float((pred == labels[test]).mean()))
                if variant == "scanned" and policy in swapped:
                    six_nine = np.isin(labels[test], (6, 9))
                    swapped[policy].append(float((pred[six_nine] == 15 - labels[test][six_nine]).mean()))
    mean = {v: {p: float(np.mean(a)) for p, a in d.items()} for v, d in accuracy.items()}
    return clean({"training_images": training_images, "accuracy": mean, "swapped_six_nine": {p: float(np.mean(a)) for p, a in swapped.items()},
                  "replicates": REPLICATES, "test_images": TEST_ROWS})
# notebook-end


SPEC = {
    "chapter": 42,
    "chapter_title": "Image Augmentation",
    "subtitle": "An augmentation is an assumption about which changes keep the label; test it before trusting it.",
    "summary": ("Shifts, mirror flips and half-turns are all easy to add and only some are valid for a given task. One demonstration "
                "trains a classifier on real digit images with each augmentation, on digits as scanned and on a constructed variant "
                "where half the images are mirrored, and measures the accuracy change against no augmentation."),
    "title": "Which augmentations help? Digits as scanned and digits in random mirror orientation",
    "question": "When does adding transformed copies of the training images raise held-out accuracy, and when does it lower it?",
    "why": ("A default augmentation stack can silently teach the model something false. The chapter asks for a label-preservation check "
            "per transform and for every policy to be compared with an unaugmented baseline on unchanged assessment images."),
    "method": ("scikit-learn's bundled digits dataset (1,797 real 8 x 8 grayscale scans). The control is the number of training "
               "images (100, 300 or 900); 500 other images are the test set. A multinomial logistic regression is trained on the "
               "originals alone, or on the originals plus one transformed copy of each: an unchanged duplicate (control for "
               "doubling the rows), a random shift of up to one pixel, a left-right flip, or a half-turn. This is run on the "
               "digits as scanned, and on a constructed variant in which each image, in training and test alike, is "
               "mirrored with probability one half and keeps its label. Results are means over 12 random splits."),
    "control": {"key": "training_images", "label": "Training images (before augmentation)",
                "values": [100, 300, 900], "default": 100,
                "value_labels": ["100", "300", "900"]},
    "source_section": "Decide Which Changes Preserve the Target",
    "symbols": ("Delta is the change in held-out accuracy from adding a transformed copy of each training image, compared with training on "
                "the originals alone; positive means the augmentation helped."),
    "explanation": ("A flip, shift or turn only adds information when the transformed picture is a legitimate example of the same class "
                    "that the test images could contain. For scanned digits none of the three is: a flip or a half-turn produces many pictures "
                    "that never occur at test time (a half-turned 6 is a 9), and even a one-pixel shift is large on an 8 x 8 grid. "
                    "When the test images are themselves randomly mirrored, the flip is exactly the missing invariance and helps most with little data."),
    "application": ("Write down what must stay invariant, check each transform against it on a few images, and compare every "
                    "candidate with the unaugmented baseline (and an unchanged-duplicate control) on untouched assessment images."),
    "assumptions": ("Real digits at 8 x 8 resolution and a linear classifier, so conclusions about shifts do not carry over to larger images or "
                    "convolutional networks, where small shifts are often valid. The mirrored variant is constructed to make the flip "
                    "valid. One transformed copy per image; no test-time augmentation."),
    "prediction": ("With 100 training images, horizontal-flip augmentation changes held-out accuracy by what, on digits as scanned and on digits "
                   "in random mirror orientation?"),
    "prediction_options": ["It helps on both", "It helps the mirrored digits and hurts the scanned ones", "It hurts on both"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "Accuracy rises by 0.091 on the mirrored digits (0.749 to 0.840) and falls by 0.042 on the scanned ones (0.882 to 0.840).",
        "incorrect": "Accuracy rises by 0.091 on the mirrored digits (0.749 to 0.840) and falls by 0.042 on the scanned ones (0.882 to 0.840): the same transform is valid for one task and invalid for the other.",
    },
    "check": "With scanned digits, a half-turn keeps the label. Which classes does it damage, and how can you tell the damage comes from the transform rather than from doubling the training rows?",
    "answer": ("A half-turn turns a 6 into a 9 while labeling it 6, so the model confuses the two: with 100 training images 20% of true 6s and "
               "9s are predicted as the other digit, against 0% with no augmentation, and accuracy falls by 0.096. An unchanged "
               "duplicate of every image changes accuracy by only +0.003, so the loss comes from the transform, not from the extra rows."),
    "provenance": ("Real data: scikit-learn's bundled digits (8 x 8 scans). Constructed: the randomly mirrored variant, built from the same images. "
                   "Measured by the chapter activity."),
    "apply": [
        "Write the invariance down (class, box, mask, orientation) and check each transform on a handful of images before training.",
        "Compare every augmentation with an unaugmented baseline and an unchanged-duplicate control on the same untouched assessment images.",
        "Look at per-class errors for confusion pairs a transform could create (6 and 9, left and right, mirrored text).",
        "Expect valid augmentation to matter most with little data; its gain shrinks as the training set grows, while an invalid transform keeps costing accuracy.",
    ],
    "honesty": ("Real small images and a linear model. Shifts hurt here, which would likely not hold for larger images or convolutional "
                "networks; the mirrored variant is constructed so that the flip is valid, and the result says nothing about natural photographs. "
                "The size of each gain and loss moves noticeably between random draws; the signs did not."),
}

EQUATIONS = [{"tex": r"\Delta = \mathrm{accuracy}_{\mathrm{augmented}} - \mathrm{accuracy}_{\mathrm{original\ only}}",
              "alt": "Delta equals the held-out accuracy with augmentation minus the held-out accuracy with the original images only",
              "basis": "The activity's own comparison for the Chapter 42 rule (Decide Which Changes Preserve the Target); the chapter has no display equation."}]
NCOLS = 2
HEIGHT = 4.4
VARIANTS = [("scanned", "Digits as scanned"), ("mirrored", "Digits in random mirror orientation")]
TRANSFORM_LABELS = [("shift", "Shift\n(up to 1 px)"), ("hflip", "Left-right\nflip"), ("rot180", "Half-turn")]


def draw(axes, result, parameter):
    xs = list(range(len(TRANSFORM_LABELS)))
    for ax, (variant, name) in zip(axes, VARIANTS):
        acc = result["accuracy"][variant]
        deltas = [acc[k] - acc["none"] for k, _ in TRANSFORM_LABELS]
        colors = [COLORS["teal"] if d > 0 else COLORS["terracotta"] for d in deltas]
        ax.bar(xs, deltas, color=colors, edgecolor=COLORS["ink"], lw=0.6, width=0.6)
        for x, d in zip(xs, deltas):
            ax.text(x, d + (0.006 if d >= 0 else -0.006), signed(d), ha="center", va="bottom" if d >= 0 else "top", fontsize=10)
        ax.axhline(0, color=COLORS["ink"], lw=1.0)
        ax.set_xticks(xs, [label for _, label in TRANSFORM_LABELS], fontsize=10)
        ax.set_ylim(-0.17, 0.17)
        ax.set_ylabel("Accuracy change vs original images only")
        ax.set_xlabel(f"{name}\nno augmentation: {fmt(acc['none'])} accuracy")


def explain(result, parameter):
    s, m = result["accuracy"]["scanned"], result["accuracy"]["mirrored"]
    flip_scanned, flip_mirrored = s["hflip"] - s["none"], m["hflip"] - m["none"]
    rot = s["rot180"] - s["none"]
    doubling = s["duplicate"] - s["none"]
    interpretation = (
        f"With {parameter} training images, a left-right flip changes accuracy by {fmt(s['hflip'])} - {fmt(s['none'])} = {signed(flip_scanned)} "
        f"on digits as scanned and by {fmt(m['hflip'])} - {fmt(m['none'])} = {signed(flip_mirrored)} on digits in random mirror orientation. "
        f"A one-pixel shift changes it by {signed(s['shift'] - s['none'])} and {signed(m['shift'] - m['none'])}, and a half-turn by {signed(rot)} and "
        f"{signed(m['rot180'] - m['none'])}. The half-turn makes {round(100 * result['swapped_six_nine']['rot180'])}% of true 6s and 9s "
        f"be predicted as the other digit, against {round(100 * result['swapped_six_nine']['none'])}% without augmentation. An unchanged duplicate "
        f"of every image changes accuracy by {signed(doubling)}.")
    steps = [
        f"Flip on scanned digits: {fmt(s['hflip'])} - {fmt(s['none'])} = {signed(flip_scanned)}.",
        f"Flip on mirrored digits: {fmt(m['hflip'])} - {fmt(m['none'])} = {signed(flip_mirrored)}.",
        f"Half-turn on scanned digits: {fmt(s['rot180'])} - {fmt(s['none'])} = {signed(rot)}.",
        f"Unchanged duplicate on scanned digits: {fmt(s['duplicate'])} - {fmt(s['none'])} = {signed(doubling)}.",
    ]
    metrics = {"No augmentation, scanned": fmt(s["none"]), "No augmentation, mirrored": fmt(m["none"]),
               "Flip, scanned": fmt(s["hflip"]), "Flip, mirrored": fmt(m["hflip"]), "Half-turn, scanned": fmt(s["rot180"]),
               "6/9 swapped by half-turn": f"{round(100 * result['swapped_six_nine']['rot180'])}%"}
    alt = (f"Two bar charts of accuracy change against no augmentation with {parameter} training images. Digits as scanned: shift "
           f"{signed(s['shift'] - s['none'])}, flip {signed(flip_scanned)}, half-turn {signed(rot)}. Digits in random mirror orientation: shift "
           f"{signed(m['shift'] - m['none'])}, flip {signed(flip_mirrored)}, half-turn {signed(m['rot180'] - m['none'])}.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for n, res in results.items():
        s, m = res["accuracy"]["scanned"], res["accuracy"]["mirrored"]
        for p in ("shift", "hflip", "rot180"):
            assert s[p] < s["none"] - 0.01, f"every augmentation hurts scanned digits at {n}: {p}"
        assert m["hflip"] > m["none"] + 0.01, f"flip helps mirrored digits at {n}"
        assert m["shift"] < m["none"] and m["rot180"] < m["none"] - 0.05, f"invalid transforms hurt mirrored digits at {n}"
        assert abs(s["duplicate"] - s["none"]) < 0.01, f"doubling alone is not the cause at {n}"
        assert res["swapped_six_nine"]["rot180"] > 0.08 and res["swapped_six_nine"]["none"] < 0.01, f"half-turn confuses 6 and 9 at {n}"
    gains = [results[n]["accuracy"]["mirrored"]["hflip"] - results[n]["accuracy"]["mirrored"]["none"] for n in (100, 300, 900)]
    assert gains[0] > gains[1] > gains[2] > 0, "the valid flip's gain shrinks with data"
    harm = [results[n]["accuracy"]["scanned"]["rot180"] - results[n]["accuracy"]["scanned"]["none"] for n in (100, 300, 900)]
    assert max(harm) < -0.05, "the half-turn keeps costing accuracy"
    small = results[100]
    assert fmt(small["accuracy"]["mirrored"]["none"]) == "0.749" and fmt(small["accuracy"]["mirrored"]["hflip"]) == "0.840"
    assert fmt(small["accuracy"]["scanned"]["none"]) == "0.882" and fmt(small["accuracy"]["scanned"]["hflip"]) == "0.840"
    assert fmt(small["accuracy"]["mirrored"]["hflip"] - small["accuracy"]["mirrored"]["none"]) == "0.091"
    assert fmt(small["accuracy"]["scanned"]["hflip"] - small["accuracy"]["scanned"]["none"]) == "-0.042"
    assert round(100 * small["swapped_six_nine"]["rot180"]) == 20 and round(100 * small["swapped_six_nine"]["none"]) == 0
    assert fmt(small["accuracy"]["scanned"]["rot180"] - small["accuracy"]["scanned"]["none"]) == "-0.096"
    assert fmt(small["accuracy"]["scanned"]["duplicate"] - small["accuracy"]["scanned"]["none"]) == "0.003"
    assert small["replicates"] == 12, "method text: means over 12 random splits"
