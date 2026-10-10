"""Chapter 43: Image Classification Tricks. Random against grouped validation when several crops come from one source image."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from scipy.ndimage import zoom
from sklearn.datasets import load_digits
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsClassifier

from kaggle_companion.activities._common import clean

SEED = 43
REPLICATES = 8          # independent source-image draws per control value; results are means over them
SOURCES, TEST_SOURCES = 300, 400
CROP = 14               # crops are 14 x 14 windows of a 16 x 16 image
OFFSETS = [(i, j) for i in range(3) for j in range(3)]    # nine possible, heavily overlapping, window positions


def load_images():
    """Real digit scans: scikit-learn's bundled 8 x 8 digits, enlarged to 16 x 16 by bilinear interpolation."""
    digits = load_digits()
    return np.stack([zoom(image / 16.0, 2, order=1) for image in digits.images]), digits.target


def make_crops(images, labels, crops_per_image, rng):
    """Cut `crops_per_image` windows out of every source image; remember which source each crop came from."""
    X, y, source = [], [], []
    for n, (image, label) in enumerate(zip(images, labels)):
        for position in rng.choice(len(OFFSETS), crops_per_image, replace=False):
            i, j = OFFSETS[position]
            X.append(image[i:i + CROP, j:j + CROP].ravel())
            y.append(label)
            source.append(n)
    return np.array(X), np.array(y), np.array(source)


def cross_validated_accuracy(splitter, X, y, groups=None):
    """Pool the held-out predictions of every fold and score them once."""
    pred = np.zeros(len(y), dtype=int)
    for fit, out in splitter.split(X, y, groups):
        pred[out] = KNeighborsClassifier(1).fit(X[fit], y[fit]).predict(X[out])
    return float((pred == y).mean())


def run(crops_per_image):
    images, labels = load_images()
    scores = {"random": [], "grouped": [], "heldout": []}
    for r in range(REPLICATES):
        rng = np.random.default_rng(SEED * 100 + r)
        order = rng.permutation(len(labels))
        train, test = order[:SOURCES], order[1000:1000 + TEST_SOURCES]       # test sources never appear in training
        X, y, source = make_crops(images[train], labels[train], crops_per_image, rng)
        X_test, y_test, _ = make_crops(images[test], labels[test], crops_per_image, rng)
        scores["random"].append(cross_validated_accuracy(KFold(5, shuffle=True, random_state=0), X, y))   # crops split at random
        scores["grouped"].append(cross_validated_accuracy(GroupKFold(5), X, y, source))                   # whole source images split
        scores["heldout"].append(float((KNeighborsClassifier(1).fit(X, y).predict(X_test) == y_test).mean()))
    mean = {k: float(np.mean(v)) for k, v in scores.items()}
    return clean({"crops_per_image": crops_per_image, "accuracy": mean, "training_crops": SOURCES * crops_per_image,
                  "random_overstates_in_share_of_worlds": float(np.mean(np.array(scores["random"]) > np.array(scores["heldout"]))),
                  "replicates": REPLICATES, "source_images": SOURCES})
# notebook-end


SPEC = {
    "chapter": 43,
    "chapter_title": "Image Classification Tricks",
    "subtitle": "Crops of one image are not independent examples: split by source image.",
    "summary": ("Cutting several overlapping crops from each image is a common way to get more training data, and a random split "
                "then puts siblings on both sides of the validation boundary. One demonstration measures how far a random-split score, a "
                "split by source image and a truly held-out score drift apart as the number of crops per image grows."),
    "title": "Random and grouped validation for crops of the same image",
    "question": "How much does a random split of crops overstate accuracy compared with splitting by source image, and how does that grow with crops per image?",
    "why": ("A validation score that cannot be reproduced on new images misleads every later decision, including which crop policy, "
            "resolution or backbone to keep. The chapter asks that the original image identity survive preprocessing so the split respects it."),
    "method": ("Real data: scikit-learn's bundled digits, enlarged to 16 x 16. 300 source images are each cut into 1, 3 or 9 "
               "overlapping 14 x 14 crops (the control). A one-nearest-neighbor classifier on pixels is scored three ways: "
               "5-fold cross-validation with crops split at random, 5-fold cross-validation with whole source images grouped, and "
               "accuracy on the crops of 400 source images never seen in training. Predictions are pooled before scoring; results are means over 8 draws."),
    "control": {"key": "crops_per_image", "label": "Crops cut from each source image",
                "values": [1, 3, 9], "default": 9,
                "value_labels": ["1 (no siblings)", "3", "9"]},
    "source_section": "Separate Development From Assessment",
    "symbols": ("Overstatement is the cross-validated accuracy minus the accuracy on new source images; it is positive when validation "
                "describes the training images better than new ones."),
    "explanation": ("Overlapping crops of one image are nearly the same picture. A random split leaves a crop's siblings in the "
                    "training folds, and a nearest-neighbor model simply copies a sibling's label. Splitting by source image removes the "
                    "siblings, so the score matches new images. With one crop per image there are no siblings and the two splits agree."),
    "application": ("Carry the source image ID through every crop and split by it for validation and early stopping; keep a separate "
                    "assessment of whole images, and never tune crop policy on a random split of crops."),
    "assumptions": ("Real 16 x 16 digit images and a one-nearest-neighbor classifier, which memorizes nearby examples and so shows the leak "
                    "clearly; a linear or heavily regularized model would leak less. Crops overlap heavily by construction. "
                    "Grouped validation trains on 80% of the sources and sits slightly below the held-out score."),
    "prediction": ("Nine overlapping crops are cut from each of 300 source images and split at random into five folds. How does the cross-validated "
                   "accuracy compare with the accuracy on crops of new source images?"),
    "prediction_options": ["About the same, within 0.01", "About 0.03 higher", "About 0.10 higher"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "Random cross-validation reports 0.995 against 0.962 on new images: 0.033 higher, an error rate of 0.5% against 3.8%.",
        "incorrect": "Random cross-validation reports 0.995 against 0.962 on new images: 0.033 higher, which turns a 3.8% error rate into a claimed 0.5%.",
    },
    "check": "At one crop per image the random split is honest, but at nine it is not. What changed, and what does the grouped split report at nine?",
    "answer": ("With one crop per image no sibling can sit in a training fold, so random and grouped splits agree (0.877 against 0.885). At nine, every "
               "crop has near-duplicate siblings in the training folds and a nearest-neighbor model copies their label, so random "
               "validation reaches 0.995. Grouping by source image gives 0.957, within 0.005 of the 0.962 on new images."),
    "provenance": "Real data: scikit-learn's bundled digits, enlarged to 16 x 16; the crop construction and the classifier are constructed and measured by the chapter activity.",
    "apply": [
        "Keep a source-image ID on every crop, tile or augmented copy, and split, early-stop and tune by that ID.",
        "Treat a validation score that rises as you add crops per image with suspicion: part of the rise is siblings, not skill.",
        "Hold out whole images for the final assessment of the selected recipe, and cut crops after the split, never before.",
        "If a quick random split is unavoidable, compare it with a grouped split on a sample and report the gap.",
    ],
    "honesty": ("Real digit images and a memorizing classifier: the overstatement is modest (0.03) here and would be larger for noisy "
                "labels or a model that memorizes more, and smaller for models that do not. Scores at one crop per image differ by noise only."),
}

EQUATIONS = [{"tex": r"\mathrm{overstatement} = \mathrm{accuracy}_{\mathrm{CV}} - \mathrm{accuracy}_{\mathrm{new\ images}}",
              "alt": "Overstatement equals the cross-validated accuracy minus the accuracy on new source images",
              "basis": "The activity's own gap measure for the Chapter 43 rule (Separate Development From Assessment); the chapter has no display equation."}]
NCOLS = 2
HEIGHT = 4.4
SCHEMES = [("random", "Random split\nof crops", COLORS["terracotta"]), ("grouped", "Split by\nsource image", COLORS["teal"]),
           ("heldout", "New source\nimages", COLORS["navy"])]


def draw(axes, result, parameter):
    left, right = axes
    xs = list(range(len(SCHEMES)))
    acc = [result["accuracy"][k] for k, _, _ in SCHEMES]
    left.axhline(result["accuracy"]["heldout"], color=COLORS["ink"], ls=(0, (4, 3)), lw=1.4, label="Accuracy on new images")
    for x, v, (_, _, color) in zip(xs, acc, SCHEMES):
        left.scatter([x], [v], s=140, color=color, edgecolor=COLORS["ink"], lw=0.8, zorder=3)
        below = v < result["accuracy"]["heldout"] - 1e-9          # keep labels clear of the reference line
        left.text(x, v - 0.008 if below else v + 0.008, fmt(v), ha="center", va="top" if below else "bottom", fontsize=10)
    left.set_xticks(xs, [name for _, name, _ in SCHEMES], fontsize=10)
    left.set_xlim(-0.5, 2.8)
    left.set_ylim(0.85, 1.02)
    left.set_ylabel("Accuracy")
    left.set_xlabel(f"How the crops are scored, {parameter} per image")
    left.legend(loc="upper right", frameon=False, fontsize=10)

    err = [100 * (1 - v) for v in acc]
    right.bar(xs, err, color=[c for _, _, c in SCHEMES], edgecolor=COLORS["ink"], lw=0.6, width=0.6)
    for x, v in zip(xs, err):
        right.text(x, v + 0.15, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    right.set_xticks(xs, [name for _, name, _ in SCHEMES], fontsize=10)
    right.set_ylim(0, max(err) * 1.25)
    right.set_ylabel("Error rate (%)")
    right.set_xlabel("Same scores as error rates")


def explain(result, parameter):
    a = result["accuracy"]
    over = a["random"] - a["heldout"]
    gap = a["grouped"] - a["heldout"]
    ratio = (1 - a["heldout"]) / (1 - a["random"])
    interpretation = (
        f"With {parameter} crop{'s' if parameter != 1 else ''} per source image ({result['training_crops']:,} training crops), a random split reports "
        f"{fmt(a['random'])} and a split by source image {fmt(a['grouped'])}, while new source images score {fmt(a['heldout'])}. "
        + (f"The random split overstates accuracy by {fmt(a['random'])} - {fmt(a['heldout'])} = {fmt(over)}; "
           if over > 0 else
           f"The random split does not overstate accuracy here: {fmt(a['random'])} - {fmt(a['heldout'])} = {fmt(over)}; ")
        + f"the grouped split is {fmt(a['grouped'])} - {fmt(a['heldout'])} = {fmt(gap)} away. "
        + (f"The random split claims an error rate {fmt(ratio, 1)} times smaller than the real one, and "
           if ratio > 1 else "Its error rate is not smaller than the real one, and it ")
        + f"overstated accuracy in {round(100 * result['random_overstates_in_share_of_worlds'])}% of {result['replicates']} draws.")
    steps = [
        f"Random-split overstatement: {fmt(a['random'])} - {fmt(a['heldout'])} = {signed(over)}.",
        f"Grouped-split difference: {fmt(a['grouped'])} - {fmt(a['heldout'])} = {signed(gap)}.",
        f"Error rate on new images against the random-split claim: {fmt(1 - a['heldout'])} / {fmt(1 - a['random'])} = {fmt(ratio, 1)}.",
        f"Training crops: {result['source_images']} sources * {parameter} = {result['training_crops']:,}.",
    ]
    metrics = {"Random split": fmt(a["random"]), "Split by source image": fmt(a["grouped"]), "New source images": fmt(a["heldout"]),
               "Random overstatement": signed(over), "Grouped difference": signed(gap)}
    alt = (f"Left: accuracy under a random split of crops ({fmt(a['random'])}), a split by source image ({fmt(a['grouped'])}) and on new "
           f"source images ({fmt(a['heldout'])}) with {parameter} crops per image. Right: the same scores as error rates.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    one, three, nine = results[1], results[3], results[9]
    assert abs(one["accuracy"]["random"] - one["accuracy"]["grouped"]) < 0.02, "no siblings at one crop per image"
    assert abs(one["accuracy"]["random"] - one["accuracy"]["heldout"]) < 0.03
    for res in (three, nine):
        a = res["accuracy"]
        assert a["random"] - a["heldout"] > 0.015 and res["random_overstates_in_share_of_worlds"] >= 0.85, "random split overstates with siblings"
        assert abs(a["grouped"] - a["heldout"]) < 0.02, "grouped split tracks new images"
    assert nine["accuracy"]["random"] - nine["accuracy"]["heldout"] > three["accuracy"]["random"] - three["accuracy"]["heldout"] - 0.005
    assert fmt(nine["accuracy"]["random"]) == "0.995" and fmt(nine["accuracy"]["heldout"]) == "0.962", "prediction numbers"
    assert fmt(nine["accuracy"]["random"] - nine["accuracy"]["heldout"]) == "0.033"
    assert f"{100 * (1 - nine['accuracy']['random']):.1f}" == "0.5" and f"{100 * (1 - nine['accuracy']['heldout']):.1f}" == "3.8"
    assert fmt(one["accuracy"]["random"]) == "0.877" and fmt(one["accuracy"]["grouped"]) == "0.885"
    assert fmt(nine["accuracy"]["grouped"]) == "0.957" and abs(nine["accuracy"]["grouped"] - nine["accuracy"]["heldout"]) < 0.005
    assert nine["accuracy"]["heldout"] > one["accuracy"]["heldout"], "more crops help real accuracy too"
