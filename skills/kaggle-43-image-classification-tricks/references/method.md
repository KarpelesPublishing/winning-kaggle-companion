# Chapter 43: Image Classification Tricks

**How much does a random split of crops overstate accuracy compared with splitting by source image, and how does that grow with crops per image?**

A validation score that cannot be reproduced on new images misleads every later decision, including which crop policy, resolution or backbone to keep. The chapter asks that the original image identity survive preprocessing so the split respects it.

## The experiment

Real data: scikit-learn's bundled digits, enlarged to 16 x 16. 300 source images are each cut into 1, 3 or 9 overlapping 14 x 14 crops (the control). A one-nearest-neighbor classifier on pixels is scored three ways: 5-fold cross-validation with crops split at random, 5-fold cross-validation with whole source images grouped, and accuracy on the crops of 400 source images never seen in training. Predictions are pooled before scoring; results are means over 8 draws.

Control: Crops cut from each source image (1 (no siblings), 3, 9; default 9).

## Measured results

| Measure | 1 (no siblings) | 3 | 9 |
|---|---|---|---|
| Random split | 0.877 | 0.964 | 0.995 |
| Split by source image | 0.885 | 0.935 | 0.957 |
| New source images | 0.893 | 0.942 | 0.962 |
| Random overstatement | -0.016 | +0.021 | +0.033 |
| Grouped difference | -0.008 | -0.008 | -0.005 |

## What the result says (default, crops cut from each source image = 9)

With 9 crops per source image (2,700 training crops), a random split reports 0.995 and a split by source image 0.957, while new source images score 0.962. The random split overstates accuracy by 0.995 - 0.962 = 0.033; the grouped split is 0.957 - 0.962 = -0.005 away. The random split claims an error rate 7.3 times smaller than the real one, and overstated accuracy in 100% of 8 draws.

- Random-split overstatement: 0.995 - 0.962 = +0.033.
- Grouped-split difference: 0.957 - 0.962 = -0.005.
- Error rate on new images against the random-split claim: 0.038 / 0.005 = 7.3.
- Training crops: 300 sources * 9 = 2,700.

## Apply it to a competition

- Keep a source-image ID on every crop, tile or augmented copy, and split, early-stop and tune by that ID.
- Treat a validation score that rises as you add crops per image with suspicion: part of the rise is siblings, not skill.
- Hold out whole images for the final assessment of the selected recipe, and cut crops after the split, never before.
- If a quick random split is unavoidable, compare it with a grouped split on a sample and report the gap.

## Assumptions and limits

Real 16 x 16 digit images and a one-nearest-neighbor classifier, which memorizes nearby examples and so shows the leak clearly; a linear or heavily regularized model would leak less. Crops overlap heavily by construction. Grouped validation trains on 80% of the sources and sits slightly below the held-out score.

Real digit images and a memorizing classifier: the overstatement is modest (0.03) here and would be larger for noisy labels or a model that memorizes more, and smaller for models that do not. Scores at one crop per image differ by noise only.

## Reproduce it

The chapter notebook `notebooks/43-image-classification-tricks.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch43` (`run`, `explain`, `draw`).

```python
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
```

Book location: Chapter 43, Separate Development From Assessment. Real data: scikit-learn's bundled digits, enlarged to 16 x 16; the crop construction and the classifier are constructed and measured by the chapter activity.
