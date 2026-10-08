"""Small numerical and fold-provenance checks used by the modeling skills."""
import numpy as np


def stable_softmax(logits):
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2 or not np.isfinite(logits).all():
        raise ValueError("supply a finite rows-by-classes array")
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(y, probabilities, bins=10):
    """Binary, equal-width ECE weighted by the observed number of rows per bin."""
    y, p = np.asarray(y), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or len(y) == 0:
        raise ValueError("supply nonempty one-dimensional arrays of matching length")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or bins < 1:
        raise ValueError("labels must be binary, probabilities in [0, 1], bins positive")
    assignments = np.minimum((p*bins).astype(int), bins-1)
    return float(sum((assignments == b).sum()/len(p) * abs(y[assignments == b].mean()-p[assignments == b].mean())
                     for b in range(bins) if (assignments == b).any()))


def eligible_teachers(validation_indices, teacher_training_indices):
    """Select teachers by training-row provenance rather than misleading fold names."""
    forbidden = set(validation_indices)
    eligible = [i for i, rows in enumerate(teacher_training_indices) if not forbidden.intersection(rows)]
    if not eligible:
        raise ValueError("no teacher is independent of these validation labels")
    return eligible
