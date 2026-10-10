"""Small shared helpers for chapter activities. Keep this file short and boring."""
import math
import os

# One thread per process: avoids the oversubscription that made small fits take tens of seconds.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

# Shared plot colors (match the reader theme's palette).
COLORS = {
    "ink": "#172a3b", "teal": "#136f75", "gold": "#8f5d0f", "navy": "#334c72",
    "terracotta": "#a24a33", "olive": "#5d6a37", "grey": "#6b757b", "light": "#c9d3d6",
}


def rng(seed):
    return np.random.default_rng(seed)


def r(value, digits=4):
    """Round a float for results and display; ints and strings pass through."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("non-finite result; describe the undefined case instead")
        return round(value, digits)
    return value


def clean(obj, digits=4):
    """Recursively convert a result to plain JSON types with rounded floats."""
    if isinstance(obj, dict):
        return {str(k): clean(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean(v, digits) for v in obj]
    return r(obj, digits)


def fmt(value, digits=3):
    """Fixed-decimal display text."""
    text = f"{float(value):.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def signed(value, digits=3):
    """Display a number with an explicit sign, using an ASCII hyphen for negatives."""
    v = float(value)
    return ("+" if v >= 0 else "-") + fmt(abs(v), digits)
