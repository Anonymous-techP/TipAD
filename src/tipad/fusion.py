
import numpy as np


def pct_rank(x):
    """Label-free monotone rank of x within itself, scaled to [0, 1]."""
    order = np.argsort(np.argsort(x))
    return order / max(1, len(x) - 1)


def soft_max(a, b, kappa):
    """Self-gated soft-max: favors whichever of a, b ranks higher (kappa -> inf gives hard max)."""
    ap, bp = np.power(a + 1e-12, kappa), np.power(b + 1e-12, kappa)
    g = ap / (ap + bp + 1e-12)
    return g * a + (1 - g) * b
