
import numpy as np

from . import predictor
from . import criticality_estimator as CE
from .fusion import pct_rank, soft_max


def signals(Z, tr, cfg, tag=""):
    """Z: (T,N) standardized observations. tr: normal-prefix length.
    Returns dict(resid, nis_kf2) or None if the training prefix is too short.
    `tag`, if given, is forwarded to the predictor for periodic epoch progress."""
    L = cfg.seq_len
    if tr - L < 8:
        return None
    out_f = predictor.forecast(Z, tr, cfg, tag=tag)
    if out_f is None:
        return None
    xs, xf = out_f
    E = (Z - xs - xf).astype(np.float64)
    ss = CE.identify(E[L:tr], cfg)
    nis = np.nan_to_num(CE.run_filter(ss, E - ss.mean, cfg)["nis"])
    nis_kf2 = np.nan_to_num(CE.persistence_burst_transform(nis, tr, cfg)["score"])
    resid = np.nan_to_num((E ** 2).sum(1))
    return {"resid": resid, "nis_kf2": nis_kf2}


def fuse(resid, nis_kf2, mode, kappa=2.0):
    """Combine the raw residual energy and the criticality score into one anomaly score."""
    if mode == "resid":
        return resid
    if mode == "nis_kf2":
        return nis_kf2
    return soft_max(pct_rank(resid), pct_rank(nis_kf2), kappa)


class TipAD:
    """Convenience wrapper bundling signals() + fuse() under a single config."""

    def __init__(self, cfg):
        self.cfg = cfg

    def score(self, Z, tr):
        """Z: (T,N) standardized observations, tr: normal-prefix length.
        Returns the fused anomaly score (T,), or None if tr is too short to fit."""
        sig = signals(Z, tr, self.cfg)
        if sig is None:
            return None
        return fuse(sig["resid"], sig["nis_kf2"], self.cfg.fuse, self.cfg.fusion_kappa)
