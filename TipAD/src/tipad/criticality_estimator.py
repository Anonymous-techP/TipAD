
import numpy as np
from dataclasses import dataclass
from scipy.stats import chi2


# ============================================================================
# 1) State-space identification
# ============================================================================
@dataclass
class SSModel:
    A: np.ndarray      # (k,k) latent transition
    H: np.ndarray      # (N,k) observation matrix (PCA loadings)
    Q: np.ndarray      # (k,k) process-noise covariance
    R: np.ndarray      # (N,N) observation-noise covariance (diagonal)
    mean: np.ndarray   # (N,) training-prefix mean, subtracted before filtering
    k: int             # latent dimension
    P0: np.ndarray     # (k,k) initial state covariance
    h0: np.ndarray     # (k,) initial latent state


def _pca(Xc: np.ndarray, energy: float, kmax: int):
    """PCA on already-centered Xc; choose k by retained-energy threshold (capped at kmax)."""
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev = S ** 2
    ratio = np.cumsum(ev) / (ev.sum() + 1e-12)
    k = int(np.searchsorted(ratio, energy) + 1)
    k = max(1, min(k, kmax, Vt.shape[0]))
    W = Vt[:k].T
    return W, k


def _var_p_ssmodel(Zt, W, mean, k, p, cfg, Xc, H) -> SSModel:
    """Latent VAR(p) in companion form. State = [h_t; ...; h_{t-p+1}] in R^{pk}."""
    tr = Zt.shape[0]
    Y = Zt[p:]
    Xlag = np.concatenate([Zt[p - i - 1:tr - i - 1] for i in range(p)], axis=1)
    G = Xlag.T @ Xlag + cfg.ridge * np.eye(p * k)
    B = np.linalg.solve(G, Xlag.T @ Y).T
    resid = Y - Xlag @ B.T
    Qk = np.cov(resid.T) if k > 1 else np.array([[resid.var()]])
    Qk = np.atleast_2d(Qk)

    pk = p * k
    A = np.zeros((pk, pk))
    A[:k, :] = B
    if p > 1:
        A[k:, :-k] = np.eye((p - 1) * k)
    Q = np.zeros((pk, pk))
    Q[:k, :k] = 0.5 * (Qk + Qk.T)
    Q += cfg.q_floor * np.eye(pk)
    Hc = np.zeros((W.shape[0], pk)); Hc[:, :k] = W

    rres = Xc - Zt @ H.T
    r_diag = np.maximum(rres.var(0), 1e-6) * cfg.r_scale
    R = np.diag(r_diag.astype(np.float64))

    h0 = np.concatenate([Zt[-i - 1] for i in range(p)]).astype(np.float64)
    P0 = Q.copy()
    return SSModel(A=A, H=Hc, Q=Q, R=R, mean=mean.astype(np.float64), k=pk, P0=P0, h0=h0)


def identify(Xtr: np.ndarray, cfg) -> SSModel:
    """Xtr: (tr, N) standardized training prefix. cfg: TipADConfig."""
    tr, N = Xtr.shape
    mean = Xtr.mean(0)
    Xc = Xtr - mean

    if cfg.latent_dim is not None:
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = max(1, min(cfg.latent_dim, Vt.shape[0]))
        W = Vt[:k].T
    else:
        W, k = _pca(Xc, cfg.pca_energy, cfg.max_latent)

    H = W
    Zt = Xc @ W

    p = int(getattr(cfg, "var_order", 1))
    if p > 1 and tr > p + 2:
        return _var_p_ssmodel(Zt, W, mean, k, p, cfg, Xc, H)

    if tr >= 3:
        Z0 = Zt[:-1]
        Z1 = Zt[1:]
        G = Z0.T @ Z0 + cfg.ridge * np.eye(k)
        A = np.linalg.solve(G, Z0.T @ Z1).T
        resid = Z1 - Z0 @ A.T
        Q = np.cov(resid.T) if k > 1 else np.array([[resid.var()]])
        Q = np.atleast_2d(Q)
    else:                               # degenerate case: training prefix too short
        A = 0.9 * np.eye(k)
        Q = np.eye(k) * 0.01

    Q = 0.5 * (Q + Q.T) + cfg.q_floor * np.eye(k)

    Xrec = Zt @ H.T
    rres = Xc - Xrec
    r_diag = np.maximum(rres.var(0), 1e-6) * cfg.r_scale
    R = np.diag(r_diag.astype(np.float64))

    h0 = Zt[-1].astype(np.float64)      # start the online pass from the final training state
    P0 = Q.copy()

    return SSModel(A=A.astype(np.float64), H=H.astype(np.float64),
                   Q=Q.astype(np.float64), R=R.astype(np.float64),
                   mean=mean.astype(np.float64), k=k, P0=P0, h0=h0)


# ============================================================================
# 2) Kalman filter
# ============================================================================
def _safe_solve_quad(S, v, jitter=1e-8):
    """Compute v^T S^-1 v robustly, returning S^-1 as well (reused for the gain)."""
    n = S.shape[0]
    Sj = S + jitter * np.eye(n)
    try:
        Sinv = np.linalg.inv(Sj)
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(Sj)
    q = float(v @ Sinv @ v)
    return max(q, 0.0), Sinv


def run_filter(ss: SSModel, Z, cfg):
    """
    ss: SSModel. Z: (T,N) standardized observations, already mean-subtracted
    (see identify: Z should be E - ss.mean). cfg: TipADConfig.

    Returns dict:
      nis    (T,)   raw per-step NIS stress
      latent (T,k)  filtered latent state h_t
      gated  (T,)   bool, whether the (frozen-off) robust gate fired at step t
    """
    T, N = Z.shape
    A, H, Q, R = ss.A, ss.H, ss.Q, ss.R
    k = ss.k
    I_k = np.eye(k)

    c_gate = chi2.ppf(cfg.chi2_gate_q, df=N)

    h = ss.h0.copy()
    P = ss.P0.copy()

    nis = np.zeros(T)
    latent = np.zeros((T, k))
    gated = np.zeros(T, dtype=bool)

    for t in range(T):
        x = Z[t].astype(np.float64)
        h_pred = A @ h
        P_pred = A @ P @ A.T + Q
        x_pred = H @ h_pred

        v = x - x_pred
        S = H @ P_pred @ H.T + R
        eps, Sinv = _safe_solve_quad(S, v)
        nis[t] = eps

        if cfg.gate_inflate and eps > c_gate:              # robust gate, frozen off for TipAD
            gated[t] = True
            R_eff = R * (eps / c_gate)
            S_eff = H @ P_pred @ H.T + R_eff
            _, Sinv = _safe_solve_quad(S_eff, v)

        K = P_pred @ H.T @ Sinv
        h = h_pred + K @ v
        P = (I_k - K @ H) @ P_pred
        P = 0.5 * (P + P.T)
        latent[t] = h

    return {"nis": nis, "latent": latent, "gated": gated, "c_gate": c_gate}


# ============================================================================
# 3) Persistence/burst decomposition
# ============================================================================
def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """Causal EMA: y_t = alpha*y_{t-1} + (1-alpha)*x_t, y_0 = x_0."""
    x = np.asarray(x, float)
    y = np.empty_like(x)
    acc = x[0]
    one_m = 1.0 - alpha
    for i in range(len(x)):
        acc = alpha * acc + one_m * x[i]
        y[i] = acc
    return y


def normalize_stress(nis, tr, cfg):
    """Calibrate NIS so the normal prefix sits near 1 (stress_norm: median|dof|none)."""
    if cfg.stress_norm == "median":
        base = np.median(nis[:tr]) + 1e-8
    elif cfg.stress_norm == "dof":
        base = np.mean(nis[:tr]) + 1e-8          # E[chi2(N)] = N
    else:
        base = 1.0
    return nis / base


def fold_ode(stress, cfg):
    """Fold-bifurcation accumulation ODE (forward Euler, sub-stepped):
        dz/dt = alpha*S - beta*z^2 - gamma*z,  z >= 0
    Only exercised when cfg.score_mode == "ode"."""
    T = len(stress)
    z = 0.0
    dt = 1.0 / cfg.ode_substeps
    out = np.zeros(T)
    for i in range(T):
        s = stress[i]
        for _ in range(cfg.ode_substeps):
            z = max(z + dt * (cfg.ode_alpha * s - cfg.ode_beta * z * z - cfg.ode_gamma * z), 0.0)
        out[i] = z
    return out


def persistence_burst_transform(nis, tr, cfg):
    """Calibrated NIS -> persistence/burst decomposition -> criticality score c_t.

    Returns dict: score / comb / persistence / burst / gate_w.
    """
    nis = np.asarray(nis, float)
    S = normalize_stress(nis, tr, cfg)
    a, b, g = cfg.persistence_ema, cfg.gate_ema, cfg.burst_gain
    Sbar = _ema(S, a)                                       # persistence baseline (low-pass)
    F = np.maximum(S - Sbar, 0.0)                           # burst excess over the baseline
    F = F / (np.std(F[:tr]) + 1e-6)                         # scale by the training-prefix std
    gF = g * F
    e_slow = _ema(Sbar ** 2, b)
    e_fast = _ema(gF ** 2, b)
    w = e_slow / (e_slow + e_fast + 1e-8)
    comb = w * Sbar + (1.0 - w) * gF
    score = fold_ode(comb, cfg) if cfg.score_mode == "ode" else comb
    return {"score": np.nan_to_num(score), "comb": comb, "persistence": Sbar, "burst": gF, "gate_w": w}
