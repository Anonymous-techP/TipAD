
from dataclasses import dataclass


@dataclass
class TipADConfig:
    seed: int = 2021

    # ---- Dual-path predictor (slow = attention, fast = causal conv) ----
    seq_len: int = 48                  # input window length L
    d_model: int = 32                  # embedding dimension
    n_heads: int = 4                   # attention heads (slow path)
    fast_kernel: int = 5               # causal conv kernel size (fast path)
    ema_init: float = 0.9              # initial value of the learnable EMA (slow/fast split)
    l2_attn: bool = True               # l2-normalize attention queries/keys
    score_ema: float = 0.9             # EMA used to form the slow/fast auxiliary targets
    w_aux: float = 0.3                 # weight of the auxiliary slow/fast supervision
    predictor_epochs: int = 40
    predictor_lr: float = 5e-3
    max_train_windows: int = 1500      # subsample cap for long training prefixes

    # ---- State-space identification (fit on the normal prefix) ----
    latent_dim: int | None = None      # None = choose k automatically via pca_energy
    pca_energy: float = 0.95           # retained PCA energy, sets latent dimension k
    max_latent: int = 10               # upper bound on k
    var_order: int = 1                 # latent VAR(p) order (1 = VAR(1))
    ridge: float = 1e-3                # ridge regularization for the VAR(1) least squares
    q_floor: float = 1e-4              # numerical floor added to the process-noise diagonal
    r_scale: float = 1.0               # scale applied to the observation-noise covariance R

    # ---- Kalman filter ----
    chi2_gate_q: float = 0.99          # chi-square quantile for the robust gate
    gate_inflate: bool = False         # robust-gating branch; frozen off for TipAD

    # ---- Criticality estimator: persistence/burst decomposition ----
    stress_norm: str = "median"        # calibrate NIS by the training-prefix median (or "dof"/"none")
    persistence_ema: float = 0.98      # alpha: EMA smoothing of the persistence baseline
    gate_ema: float = 0.90             # beta: EMA smoothing of the energy gate
    burst_gain: float = 0.5            # gamma: burst-channel gain
    score_mode: str = "comb"           # "comb" = persistence/burst mix; "ode" = additionally route through fold_ode
    ode_alpha: float = 1.0
    ode_beta: float = 0.5
    ode_gamma: float = 0.001
    ode_substeps: int = 4

    # ---- Adaptive fusion ----
    fuse: str = "softmax"              # "softmax" | "resid" | "nis_kf2" | "rankmax"
    fusion_kappa: float = 2.0          # kappa: self-gated softmax sharpness


DEFAULT = TipADConfig()
