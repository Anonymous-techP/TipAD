import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableEMA(nn.Module):
    """Learnable exponential moving average along time (slow baseline); a = sigmoid(p) in (0,1)."""

    def __init__(self, init=0.9):
        super().__init__()
        p0 = float(np.log(init / (1 - init)))
        self.p = nn.Parameter(torch.tensor(p0, dtype=torch.float32))

    def forward(self, x):                        # x: (B,L,d)
        a = torch.sigmoid(self.p)
        outs = [x[:, 0]]
        for t in range(1, x.size(1)):
            outs.append(a * outs[-1] + (1 - a) * x[:, t])
        return torch.stack(outs, dim=1)


class L2Attn(nn.Module):
    """Single-layer multi-head self-attention with optional l2-normalized Q, K
    (scale-invariant matching)."""

    def __init__(self, d, heads=4, l2=True):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk, self.l2 = heads, d // heads, l2
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.scale = nn.Parameter(torch.tensor(float(np.sqrt(self.dk))))

    def forward(self, x):                         # (B,L,d)
        B, L, d = x.shape
        q = self.q(x).view(B, L, self.h, self.dk).transpose(1, 2)
        k = self.k(x).view(B, L, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(B, L, self.h, self.dk).transpose(1, 2)
        if self.l2:
            q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
            att = (q @ k.transpose(-2, -1)) * self.scale
        else:
            att = (q @ k.transpose(-2, -1)) / np.sqrt(self.dk)
        att = torch.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, L, d)
        return self.o(out)


class DualPathPredictor(nn.Module):
    """Slow head = attention (long-range), fast head = causal small conv (local).
    forward returns (x_hat_slow, x_hat_fast)."""

    def __init__(self, N, d=32, heads=4, ema_init=0.9, l2=True, fast_kernel=5):
        super().__init__()
        self.embed = nn.Linear(N, d)
        self.ema = LearnableEMA(ema_init)
        self.slow_attn = L2Attn(d, heads, l2)
        self.fast_k = fast_kernel
        self.fast_conv = nn.Conv1d(d, d, kernel_size=fast_kernel)
        self.head_slow = nn.Linear(d, N)
        self.head_fast = nn.Linear(d, N)

    def forward(self, w):                         # w: (B,L,N)
        e = self.embed(w)
        slow_in = self.ema(e)
        fast_in = e - slow_in
        sc = self.slow_attn(slow_in)[:, -1]
        x = fast_in.transpose(1, 2)                # (B,d,L)
        x = F.pad(x, (self.fast_k - 1, 0))          # left-pad: causal
        fc = self.fast_conv(x)[:, :, -1]
        return self.head_slow(sc), self.head_fast(fc)


def causal_ema(X, a):
    """Causal EMA over a (T,N) array, used to form the auxiliary slow/fast targets."""
    out = np.empty_like(X)
    out[0] = X[0]
    for t in range(1, len(X)):
        out[t] = a * out[t - 1] + (1 - a) * X[t]
    return out


def forecast(Z, tr, cfg, infer_chunk=2048):
    """Train the dual-path predictor on the normal prefix [0, tr), then predict
    over the full sequence. Z: (T,N) standardized. Returns (x_hat_slow, x_hat_fast),
    both (T,N); the first L steps are filled with the target components (residual 0).
    Returns None if the training prefix is too short."""
    T, N = Z.shape
    L = cfg.seq_len
    n_train = tr - L
    if n_train < 8:
        return None

    torch.manual_seed(cfg.seed)
    Zt = torch.tensor(Z, dtype=torch.float32)
    Zsm = causal_ema(Z, cfg.score_ema)              # slow auxiliary target u
    Zfa = Z - Zsm                                   # fast auxiliary target f
    Ut = torch.tensor(Zsm, dtype=torch.float32)
    Ft = torch.tensor(Zfa, dtype=torch.float32)

    idx = (np.linspace(0, n_train - 1, cfg.max_train_windows).astype(int).tolist()
           if n_train > cfg.max_train_windows else list(range(n_train)))
    Xb = torch.stack([Zt[i:i + L] for i in idx])
    Yb = torch.stack([Zt[i + L] for i in idx])
    Ys = torch.stack([Ut[i + L] for i in idx])
    Yf = torch.stack([Ft[i + L] for i in idx])

    m = DualPathPredictor(N, cfg.d_model, cfg.n_heads, cfg.ema_init, cfg.l2_attn, cfg.fast_kernel)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.predictor_lr)
    mse = nn.MSELoss()
    m.train()
    for _ in range(cfg.predictor_epochs):
        opt.zero_grad()
        xs, xf = m(Xb)
        loss = mse(xs + xf, Yb) + cfg.w_aux * (mse(xs, Ys) + mse(xf, Yf))
        loss.backward()
        opt.step()

    m.eval()
    xs_full = Zsm.copy().astype(np.float32)
    xf_full = Zfa.copy().astype(np.float32)
    starts = list(range(0, T - L))
    with torch.no_grad():
        for c in range(0, len(starts), infer_chunk):
            ss = starts[c:c + infer_chunk]
            Wb = torch.stack([Zt[s:s + L] for s in ss])
            xs, xf = m(Wb)
            xs, xf = xs.numpy(), xf.numpy()
            for j, s in enumerate(ss):
                xs_full[s + L] = xs[j]
                xf_full[s + L] = xf[j]
    return xs_full, xf_full
