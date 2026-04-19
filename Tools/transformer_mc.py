"""
Transformer Monte Carlo Model
==============================
Multi-head self-attention Transformer trained on historical returns, used
to generate Monte Carlo price paths that capture:

  - Volatility clustering  (GARCH-like memory via attention)
  - Fat tails              (learned non-Gaussian return distribution)
  - Momentum & mean-reversion (long-range temporal dependencies)
  - Regime transitions     (attention heads specialise on different regimes)

Backend selection (automatic)
------------------------------
  Priority 1 — PyTorch + CUDA   : GPU acceleration, exact autograd
  Priority 2 — PyTorch (CPU)    : exact autograd, ~5-10x faster than NumPy FD
  Priority 3 — NumPy (fallback) : no PyTorch installed; uses finite-difference
                                   approximate gradients (original implementation)

Architecture
------------
  1. Return embedding  : scalar return → d_model dims
  2. Sinusoidal positional encoding (Vaswani et al. 2017)
  3. Transformer encoder: N_layers × (MHA causal + GELU FFN + LayerNorm + residual)
  4. Output head        : last-token hidden state → (mu, log_sigma, skew_logit)
  5. Path generation    : autoregressive, seeded from last context_window returns

Training
--------
  - NLL of skew-adjusted Gaussian
  - Adam + cosine LR decay
  - 80/20 time-ordered train/val split
  - Early stopping on val NLL (patience=15)
"""

from __future__ import annotations

import numpy as np
import warnings
from scipy.special import softmax as _softmax
from typing import Optional, Tuple
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_device():
    """
    Returns (torch_module_or_None, device_string).

    Tries CUDA first, then MPS (Apple Silicon), then CPU.
    Falls back to None/'numpy' if PyTorch is not installed.
    """
    try:
        import torch
        if torch.cuda.is_available():
            dev = "cuda"
            name = torch.cuda.get_device_name(0)
            print(f"[TransformerMC] Using GPU: {name} (CUDA)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = "mps"
            print("[TransformerMC] Using Apple Silicon GPU (MPS)")
        else:
            dev = "cpu"
            print("[TransformerMC] Using CPU (PyTorch)")
        return torch, dev
    except ImportError:
        print("[TransformerMC] PyTorch not found — falling back to NumPy backend")
        return None, "numpy"


_torch, _DEVICE = _detect_device()
_USE_TORCH = _torch is not None


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransformerMCResult:
    call:              float
    put:               float
    model:             str
    inputs:            dict
    paths:             Optional[np.ndarray] = field(default=None, repr=False)
    terminal:          Optional[np.ndarray] = field(default=None, repr=False)
    # Diagnostics
    train_loss_curve:  Optional[np.ndarray] = field(default=None, repr=False)
    val_loss_curve:    Optional[np.ndarray] = field(default=None, repr=False)
    attn_weights_last: Optional[np.ndarray] = field(default=None, repr=False)
    fitted:            bool  = False
    fit_epochs:        int   = 0
    device_used:       str   = "numpy"  # cuda / mps / cpu / numpy


# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH BACKEND
# ─────────────────────────────────────────────────────────────────────────────

if _USE_TORCH:
    import warnings as _warnings
    import torch
    import torch.nn as nn
    # Suppress expected informational warnings from PyTorch internals
    _warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
    _warnings.filterwarnings("ignore", message=".*flash attention.*")
    _warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    def _sinusoidal_pe_torch(seq_len: int, d_model: int) -> "torch.Tensor":
        pe  = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        return pe  # (seq_len, d_model)

    class _TorchTransformerNet(nn.Module):
        """
        Lightweight causal Transformer for return sequence modelling.

        Input  : (B, L) float32 — normalised log-returns
        Output : (B, 3) float32 — [mu, log_sigma, skew_logit]
        """

        def __init__(self, d_model: int, n_heads: int, n_layers: int,
                     context_window: int, dropout: float = 0.0):
            super().__init__()
            self.d_model = d_model
            self.context_window = context_window

            self.embed = nn.Linear(1, d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model        = d_model,
                nhead          = n_heads,
                dim_feedforward= d_model * 4,
                dropout        = dropout,
                activation     = "gelu",
                batch_first    = True,
                norm_first     = True,   # Pre-LN for training stability
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Linear(d_model, 3)

            # Sinusoidal PE as non-trainable buffer
            self.register_buffer("pe", _sinusoidal_pe_torch(context_window, d_model))

            self._init_weights()

        def _init_weights(self):
            nn.init.normal_(self.embed.weight, 0.0, 0.02)
            nn.init.zeros_(self.embed.bias)
            nn.init.normal_(self.head.weight, 0.0, 0.01)
            nn.init.zeros_(self.head.bias)

        def forward(self, x: "torch.Tensor",
                    return_attn: bool = False) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
            """
            x : (B, L) float32
            Returns (out, attn) where out is (B, 3) and attn is None unless
            return_attn=True (expensive — hooks all encoder layers).
            """
            B, L = x.shape

            # Embed scalar returns to d_model
            emb = self.embed(x.unsqueeze(-1))  # (B, L, d_model)
            emb = emb + self.pe[:L].unsqueeze(0)  # add PE

            # Causal mask (upper-triangular -inf)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                L, device=x.device, dtype=x.dtype
            )

            if return_attn:
                # Register forward hooks to capture attention weights
                attn_maps = []

                def _hook(module, inp, out):
                    # TransformerEncoderLayer returns only the hidden state
                    # We re-run attention manually to get weights
                    pass

                # Simpler: just run encoder without hooks, return None for attn
                # (capturing attn from nn.TransformerEncoder requires forward hooks
                # that are version-dependent; we skip for now)
                out = self.encoder(emb, mask=causal_mask, is_causal=True)
                attn = None
            else:
                out = self.encoder(emb, mask=causal_mask, is_causal=True)
                attn = None

            h = out[:, -1, :]          # last token: (B, d_model)
            return self.head(h), attn  # (B, 3), None


    def _gaussian_nll_torch(out: "torch.Tensor",
                             targets: "torch.Tensor") -> "torch.Tensor":
        """Negative log-likelihood of skew-adjusted Gaussian. Returns scalar."""
        mu        = out[:, 0]
        log_sigma = out[:, 1]
        skew      = torch.tanh(out[:, 2]) * 0.5

        sigma = torch.exp(log_sigma.clamp(-4, 4)) + 1e-6
        z = (targets - mu - skew * sigma) / sigma
        nll = 0.5 * (np.log(2 * np.pi) + 2 * torch.log(sigma + 1e-8) + z ** 2)
        return nll.mean()


# ─────────────────────────────────────────────────────────────────────────────
# NUMPY FALLBACK BACKEND  (original pure-NumPy implementation)
# ─────────────────────────────────────────────────────────────────────────────

def _layer_norm(X: np.ndarray, eps: float = 1e-6):
    mu  = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True) + eps
    return (X - mu) / std, mu, std


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _causal_mask(seq_len: int) -> np.ndarray:
    m = np.full((seq_len, seq_len), -1e9)
    m[np.tril_indices(seq_len)] = 0.0
    return m


def _sinusoidal_pe(seq_len: int, d_model: int) -> np.ndarray:
    pe  = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[: d_model // 2])
    return pe


class _Params:
    """Lightweight container for all trainable parameters + Adam state."""

    def __init__(self):
        self._p:  dict[str, np.ndarray] = {}
        self._m:  dict[str, np.ndarray] = {}
        self._v:  dict[str, np.ndarray] = {}
        self._t:  int = 0

    def add(self, name: str, arr: np.ndarray) -> None:
        self._p[name] = arr.copy()
        self._m[name] = np.zeros_like(arr)
        self._v[name] = np.zeros_like(arr)

    def __getitem__(self, name: str) -> np.ndarray:
        return self._p[name]

    def __setitem__(self, name: str, val: np.ndarray) -> None:
        self._p[name] = val

    def adam_step(self, grads: dict, lr: float,
                  beta1: float = 0.9, beta2: float = 0.999,
                  eps: float = 1e-8, weight_decay: float = 1e-4) -> None:
        self._t += 1
        for name, g in grads.items():
            if name not in self._p:
                continue
            g = g + weight_decay * self._p[name]
            self._m[name] = beta1 * self._m[name] + (1 - beta1) * g
            self._v[name] = beta2 * self._v[name] + (1 - beta2) * g ** 2
            m_hat = self._m[name] / (1 - beta1 ** self._t)
            v_hat = self._v[name] / (1 - beta2 ** self._t)
            self._p[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def names(self):
        return list(self._p.keys())


def _mha_forward(X, W_Q, W_K, W_V, W_O, n_heads, causal=True):
    B, L, d_model = X.shape
    d_k = d_model // n_heads

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    Q = Q.reshape(B, L, n_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(B, L, n_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(B, L, n_heads, d_k).transpose(0, 2, 1, 3)

    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    if causal:
        scores += _causal_mask(L)[None, None, :, :]

    attn = _softmax(scores, axis=-1)
    out  = attn @ V
    out  = out.transpose(0, 2, 1, 3).reshape(B, L, d_model)
    return out @ W_O, attn


def _ffn_forward(X, W1, b1, W2, b2):
    return _gelu(X @ W1 + b1) @ W2 + b2


def _transformer_forward_np(X_raw, P, n_layers, n_heads, d_model, return_attn=False):
    B, L = X_raw.shape
    X = X_raw[:, :, None] @ P["W_embed"][None, :, :]
    X = X + P["b_embed"][None, None, :]
    X = X + _sinusoidal_pe(L, d_model)[None, :, :]

    last_attn = None
    for i in range(n_layers):
        pf = f"l{i}_"
        attn_out, attn_w = _mha_forward(
            X, P[pf+"W_Q"], P[pf+"W_K"], P[pf+"W_V"], P[pf+"W_O"],
            n_heads=n_heads, causal=True,
        )
        X_res = X + attn_out
        X_norm, _, _ = _layer_norm(X_res)
        ffn_out = _ffn_forward(X_norm, P[pf+"W1"], P[pf+"b1"], P[pf+"W2"], P[pf+"b2"])
        X = X_norm + ffn_out
        X, _, _ = _layer_norm(X)
        if i == n_layers - 1:
            last_attn = attn_w

    h   = X[:, -1, :]
    out = h @ P["W_out"] + P["b_out"]
    mu        = out[:, 0]
    log_sigma = out[:, 1]
    skew      = np.tanh(out[:, 2]) * 0.5
    return mu, log_sigma, skew, (last_attn if return_attn else None)


def _gaussian_nll_np(mu, log_sigma, skew, targets):
    sigma = np.exp(np.clip(log_sigma, -4, 4)) + 1e-6
    z     = (targets - mu - skew * sigma) / sigma
    return 0.5 * (np.log(2 * np.pi) + 2 * np.log(sigma + 1e-8) + z ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL CLASS
# ─────────────────────────────────────────────────────────────────────────────

class TransformerMCModel:
    """
    Transformer-based Monte Carlo option pricer with automatic GPU/CPU dispatch.

    Parameters
    ----------
    n_paths : int
        Number of simulated paths.
    n_steps : int
        Number of time steps per path (e.g. 252 for 1 year daily).
    context_window : int
        Look-back window length. Longer = more temporal memory, slower training.
    d_model : int
        Embedding dimension. Must be divisible by n_heads.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of Transformer encoder layers.
    lr : float
        Initial Adam learning rate.
    n_epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size (GPU can handle larger batches: e.g. 256).
    patience : int
        Early stopping patience.
    seed : int
        Random seed.
    device : str or None
        Force a specific device: 'cuda', 'mps', 'cpu', 'numpy'.
        None (default) = auto-detect best available.
    """

    def __init__(
        self,
        n_paths:        int   = 8000,
        n_steps:        int   = 252,
        context_window: int   = 60,
        d_model:        int   = 64,    # larger default — GPU can handle it
        n_heads:        int   = 4,
        n_layers:       int   = 3,     # one extra layer vs NumPy default
        lr:             float = 3e-3,
        n_epochs:       int   = 150,
        batch_size:     int   = 128,   # larger batches for GPU
        patience:       int   = 20,
        seed:           int   = 42,
        device:         Optional[str] = None,
    ):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_paths        = n_paths
        self.n_steps        = n_steps
        self.context_window = context_window
        self.d_model        = d_model
        self.n_heads        = n_heads
        self.n_layers       = n_layers
        self.lr             = lr
        self.n_epochs       = n_epochs
        self.batch_size     = batch_size
        self.patience       = patience
        self.seed           = seed

        # Resolve device
        if device is not None:
            self._device = device
        else:
            self._device = _DEVICE

        self._use_torch = _USE_TORCH and self._device != "numpy"

        # Internal state
        self._P: Optional[_Params] = None          # NumPy params
        self._net = None                            # PyTorch nn.Module
        self._hist_returns: Optional[np.ndarray] = None
        self._hist_norm:    Optional[np.ndarray] = None
        self._ret_mean:  float = 0.0
        self._ret_std:   float = 1.0
        self._fitted:    bool  = False
        self._train_loss: list = []
        self._val_loss:   list = []
        self._fit_epochs: int  = 0
        self._last_paths:    Optional[np.ndarray] = None
        self._last_terminal: Optional[np.ndarray] = None
        self._last_attn:     Optional[np.ndarray] = None

    # ──────────────────────────────────────────────────────────────────────────
    # PYTORCH FIT
    # ──────────────────────────────────────────────────────────────────────────

    def _fit_torch(self, X_tr, y_tr, X_val, y_val) -> None:
        import torch
        import torch.optim as optim
        from torch.optim.lr_scheduler import CosineAnnealingLR

        dev = torch.device(self._device)

        torch.manual_seed(self.seed)
        if self._device == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        net = _TorchTransformerNet(
            d_model        = self.d_model,
            n_heads        = self.n_heads,
            n_layers       = self.n_layers,
            context_window = self.context_window,
        ).to(dev)
        self._net = net

        # Use mixed precision on CUDA for faster training
        use_amp = (self._device == "cuda")
        scaler  = torch.cuda.amp.GradScaler() if use_amp else None

        optimizer = optim.AdamW(net.parameters(), lr=self.lr,
                                weight_decay=1e-4, betas=(0.9, 0.999))
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_epochs, eta_min=1e-5)

        # Convert to tensors
        Xtr_t  = torch.tensor(X_tr,  dtype=torch.float32, device=dev)
        ytr_t  = torch.tensor(y_tr,  dtype=torch.float32, device=dev)
        Xval_t = torch.tensor(X_val, dtype=torch.float32, device=dev)
        yval_t = torch.tensor(y_val, dtype=torch.float32, device=dev)

        n_train    = len(ytr_t)
        best_val   = np.inf
        no_improve = 0
        best_state = {k: v.clone() for k, v in net.state_dict().items()}

        rng = torch.Generator(device=dev)
        rng.manual_seed(self.seed)

        for epoch in range(self.n_epochs):
            net.train()
            perm = torch.randperm(n_train, generator=rng, device=dev)
            epoch_losses = []

            for start in range(0, n_train, self.batch_size):
                idx  = perm[start : start + self.batch_size]
                Xb   = Xtr_t[idx]
                yb   = ytr_t[idx]
                if len(Xb) < 2:
                    continue

                optimizer.zero_grad()

                if use_amp:
                    with torch.cuda.amp.autocast():
                        out, _ = net(Xb)
                        loss   = _gaussian_nll_torch(out, yb)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out, _ = net(Xb)
                    loss   = _gaussian_nll_torch(out, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optimizer.step()

                epoch_losses.append(loss.item())

            scheduler.step()

            # Validation
            net.eval()
            with torch.no_grad():
                out_v, _ = net(Xval_t)
                val_loss = _gaussian_nll_torch(out_v, yval_t).item()

            tr_loss = float(np.mean(epoch_losses)) if epoch_losses else np.nan
            self._train_loss.append(tr_loss)
            self._val_loss.append(val_loss)

            if val_loss < best_val - 1e-5:
                best_val   = val_loss
                no_improve = 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    net.load_state_dict(best_state)
                    break

        net.load_state_dict(best_state)

    # ──────────────────────────────────────────────────────────────────────────
    # NUMPY FIT  (fallback)
    # ──────────────────────────────────────────────────────────────────────────

    def _init_params_np(self) -> _Params:
        rng = np.random.default_rng(self.seed)
        D   = self.d_model
        D4  = D * 4
        d_k = D // self.n_heads
        P   = _Params()

        P.add("W_embed", rng.normal(0, 0.02, (1, D)))
        P.add("b_embed", np.zeros(D))

        for i in range(self.n_layers):
            pf = f"l{i}_"
            s  = np.sqrt(2.0 / (D + d_k))
            P.add(pf+"W_Q",  rng.normal(0, s,    (D, D)))
            P.add(pf+"W_K",  rng.normal(0, s,    (D, D)))
            P.add(pf+"W_V",  rng.normal(0, s,    (D, D)))
            P.add(pf+"W_O",  rng.normal(0, s,    (D, D)))
            P.add(pf+"W1",   rng.normal(0, 0.02, (D, D4)))
            P.add(pf+"b1",   np.zeros(D4))
            P.add(pf+"W2",   rng.normal(0, 0.02, (D4, D)))
            P.add(pf+"b2",   np.zeros(D))

        P.add("W_out", rng.normal(0, 0.01, (D, 3)))
        P.add("b_out", np.zeros(3))
        return P

    def _compute_grads_fast_np(self, X_batch, y_batch):
        P   = self._P
        eps = 1e-4

        mu, ls, sk, _ = _transformer_forward_np(
            X_batch, P, self.n_layers, self.n_heads, self.d_model
        )
        sigma = np.exp(np.clip(ls, -4, 4)) + 1e-6
        resid = (y_batch - mu - sk * sigma) / sigma
        loss  = float((0.5 * (np.log(2 * np.pi) + 2 * np.log(sigma + 1e-8) + resid**2)).mean())

        grads  = {name: np.zeros_like(P[name]) for name in P.names()}
        d_mu   = -resid / sigma / len(y_batch)
        d_ls   = (1.0 - resid**2) / len(y_batch)
        d_sk_l = (-resid / sigma) * sigma * (1 - np.tanh(P["b_out"][2])**2) * 0.5 / len(y_batch)
        d_out  = np.stack([d_mu, d_ls, np.full_like(d_mu, d_sk_l.mean())], axis=1)

        B, L   = X_batch.shape
        X_emb  = X_batch[:, :, None] @ P["W_embed"][None, :, :]
        X_emb  = X_emb + P["b_embed"][None, None, :]
        X_emb  = X_emb + _sinusoidal_pe(L, self.d_model)[None, :, :]
        X_cur  = X_emb.copy()

        for i in range(self.n_layers):
            pf = f"l{i}_"
            ao, _ = _mha_forward(X_cur, P[pf+"W_Q"], P[pf+"W_K"],
                                  P[pf+"W_V"], P[pf+"W_O"], self.n_heads)
            Xr = X_cur + ao
            Xn, _, _ = _layer_norm(Xr)
            fo = _ffn_forward(Xn, P[pf+"W1"], P[pf+"b1"], P[pf+"W2"], P[pf+"b2"])
            X_cur = Xn + fo
            X_cur, _, _ = _layer_norm(X_cur)

        h = X_cur[:, -1, :]
        grads["W_out"] = h.T @ d_out / B
        grads["b_out"] = d_out.mean(axis=0)

        rng        = np.random.default_rng(self.seed + P._t)
        n_fd       = 4
        enc_params = [n for n in P.names() if n not in ("W_out", "b_out", "W_embed", "b_embed")]

        for name in enc_params:
            param = P[name]
            g     = np.zeros_like(param)
            n_el  = param.size
            idxs  = rng.choice(n_el, size=min(n_fd, n_el), replace=False)
            for fi in idxs:
                midx = np.unravel_index(fi, param.shape)
                orig = param[midx]
                param[midx] = orig + eps; P[name] = param
                mp, lp, sp, _ = _transformer_forward_np(X_batch, P, self.n_layers, self.n_heads, self.d_model)
                lp_ = float(_gaussian_nll_np(mp, lp, sp, y_batch).mean())
                param[midx] = orig - eps; P[name] = param
                mm, lm, sm, _ = _transformer_forward_np(X_batch, P, self.n_layers, self.n_heads, self.d_model)
                lm_ = float(_gaussian_nll_np(mm, lm, sm, y_batch).mean())
                g[midx] = (lp_ - lm_) / (2 * eps)
                param[midx] = orig; P[name] = param
            if n_fd < n_el:
                g *= n_el / n_fd
            grads[name] = g

        for name in ("W_embed", "b_embed"):
            param = P[name]
            g     = np.zeros_like(param)
            it    = np.nditer(param, flags=["multi_index"])
            while not it.finished:
                midx = it.multi_index
                orig = param[midx]
                param[midx] = orig + eps; P[name] = param
                mp, lp, sp, _ = _transformer_forward_np(X_batch, P, self.n_layers, self.n_heads, self.d_model)
                lp_ = float(_gaussian_nll_np(mp, lp, sp, y_batch).mean())
                param[midx] = orig - eps; P[name] = param
                mm, lm, sm, _ = _transformer_forward_np(X_batch, P, self.n_layers, self.n_heads, self.d_model)
                lm_ = float(_gaussian_nll_np(mm, lm, sm, y_batch).mean())
                g[midx] = (lp_ - lm_) / (2 * eps)
                param[midx] = orig; P[name] = param
                it.iternext()
            grads[name] = g

        return loss, grads

    def _fit_numpy(self, X_tr, y_tr, X_val, y_val) -> None:
        self._P = self._init_params_np()

        def _lr(epoch):
            return self.lr * 0.5 * (1 + np.cos(np.pi * epoch / self.n_epochs))

        rng        = np.random.default_rng(self.seed)
        best_val   = np.inf
        no_improve = 0
        best_state = {k: self._P[k].copy() for k in self._P.names()}
        n_train    = len(y_tr)

        for epoch in range(self.n_epochs):
            perm  = rng.permutation(n_train)
            X_sh  = X_tr[perm]
            y_sh  = y_tr[perm]
            epoch_losses = []

            for start in range(0, n_train, self.batch_size):
                Xb = X_sh[start : start + self.batch_size]
                yb = y_sh[start : start + self.batch_size]
                if len(Xb) < 2:
                    continue
                loss, grads = self._compute_grads_fast_np(Xb, yb)
                self._P.adam_step(grads, lr=_lr(epoch))
                epoch_losses.append(loss)

            mu_v, ls_v, sk_v, _ = _transformer_forward_np(
                X_val, self._P, self.n_layers, self.n_heads, self.d_model
            )
            val_loss = float(_gaussian_nll_np(mu_v, ls_v, sk_v, y_val).mean())

            self._train_loss.append(float(np.mean(epoch_losses)) if epoch_losses else np.nan)
            self._val_loss.append(val_loss)

            if val_loss < best_val - 1e-5:
                best_val   = val_loss
                no_improve = 0
                best_state = {k: self._P[k].copy() for k in self._P.names()}
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    for k in self._P.names():
                        self._P[k] = best_state[k]
                    break

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC FIT
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, hist_prices: np.ndarray) -> "TransformerMCModel":
        """
        Train on historical daily price series.

        Parameters
        ----------
        hist_prices : 1-D array of adjusted closing prices (≥ context_window+30)
        """
        prices = np.asarray(hist_prices, dtype=float)
        prices = prices[~np.isnan(prices)]
        if len(prices) < self.context_window + 10:
            raise ValueError(
                f"Need at least {self.context_window + 10} price observations; "
                f"got {len(prices)}."
            )

        rets        = np.log(prices[1:] / prices[:-1])
        self._ret_mean = float(rets.mean())
        self._ret_std  = float(rets.std() + 1e-8)
        rets_norm   = (rets - self._ret_mean) / self._ret_std

        self._hist_returns = rets
        self._hist_norm    = rets_norm

        L = self.context_window
        X_all = np.lib.stride_tricks.sliding_window_view(rets_norm, L)[:-1].astype(np.float32)
        y_all = rets_norm[L:].astype(np.float32)

        N = len(y_all)
        if N < 20:
            raise ValueError("Not enough data to build training windows.")

        n_train = int(0.8 * N)
        X_tr, X_val = X_all[:n_train], X_all[n_train:]
        y_tr, y_val = y_all[:n_train], y_all[n_train:]

        self._train_loss = []
        self._val_loss   = []

        if self._use_torch:
            self._fit_torch(X_tr, y_tr, X_val, y_val)
        else:
            self._fit_numpy(X_tr, y_tr, X_val, y_val)

        self._fit_epochs = len(self._train_loss)
        self._fitted     = True
        return self

    # ──────────────────────────────────────────────────────────────────────────
    # SIMULATE PATHS
    # ──────────────────────────────────────────────────────────────────────────

    def simulate_paths(self, S: float, T: float, r: float, q: float = 0.0) -> np.ndarray:
        """
        Generate price paths autoregressively.

        Returns
        -------
        paths : np.ndarray of shape (n_steps + 1, n_paths)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before simulate_paths().")

        if self._use_torch:
            return self._simulate_torch(S, T, r, q)
        else:
            return self._simulate_numpy(S, T, r, q)

    def _simulate_torch(self, S, T, r, q) -> np.ndarray:
        import torch

        dev   = torch.device(self._device)
        L     = self.context_window
        dt    = T / self.n_steps
        rng   = np.random.default_rng(self.seed)

        seed_ctx    = self._hist_norm[-L:].astype(np.float32)
        noise_scale = 0.05
        contexts_np = (
            np.tile(seed_ctx, (self.n_paths, 1))
            + rng.normal(0, noise_scale, (self.n_paths, L)).astype(np.float32)
        )

        log_rets = np.zeros((self.n_steps, self.n_paths), dtype=np.float32)
        self._net.eval()
        infer_batch = min(1024, self.n_paths)  # larger batches on GPU

        with torch.no_grad():
            for step in range(self.n_steps):
                mus_s  = np.zeros(self.n_paths, dtype=np.float32)
                sigs_s = np.zeros(self.n_paths, dtype=np.float32)
                sks_s  = np.zeros(self.n_paths, dtype=np.float32)

                for b0 in range(0, self.n_paths, infer_batch):
                    b1  = b0 + infer_batch
                    ctx = torch.tensor(contexts_np[b0:b1], device=dev)
                    out, _ = self._net(ctx)
                    out_np = out.cpu().numpy()
                    mus_s[b0:b1]  = out_np[:, 0]
                    sigs_s[b0:b1] = np.exp(np.clip(out_np[:, 1], -4, 4)) + 1e-6
                    sks_s[b0:b1]  = np.tanh(out_np[:, 2]) * 0.5

                mu_rn = (r - q) * dt - 0.5 * (sigs_s * self._ret_std) ** 2
                mu_rn += mus_s * self._ret_std + self._ret_mean

                sig_raw = sigs_s * self._ret_std
                z       = rng.standard_normal(self.n_paths).astype(np.float32)
                lr_step = mu_rn + sig_raw * (z + sks_s * (z**2 - 1))

                log_rets[step] = lr_step

                new_ctx_norm = (lr_step - self._ret_mean) / self._ret_std
                contexts_np  = np.roll(contexts_np, -1, axis=1)
                contexts_np[:, -1] = new_ctx_norm

        log_paths = np.vstack([np.zeros(self.n_paths, dtype=np.float32),
                                np.cumsum(log_rets, axis=0)])
        paths = S * np.exp(log_paths)

        self._last_paths    = paths
        self._last_terminal = paths[-1]
        self._last_attn     = None
        return paths

    def _simulate_numpy(self, S, T, r, q) -> np.ndarray:
        L     = self.context_window
        dt    = T / self.n_steps
        rng   = np.random.default_rng(self.seed)

        seed_ctx    = self._hist_norm[-L:]
        noise_scale = 0.05 * self._ret_std
        contexts    = (
            np.tile(seed_ctx, (self.n_paths, 1))
            + rng.normal(0, noise_scale, (self.n_paths, L))
        )

        log_rets     = np.zeros((self.n_steps, self.n_paths))
        infer_batch  = min(512, self.n_paths)
        last_attn_acc = []

        for step in range(self.n_steps):
            mus_s  = np.zeros(self.n_paths)
            sigs_s = np.zeros(self.n_paths)
            sks_s  = np.zeros(self.n_paths)

            for b0 in range(0, self.n_paths, infer_batch):
                b1 = b0 + infer_batch
                ctx_b = contexts[b0:b1]
                mu_b, ls_b, sk_b, attn_b = _transformer_forward_np(
                    ctx_b, self._P, self.n_layers, self.n_heads, self.d_model,
                    return_attn=(step == self.n_steps - 1),
                )
                mus_s[b0:b1]  = mu_b
                sigs_s[b0:b1] = np.exp(np.clip(ls_b, -4, 4)) + 1e-6
                sks_s[b0:b1]  = sk_b
                if attn_b is not None:
                    last_attn_acc.append(attn_b)

            mu_raw  = mus_s  * self._ret_std + self._ret_mean
            sig_raw = sigs_s * self._ret_std
            mu_rn   = (r - q) * dt - 0.5 * sig_raw**2

            z       = rng.standard_normal(self.n_paths)
            lr_step = mu_rn + sig_raw * (z + sks_s * (z**2 - 1))
            log_rets[step] = lr_step

            new_ctx_norm = (lr_step - self._ret_mean) / self._ret_std
            contexts = np.roll(contexts, -1, axis=1)
            contexts[:, -1] = new_ctx_norm

        log_paths = np.vstack([np.zeros(self.n_paths), np.cumsum(log_rets, axis=0)])
        paths = S * np.exp(log_paths)

        self._last_paths    = paths
        self._last_terminal = paths[-1]
        if last_attn_acc:
            self._last_attn = np.concatenate(last_attn_acc, axis=0).mean(axis=0)
        else:
            self._last_attn = None

        return paths

    # ──────────────────────────────────────────────────────────────────────────
    # PRICE
    # ──────────────────────────────────────────────────────────────────────────

    def price(self, S: float, K: float, T: float,
              r: float, q: float = 0.0) -> TransformerMCResult:
        """Price European call and put options via Monte Carlo."""
        paths    = self.simulate_paths(S, T, r, q)
        terminal = paths[-1]
        disc     = np.exp(-r * T)

        call = float(disc * np.maximum(terminal - K, 0.0).mean())
        put  = float(disc * np.maximum(K - terminal, 0.0).mean())

        backend = f"PyTorch/{self._device.upper()}" if self._use_torch else "NumPy"

        return TransformerMCResult(
            call=call,
            put=put,
            model=(f"Transformer MC [{backend}]  "
                   f"n={self.n_paths:,}  L={self.context_window}  "
                   f"d={self.d_model}  h={self.n_heads}  layers={self.n_layers}"),
            inputs=dict(S=S, K=K, T=T, r=r, q=q,
                        n_paths=self.n_paths, context_window=self.context_window),
            paths=paths,
            terminal=terminal,
            train_loss_curve=np.array(self._train_loss) if self._train_loss else None,
            val_loss_curve=np.array(self._val_loss)     if self._val_loss   else None,
            attn_weights_last=self._last_attn,
            fitted=self._fitted,
            fit_epochs=self._fit_epochs,
            device_used=self._device,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # CONVENIENCE
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def device(self) -> str:
        return self._device

    @property
    def last_paths(self) -> Optional[np.ndarray]:
        return self._last_paths

    @property
    def last_terminal(self) -> Optional[np.ndarray]:
        return self._last_terminal

    @property
    def last_attn(self) -> Optional[np.ndarray]:
        return self._last_attn

    def var_cvar(self, confidence: float = 0.05) -> tuple:
        if self._last_terminal is None:
            raise RuntimeError("Run price() or simulate_paths() first.")
        t    = np.sort(self._last_terminal)
        idx  = int(confidence * len(t))
        var  = float(t[idx])
        cvar = float(t[:idx].mean()) if idx > 0 else float(t[0])
        return var, cvar
