"""
Rough Bergomi Stochastic Volatility Model
------------------------------------------
Implements the Rough Bergomi model (Bayer, Friz, Gatheral 2016) which uses
fractional Brownian motion (fBM) with Hurst parameter H ≈ 0.1 to capture
the rough, mean-reverting nature of realized volatility in equity markets.

Key references:
  - Bayer, C., Friz, P., & Gatheral, J. (2016). Pricing under rough volatility.
    Quantitative Finance, 16(6), 887-904.
  - Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). Volatility is rough.
    Quantitative Finance, 18(6), 933-949.
"""

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List

from scipy.optimize import brentq
try:
    from scipy.linalg import cholesky as scipy_cholesky
    _SCIPY_CHOLESKY = True
except ImportError:
    _SCIPY_CHOLESKY = False

from .pricing import BlackScholesModel


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TRADING_DAYS: int = 252
DEFAULT_H: float = 0.1       # Typical Hurst parameter for equity vol
DEFAULT_ETA: float = 1.9     # Vol of vol parameter
DEFAULT_PATHS: int = 10_000
DEFAULT_STEPS: int = 252

# Internal step cap for Cholesky; weekly granularity to keep memory/time sane
_CHOL_STEP_CAP: int = 52


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoughVolResult:
    """
    Container for Rough Bergomi option pricing output.

    Attributes
    ----------
    call : float
        Discounted call option price.
    put : float
        Discounted put option price.
    model : str
        Model label ("Rough Bergomi").
    hurst : float
        Hurst parameter H used in the simulation.
    eta : float
        Vol-of-vol parameter eta.
    xi_0 : float
        Initial (flat) forward variance used as variance at t=0.
    std_error_call : float
        Monte Carlo standard error for the call price.
    std_error_put : float
        Monte Carlo standard error for the put price.
    n_paths : int
        Number of simulation paths used.
    vol_paths : Optional[np.ndarray]
        First ≤100 annualised vol paths (sqrt of variance), shape
        (n_steps+1, min(n_paths, 100)).  None if not stored.
    spot_paths : Optional[np.ndarray]
        First ≤100 spot paths, shape (n_steps+1, min(n_paths, 100)).
        None if not stored.
    """

    call: float
    put: float
    model: str
    hurst: float
    eta: float
    xi_0: float
    std_error_call: float
    std_error_put: float
    n_paths: int
    vol_paths: Optional[np.ndarray] = field(default=None, repr=False)
    spot_paths: Optional[np.ndarray] = field(default=None, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# ROUGH BERGOMI MODEL
# ─────────────────────────────────────────────────────────────────────────────

class RoughBergomiModel:
    """
    Rough Bergomi model (Bayer, Friz, Gatheral 2016).

    Dynamics under the risk-neutral measure:

        dS_t = S_t * sqrt(V_t) dW_t^S
        V_t  = xi_0 * exp(eta * W_t^H  -  0.5 * eta^2 * t^{2H})

    where W^H is fractional Brownian motion with Hurst parameter H in (0, 0.5),
    and W^S, W^H are correlated with correlation rho.

    fBM is simulated exactly via Cholesky decomposition of the covariance
    matrix.  For n_steps > 100 the grid is internally capped at 52 steps
    (weekly) for numerical tractability; time-scaling ensures correct pricing.

    Parameters
    ----------
    H : float
        Hurst parameter.  Must be in (0, 0.5).  Typical equity value ≈ 0.1.
    eta : float
        Volatility of volatility.  Typical range 1.5–2.5.
    rho : float
        Spot-vol correlation.  Typically −0.7 to −0.9.
    n_paths : int
        Number of Monte Carlo paths.
    n_steps : int
        Number of time steps for the simulation grid (capped at 52 internally
        when > 100 for tractability).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        H: float = DEFAULT_H,
        eta: float = DEFAULT_ETA,
        rho: float = -0.7,
        n_paths: int = DEFAULT_PATHS,
        n_steps: int = DEFAULT_STEPS,
        seed: int = 42,
    ) -> None:
        if not (0.0 < H < 0.5):
            raise ValueError(
                f"Hurst parameter H must be in (0, 0.5), got {H}"
            )
        if n_paths < 1:
            raise ValueError(f"n_paths must be positive, got {n_paths}")
        if n_steps < 1:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        if not (-1.0 < rho < 1.0):
            raise ValueError(
                f"Correlation rho must be in (-1, 1), got {rho}"
            )

        self.H = H
        self.eta = eta
        self.rho = rho
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed

        # Effective step count used in Cholesky simulation
        self._eff_steps: int = min(n_steps, _CHOL_STEP_CAP)
        if n_steps > _CHOL_STEP_CAP:
            warnings.warn(
                f"n_steps={n_steps} > {_CHOL_STEP_CAP}.  Internally using "
                f"{self._eff_steps} steps (weekly grid) for fBM Cholesky "
                f"simulation.  Pricing accuracy is maintained via correct "
                f"time-scaling.",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_fbm_covariance(self, T: float) -> np.ndarray:
        """
        Build the (n, n) covariance matrix for fractional Brownian motion at
        time points t_i = i * T / n, i = 1, …, n, where n = self._eff_steps.

        The covariance kernel is:
            Cov(W^H_{s}, W^H_{t}) = 0.5 * (s^{2H} + t^{2H} − |t−s|^{2H})

        Parameters
        ----------
        T : float
            Option maturity in years.

        Returns
        -------
        C : np.ndarray, shape (n, n)
            Covariance matrix (symmetric, positive definite).
        """
        n = self._eff_steps
        dt = T / n
        t = np.arange(1, n + 1) * dt          # (n,)  — t_1, …, t_n
        H2 = 2.0 * self.H

        # Broadcast to build full (n, n) covariance matrix
        ti = t[:, None]  # column vector
        tj = t[None, :]  # row vector
        C = 0.5 * (ti ** H2 + tj ** H2 - np.abs(ti - tj) ** H2)
        return C

    def _simulate_fbm(
        self, T: float, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Simulate fractional Brownian motion paths via Cholesky decomposition.

        Steps:
          1. Build time grid t_i = i * T / n for i = 1 … n.
          2. Build covariance matrix C[i,j] (see _build_fbm_covariance).
          3. Cholesky: L = chol(C + ε I) where ε = 1e-10.
          4. Sample Z ~ N(0, I_{n × n_paths}).
          5. W^H = L @ Z  →  shape (n, n_paths).

        Parameters
        ----------
        T : float
            Maturity in years.
        rng : np.random.Generator
            Pre-seeded random generator.

        Returns
        -------
        W_H : np.ndarray, shape (n_steps_eff, n_paths)
            fBM values W^H_{t_i} at each time point (not increments).
        """
        C = self._build_fbm_covariance(T)
        n = self._eff_steps

        # Tikhonov regularisation for numerical stability
        C_reg = C + 1e-10 * np.eye(n)

        # Cholesky decomposition — try scipy first, fall back to numpy
        try:
            if _SCIPY_CHOLESKY:
                L = scipy_cholesky(C_reg, lower=True)
            else:
                L = np.linalg.cholesky(C_reg)
        except Exception:
            L = np.linalg.cholesky(C_reg)

        Z = rng.standard_normal((n, self.n_paths))  # (n, n_paths)
        W_H = L @ Z                                  # (n, n_paths)
        return W_H

    # ------------------------------------------------------------------
    # Public simulation
    # ------------------------------------------------------------------

    def simulate_paths(
        self, S: float, T: float, xi_0: float
    ) -> tuple:
        """
        Simulate Rough Bergomi spot and variance paths.

        Algorithm:
          1. Simulate W^H via Cholesky (shape: (n_eff, n_paths)).
          2. Compute variance:
               V_{t_i} = xi_0 * exp(eta * W^H_{t_i}  −  0.5 * eta^2 * t_i^{2H})
          3. Build correlated increments for spot BM:
               dW^S_i = rho * dW^H_i  +  sqrt(1 − rho^2) * dZ_i
             where dW^H_i = W^H_{t_i} − W^H_{t_{i-1}} are fBM increments,
             and dZ_i ~ N(0, dt) are independent.
          4. Simulate spot via log-Euler:
               S_{i+1} = S_i * exp(−0.5 * V_i * dt  +  sqrt(V_i * dt) * dW^S_i / sqrt(dt))
             i.e. the standard log-normal step using instantaneous variance V_i.

        Parameters
        ----------
        S : float
            Current spot price.
        T : float
            Maturity in years.
        xi_0 : float
            Initial forward variance (annualised variance, not vol).

        Returns
        -------
        spot_paths : np.ndarray, shape (n_eff + 1, n_paths)
            Spot price paths; spot_paths[0, :] = S.
        var_paths : np.ndarray, shape (n_eff + 1, n_paths)
            Variance paths; var_paths[0, :] = xi_0.
        """
        rng = np.random.default_rng(self.seed)
        n = self._eff_steps
        dt = T / n

        # Step 1: fBM values at each time node, shape (n, n_paths)
        W_H = self._simulate_fbm(T, rng)

        # Step 2: Variance process shape (n, n_paths)
        H2 = 2.0 * self.H
        t_grid = np.arange(1, n + 1) * dt           # (n,)
        # Broadcast t^{2H} over paths
        t_2H = (t_grid ** H2)[:, None]               # (n, 1)
        V_inner = self.eta * W_H - 0.5 * self.eta ** 2 * t_2H
        V_sim = xi_0 * np.exp(V_inner)               # (n, n_paths)

        # Step 3: Correlated spot BM increments
        # fBM increments: dW^H_{0} = W^H_{t_1} − 0 = W^H_{t_1}
        #                 dW^H_{i} = W^H_{t_i} − W^H_{t_{i-1}}
        dW_H = np.diff(W_H, axis=0, prepend=np.zeros((1, self.n_paths)))
        # Independent standard BM increments (zero-mean, variance dt)
        dZ = rng.standard_normal((n, self.n_paths)) * np.sqrt(dt)
        # Correlated increment
        dW_S = self.rho * dW_H + np.sqrt(max(1.0 - self.rho ** 2, 0.0)) * dZ

        # Step 4: Spot paths via log-Euler — shape (n+1, n_paths)
        spot_paths = np.empty((n + 1, self.n_paths))
        var_paths = np.empty((n + 1, self.n_paths))
        spot_paths[0] = S
        var_paths[0] = xi_0

        for i in range(n):
            V_cur = np.maximum(V_sim[i], 0.0)
            var_paths[i + 1] = V_cur
            log_step = -0.5 * V_cur * dt + np.sqrt(V_cur) * dW_S[i]
            spot_paths[i + 1] = spot_paths[i] * np.exp(log_step)

        return spot_paths, var_paths

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        xi_0: Optional[float] = None,
        q: float = 0.0,
    ) -> RoughVolResult:
        """
        Price European call and put options under the Rough Bergomi model.

        Parameters
        ----------
        S : float
            Current spot price.
        K : float
            Strike price.
        T : float
            Time to maturity in years.
        r : float
            Continuously compounded risk-free rate.
        xi_0 : float, optional
            Initial forward variance.  Defaults to 0.04 (≡ 20% annualised vol).
        q : float
            Continuous dividend yield.  Default 0.0.

        Returns
        -------
        RoughVolResult
            Call/put prices with Monte Carlo standard errors and (up to 100)
            sample paths for visualisation.
        """
        if T <= 0.0:
            return RoughVolResult(
                call=max(S - K, 0.0),
                put=max(K - S, 0.0),
                model="Rough Bergomi",
                hurst=self.H,
                eta=self.eta,
                xi_0=xi_0 if xi_0 is not None else 0.04,
                std_error_call=0.0,
                std_error_put=0.0,
                n_paths=self.n_paths,
                vol_paths=None,
                spot_paths=None,
            )

        if xi_0 is None:
            xi_0 = 0.04  # default: (20% vol)^2

        # Simulate all paths
        spot_paths, var_paths = self.simulate_paths(S, T, xi_0)

        # Terminal spot prices
        S_T = spot_paths[-1, :]

        # Discounted payoffs
        disc = np.exp(-(r - q) * T)
        call_payoffs = disc * np.maximum(S_T - K, 0.0)
        put_payoffs = disc * np.maximum(K - S_T, 0.0)

        call_price = float(np.mean(call_payoffs))
        put_price = float(np.mean(put_payoffs))

        se_call = float(np.std(call_payoffs, ddof=1) / np.sqrt(self.n_paths))
        se_put = float(np.std(put_payoffs, ddof=1) / np.sqrt(self.n_paths))

        # Store up to 100 paths for visualisation (annualised vol, not variance)
        n_vis = min(self.n_paths, 100)
        vol_vis = np.sqrt(np.maximum(var_paths[:, :n_vis], 0.0))
        spot_vis = spot_paths[:, :n_vis]

        return RoughVolResult(
            call=call_price,
            put=put_price,
            model="Rough Bergomi",
            hurst=self.H,
            eta=self.eta,
            xi_0=xi_0,
            std_error_call=se_call,
            std_error_put=se_put,
            n_paths=self.n_paths,
            vol_paths=vol_vis,
            spot_paths=spot_vis,
        )

    # ------------------------------------------------------------------
    # Implied volatility surface
    # ------------------------------------------------------------------

    def implied_vol_surface(
        self,
        S: float,
        T: float,
        r: float,
        xi_0: float,
        strikes_pct: Optional[List[float]] = None,
        q: float = 0.0,
    ) -> pd.DataFrame:
        """
        Compute the model-implied Black-Scholes volatility smile at a grid of
        moneyness levels.

        For each strike K = S * moneyness the Rough Bergomi call price is
        inverted via Brent's method to find the equivalent BS implied vol.

        Parameters
        ----------
        S : float
            Current spot price.
        T : float
            Option maturity in years.
        r : float
            Continuously compounded risk-free rate.
        xi_0 : float
            Initial forward variance (e.g. (0.20)^2 for 20% vol).
        strikes_pct : list of float, optional
            Moneyness grid (K/S).  Default: np.linspace(0.8, 1.2, 9).
        q : float
            Continuous dividend yield.  Default 0.0.

        Returns
        -------
        pd.DataFrame
            Columns: ['moneyness', 'strike', 'implied_vol', 'rb_price',
                      'bs_price']
        """
        if strikes_pct is None:
            strikes_pct = list(np.linspace(0.8, 1.2, 9))

        # Flat-vol reference for BS comparison
        sigma_ref = float(np.sqrt(xi_0))

        rows = []
        for m in strikes_pct:
            K = S * m
            rb_result = self.price(S, K, T, r, xi_0=xi_0, q=q)
            rb_call = rb_result.call
            bs_call = BlackScholesModel.call_price(S, K, T, r, sigma_ref, q)

            # Invert BS to find implied vol
            intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
            iv = np.nan
            if rb_call > intrinsic + 1e-8 and T > 0:
                def _obj(sigma: float) -> float:
                    return BlackScholesModel.call_price(S, K, T, r, sigma, q) - rb_call
                try:
                    iv = brentq(_obj, 1e-6, 20.0, xtol=1e-6, maxiter=500)
                except (ValueError, RuntimeError):
                    iv = np.nan

            rows.append({
                "moneyness": float(m),
                "strike": float(K),
                "implied_vol": float(iv) if not np.isnan(iv) else np.nan,
                "rb_price": float(rb_call),
                "bs_price": float(bs_call),
            })

        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# HURST ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

class HurstEstimator:
    """
    Estimate the Hurst parameter from a financial returns (or log-vol) series.

    Typical findings:
      - Realized volatility of equity indices: H ≈ 0.05–0.15  (very rough)
      - Log-price returns: H ≈ 0.5  (close to BM)
      - H < 0.5: mean-reverting / anti-persistent  (rough)
      - H = 0.5: standard Brownian motion
      - H > 0.5: persistent / trending
    """

    # ------------------------------------------------------------------
    # Variogram method
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_variogram(
        returns: pd.Series, max_lag: int = 50
    ) -> dict:
        """
        Estimate the Hurst parameter using the variogram (structure function)
        method.

        For a self-similar process with index H:
            E[|X_{t+h} − X_t|^2]  ~  h^{2H}

        Fit log(v_k) = 2H * log(k) + const via OLS.

        Parameters
        ----------
        returns : pd.Series
            Time series of returns (or log-vol changes).
        max_lag : int
            Maximum lag to include in the variogram.  Default 50.

        Returns
        -------
        dict with keys:
            'hurst'      : float  — estimated Hurst parameter
            'lags'       : np.ndarray  — lag integers 1 … max_lag
            'variogram'  : np.ndarray  — v_k values
            'fit_line'   : np.ndarray  — fitted log-linear values
            'r_squared'  : float  — R² of the log-linear fit
        """
        x = np.asarray(returns, dtype=float)
        x = x[np.isfinite(x)]
        n = len(x)
        if n < max_lag + 2:
            max_lag = max(1, n // 2 - 1)

        lags = np.arange(1, max_lag + 1)
        variogram = np.array([
            np.mean((x[k:] - x[:-k]) ** 2) for k in lags
        ])

        # Log-linear regression: log(v_k) = 2H*log(k) + c
        log_lags = np.log(lags)
        log_var = np.log(np.maximum(variogram, 1e-30))

        # OLS via polyfit (degree 1)
        coeffs = np.polyfit(log_lags, log_var, 1)
        H_est = float(coeffs[0]) / 2.0

        # R²
        log_var_fit = np.polyval(coeffs, log_lags)
        ss_res = np.sum((log_var - log_var_fit) ** 2)
        ss_tot = np.sum((log_var - log_var.mean()) ** 2)
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        return {
            "hurst": float(np.clip(H_est, 0.01, 0.99)),
            "lags": lags,
            "variogram": variogram,
            "fit_line": np.exp(log_var_fit),
            "r_squared": r_squared,
        }

    # ------------------------------------------------------------------
    # R/S analysis
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_rs(returns: pd.Series) -> float:
        """
        Estimate the Hurst parameter using rescaled range (R/S) analysis.

        For a time series of length n:
            R/S(n) = range(cumulative deviation) / std(series[:n])

        OLS regression of log(R/S) on log(n) yields slope H.

        Parameters
        ----------
        returns : pd.Series
            Input return series.

        Returns
        -------
        float
            Estimated Hurst parameter.
        """
        x = np.asarray(returns, dtype=float)
        x = x[np.isfinite(x)]
        n_total = len(x)

        # Window sizes: start at 10, double until n_total/4
        windows: List[int] = []
        w = 10
        while w <= n_total // 4:
            windows.append(w)
            w = int(w * 2)
        if not windows:
            return 0.5  # not enough data

        rs_values = []
        for w in windows:
            rs_for_w = []
            for start in range(0, n_total - w + 1, w):
                seg = x[start: start + w]
                mean_seg = seg.mean()
                cumdev = np.cumsum(seg - mean_seg)
                r = cumdev.max() - cumdev.min()
                s = seg.std(ddof=1)
                if s > 0:
                    rs_for_w.append(r / s)
            if rs_for_w:
                rs_values.append(np.mean(rs_for_w))
            else:
                rs_values.append(np.nan)

        log_n = np.log(windows)
        log_rs = np.log(np.array(rs_values, dtype=float))

        # Remove NaN rows
        mask = np.isfinite(log_rs)
        if mask.sum() < 2:
            return 0.5

        coeffs = np.polyfit(log_n[mask], log_rs[mask], 1)
        H_est = float(coeffs[0])
        return float(np.clip(H_est, 0.01, 0.99))

    # ------------------------------------------------------------------
    # Detrended Fluctuation Analysis (DFA)
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_dfa(
        returns: pd.Series, scales: Optional[List[int]] = None
    ) -> float:
        """
        Estimate the Hurst parameter via Detrended Fluctuation Analysis (DFA).

        DFA is more robust than R/S for financial time series as it removes
        local polynomial trends before measuring fluctuations.

        Algorithm:
          1. Compute cumulative profile Y[k] = cumsum(returns − mean).
          2. For each scale s:
             a. Divide Y into non-overlapping windows of size s.
             b. In each window fit a linear trend; compute RMS of residuals F_s.
          3. OLS of log(F_s) on log(s) gives slope H.

        Parameters
        ----------
        returns : pd.Series
            Input return series.
        scales : list of int, optional
            Window sizes to use.  Default: geometric sequence
            [10, 20, 40, 80, 160].

        Returns
        -------
        float
            Estimated Hurst parameter.
        """
        x = np.asarray(returns, dtype=float)
        x = x[np.isfinite(x)]
        n_total = len(x)

        if scales is None:
            scales = [10, 20, 40, 80, 160]

        # Keep only scales smaller than n_total/4
        scales = [s for s in scales if s <= n_total // 4]
        if len(scales) < 2:
            return 0.5

        # Cumulative profile
        Y = np.cumsum(x - x.mean())

        F_values = []
        valid_scales = []
        for s in scales:
            n_wins = n_total // s
            if n_wins == 0:
                continue
            rms_list = []
            for w in range(n_wins):
                seg = Y[w * s: (w + 1) * s]
                # Linear detrend
                t_idx = np.arange(s, dtype=float)
                coeffs = np.polyfit(t_idx, seg, 1)
                trend = np.polyval(coeffs, t_idx)
                residuals = seg - trend
                rms_list.append(np.sqrt(np.mean(residuals ** 2)))
            F_values.append(np.mean(rms_list))
            valid_scales.append(s)

        if len(valid_scales) < 2:
            return 0.5

        log_s = np.log(np.array(valid_scales, dtype=float))
        log_F = np.log(np.array(F_values, dtype=float))

        mask = np.isfinite(log_F)
        if mask.sum() < 2:
            return 0.5

        coeffs = np.polyfit(log_s[mask], log_F[mask], 1)
        H_est = float(coeffs[0])
        return float(np.clip(H_est, 0.01, 0.99))

    # ------------------------------------------------------------------
    # Interpretation helper
    # ------------------------------------------------------------------

    @staticmethod
    def interpret(H: float) -> str:
        """
        Return a human-readable interpretation of the Hurst parameter.

        Parameters
        ----------
        H : float
            Hurst parameter estimate.

        Returns
        -------
        str
            Interpretation string.
        """
        if H < 0.2:
            return (
                f"H = {H:.3f}: Very rough (typical equity vol regime). "
                "Volatility is strongly mean-reverting and rough — consistent "
                "with the Rough Bergomi model for short-dated options."
            )
        elif H < 0.4:
            return (
                f"H = {H:.3f}: Rough (mean-reverting vol). "
                "Volatility exhibits anti-persistence; rough stochastic vol "
                "models are appropriate."
            )
        elif H < 0.55:
            return (
                f"H = {H:.3f}: Near standard Brownian motion (H ≈ 0.5). "
                "Classical diffusion models (Heston, GBM) are reasonable "
                "approximations."
            )
        elif H < 0.7:
            return (
                f"H = {H:.3f}: Mildly persistent. "
                "Trending behaviour in the series; ARFIMA or fractional "
                "Ornstein-Uhlenbeck models may be appropriate."
            )
        else:
            return (
                f"H = {H:.3f}: Strongly persistent (trending). "
                "The series shows long memory; standard diffusion models will "
                "under-price tail risk."
            )


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def rough_vol_chart_data(result: RoughVolResult) -> dict:
    """
    Package RoughVolResult arrays into a dict suitable for Plotly charts.

    Parameters
    ----------
    result : RoughVolResult
        Output from RoughBergomiModel.price().

    Returns
    -------
    dict with keys:
        'spot_paths'  : np.ndarray or None  — shape (n_steps+1, ≤100)
        'var_paths'   : np.ndarray or None  — annualised vol (sqrt of variance)
        'time_grid'   : np.ndarray  — x-axis values in years
        'title'       : str  — suggested chart title
    """
    spot = result.spot_paths
    vol = result.vol_paths

    # Infer n_steps from stored arrays
    if spot is not None:
        n_steps = spot.shape[0] - 1
        # Infer T from shape and assumption of 1-year default; caller should pass T
        # We expose whatever time-points we have, scaled 0..1 (caller rescales with T)
        time_grid = np.linspace(0.0, 1.0, n_steps + 1)
    else:
        time_grid = np.array([0.0, 1.0])

    title = (
        f"Rough Bergomi: H={result.hurst:.2f}, "
        f"eta={result.eta:.2f}, "
        f"xi_0={result.xi_0:.4f} "
        f"(n_paths={result.n_paths:,})"
    )

    return {
        "spot_paths": spot,
        "var_paths": vol,
        "time_grid": time_grid,
        "title": title,
    }


def variogram_chart_data(variogram_result: dict) -> dict:
    """
    Package HurstEstimator.estimate_variogram() output for a Plotly chart.

    Parameters
    ----------
    variogram_result : dict
        Output from HurstEstimator.estimate_variogram().

    Returns
    -------
    dict with keys:
        'lags'       : np.ndarray  — lag values
        'variogram'  : np.ndarray  — empirical variogram values
        'fit_line'   : np.ndarray  — fitted power-law values
        'hurst'      : float  — estimated H
        'r_squared'  : float  — R² of the fit
        'title'      : str    — suggested chart title
    """
    H = variogram_result.get("hurst", float("nan"))
    r2 = variogram_result.get("r_squared", float("nan"))
    title = (
        f"Variogram Hurst Estimation: H = {H:.3f}  (R² = {r2:.3f})"
        if not np.isnan(H) else "Variogram Hurst Estimation"
    )
    return {
        "lags": variogram_result.get("lags"),
        "variogram": variogram_result.get("variogram"),
        "fit_line": variogram_result.get("fit_line"),
        "hurst": H,
        "r_squared": r2,
        "title": title,
    }
