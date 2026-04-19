"""
Option Pricing Models
---------------------
Black-Scholes, Binomial Tree (CRR), Monte Carlo (GBM), Heston (stochastic vol),
Merton Jump-Diffusion (1976), Kou Double-Exponential Jump-Diffusion (2002)
"""

import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptionParams:
    S: float          # Spot price
    K: float          # Strike price
    T: float          # Time to expiry in years
    r: float          # Annualised risk-free rate
    sigma: float      # Annualised volatility
    q: float = 0.0    # Continuous dividend yield
    option_type: str = "call"   # "call" or "put"


@dataclass
class PricingResult:
    call: float
    put: float
    model: str
    inputs: dict


# ─────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES MODEL
# ─────────────────────────────────────────────────────────────────────────────

class BlackScholesModel:
    """Analytical Black-Scholes-Merton pricing for European options."""

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        if T <= 0 or sigma <= 0:
            return np.nan
        return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        if T <= 0 or sigma <= 0:
            return np.nan
        return BlackScholesModel.d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        if T <= 0:
            return max(S - K, 0.0)
        d1 = BlackScholesModel.d1(S, K, T, r, sigma, q)
        d2 = d1 - sigma * np.sqrt(T)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        if T <= 0:
            return max(K - S, 0.0)
        d1 = BlackScholesModel.d1(S, K, T, r, sigma, q)
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    @classmethod
    def price(cls, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> PricingResult:
        call = cls.call_price(S, K, T, r, sigma, q)
        put = cls.put_price(S, K, T, r, sigma, q)
        return PricingResult(
            call=call, put=put, model="Black-Scholes",
            inputs=dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
        )

    @staticmethod
    def implied_volatility(
        market_price: float, S: float, K: float, T: float,
        r: float, q: float = 0.0, option_type: str = "call"
    ) -> Optional[float]:
        """Newton-Raphson / Brent's method to solve IV from market price."""
        if T <= 0:
            return np.nan
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        if market_price <= intrinsic + 1e-8:
            return np.nan

        pricer = BlackScholesModel.call_price if option_type == "call" else BlackScholesModel.put_price

        def objective(sigma: float) -> float:
            return pricer(S, K, T, r, sigma, q) - market_price

        try:
            iv = brentq(objective, 1e-6, 20.0, xtol=1e-6, maxiter=500)
            return iv
        except (ValueError, RuntimeError):
            return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# BINOMIAL TREE MODEL (CRR)
# ─────────────────────────────────────────────────────────────────────────────

class BinomialTreeModel:
    """Cox-Ross-Rubinstein binomial tree; supports American & European."""

    def __init__(self, n_steps: int = 200):
        self.n_steps = n_steps

    def price(
        self,
        S: float, K: float, T: float, r: float, sigma: float,
        q: float = 0.0, american: bool = False
    ) -> PricingResult:
        n = self.n_steps
        dt = T / n
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        disc = np.exp(-r * dt)
        p = (np.exp((r - q) * dt) - d) / (u - d)
        p = np.clip(p, 0, 1)

        call = self._build_tree(S, K, T, r, sigma, q, p, u, d, disc, n, dt, "call", american)
        put = self._build_tree(S, K, T, r, sigma, q, p, u, d, disc, n, dt, "put", american)

        label = f"Binomial Tree ({'American' if american else 'European'}, n={n})"
        return PricingResult(call=call, put=put, model=label,
                             inputs=dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q, n_steps=n, american=american))

    @staticmethod
    def _build_tree(
        S, K, T, r, sigma, q, p, u, d, disc, n, dt,
        option_type: str, american: bool
    ) -> float:
        # Terminal node payoffs
        ST = S * (u ** np.arange(n, -1, -1)) * (d ** np.arange(0, n + 1))
        if option_type == "call":
            V = np.maximum(ST - K, 0.0)
        else:
            V = np.maximum(K - ST, 0.0)

        # Backward induction
        for i in range(n - 1, -1, -1):
            ST = S * (u ** np.arange(i, -1, -1)) * (d ** np.arange(0, i + 1))
            V = disc * (p * V[:-1] + (1 - p) * V[1:])
            if american:
                if option_type == "call":
                    V = np.maximum(V, ST - K)
                else:
                    V = np.maximum(V, K - ST)
        return float(V[0])


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO MODEL (GBM)
# ─────────────────────────────────────────────────────────────────────────────

class MonteCarloModel:
    """GBM Monte Carlo simulation for European option pricing."""

    def __init__(self, n_paths: int = 10000, n_steps: int = 252, seed: int = 42):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        self._last_paths: Optional[np.ndarray] = None
        self._last_terminal: Optional[np.ndarray] = None

    def simulate_paths(
        self, S: float, T: float, r: float, sigma: float, q: float = 0.0
    ) -> np.ndarray:
        """Simulate GBM paths. Returns array of shape (n_steps+1, n_paths)."""
        rng = np.random.default_rng(self.seed)
        dt = T / self.n_steps
        drift = (r - q - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt)

        log_returns = rng.normal(drift, diffusion, size=(self.n_steps, self.n_paths))
        log_paths = np.cumsum(np.vstack([np.zeros(self.n_paths), log_returns]), axis=0)
        paths = S * np.exp(log_paths)
        self._last_paths = paths
        self._last_terminal = paths[-1]
        return paths

    def price(
        self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
    ) -> PricingResult:
        paths = self.simulate_paths(S, T, r, sigma, q)
        terminal = paths[-1]

        call_payoffs = np.maximum(terminal - K, 0.0)
        put_payoffs = np.maximum(K - terminal, 0.0)

        disc = np.exp(-r * T)
        call = disc * np.mean(call_payoffs)
        put = disc * np.mean(put_payoffs)

        return PricingResult(
            call=call, put=put,
            model=f"Monte Carlo (GBM, n={self.n_paths:,})",
            inputs=dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q, n_paths=self.n_paths)
        )

    @property
    def last_paths(self) -> Optional[np.ndarray]:
        return self._last_paths

    @property
    def last_terminal(self) -> Optional[np.ndarray]:
        return self._last_terminal

    def var_cvar(self, confidence: float = 0.05) -> tuple:
        """VaR and CVaR of terminal price distribution."""
        if self._last_terminal is None:
            raise RuntimeError("Run simulate_paths first.")
        sorted_t = np.sort(self._last_terminal)
        var_idx = int(confidence * len(sorted_t))
        var = sorted_t[var_idx]
        cvar = sorted_t[:var_idx].mean() if var_idx > 0 else sorted_t[0]
        return float(var), float(cvar)


# ─────────────────────────────────────────────────────────────────────────────
# HESTON MODEL (Stochastic Volatility) – Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────

class HestonModel:
    """
    Heston stochastic volatility model.
    dS = (r-q)S dt + sqrt(V) S dW_S
    dV = kappa*(theta - V) dt + sigma_v * sqrt(V) dW_V
    corr(dW_S, dW_V) = rho
    """

    def __init__(
        self,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma_v: float = 0.3,
        rho: float = -0.7,
        v0: float = 0.04,
        n_paths: int = 5000,
        n_steps: int = 252,
        seed: int = 42,
    ):
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.v0 = v0
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed

    def simulate_paths(
        self, S: float, T: float, r: float, q: float = 0.0
    ) -> tuple:
        """Returns (S_paths, V_paths) of shape (n_steps+1, n_paths)."""
        rng = np.random.default_rng(self.seed)
        dt = T / self.n_steps
        n, m = self.n_steps, self.n_paths

        Z1 = rng.standard_normal((n, m))
        Z2 = rng.standard_normal((n, m))
        W_S = Z1
        W_V = self.rho * Z1 + np.sqrt(1 - self.rho ** 2) * Z2

        S_paths = np.empty((n + 1, m))
        V_paths = np.empty((n + 1, m))
        S_paths[0] = S
        V_paths[0] = self.v0

        for i in range(n):
            V_cur = np.maximum(V_paths[i], 0.0)
            sqrt_V = np.sqrt(V_cur)
            # Euler-Maruyama discretisation
            V_paths[i + 1] = (
                V_cur
                + self.kappa * (self.theta - V_cur) * dt
                + self.sigma_v * sqrt_V * np.sqrt(dt) * W_V[i]
            )
            V_paths[i + 1] = np.maximum(V_paths[i + 1], 0.0)
            S_paths[i + 1] = S_paths[i] * np.exp(
                (r - q - 0.5 * V_cur) * dt + sqrt_V * np.sqrt(dt) * W_S[i]
            )

        return S_paths, V_paths

    def price(
        self, S: float, K: float, T: float, r: float, q: float = 0.0
    ) -> PricingResult:
        S_paths, _ = self.simulate_paths(S, T, r, q)
        terminal = S_paths[-1]

        disc = np.exp(-r * T)
        call = disc * np.mean(np.maximum(terminal - K, 0.0))
        put = disc * np.mean(np.maximum(K - terminal, 0.0))

        return PricingResult(
            call=call, put=put,
            model=f"Heston (MC, n={self.n_paths:,})",
            inputs=dict(S=S, K=K, T=T, r=r, q=q, kappa=self.kappa,
                        theta=self.theta, sigma_v=self.sigma_v, rho=self.rho, v0=self.v0)
        )


# ─────────────────────────────────────────────────────────────────────────────
# MERTON JUMP-DIFFUSION MODEL (1976)
# ─────────────────────────────────────────────────────────────────────────────

class MertonJumpDiffusionModel:
    """
    Merton (1976) Jump-Diffusion Model.

    Extends GBM by adding a compound Poisson process of log-normally distributed
    jumps.  The price is expressed as a series expansion of Black-Scholes prices
    — each term corresponds to exactly n jumps occurring over the life of the
    option.

    Parameters
    ----------
    lam : float
        Jump intensity — expected number of jumps per year.  Default 1.0.
    mu_j : float
        Mean log-jump size (log of the jump multiplier).  Default -0.05
        (roughly a 5 % average downward jump in the asset price).
    sigma_j : float
        Standard deviation of the log-jump size.  Default 0.10.
    n_terms : int
        Number of terms in the series expansion.  Convergence is typically
        achieved by 20–30 terms.  Default 30.
    """

    def __init__(
        self,
        lam: float = 1.0,
        mu_j: float = -0.05,
        sigma_j: float = 0.10,
        n_terms: int = 30,
    ):
        self.lam = lam
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.n_terms = n_terms

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bs_call(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
        """Black-Scholes call price used inside the series summation."""
        if T <= 0:
            return max(S - K, 0.0)
        if sigma <= 0:
            return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return (S * math.exp(-q * T) * norm.cdf(d1)
                - K * math.exp(-r * T) * norm.cdf(d2))

    @staticmethod
    def _bs_put(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
        """Black-Scholes put price used inside the series summation."""
        if T <= 0:
            return max(K - S, 0.0)
        if sigma <= 0:
            return max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return (K * math.exp(-r * T) * norm.cdf(-d2)
                - S * math.exp(-q * T) * norm.cdf(-d1))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> PricingResult:
        """
        Price a European option using the Merton (1976) series expansion.

        The formula is:

            kappa_j       = exp(mu_j + 0.5*sigma_j^2) - 1
            lambda_prime  = lam * (1 + kappa_j)          [risk-neutral intensity]

            C = sum_{n=0}^{N}  w_n * C_BS(sigma_n, r_n)

        where the Poisson weights are:

            w_n = exp(-lambda_prime*T) * (lambda_prime*T)^n / n!

        and the per-term BS parameters absorb the jump distribution:

            r_n     = r - lam*kappa_j + n*ln(1+kappa_j)/T   (clipped to [-0.5, 0.5])
            sigma_n = sqrt(sigma^2 + n*sigma_j^2/T)

        When lam=0 the series collapses to a single n=0 term which is exactly
        the standard Black-Scholes price.

        Parameters
        ----------
        S, K, T, r, sigma, q : standard option parameters

        Returns
        -------
        PricingResult
        """
        # Edge cases
        if T <= 0:
            return PricingResult(
                call=max(S - K, 0.0),
                put=max(K - S, 0.0),
                model="Merton Jump-Diffusion",
                inputs=dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q,
                            lam=self.lam, mu_j=self.mu_j, sigma_j=self.sigma_j),
            )
        if K <= 0:
            return PricingResult(
                call=float(S),
                put=0.0,
                model="Merton Jump-Diffusion",
                inputs=dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q,
                            lam=self.lam, mu_j=self.mu_j, sigma_j=self.sigma_j),
            )

        # Pre-compute jump-distribution constants
        kappa_j = math.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1.0
        lambda_prime = self.lam * (1.0 + kappa_j)

        # Drift correction term shared by all n
        drift_correction = self.lam * kappa_j

        # log(1+kappa_j) = mu_j + 0.5*sigma_j^2
        log_one_plus_kappa = math.log(1.0 + kappa_j) if kappa_j > -1 else self.mu_j

        lp_T = lambda_prime * T  # lambda_prime * T (Poisson rate parameter)

        call_price = 0.0
        put_price = 0.0

        exp_neg_lp_T = math.exp(-lp_T)

        # Accumulate series terms
        poisson_weight_unnorm = exp_neg_lp_T   # starts at exp(-lp_T) * (lp_T)^0 / 0!
        for n in range(self.n_terms):
            if n > 0:
                # Multiply by (lp_T / n) to get next Poisson weight efficiently
                poisson_weight_unnorm *= lp_T / n

            w = poisson_weight_unnorm

            # Per-term risk-free rate, clipped to avoid numerical blow-up
            r_n = r - drift_correction + n * log_one_plus_kappa / T
            r_n = max(-0.5, min(0.5, r_n))

            # Per-term volatility (variance adds n * sigma_j^2 / T from jumps)
            var_n = sigma ** 2 + n * self.sigma_j ** 2 / T
            sigma_n = math.sqrt(max(var_n, 1e-12))

            call_price += w * self._bs_call(S, K, T, r_n, sigma_n, q)
            put_price += w * self._bs_put(S, K, T, r_n, sigma_n, q)

        return PricingResult(
            call=float(call_price),
            put=float(put_price),
            model="Merton Jump-Diffusion",
            inputs=dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q,
                        lam=self.lam, mu_j=self.mu_j, sigma_j=self.sigma_j,
                        n_terms=self.n_terms),
        )

    def implied_vol_equivalent(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> dict:
        """
        Compare Merton jump-diffusion prices with standard Black-Scholes prices
        and return the jump premium for both calls and puts.

        Returns
        -------
        dict with keys:
            merton_call        : float
            merton_put         : float
            bs_call            : float
            bs_put             : float
            jump_premium_call  : float   (merton_call - bs_call)
            jump_premium_put   : float   (merton_put  - bs_put)
        """
        merton_result = self.price(S, K, T, r, sigma, q)
        bs_call = BlackScholesModel.call_price(S, K, T, r, sigma, q)
        bs_put = BlackScholesModel.put_price(S, K, T, r, sigma, q)

        return {
            "merton_call": merton_result.call,
            "merton_put": merton_result.put,
            "bs_call": bs_call,
            "bs_put": bs_put,
            "jump_premium_call": merton_result.call - bs_call,
            "jump_premium_put": merton_result.put - bs_put,
        }


# ─────────────────────────────────────────────────────────────────────────────
# KOU DOUBLE-EXPONENTIAL JUMP-DIFFUSION MODEL (2002)
# ─────────────────────────────────────────────────────────────────────────────

class KouJumpDiffusionModel:
    """
    Kou (2002) Double-Exponential Jump-Diffusion Model.

    Unlike Merton's log-normal jumps, Kou models upward and downward jumps
    with separate exponential distributions, producing an asymmetric leptokurtic
    return distribution.  The option is priced via Monte Carlo simulation.

    Jump size distribution (log of jump multiplier):
        With probability p    : Y ~  Exp(eta1)   [upward jump,   Y > 0]
        With probability 1-p  : Y ~ -Exp(eta2)   [downward jump, Y < 0]

    Parameters
    ----------
    lam : float
        Jump intensity (expected jumps per year).  Default 1.0.
    p : float
        Probability that a jump is upward.  Default 0.4.
    eta1 : float
        Rate parameter of the upward-jump exponential distribution.
        Higher eta1 implies smaller average upward jumps (mean = 1/eta1).
        Default 10.0.
    eta2 : float
        Rate parameter of the downward-jump exponential distribution.
        Higher eta2 implies smaller average downward jumps (mean = 1/eta2).
        Default 5.0.
    n_paths : int
        Number of Monte Carlo paths.  Default 50 000.
    n_steps : int
        Time steps per year of maturity.  Default 252.
    seed : int
        Random seed for reproducibility.  Default 42.
    """

    def __init__(
        self,
        lam: float = 1.0,
        p: float = 0.4,
        eta1: float = 10.0,
        eta2: float = 5.0,
        n_paths: int = 50_000,
        n_steps: int = 252,
        seed: int = 42,
    ):
        self.lam = lam
        self.p = p
        self.eta1 = eta1
        self.eta2 = eta2
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed

    # ------------------------------------------------------------------
    # Path simulation
    # ------------------------------------------------------------------

    def simulate_paths(
        self,
        S: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> np.ndarray:
        """
        Simulate Kou jump-diffusion paths using Euler discretisation.

        For each time step the log-return is:

            log_ret = (r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*dW
                      + sum_of_jump_sizes - lam*mean_jump*dt

        where the compensator  lam*mean_jump*dt  ensures the discounted
        price process is a martingale under the risk-neutral measure:

            mean_jump = p/eta1 - (1-p)/eta2

        Jumps in each step are drawn from a double-exponential distribution
        conditioned on the Poisson count for that step.  For efficiency,
        steps with zero jumps (the vast majority) skip the jump calculation
        entirely, and the per-step jump count is capped at 5.

        Parameters
        ----------
        S      : float  Current spot price.
        T      : float  Time to maturity in years.
        r      : float  Continuously compounded risk-free rate.
        sigma  : float  Diffusion volatility (annualised).
        q      : float  Continuous dividend yield.  Default 0.0.

        Returns
        -------
        paths : np.ndarray of shape (n_total_steps+1, n_paths)
        """
        rng = np.random.default_rng(self.seed)

        n_total_steps = max(int(T * self.n_steps), 1)
        dt = T / n_total_steps

        # Risk-neutral drift compensation for jumps
        mean_jump = self.p / self.eta1 - (1.0 - self.p) / self.eta2

        # Diffusion components (all steps at once for speed)
        dW = rng.standard_normal((n_total_steps, self.n_paths))

        # Poisson jump counts per step
        n_jumps_array = rng.poisson(self.lam * dt, size=(n_total_steps, self.n_paths))

        # Pre-allocate log-return array
        log_returns = ((r - q - 0.5 * sigma ** 2) * dt
                       + sigma * math.sqrt(dt) * dW
                       - self.lam * mean_jump * dt)  # drift correction (scalar broadcast)

        # Add jump contributions only for steps that have at least one jump
        # Cap jumps at MAX_JUMPS_PER_STEP for numerical efficiency
        MAX_JUMPS_PER_STEP = 5

        # Find which (step, path) cells have jumps
        jump_mask = n_jumps_array > 0
        if jump_mask.any():
            step_indices, path_indices = np.where(jump_mask)
            capped_counts = np.minimum(n_jumps_array[step_indices, path_indices],
                                       MAX_JUMPS_PER_STEP)

            # Generate all jump sizes in one vectorised call.
            # We generate max_total_jumps draws and index via cumsum offsets.
            max_total_jumps = int(capped_counts.sum())
            if max_total_jumps > 0:
                # Direction: True -> upward
                up_mask = rng.random(max_total_jumps) < self.p
                magnitudes = np.where(
                    up_mask,
                    rng.exponential(1.0 / self.eta1, size=max_total_jumps),
                    -rng.exponential(1.0 / self.eta2, size=max_total_jumps),
                )
                # Accumulate per (step, path) cell using reduceat
                offsets = np.concatenate([[0], np.cumsum(capped_counts[:-1])])
                jump_totals = np.add.reduceat(magnitudes, offsets)
                log_returns[step_indices, path_indices] += jump_totals

        # Build paths from cumulative log-returns
        log_paths = np.empty((n_total_steps + 1, self.n_paths))
        log_paths[0] = 0.0
        np.cumsum(log_returns, axis=0, out=log_paths[1:])
        paths = S * np.exp(log_paths)
        return paths

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> PricingResult:
        """
        Price a European option by discounting the mean simulated payoff.

        Parameters
        ----------
        S, K, T, r, sigma, q : standard option parameters.

        Returns
        -------
        PricingResult
        """
        if T <= 0:
            return PricingResult(
                call=max(S - K, 0.0),
                put=max(K - S, 0.0),
                model=f"Kou JD (MC, n={self.n_paths:,})",
                inputs=dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q,
                            lam=self.lam, p=self.p, eta1=self.eta1, eta2=self.eta2),
            )

        paths = self.simulate_paths(S, T, r, sigma, q)
        terminal = paths[-1]

        disc = math.exp(-r * T)
        call = disc * float(np.mean(np.maximum(terminal - K, 0.0)))
        put = disc * float(np.mean(np.maximum(K - terminal, 0.0)))

        return PricingResult(
            call=call,
            put=put,
            model=f"Kou JD (MC, n={self.n_paths:,})",
            inputs=dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q,
                        lam=self.lam, p=self.p, eta1=self.eta1, eta2=self.eta2,
                        n_paths=self.n_paths),
        )


# ─────────────────────────────────────────────────────────────────────────────
# PRICING ENGINE (aggregates all models)
# ─────────────────────────────────────────────────────────────────────────────

class PricingEngine:
    """Unified entry-point that runs all or selected pricing models."""

    def __init__(
        self,
        mc_paths: int = 10000,
        mc_steps: int = 252,
        binomial_steps: int = 200,
        heston_params: Optional[dict] = None,
        merton_params: Optional[dict] = None,
        kou_params: Optional[dict] = None,
        seed: int = 42,
    ):
        self.bs = BlackScholesModel()
        self.binomial = BinomialTreeModel(n_steps=binomial_steps)
        self.mc = MonteCarloModel(n_paths=mc_paths, n_steps=mc_steps, seed=seed)

        hp = heston_params or {}
        self.heston = HestonModel(
            kappa=hp.get("kappa", 2.0),
            theta=hp.get("theta", 0.04),
            sigma_v=hp.get("sigma_v", 0.3),
            rho=hp.get("rho", -0.7),
            v0=hp.get("v0", 0.04),
            n_paths=hp.get("n_paths", 5000),
            n_steps=mc_steps,
            seed=seed,
        )

        mp = merton_params or {}
        self.merton = MertonJumpDiffusionModel(
            lam=mp.get("lam", 1.0),
            mu_j=mp.get("mu_j", -0.05),
            sigma_j=mp.get("sigma_j", 0.10),
            n_terms=mp.get("n_terms", 30),
        )

        kp = kou_params or {}
        self.kou = KouJumpDiffusionModel(
            lam=kp.get("lam", 1.0),
            p=kp.get("p", 0.4),
            eta1=kp.get("eta1", 10.0),
            eta2=kp.get("eta2", 5.0),
            n_paths=kp.get("n_paths", 50_000),
            n_steps=mc_steps,
            seed=seed,
        )

    def price_all(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        american: bool = False,
        include_heston: bool = True,
        include_jump_diffusion: bool = True,
    ) -> dict:
        """
        Returns dict of {model_name: PricingResult} for all active models.

        Keys
        ----
        "Black-Scholes"        : always included
        "Binomial Tree"        : always included
        "Monte Carlo (GBM)"    : always included
        "Heston (Stoch. Vol)"  : included when include_heston=True
        "Merton JD"            : included when include_jump_diffusion=True
        "Kou JD"               : included when include_jump_diffusion=True

        Each model is wrapped in a try/except so a numerical failure in one
        model does not prevent the others from being returned.
        """
        results: dict = {}

        results["Black-Scholes"] = self.bs.price(S, K, T, r, sigma, q)

        try:
            results["Binomial Tree"] = self.binomial.price(
                S, K, T, r, sigma, q, american=american
            )
        except Exception:
            pass

        try:
            results["Monte Carlo (GBM)"] = self.mc.price(S, K, T, r, sigma, q)
        except Exception:
            pass

        if include_heston:
            try:
                results["Heston (Stoch. Vol)"] = self.heston.price(S, K, T, r, q)
            except Exception:
                pass

        if include_jump_diffusion:
            try:
                results["Merton JD"] = self.merton.price(S, K, T, r, sigma, q)
            except Exception:
                pass

            try:
                results["Kou JD"] = self.kou.price(S, K, T, r, sigma, q)
            except Exception:
                pass

        return results
