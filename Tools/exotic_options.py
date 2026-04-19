"""
Exotic Options Pricer
---------------------
Monte Carlo pricing for barrier, Asian, and lookback exotic options.

Classes
-------
BarrierOptionPricer  – up/down in/out barrier options with rebate support
AsianOptionPricer    – arithmetic & geometric Asian (average-price) options
LookbackOptionPricer – floating- and fixed-strike lookback options

Each pricer returns an ExoticResult dataclass containing price, std error,
vanilla BS comparison, exotic premium, Greeks (finite-difference bumps), and
a convergence array for charting.

All simulation uses np.random.default_rng(seed) for full reproducibility and
common-random-number (CRN) Greek estimation.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.stats import norm

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TRADING_DAYS: int = 252
MIN_PATHS: int = 1_000
MAX_PATHS: int = 200_000
DEFAULT_PATHS: int = 50_000
DEFAULT_STEPS: int = 252
BUMP_SPOT_PCT: float = 0.01       # 1% bump for delta/gamma
BUMP_VOL_ABS: float = 0.001       # 0.1% absolute bump for vega
BUMP_TIME_DAYS: float = 1.0       # 1 calendar day bump for theta


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExoticResult:
    """
    Container for the output of an exotic option pricer.

    Attributes
    ----------
    option_type    : "call" or "put"
    exotic_type    : "barrier", "asian", or "lookback"
    exotic_subtype : e.g. "up-and-out", "arithmetic", "floating_call"
    price          : Monte Carlo price (discounted expected payoff)
    std_error      : MC standard error = std(payoffs) / sqrt(n_paths)
    vanilla_price  : Black-Scholes vanilla price for comparison
    exotic_premium : price - vanilla_price  (negative for knock-outs)
    greeks         : dict with keys delta, gamma, vega, theta, vega_per_pct
    n_paths        : number of MC paths used
    convergence    : running average price sampled every 100 paths
    """
    option_type: str
    exotic_type: str
    exotic_subtype: str
    price: float
    std_error: float
    vanilla_price: float
    exotic_premium: float
    greeks: dict
    n_paths: int
    convergence: np.ndarray


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL GBM PATH SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_gbm(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion price paths via the log-Euler scheme.

    The log-Euler discretisation is exact for GBM (no discretisation error in
    the marginal distribution):

        S_{t+dt} = S_t * exp((r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    where Z ~ N(0,1).

    Parameters
    ----------
    S        : current spot price
    T        : time to expiry in years
    r        : annualised risk-free rate
    sigma    : annualised volatility
    q        : continuous dividend yield
    n_paths  : number of simulation paths
    n_steps  : number of time steps
    rng      : numpy Generator (controls reproducibility)

    Returns
    -------
    np.ndarray of shape (n_steps + 1, n_paths)
        Row 0 is S (the initial spot), row n_steps is the terminal price.
    """
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * math.sqrt(dt)

    # Shape: (n_steps, n_paths)
    Z = rng.standard_normal((n_steps, n_paths))
    log_increments = drift + diffusion * Z          # (n_steps, n_paths)

    # Prepend a row of zeros to represent log(S0/S0) = 0
    log_paths = np.vstack([np.zeros(n_paths), log_increments])
    log_paths = np.cumsum(log_paths, axis=0)        # (n_steps+1, n_paths)

    return S * np.exp(log_paths)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL BLACK-SCHOLES HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bs_price(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call", q: float = 0.0
) -> float:
    """Black-Scholes-Merton price for a European call or put."""
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    if sigma <= 0:
        disc_fwd = S * math.exp(-q * T) - K * math.exp(-r * T)
        if option_type == "call":
            return max(disc_fwd, 0.0)
        return max(-disc_fwd, 0.0)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "call":
        return (S * math.exp(-q * T) * norm.cdf(d1)
                - K * math.exp(-r * T) * norm.cdf(d2))
    return (K * math.exp(-r * T) * norm.cdf(-d2)
            - S * math.exp(-q * T) * norm.cdf(-d1))


def _convergence_array(discounted_payoffs: np.ndarray, step: int = 100) -> np.ndarray:
    """
    Build a convergence array by computing the running mean of discounted payoffs
    sampled every *step* paths.

    Parameters
    ----------
    discounted_payoffs : 1-D array of per-path discounted payoffs
    step               : sampling interval (default 100)

    Returns
    -------
    np.ndarray of shape (ceil(n_paths / step),)
    """
    n = len(discounted_payoffs)
    sample_indices = np.arange(step, n + 1, step)
    if len(sample_indices) == 0 or sample_indices[-1] < n:
        sample_indices = np.append(sample_indices, n)
    return np.array([discounted_payoffs[:i].mean() for i in sample_indices])


# ─────────────────────────────────────────────────────────────────────────────
# BARRIER OPTION PRICER
# ─────────────────────────────────────────────────────────────────────────────

class BarrierOptionPricer:
    """
    Monte Carlo pricer for continuous barrier options.

    Supported barrier types
    -----------------------
    "up-and-out"  : option ceases (knocked out) if the spot ever rises above H
    "up-and-in"   : option activates only if the spot ever rises above H
    "down-and-out": option ceases if the spot ever falls below H
    "down-and-in" : option activates only if the spot ever falls below H

    For knocked-out paths a *rebate* (default 0) is paid at expiry.

    Parameters
    ----------
    barrier      : float – the barrier level H
    barrier_type : str – one of the four types listed above
    rebate       : float – cash payment on knock-out (default 0.0)
    n_paths      : int – number of MC simulation paths
    n_steps      : int – number of time steps per path
    seed         : int – RNG seed for reproducibility
    """

    _VALID_TYPES = {"up-and-out", "up-and-in", "down-and-out", "down-and-in"}

    def __init__(
        self,
        barrier: float,
        barrier_type: str = "down-and-out",
        rebate: float = 0.0,
        n_paths: int = DEFAULT_PATHS,
        n_steps: int = DEFAULT_STEPS,
        seed: int = 42,
    ) -> None:
        if barrier is None:
            raise ValueError(
                "barrier must be a positive float (e.g. barrier=90.0 for a "
                "down-and-out option with H=90)."
            )
        if barrier_type not in self._VALID_TYPES:
            raise ValueError(
                f"barrier_type must be one of {self._VALID_TYPES}, "
                f"got '{barrier_type}'."
            )
        self.barrier = float(barrier)
        self.barrier_type = barrier_type
        self.rebate = float(rebate)
        self.n_paths = int(np.clip(n_paths, MIN_PATHS, MAX_PATHS))
        self.n_steps = max(n_steps, 1)
        self.seed = seed

    # ------------------------------------------------------------------
    # Public pricing method
    # ------------------------------------------------------------------

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        q: float = 0.0,
    ) -> ExoticResult:
        """
        Price the barrier option via Monte Carlo.

        Parameters
        ----------
        S           : spot price
        K           : strike price
        T           : time to expiry (years)
        r           : risk-free rate
        sigma       : volatility
        option_type : "call" or "put"
        q           : continuous dividend yield

        Returns
        -------
        ExoticResult
        """
        # ── edge cases ────────────────────────────────────────────────
        if T <= 0:
            intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
            vanilla = _bs_price(S, K, T, r, sigma, option_type, q)
            return ExoticResult(
                option_type=option_type,
                exotic_type="barrier",
                exotic_subtype=self.barrier_type,
                price=intrinsic,
                std_error=0.0,
                vanilla_price=vanilla,
                exotic_premium=intrinsic - vanilla,
                greeks={"delta": 0.0, "gamma": 0.0, "vega": 0.0,
                        "theta": 0.0, "vega_per_pct": 0.0},
                n_paths=self.n_paths,
                convergence=np.array([intrinsic]),
            )

        H = self.barrier
        # Knocked out immediately if spot is already past barrier
        if self.barrier_type in ("up-and-out", "up-and-in") and S >= H:
            immediate_price = self.rebate * math.exp(-r * T) if self.barrier_type == "up-and-out" else 0.0
            vanilla = _bs_price(S, K, T, r, sigma, option_type, q)
            return ExoticResult(
                option_type=option_type,
                exotic_type="barrier",
                exotic_subtype=self.barrier_type,
                price=immediate_price,
                std_error=0.0,
                vanilla_price=vanilla,
                exotic_premium=immediate_price - vanilla,
                greeks={"delta": 0.0, "gamma": 0.0, "vega": 0.0,
                        "theta": 0.0, "vega_per_pct": 0.0},
                n_paths=self.n_paths,
                convergence=np.array([immediate_price]),
            )
        if self.barrier_type in ("down-and-out", "down-and-in") and S <= H:
            immediate_price = self.rebate * math.exp(-r * T) if self.barrier_type == "down-and-out" else 0.0
            vanilla = _bs_price(S, K, T, r, sigma, option_type, q)
            return ExoticResult(
                option_type=option_type,
                exotic_type="barrier",
                exotic_subtype=self.barrier_type,
                price=immediate_price,
                std_error=0.0,
                vanilla_price=vanilla,
                exotic_premium=immediate_price - vanilla,
                greeks={"delta": 0.0, "gamma": 0.0, "vega": 0.0,
                        "theta": 0.0, "vega_per_pct": 0.0},
                n_paths=self.n_paths,
                convergence=np.array([immediate_price]),
            )

        # ── core simulation ───────────────────────────────────────────
        disc_payoffs = self._mc_payoffs(S, K, T, r, sigma, option_type, q, self.seed)
        disc = math.exp(-r * T)

        price_val = float(np.mean(disc_payoffs))
        std_err = float(np.std(disc_payoffs, ddof=1) / math.sqrt(self.n_paths))
        vanilla = _bs_price(S, K, T, r, sigma, option_type, q)
        convergence = _convergence_array(disc_payoffs)

        # ── Greeks ────────────────────────────────────────────────────
        greeks = compute_exotic_greeks(self, S, K, T, r, sigma, option_type, q)

        return ExoticResult(
            option_type=option_type,
            exotic_type="barrier",
            exotic_subtype=self.barrier_type,
            price=price_val,
            std_error=std_err,
            vanilla_price=vanilla,
            exotic_premium=price_val - vanilla,
            greeks=greeks,
            n_paths=self.n_paths,
            convergence=convergence,
        )

    # ------------------------------------------------------------------
    # Internal MC helper (returns discounted payoffs array)
    # ------------------------------------------------------------------

    def _mc_payoffs(
        self,
        S: float, K: float, T: float, r: float,
        sigma: float, option_type: str, q: float,
        seed: int,
    ) -> np.ndarray:
        """Simulate paths and return array of discounted barrier payoffs."""
        rng = np.random.default_rng(seed)
        paths = _simulate_gbm(S, T, r, sigma, q, self.n_paths, self.n_steps, rng)
        H = self.barrier
        btype = self.barrier_type

        # Determine breach: shape (n_paths,)
        if btype in ("up-and-out", "up-and-in"):
            breached = np.any(paths[1:] >= H, axis=0)   # exclude t=0 row
        else:  # down-and-out / down-and-in
            breached = np.any(paths[1:] <= H, axis=0)

        # Vanilla payoff at expiry
        terminal = paths[-1]
        if option_type == "call":
            vanilla_payoff = np.maximum(terminal - K, 0.0)
        else:
            vanilla_payoff = np.maximum(K - terminal, 0.0)

        # Apply barrier condition
        if btype in ("up-and-out", "down-and-out"):
            # Survive = not breached; knocked-out paths pay rebate
            payoff = np.where(breached, self.rebate, vanilla_payoff)
        else:  # up-and-in / down-and-in
            # Only activated paths pay; others pay zero
            payoff = np.where(breached, vanilla_payoff, 0.0)

        disc = math.exp(-r * T)
        return disc * payoff

    # ------------------------------------------------------------------
    # Analytical closed-form (up-and-out call / down-and-out put)
    # ------------------------------------------------------------------

    @staticmethod
    def analytical_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        barrier: float,
        barrier_type: str,
        option_type: str = "call",
        q: float = 0.0,
        rebate: float = 0.0,
    ) -> Optional[float]:
        """
        Closed-form price for continuous barrier options using the reflection
        principle.  Only "up-and-out" call and "down-and-out" put are
        implemented; for all other combinations this returns None (use MC).

        Up-and-out call (requires H > K):
            C_uao = C_BS(S,K) - (H/S)^(2*(r-q)/sigma^2 - 1) * C_BS(H^2/S, K)

        If S >= H (already knocked out), returns the discounted rebate.

        Parameters
        ----------
        S           : spot price
        K           : strike
        T           : time to expiry (years)
        r           : risk-free rate
        sigma       : volatility
        barrier     : barrier level H
        barrier_type: one of the four barrier types
        option_type : "call" or "put"
        q           : dividend yield
        rebate      : payout on knock-out

        Returns
        -------
        float price, or None if no closed form available
        """
        H = float(barrier)

        # ── up-and-out call ───────────────────────────────────────────
        if barrier_type == "up-and-out" and option_type == "call":
            if S >= H:
                return rebate * math.exp(-r * T)
            if H <= K:
                # Barrier below (or at) strike: option is always knocked out
                # before it can be in the money; price = 0 (no rebate variant)
                return rebate * math.exp(-r * T)
            exponent = 2.0 * (r - q) / (sigma ** 2) - 1.0
            c1 = _bs_price(S, K, T, r, sigma, "call", q)
            c2 = _bs_price(H ** 2 / S, K, T, r, sigma, "call", q)
            return c1 - (H / S) ** exponent * c2

        # ── down-and-out put ──────────────────────────────────────────
        if barrier_type == "down-and-out" and option_type == "put":
            if S <= H:
                return rebate * math.exp(-r * T)
            if H >= K:
                return rebate * math.exp(-r * T)
            exponent = 2.0 * (r - q) / (sigma ** 2) - 1.0
            p1 = _bs_price(S, K, T, r, sigma, "put", q)
            p2 = _bs_price(H ** 2 / S, K, T, r, sigma, "put", q)
            return p1 - (H / S) ** exponent * p2

        # No closed form for other combinations
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ASIAN OPTION PRICER
# ─────────────────────────────────────────────────────────────────────────────

class AsianOptionPricer:
    """
    Monte Carlo pricer for Asian (average-price) options.

    Supported average types
    -----------------------
    "arithmetic" : payoff based on the arithmetic mean of monitored prices
    "geometric"  : payoff based on the geometric mean (also has closed form)

    Monitoring frequencies
    ----------------------
    "daily"   : every step  (all n_steps observations)
    "weekly"  : every 5 steps
    "monthly" : every 21 steps

    Variance Reduction (arithmetic only)
    -------------------------------------
    When use_control_variate=True the geometric Asian (which has a closed-form
    price) is used as a control variate:

        price_arith = price_arith_MC + (price_geo_CF - price_geo_MC)

    This exploits the near-perfect correlation between arithmetic and geometric
    average payoffs to substantially reduce variance.

    Parameters
    ----------
    avg_type            : "arithmetic" or "geometric"
    avg_frequency       : "daily", "weekly", or "monthly"
    use_control_variate : bool – enable geometric control variate (default True)
    n_paths             : int
    n_steps             : int
    seed                : int
    """

    _FREQ_MAP = {"daily": 1, "weekly": 5, "monthly": 21}

    def __init__(
        self,
        avg_type: str = "arithmetic",
        avg_frequency: str = "daily",
        use_control_variate: bool = True,
        n_paths: int = DEFAULT_PATHS,
        n_steps: int = DEFAULT_STEPS,
        seed: int = 42,
    ) -> None:
        if avg_type not in ("arithmetic", "geometric"):
            raise ValueError(f"avg_type must be 'arithmetic' or 'geometric', got '{avg_type}'.")
        if avg_frequency not in self._FREQ_MAP:
            raise ValueError(
                f"avg_frequency must be one of {list(self._FREQ_MAP)}, got '{avg_frequency}'."
            )
        self.avg_type = avg_type
        self.avg_frequency = avg_frequency
        self.use_control_variate = use_control_variate
        self.n_paths = int(np.clip(n_paths, MIN_PATHS, MAX_PATHS))
        self.n_steps = max(n_steps, 1)
        self.seed = seed

    # ------------------------------------------------------------------
    # Public pricing method
    # ------------------------------------------------------------------

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        q: float = 0.0,
    ) -> ExoticResult:
        """
        Price the Asian option via Monte Carlo.

        Parameters
        ----------
        S           : spot price
        K           : strike price
        T           : time to expiry (years)
        r           : risk-free rate
        sigma       : volatility
        option_type : "call" or "put"
        q           : continuous dividend yield

        Returns
        -------
        ExoticResult
        """
        if T <= 0:
            intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
            vanilla = _bs_price(S, K, T, r, sigma, option_type, q)
            return ExoticResult(
                option_type=option_type,
                exotic_type="asian",
                exotic_subtype=self.avg_type,
                price=intrinsic,
                std_error=0.0,
                vanilla_price=vanilla,
                exotic_premium=intrinsic - vanilla,
                greeks={"delta": 0.0, "gamma": 0.0, "vega": 0.0,
                        "theta": 0.0, "vega_per_pct": 0.0},
                n_paths=self.n_paths,
                convergence=np.array([intrinsic]),
            )

        disc_payoffs, price_val, std_err = self._mc_price(S, K, T, r, sigma, option_type, q, self.seed)
        vanilla = _bs_price(S, K, T, r, sigma, option_type, q)
        convergence = _convergence_array(disc_payoffs)
        greeks = compute_exotic_greeks(self, S, K, T, r, sigma, option_type, q)

        return ExoticResult(
            option_type=option_type,
            exotic_type="asian",
            exotic_subtype=self.avg_type,
            price=price_val,
            std_error=std_err,
            vanilla_price=vanilla,
            exotic_premium=price_val - vanilla,
            greeks=greeks,
            n_paths=self.n_paths,
            convergence=convergence,
        )

    # ------------------------------------------------------------------
    # Internal MC helper
    # ------------------------------------------------------------------

    def _mc_price(
        self,
        S: float, K: float, T: float, r: float,
        sigma: float, option_type: str, q: float,
        seed: int,
    ):
        """
        Simulate paths and return (disc_payoffs_array, price, std_error).
        Applies control-variate correction for arithmetic average if requested.
        """
        rng = np.random.default_rng(seed)
        paths = _simulate_gbm(S, T, r, sigma, q, self.n_paths, self.n_steps, rng)

        # Determine monitoring indices (exclude the initial t=0 row)
        freq = self._FREQ_MAP[self.avg_frequency]
        all_indices = np.arange(1, self.n_steps + 1)
        monitor_idx = all_indices[::freq]         # every `freq` steps
        if len(monitor_idx) == 0:
            monitor_idx = np.array([self.n_steps])
        monitored = paths[monitor_idx, :]          # (m_obs, n_paths)

        disc = math.exp(-r * T)

        if self.avg_type == "geometric":
            # Geometric mean along monitored observations
            log_avg = np.mean(np.log(monitored), axis=0)
            avg = np.exp(log_avg)
            if option_type == "call":
                payoff = np.maximum(avg - K, 0.0)
            else:
                payoff = np.maximum(K - avg, 0.0)
            disc_payoffs = disc * payoff
            price_val = float(np.mean(disc_payoffs))
            std_err = float(np.std(disc_payoffs, ddof=1) / math.sqrt(self.n_paths))
            return disc_payoffs, price_val, std_err

        # ── arithmetic average ────────────────────────────────────────
        arith_avg = np.mean(monitored, axis=0)     # (n_paths,)
        if option_type == "call":
            arith_payoff = np.maximum(arith_avg - K, 0.0)
        else:
            arith_payoff = np.maximum(K - arith_avg, 0.0)
        disc_arith = disc * arith_payoff

        if self.use_control_variate:
            # Geometric control variate (same paths, same seed segment)
            log_avg = np.mean(np.log(monitored), axis=0)
            geo_avg = np.exp(log_avg)
            if option_type == "call":
                geo_payoff = np.maximum(geo_avg - K, 0.0)
            else:
                geo_payoff = np.maximum(K - geo_avg, 0.0)
            disc_geo_mc = disc * geo_payoff

            # Closed-form geometric Asian price (n_steps used as proxy for
            # number of monitoring points)
            n_obs = len(monitor_idx)
            geo_cf = self.geometric_closed_form(S, K, T, r, sigma, option_type, q, n_steps=n_obs)

            # Control-variate corrected payoffs
            disc_payoffs = disc_arith + (geo_cf - float(np.mean(disc_geo_mc)))
            # disc_payoffs is now a scalar correction on top of the MC vector;
            # rebuild as vector for std_error and convergence
            correction = geo_cf - float(np.mean(disc_geo_mc))
            disc_payoffs = disc_arith + correction  # uniform additive shift
        else:
            disc_payoffs = disc_arith

        price_val = float(np.mean(disc_payoffs))
        std_err = float(np.std(disc_payoffs, ddof=1) / math.sqrt(self.n_paths))
        return disc_payoffs, price_val, std_err

    # ------------------------------------------------------------------
    # Geometric Asian closed-form
    # ------------------------------------------------------------------

    @staticmethod
    def geometric_closed_form(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        q: float = 0.0,
        n_steps: int = 252,
    ) -> float:
        """
        Closed-form price for a discretely-monitored geometric average-price
        Asian option (Kemna & Vorst 1990 adjusted Black-Scholes).

        Adjusted parameters:
            n = n_steps  (number of monitoring dates)
            sigma_adj = sigma * sqrt((2n + 1) / (6*(n + 1)))
            r_adj     = 0.5*(r - q - 0.5*sigma^2) + 0.5*sigma_adj^2

        The option is then priced as a standard European option using
        sigma_adj as volatility and r_adj as the cost-of-carry (with q=0).

        Parameters
        ----------
        S        : spot price
        K        : strike
        T        : time to expiry (years)
        r        : risk-free rate
        sigma    : volatility
        option_type : "call" or "put"
        q        : dividend yield
        n_steps  : number of monitoring dates

        Returns
        -------
        float
        """
        if T <= 0:
            if option_type == "call":
                return max(S - K, 0.0)
            return max(K - S, 0.0)

        n = max(int(n_steps), 1)
        sigma_adj = sigma * math.sqrt((2 * n + 1) / (6 * (n + 1)))
        r_adj = 0.5 * (r - q - 0.5 * sigma ** 2) + 0.5 * sigma_adj ** 2

        # Price using adjusted parameters; treat as European with no dividend
        # (cost-of-carry already embedded in r_adj)
        return _bs_price(S, K, T, r_adj, sigma_adj, option_type, q=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# LOOKBACK OPTION PRICER
# ─────────────────────────────────────────────────────────────────────────────

class LookbackOptionPricer:
    """
    Monte Carlo pricer for lookback options.

    Supported subtypes
    ------------------
    "floating_call" : payoff = S_T - min(S_t)  [floating strike call]
    "floating_put"  : payoff = max(S_t) - S_T   [floating strike put]
    "fixed_call"    : payoff = max(max(S_t) - K, 0)  [fixed strike call]
    "fixed_put"     : payoff = max(K - min(S_t), 0)   [fixed strike put]

    For floating-strike subtypes, the strike K is ignored.

    Parameters
    ----------
    subtype  : one of the four subtypes above
    n_paths  : int
    n_steps  : int
    seed     : int
    """

    _VALID_SUBTYPES = {"floating_call", "floating_put", "fixed_call", "fixed_put"}

    def __init__(
        self,
        subtype: str = "floating_call",
        n_paths: int = DEFAULT_PATHS,
        n_steps: int = DEFAULT_STEPS,
        seed: int = 42,
    ) -> None:
        if subtype not in self._VALID_SUBTYPES:
            raise ValueError(
                f"subtype must be one of {self._VALID_SUBTYPES}, got '{subtype}'."
            )
        self.subtype = subtype
        self.n_paths = int(np.clip(n_paths, MIN_PATHS, MAX_PATHS))
        self.n_steps = max(n_steps, 1)
        self.seed = seed

    # ------------------------------------------------------------------
    # Public pricing method
    # ------------------------------------------------------------------

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> ExoticResult:
        """
        Price the lookback option via Monte Carlo.

        Parameters
        ----------
        S     : spot price
        K     : strike (used only for fixed-strike subtypes)
        T     : time to expiry (years)
        r     : risk-free rate
        sigma : volatility
        q     : continuous dividend yield

        Returns
        -------
        ExoticResult
        """
        # Determine option_type string for ExoticResult
        option_type = "call" if "call" in self.subtype else "put"

        if T <= 0:
            # At expiry, running max/min equal S
            if self.subtype == "floating_call":
                intrinsic = 0.0  # S_T - min = S - S = 0
            elif self.subtype == "floating_put":
                intrinsic = 0.0
            elif self.subtype == "fixed_call":
                intrinsic = max(S - K, 0.0)
            else:  # fixed_put
                intrinsic = max(K - S, 0.0)
            vanilla = _bs_price(S, K, T, r, sigma, option_type, q)
            return ExoticResult(
                option_type=option_type,
                exotic_type="lookback",
                exotic_subtype=self.subtype,
                price=intrinsic,
                std_error=0.0,
                vanilla_price=vanilla,
                exotic_premium=intrinsic - vanilla,
                greeks={"delta": 0.0, "gamma": 0.0, "vega": 0.0,
                        "theta": 0.0, "vega_per_pct": 0.0},
                n_paths=self.n_paths,
                convergence=np.array([intrinsic]),
            )

        disc_payoffs = self._mc_payoffs(S, K, T, r, sigma, q, self.seed)
        price_val = float(np.mean(disc_payoffs))
        std_err = float(np.std(disc_payoffs, ddof=1) / math.sqrt(self.n_paths))
        vanilla = _bs_price(S, K, T, r, sigma, option_type, q)
        convergence = _convergence_array(disc_payoffs)

        greeks = compute_exotic_greeks(self, S, K, T, r, sigma, option_type, q)

        return ExoticResult(
            option_type=option_type,
            exotic_type="lookback",
            exotic_subtype=self.subtype,
            price=price_val,
            std_error=std_err,
            vanilla_price=vanilla,
            exotic_premium=price_val - vanilla,
            greeks=greeks,
            n_paths=self.n_paths,
            convergence=convergence,
        )

    # ------------------------------------------------------------------
    # Internal MC helper
    # ------------------------------------------------------------------

    def _mc_payoffs(
        self,
        S: float, K: float, T: float, r: float,
        sigma: float, q: float,
        seed: int,
    ) -> np.ndarray:
        """Simulate paths and return discounted lookback payoffs."""
        rng = np.random.default_rng(seed)
        paths = _simulate_gbm(S, T, r, sigma, q, self.n_paths, self.n_steps, rng)
        # paths shape: (n_steps+1, n_paths); include t=0 (S itself) in max/min
        terminal = paths[-1]               # (n_paths,)
        path_max = np.max(paths, axis=0)   # (n_paths,)
        path_min = np.min(paths, axis=0)   # (n_paths,)

        if self.subtype == "floating_call":
            payoff = terminal - path_min        # always >= 0
        elif self.subtype == "floating_put":
            payoff = path_max - terminal        # always >= 0
        elif self.subtype == "fixed_call":
            payoff = np.maximum(path_max - K, 0.0)
        else:  # fixed_put
            payoff = np.maximum(K - path_min, 0.0)

        disc = math.exp(-r * T)
        return disc * payoff

    # ------------------------------------------------------------------
    # Analytical floating-strike lookback (Goldman-Sosin-Gatto, 1979)
    # ------------------------------------------------------------------

    @staticmethod
    def analytical_floating(
        S: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        q: float = 0.0,
    ) -> float:
        """
        Goldman-Sosin-Gatto (1979) closed-form for a floating-strike lookback
        option, assuming the option was just initiated so S_min = S_max = S.

        Floating call (S_min = S at inception):
            a1 = (r - q + 0.5*sigma^2) * sqrt(T) / sigma
            C  = S*exp(-q*T)*N(a1)
                 - S*exp(-r*T)*(1 + sigma^2/(2*(r-q)))*N(a1 - sigma*sqrt(T))
                 + S*sigma^2/(2*(r-q))*exp(-r*T)

        Floating put (S_max = S at inception):
            P  = -S*exp(-q*T)*N(-a1)
                 + S*exp(-r*T)*(1 + sigma^2/(2*(r-q)))*N(-a1 + sigma*sqrt(T))
                 - S*sigma^2/(2*(r-q))*exp(-r*T)

        For r - q ≈ 0 (< 1e-6) a limiting formula is used to avoid division
        by zero:
            C_limit = S*exp(-q*T)*N(a1_lim) - S*exp(-r*T)*N(a1_lim - sigma*sqrt(T))
                      + S*sigma*sqrt(T)*[phi(a1_lim) + a1_lim*(N(a1_lim)-1)]  [approx]

        where a1_lim = 0.5*sigma*sqrt(T).

        Parameters
        ----------
        S           : spot (= S_min for call, = S_max for put at inception)
        T           : time to expiry (years)
        r           : risk-free rate
        sigma       : volatility
        option_type : "call" or "put"
        q           : dividend yield

        Returns
        -------
        float
        """
        if T <= 0:
            return 0.0  # floating: at expiry running min/max = S_T

        sqrt_T = math.sqrt(T)

        rq = r - q
        abs_rq = abs(rq)

        if abs_rq < 1e-6:
            # Limiting case: r ≈ q
            # Use first-order Taylor around r-q = 0
            a1 = 0.5 * sigma * sqrt_T
            nd1 = norm.cdf(a1)
            nd1_m = norm.cdf(a1 - sigma * sqrt_T)
            phi_a1 = norm.pdf(a1)
            exp_rT = math.exp(-r * T)
            exp_qT = math.exp(-q * T)
            if option_type == "call":
                # C ≈ S*[exp(-q*T)*N(a1) - exp(-r*T)*N(a1-sig*sqT)]
                #     + S*sigma*sqT * phi(a1)
                price = (S * exp_qT * nd1
                         - S * exp_rT * nd1_m
                         + S * sigma * sqrt_T * phi_a1)
            else:
                nd1_neg = norm.cdf(-a1)
                nd1_m_neg = norm.cdf(-(a1 - sigma * sqrt_T))
                phi_a1_neg = norm.pdf(-a1)
                price = (S * exp_rT * nd1_m_neg
                         - S * exp_qT * nd1_neg
                         + S * sigma * sqrt_T * phi_a1_neg)
            return max(price, 0.0)

        # Standard formula
        a1 = (rq + 0.5 * sigma ** 2) * sqrt_T / sigma
        a2 = a1 - sigma * sqrt_T
        coeff = sigma ** 2 / (2.0 * rq)
        exp_rT = math.exp(-r * T)
        exp_qT = math.exp(-q * T)

        if option_type == "call":
            price = (S * exp_qT * norm.cdf(a1)
                     - S * exp_rT * (1.0 + coeff) * norm.cdf(a2)
                     + S * coeff * exp_rT)
        else:
            price = (-S * exp_qT * norm.cdf(-a1)
                     + S * exp_rT * (1.0 + coeff) * norm.cdf(-a2)
                     - S * coeff * exp_rT)

        return max(price, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# GREEK CALCULATION VIA BUMP-AND-REPRICE (COMMON RANDOM NUMBERS)
# ─────────────────────────────────────────────────────────────────────────────

def compute_exotic_greeks(
    pricer,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> dict:
    """
    Compute Delta, Gamma, Vega, and Theta for an exotic option pricer via
    central finite differences (bump-and-reprice).

    Common Random Numbers (CRN) are ensured by using the same seed in each
    bumped call, so that Monte Carlo noise largely cancels across bumps and
    Greek estimates are more stable.

    Formulas
    --------
    dS   = S * BUMP_SPOT_PCT   (1% of spot)
    dv   = BUMP_VOL_ABS         (0.1% absolute vol)
    dt   = BUMP_TIME_DAYS / 365 (1 calendar day)

    Delta      = (V(S+dS) - V(S-dS)) / (2*dS)
    Gamma      = (V(S+dS) - 2*V(S) + V(S-dS)) / dS^2
    Vega       = (V(sigma+dv) - V(sigma-dv)) / (2*dv)   [raw, per 1% = *0.01]
    Theta      = (V(T - dt) - V(T)) / dt                 [per calendar day]
    vega_per_pct = Vega * 0.01                            [per 1% vol change]

    Parameters
    ----------
    pricer      : any pricer with a .price() method returning ExoticResult
                  (BarrierOptionPricer, AsianOptionPricer, LookbackOptionPricer)
    S           : spot price
    K           : strike
    T           : time to expiry (years)
    r           : risk-free rate
    sigma       : volatility
    option_type : "call" or "put"  (ignored for lookback pricers that derive
                  option_type from subtype)
    q           : dividend yield

    Returns
    -------
    dict with keys: delta, gamma, vega, theta, vega_per_pct
    """
    dS = S * BUMP_SPOT_PCT
    dv = BUMP_VOL_ABS
    dt = BUMP_TIME_DAYS / 365.0

    def _call_pricer(s, sig, t_exp) -> float:
        """Thin wrapper that handles both lookback (no option_type) and others."""
        if isinstance(pricer, LookbackOptionPricer):
            return pricer.price(s, K, t_exp, r, sig, q).price
        return pricer.price(s, K, t_exp, r, sig, option_type, q).price

    # Base price (use cached or re-compute)
    v0 = _call_pricer(S, sigma, T)

    # Delta & Gamma (spot bumps)
    v_up = _call_pricer(S + dS, sigma, T)
    v_dn = _call_pricer(S - dS, sigma, T)
    delta = (v_up - v_dn) / (2.0 * dS)
    gamma = (v_up - 2.0 * v0 + v_dn) / (dS ** 2)

    # Vega (vol bumps)
    sig_up = max(sigma + dv, 1e-6)
    sig_dn = max(sigma - dv, 1e-6)
    v_vup = _call_pricer(S, sig_up, T)
    v_vdn = _call_pricer(S, sig_dn, T)
    vega_raw = (v_vup - v_vdn) / (2.0 * dv)
    vega_per_pct = vega_raw * 0.01

    # Theta (time bump — T shrinks by 1 day)
    t_bumped = max(T - dt, 1e-6)
    v_theta = _call_pricer(S, sigma, t_bumped)
    theta = (v_theta - v0) / dt  # negative for long positions

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega_raw),
        "theta": float(theta),
        "vega_per_pct": float(vega_per_pct),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Constants
    "TRADING_DAYS",
    "MIN_PATHS",
    "MAX_PATHS",
    "DEFAULT_PATHS",
    "DEFAULT_STEPS",
    "BUMP_SPOT_PCT",
    "BUMP_VOL_ABS",
    "BUMP_TIME_DAYS",
    # Dataclass
    "ExoticResult",
    # Pricers
    "BarrierOptionPricer",
    "AsianOptionPricer",
    "LookbackOptionPricer",
    # Helpers
    "compute_exotic_greeks",
]
