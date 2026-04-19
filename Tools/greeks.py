"""
Options Greeks Calculator
--------------------------
Analytical Black-Scholes Greeks + profile functions for visualisation.
Includes: Delta, Gamma, Theta, Vega, Rho + second-order (Charm, Vanna, Volga).
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass


@dataclass
class GreeksResult:
    delta: float
    gamma: float
    theta: float   # per calendar day
    vega: float    # per 1% move in vol
    rho: float     # per 1% move in rate
    charm: float   # dDelta/dT  (per day)
    vanna: float   # dDelta/dSigma
    volga: float   # dVega/dSigma
    lambda_: float  # leverage / elasticity


class GreeksCalculator:
    """All first- and second-order Greeks from Black-Scholes closed form."""

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _d1d2(S, K, T, r, sigma, q=0.0):
        if T <= 0 or sigma <= 0:
            return np.nan, np.nan
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    # ── first-order ──────────────────────────────────────────────────────────

    @classmethod
    def delta(cls, S, K, T, r, sigma, q=0.0, option_type="call") -> float:
        d1, _ = cls._d1d2(S, K, T, r, sigma, q)
        if np.isnan(d1):
            return 1.0 if (option_type == "call" and S > K) else 0.0
        if option_type == "call":
            return np.exp(-q * T) * norm.cdf(d1)
        return -np.exp(-q * T) * norm.cdf(-d1)

    @classmethod
    def gamma(cls, S, K, T, r, sigma, q=0.0) -> float:
        d1, _ = cls._d1d2(S, K, T, r, sigma, q)
        if np.isnan(d1) or S <= 0 or sigma <= 0 or T <= 0:
            return 0.0
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @classmethod
    def theta(cls, S, K, T, r, sigma, q=0.0, option_type="call") -> float:
        """Theta per calendar day (negative = time decay)."""
        if T <= 0:
            return 0.0
        d1, d2 = cls._d1d2(S, K, T, r, sigma, q)
        if np.isnan(d1):
            return 0.0
        common = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option_type == "call":
            val = (common
                   - r * K * np.exp(-r * T) * norm.cdf(d2)
                   + q * S * np.exp(-q * T) * norm.cdf(d1))
        else:
            val = (common
                   + r * K * np.exp(-r * T) * norm.cdf(-d2)
                   - q * S * np.exp(-q * T) * norm.cdf(-d1))
        return val / 365.0   # convert to per calendar day

    @classmethod
    def vega(cls, S, K, T, r, sigma, q=0.0) -> float:
        """Vega per 1% move in implied vol."""
        if T <= 0:
            return 0.0
        d1, _ = cls._d1d2(S, K, T, r, sigma, q)
        if np.isnan(d1):
            return 0.0
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) * 0.01

    @classmethod
    def rho(cls, S, K, T, r, sigma, q=0.0, option_type="call") -> float:
        """Rho per 1% move in risk-free rate."""
        if T <= 0:
            return 0.0
        _, d2 = cls._d1d2(S, K, T, r, sigma, q)
        if np.isnan(d2):
            return 0.0
        if option_type == "call":
            return K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01
        return -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01

    # ── second-order ─────────────────────────────────────────────────────────

    @classmethod
    def charm(cls, S, K, T, r, sigma, q=0.0, option_type="call") -> float:
        """Charm = dDelta/dTime. Per calendar day."""
        if T <= 0 or sigma <= 0:
            return 0.0
        d1, d2 = cls._d1d2(S, K, T, r, sigma, q)
        if np.isnan(d1):
            return 0.0
        common = (
            np.exp(-q * T) * norm.pdf(d1)
            * (2 * (r - q) * T - d2 * sigma * np.sqrt(T))
            / (2 * T * sigma * np.sqrt(T))
        )
        if option_type == "call":
            val = q * np.exp(-q * T) * norm.cdf(d1) - common
        else:
            val = -q * np.exp(-q * T) * norm.cdf(-d1) + common
        return val / 365.0

    @classmethod
    def vanna(cls, S, K, T, r, sigma, q=0.0) -> float:
        """Vanna = dDelta/dSigma = dVega/dS. Per 1% vol move."""
        if T <= 0 or sigma <= 0:
            return 0.0
        d1, d2 = cls._d1d2(S, K, T, r, sigma, q)
        if np.isnan(d1):
            return 0.0
        return (np.exp(-q * T) * norm.pdf(d1) * (-d2) / sigma) * 0.01

    @classmethod
    def volga(cls, S, K, T, r, sigma, q=0.0) -> float:
        """Volga = dVega/dSigma (vol convexity). Per 1% vol move."""
        if T <= 0 or sigma <= 0:
            return 0.0
        d1, d2 = cls._d1d2(S, K, T, r, sigma, q)
        if np.isnan(d1):
            return 0.0
        return cls.vega(S, K, T, r, sigma, q) * d1 * d2 / sigma

    @classmethod
    def lambda_(cls, S, K, T, r, sigma, q=0.0, option_type="call", price=None) -> float:
        """Leverage / elasticity = Delta * S / Option_Price."""
        dlt = cls.delta(S, K, T, r, sigma, q, option_type)
        if price is None or price <= 0:
            return np.nan
        return dlt * S / price

    # ── full bundle ───────────────────────────────────────────────────────────

    @classmethod
    def all_greeks(
        cls, S: float, K: float, T: float, r: float, sigma: float,
        q: float = 0.0, option_type: str = "call", call_price: float = None
    ) -> GreeksResult:
        from .pricing import BlackScholesModel
        if call_price is None:
            call_price = BlackScholesModel.call_price(S, K, T, r, sigma, q)
        put_price = BlackScholesModel.put_price(S, K, T, r, sigma, q)
        opt_price = call_price if option_type == "call" else put_price

        return GreeksResult(
            delta=cls.delta(S, K, T, r, sigma, q, option_type),
            gamma=cls.gamma(S, K, T, r, sigma, q),
            theta=cls.theta(S, K, T, r, sigma, q, option_type),
            vega=cls.vega(S, K, T, r, sigma, q),
            rho=cls.rho(S, K, T, r, sigma, q, option_type),
            charm=cls.charm(S, K, T, r, sigma, q, option_type),
            vanna=cls.vanna(S, K, T, r, sigma, q),
            volga=cls.volga(S, K, T, r, sigma, q),
            lambda_=cls.lambda_(S, K, T, r, sigma, q, option_type, opt_price),
        )

    # ── profile generators (vectorised over a parameter) ──────────────────────

    @classmethod
    def spot_profiles(
        cls, K: float, T: float, r: float, sigma: float, q: float = 0.0,
        spot_range: tuple = (0.70, 1.30), n: int = 100
    ) -> dict:
        """Returns dict of arrays: spot, and each greek for call + put."""
        S_arr = np.linspace(K * spot_range[0], K * spot_range[1], n)
        result = {"spot": S_arr}
        for otype in ("call", "put"):
            prefix = otype
            result[f"{prefix}_delta"] = np.array([cls.delta(s, K, T, r, sigma, q, otype) for s in S_arr])
            result[f"{prefix}_gamma"] = np.array([cls.gamma(s, K, T, r, sigma, q) for s in S_arr])
            result[f"{prefix}_theta"] = np.array([cls.theta(s, K, T, r, sigma, q, otype) for s in S_arr])
            result[f"{prefix}_vega"] = np.array([cls.vega(s, K, T, r, sigma, q) for s in S_arr])
            result[f"{prefix}_rho"] = np.array([cls.rho(s, K, T, r, sigma, q, otype) for s in S_arr])
        return result

    @classmethod
    def vol_profiles(
        cls, S: float, K: float, T: float, r: float, q: float = 0.0,
        vol_range: tuple = (0.05, 0.80), n: int = 100
    ) -> dict:
        """Greeks as function of implied volatility."""
        vol_arr = np.linspace(vol_range[0], vol_range[1], n)
        result = {"vol": vol_arr}
        for otype in ("call", "put"):
            result[f"{otype}_delta"] = np.array([cls.delta(S, K, T, r, v, q, otype) for v in vol_arr])
            result[f"{otype}_gamma"] = np.array([cls.gamma(S, K, T, r, v, q) for v in vol_arr])
            result[f"{otype}_vega"] = np.array([cls.vega(S, K, T, r, v, q) for v in vol_arr])
            result[f"{otype}_theta"] = np.array([cls.theta(S, K, T, r, v, q, otype) for v in vol_arr])
        return result

    @classmethod
    def time_profiles(
        cls, S: float, K: float, r: float, sigma: float, q: float = 0.0,
        dte_range: tuple = (1, 365), n: int = 100
    ) -> dict:
        """Greeks as function of time to expiry (in days)."""
        dte_arr = np.linspace(dte_range[0], dte_range[1], n)
        T_arr = dte_arr / 365.0
        result = {"dte": dte_arr}
        for otype in ("call", "put"):
            result[f"{otype}_delta"] = np.array([cls.delta(S, K, t, r, sigma, q, otype) for t in T_arr])
            result[f"{otype}_gamma"] = np.array([cls.gamma(S, K, t, r, sigma, q) for t in T_arr])
            result[f"{otype}_theta"] = np.array([cls.theta(S, K, t, r, sigma, q, otype) for t in T_arr])
            result[f"{otype}_vega"] = np.array([cls.vega(S, K, t, r, sigma, q) for t in T_arr])
        return result
