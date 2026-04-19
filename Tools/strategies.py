"""
Options Strategies
------------------
Define multi-leg strategies, compute P&L at expiry, break-even, max P&L.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from .pricing import BlackScholesModel
from .greeks import GreeksCalculator


# ─────────────────────────────────────────────────────────────────────────────
# LEG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Leg:
    option_type: str    # "call" or "put"
    direction: str      # "long" or "short"
    strike: float
    expiry_T: float     # years to expiry
    n_contracts: int = 1
    premium: float = 0.0   # per unit (populated by pricing)

    @property
    def sign(self) -> int:
        return 1 if self.direction == "long" else -1

    def payoff_at_expiry(self, S_T: np.ndarray) -> np.ndarray:
        if self.option_type == "call":
            intrinsic = np.maximum(S_T - self.strike, 0.0)
        else:
            intrinsic = np.maximum(self.strike - S_T, 0.0)
        return self.sign * self.n_contracts * (intrinsic - self.premium)


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptionsStrategy:
    name: str
    legs: List[Leg] = field(default_factory=list)

    def total_premium(self) -> float:
        """Net debit/credit paid today (positive = debit, negative = credit)."""
        return sum(leg.sign * leg.n_contracts * leg.premium for leg in self.legs)

    def pnl_at_expiry(self, S_T: np.ndarray) -> np.ndarray:
        return sum(leg.payoff_at_expiry(S_T) for leg in self.legs)

    def breakeven_points(self, S_min: float, S_max: float, n: int = 5000) -> List[float]:
        """Find zero-crossings of P&L curve."""
        S_arr = np.linspace(S_min, S_max, n)
        pnl = self.pnl_at_expiry(S_arr)
        crossings = []
        for i in range(len(pnl) - 1):
            if pnl[i] * pnl[i + 1] < 0:
                # linear interpolation
                bp = S_arr[i] - pnl[i] * (S_arr[i + 1] - S_arr[i]) / (pnl[i + 1] - pnl[i])
                crossings.append(round(bp, 2))
        return crossings

    def max_profit(self, S_min: float, S_max: float, n: int = 5000) -> float:
        S_arr = np.linspace(S_min, S_max, n)
        return float(np.max(self.pnl_at_expiry(S_arr)))

    def max_loss(self, S_min: float, S_max: float, n: int = 5000) -> float:
        S_arr = np.linspace(S_min, S_max, n)
        return float(np.min(self.pnl_at_expiry(S_arr)))

    def net_greeks(self, S: float, r: float, sigma: float, q: float = 0.0) -> dict:
        """Aggregate Greeks across all legs at current spot."""
        agg = dict(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
        for leg in self.legs:
            g = GreeksCalculator.all_greeks(S, leg.strike, leg.expiry_T, r, sigma, q, leg.option_type)
            multiplier = leg.sign * leg.n_contracts
            agg["delta"] += multiplier * g.delta
            agg["gamma"] += multiplier * g.gamma
            agg["theta"] += multiplier * g.theta
            agg["vega"] += multiplier * g.vega
            agg["rho"] += multiplier * g.rho
        return agg


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY LIBRARY
# ─────────────────────────────────────────────────────────────────────────────

class StrategyLibrary:
    """Factory that creates pre-defined strategies with BS-priced premiums."""

    def __init__(self, S: float, T: float, r: float, sigma: float, q: float = 0.0):
        self.S = S
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q

    def _price(self, K: float, option_type: str) -> float:
        if option_type == "call":
            return BlackScholesModel.call_price(self.S, K, self.T, self.r, self.sigma, self.q)
        return BlackScholesModel.put_price(self.S, K, self.T, self.r, self.sigma, self.q)

    def _leg(self, option_type, direction, K, n=1):
        prem = self._price(K, option_type)
        return Leg(option_type=option_type, direction=direction, strike=K,
                   expiry_T=self.T, n_contracts=n, premium=prem)

    # ── individual strategies ─────────────────────────────────────────────────

    def long_call(self, K=None) -> OptionsStrategy:
        K = K or self.S
        return OptionsStrategy("Long Call", [self._leg("call", "long", K)])

    def long_put(self, K=None) -> OptionsStrategy:
        K = K or self.S
        return OptionsStrategy("Long Put", [self._leg("put", "long", K)])

    def short_call(self, K=None) -> OptionsStrategy:
        K = K or self.S
        return OptionsStrategy("Short Call", [self._leg("call", "short", K)])

    def short_put(self, K=None) -> OptionsStrategy:
        K = K or self.S
        return OptionsStrategy("Short Put", [self._leg("put", "short", K)])

    def covered_call(self, K=None) -> OptionsStrategy:
        """Long stock + short call (stock represented as call with K=0 proxy)."""
        K = K or self.S * 1.05
        short_c = self._leg("call", "short", K)
        # Represent long stock as a synthetic call that always pays S_T
        # We model stock ownership by adding a flat payoff leg separately
        strat = OptionsStrategy("Covered Call", [short_c])
        # We'll handle the stock leg in pnl specially: stock_pnl = S_T - S0
        strat._stock_cost = self.S
        return strat

    def protective_put(self, K=None) -> OptionsStrategy:
        K = K or self.S * 0.95
        return OptionsStrategy("Protective Put", [self._leg("put", "long", K)])

    def bull_call_spread(self, K_low=None, K_high=None) -> OptionsStrategy:
        K_low = K_low or self.S * 0.97
        K_high = K_high or self.S * 1.03
        return OptionsStrategy("Bull Call Spread", [
            self._leg("call", "long", K_low),
            self._leg("call", "short", K_high),
        ])

    def bear_put_spread(self, K_high=None, K_low=None) -> OptionsStrategy:
        K_high = K_high or self.S * 1.03
        K_low = K_low or self.S * 0.97
        return OptionsStrategy("Bear Put Spread", [
            self._leg("put", "long", K_high),
            self._leg("put", "short", K_low),
        ])

    def straddle(self, K=None) -> OptionsStrategy:
        K = K or self.S
        return OptionsStrategy("Long Straddle", [
            self._leg("call", "long", K),
            self._leg("put", "long", K),
        ])

    def short_straddle(self, K=None) -> OptionsStrategy:
        K = K or self.S
        return OptionsStrategy("Short Straddle", [
            self._leg("call", "short", K),
            self._leg("put", "short", K),
        ])

    def strangle(self, K_put=None, K_call=None) -> OptionsStrategy:
        K_put = K_put or self.S * 0.95
        K_call = K_call or self.S * 1.05
        return OptionsStrategy("Long Strangle", [
            self._leg("call", "long", K_call),
            self._leg("put", "long", K_put),
        ])

    def short_strangle(self, K_put=None, K_call=None) -> OptionsStrategy:
        K_put = K_put or self.S * 0.95
        K_call = K_call or self.S * 1.05
        return OptionsStrategy("Short Strangle", [
            self._leg("call", "short", K_call),
            self._leg("put", "short", K_put),
        ])

    def iron_condor(self, K1=None, K2=None, K3=None, K4=None) -> OptionsStrategy:
        """Sell OTM put spread + sell OTM call spread."""
        K1 = K1 or self.S * 0.90
        K2 = K2 or self.S * 0.95
        K3 = K3 or self.S * 1.05
        K4 = K4 or self.S * 1.10
        return OptionsStrategy("Iron Condor", [
            self._leg("put", "long", K1),
            self._leg("put", "short", K2),
            self._leg("call", "short", K3),
            self._leg("call", "long", K4),
        ])

    def butterfly_spread(self, K_low=None, K_atm=None, K_high=None) -> OptionsStrategy:
        K_low = K_low or self.S * 0.95
        K_atm = K_atm or self.S
        K_high = K_high or self.S * 1.05
        return OptionsStrategy("Long Butterfly", [
            self._leg("call", "long", K_low),
            self._leg("call", "short", K_atm, n=2),
            self._leg("call", "long", K_high),
        ])

    def risk_reversal(self, K_put=None, K_call=None) -> OptionsStrategy:
        K_put = K_put or self.S * 0.95
        K_call = K_call or self.S * 1.05
        return OptionsStrategy("Risk Reversal", [
            self._leg("put", "short", K_put),
            self._leg("call", "long", K_call),
        ])

    # ── factory by name ───────────────────────────────────────────────────────

    STRATEGY_NAMES = [
        "Long Call", "Long Put", "Short Call", "Short Put",
        "Bull Call Spread", "Bear Put Spread",
        "Long Straddle", "Short Straddle",
        "Long Strangle", "Short Strangle",
        "Iron Condor", "Long Butterfly",
        "Protective Put", "Risk Reversal",
    ]

    def build(self, name: str, **kwargs) -> OptionsStrategy:
        mapping = {
            "Long Call": self.long_call,
            "Long Put": self.long_put,
            "Short Call": self.short_call,
            "Short Put": self.short_put,
            "Bull Call Spread": self.bull_call_spread,
            "Bear Put Spread": self.bear_put_spread,
            "Long Straddle": self.straddle,
            "Short Straddle": self.short_straddle,
            "Long Strangle": self.strangle,
            "Short Strangle": self.short_strangle,
            "Iron Condor": self.iron_condor,
            "Long Butterfly": self.butterfly_spread,
            "Protective Put": self.protective_put,
            "Risk Reversal": self.risk_reversal,
        }
        fn = mapping.get(name)
        if fn is None:
            raise ValueError(f"Unknown strategy: {name}")
        return fn(**kwargs)
