"""
Risk engine — filters trades, adjusts exposure, simulates drawdowns.

Uses risk_analytics.py for VaR/CVaR/stress tests.
Wraps the StrategySignal through risk gates before execution.
"""

import sys
import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from Tools.risk_analytics import RiskAnalytics

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2)


@dataclass
class RiskFilter:
    passed: bool
    reason: str
    adjusted_position_pct: float
    var_95: float
    cvar_95: float
    max_drawdown_estimate: float
    stress_worst_case: float


class RiskEngine:
    """
    Applies risk management rules before trade approval:
    1. VaR gate: reject if daily VaR > max_var_pct of position
    2. Drawdown simulation: estimate max drawdown from returns distribution
    3. Stress test: compute worst-case P&L across historical scenarios
    4. Position adjustment: scale down if risk exceeds limits
    """

    def __init__(self, config: dict):
        self._cfg = config.get("risk", {})
        self._max_var = self._cfg.get("max_var_pct", 0.05)
        self._max_dd = self._cfg.get("max_drawdown_pct", 0.20)
        self._min_rr = self._cfg.get("min_rr_ratio", 1.5)

    async def evaluate(
        self,
        returns: np.ndarray,
        price: float,
        direction: str,
        entry: float,
        stop: float,
        take_profit: float,
        position_pct: float,
        delta: float = 0.5,
        gamma: float = 0.0,
        vega: float = 0.0,
        theta: float = 0.0,
        option_value: float = 0.0,
    ) -> RiskFilter:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: self._evaluate_sync(
                returns, price, direction, entry, stop, take_profit,
                position_pct, delta, gamma, vega, theta, option_value
            )
        )

    def _evaluate_sync(
        self,
        returns: np.ndarray,
        price: float,
        direction: str,
        entry: float,
        stop: float,
        take_profit: float,
        position_pct: float,
        delta: float,
        gamma: float,
        vega: float,
        theta: float,
        option_value: float,
    ) -> RiskFilter:
        if len(returns) < 20:
            return RiskFilter(
                passed=True,
                reason="insufficient_history",
                adjusted_position_pct=position_pct,
                var_95=0.0,
                cvar_95=0.0,
                max_drawdown_estimate=0.0,
                stress_worst_case=0.0,
            )

        # Convert numpy array to pandas Series for RiskAnalytics (expects Series)
        returns_series = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        analytics = RiskAnalytics(returns_series, price)

        # VaR / CVaR
        try:
            var_result = analytics.compute_var(confidence=0.95, horizon_days=1)
            var_95 = float(var_result.historical_var)
            cvar_95 = float(var_result.historical_cvar)
        except Exception as e:
            logger.debug("VaR computation failed: %s", e)
            var_95 = float(np.percentile(returns, 5)) * (-1)
            cvar_95 = float(np.mean(returns[returns < np.percentile(returns, 5)])) * (-1)

        # Max drawdown from rolling returns
        try:
            dd_stats = analytics.max_drawdown()
            max_dd = float(dd_stats.get("max_drawdown", 0.0))
        except Exception:
            cumret = np.cumprod(1 + returns)
            rolling_max = np.maximum.accumulate(cumret)
            drawdown = (cumret - rolling_max) / rolling_max
            max_dd = float(abs(drawdown.min()))

        # Stress test worst case
        stress_worst = 0.0
        if option_value > 0:
            try:
                stress_results = analytics.stress_test(
                    option_value, delta, gamma, vega, theta, 0.0, price
                )
                stress_worst = float(min(s.pnl for s in stress_results))
            except Exception:
                stress_worst = -price * 0.20  # assume 20% drawdown in worst case
        else:
            stress_worst = -price * max_dd

        # R:R check
        risk_per_unit = abs(entry - stop)
        reward_per_unit = abs(take_profit - entry)
        rr = reward_per_unit / (risk_per_unit + 1e-6)

        # Apply risk gates
        adjusted_pct = position_pct
        reasons = []

        if rr < self._min_rr:
            reasons.append(f"RR {rr:.2f} < min {self._min_rr}")
            if rr < 1.0:
                return RiskFilter(
                    passed=False,
                    reason=f"RR too low: {rr:.2f}",
                    adjusted_position_pct=0.0,
                    var_95=var_95,
                    cvar_95=cvar_95,
                    max_drawdown_estimate=max_dd,
                    stress_worst_case=stress_worst,
                )
            adjusted_pct *= (rr / self._min_rr)

        if var_95 > self._max_var:
            scale = self._max_var / (var_95 + 1e-6)
            adjusted_pct *= scale
            reasons.append(f"VaR {var_95:.1%} > limit {self._max_var:.1%}, scaled by {scale:.2f}")

        if max_dd > self._max_dd:
            adjusted_pct *= 0.5
            reasons.append(f"Max DD {max_dd:.1%} > limit {self._max_dd:.1%}, halved")

        if direction == "FLAT":
            adjusted_pct = 0.0

        reason_str = "; ".join(reasons) if reasons else "all_checks_passed"

        return RiskFilter(
            passed=True,
            reason=reason_str,
            adjusted_position_pct=round(max(0.0, min(adjusted_pct, 10.0)), 2),
            var_95=round(var_95, 4),
            cvar_95=round(cvar_95, 4),
            max_drawdown_estimate=round(max_dd, 4),
            stress_worst_case=round(stress_worst, 2),
        )

    def compute_sharpe(self, returns: np.ndarray, rf_daily: float = 0.0) -> float:
        if len(returns) < 10:
            return 0.0
        excess = returns - rf_daily
        return float(np.mean(excess) / (np.std(excess) + 1e-10) * np.sqrt(252))

    def compute_win_rate(self, returns: np.ndarray) -> float:
        return float(np.mean(returns > 0)) if len(returns) > 0 else 0.5

    def compute_profit_factor(self, returns: np.ndarray) -> float:
        wins = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return float(wins / (losses + 1e-10))
