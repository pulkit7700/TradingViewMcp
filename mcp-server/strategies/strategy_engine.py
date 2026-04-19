"""
Strategy engine — converts pipeline outputs into actionable strategy signals.

Takes multi-factor fusion results and applies:
1. Regime gating (block trades against macro trend)
2. Volatility adjustment (widen stops in high-vol)
3. Options-informed sizing (gamma exposure affects position size)
4. Probabilistic confidence scoring
"""

import sys
import os
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logger = logging.getLogger(__name__)


@dataclass
class StrategySignal:
    ticker: str
    direction: str          # "LONG" | "SHORT" | "FLAT"
    confidence: float       # 0–1
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float  # % of portfolio
    risk_reward: float
    regime: str
    volatility_state: str
    component_signals: dict = field(default_factory=dict)
    rationale: str = ""
    pine_params: dict = field(default_factory=dict)  # params for Pine Script generation

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "entry_price": round(self.entry_price, 4),
            "stop_loss": round(self.stop_loss, 4),
            "take_profit": round(self.take_profit, 4),
            "position_size_pct": round(self.position_size_pct, 2),
            "risk_reward": round(self.risk_reward, 4),
            "regime": self.regime,
            "volatility_state": self.volatility_state,
            "component_signals": self.component_signals,
            "rationale": self.rationale,
            "pine_params": self.pine_params,
        }


class StrategyEngine:
    """
    Converts raw pipeline outputs into a validated StrategySignal.

    Key logic:
    - Regime gate: block SHORT in Bull regime (and vice versa) unless confidence > 0.8
    - Vol adjustment: widen stops proportional to vol percentile
    - GEX filter: positive gamma exposure → dampen position size
    - Min confidence gate: below threshold → FLAT
    """

    def __init__(self, config: dict):
        self._cfg = config
        self._risk = config.get("risk", {})
        self._strategy_cfg = config.get("strategy", {})
        self._min_confidence = self._strategy_cfg.get("min_confidence", 0.40)
        self._min_strength = self._strategy_cfg.get("min_signal_strength", 0.35)

    def build_signal(self, ticker: str, pipeline_output: dict) -> StrategySignal:
        """
        Construct StrategySignal from pipeline output dict.
        Works with any pipeline (volatility, options_flow, transformer, swarm, multi_factor).
        """
        results = pipeline_output.get("results", {})
        signal = results.get("signal", {})
        meta = results.get("meta_signal", signal)  # prefer meta_signal if present
        sizing = results.get("position_sizing", {})
        regime_data = results.get("regime", {})
        garch_data = results.get("garch", {})

        direction = meta.get("direction", "FLAT")
        confidence = float(meta.get("confidence", 0.0))

        # Gate 1: minimum confidence
        if confidence < self._min_confidence:
            direction = "FLAT"

        # Gate 2: regime gating
        current_regime = regime_data.get("current_regime", "Neutral")
        direction = self._apply_regime_gate(direction, current_regime, confidence)

        # Price levels
        entry = float(sizing.get("entry_price", 0.0)) or self._fallback_price(results)
        stop = float(sizing.get("stop_loss", entry * 0.97))
        tp = float(sizing.get("take_profit", entry * 1.05))
        pos_pct = float(sizing.get("position_size_pct", 5.0))
        rr = float(sizing.get("risk_reward", 1.5))
        atr = float(sizing.get("atr", 0.0))

        # Gate 3: vol adjustment
        vol_state, vol_mult = self._classify_volatility(garch_data)
        if vol_state == "HIGH" and direction != "FLAT":
            # Widen stops in high vol
            stop_dist = abs(entry - stop) * vol_mult
            stop = (entry - stop_dist) if direction == "LONG" else (entry + stop_dist)
            tp_dist = abs(tp - entry) * vol_mult
            tp = (entry + tp_dist) if direction == "LONG" else (entry - tp_dist)
            # Reduce position size in high vol
            pos_pct *= (1.0 / vol_mult)

        # Gate 4: GEX filter
        flow_data = results.get("options_flow", {})
        net_gex = float(flow_data.get("net_gex", 0.0))
        if net_gex > 0:
            pos_pct *= 0.8  # positive GEX → dampened moves → smaller position

        # Cap position size
        max_pos = self._risk.get("max_position_pct", 0.10) * 100
        pos_pct = min(pos_pct, max_pos)

        if direction == "FLAT":
            stop = entry
            tp = entry
            pos_pct = 0.0

        rationale = self._build_rationale(meta, current_regime, vol_state)

        pine_params = self._derive_pine_params(
            direction=direction,
            confidence=confidence,
            regime=current_regime,
            vol_state=vol_state,
            atr=atr,
            component_scores=meta.get("component_scores", {}),
        )

        return StrategySignal(
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            entry_price=round(entry, 4),
            stop_loss=round(stop, 4),
            take_profit=round(tp, 4),
            position_size_pct=round(pos_pct, 2),
            risk_reward=round(rr, 4),
            regime=current_regime,
            volatility_state=vol_state,
            component_signals=meta.get("component_scores", {}),
            rationale=rationale,
            pine_params=pine_params,
        )

    # ─── Private Methods ──────────────────────────────────────────────────────

    def _apply_regime_gate(self, direction: str, regime: str, confidence: float) -> str:
        """Block trades that oppose the macro regime unless confidence is very high."""
        HIGH_CONF_OVERRIDE = 0.80

        if "Bull" in regime and direction == "SHORT" and confidence < HIGH_CONF_OVERRIDE:
            logger.debug("Regime gate: blocked SHORT in Bull regime (conf=%.2f)", confidence)
            return "FLAT"
        if "Bear" in regime and direction == "LONG" and confidence < HIGH_CONF_OVERRIDE:
            logger.debug("Regime gate: blocked LONG in Bear regime (conf=%.2f)", confidence)
            return "FLAT"
        if "Crash" in regime and direction == "LONG":
            logger.debug("Regime gate: blocked LONG in Crash regime")
            return "FLAT"
        return direction

    def _classify_volatility(self, garch_data: dict) -> tuple[str, float]:
        """Classify volatility state and return adjustment multiplier."""
        current_vol = garch_data.get("current_vol") or garch_data.get("current_conditional_vol")
        if current_vol is None:
            return "UNKNOWN", 1.0

        current_vol = float(current_vol)
        # Annualize if not already (daily vol threshold ~1.5%)
        if current_vol < 0.05:
            current_vol_ann = current_vol * np.sqrt(252)
        else:
            current_vol_ann = current_vol

        if current_vol_ann > 0.40:
            return "EXTREME", 1.5
        elif current_vol_ann > 0.25:
            return "HIGH", 1.25
        elif current_vol_ann > 0.15:
            return "NORMAL", 1.0
        else:
            return "LOW", 0.9

    def _fallback_price(self, results: dict) -> float:
        """Try to find price from any result dict."""
        for key in ["trade_plan", "swarm_aggregation"]:
            val = results.get(key, {})
            if isinstance(val, dict) and "entry" in val:
                return float(val["entry"])
        return 100.0

    def _build_rationale(self, meta: dict, regime: str, vol_state: str) -> str:
        parts = []
        direction = meta.get("direction", "FLAT")
        confidence = meta.get("confidence", 0.0)
        agreement = meta.get("agreement", 0.0)
        n_bull = meta.get("n_bullish_signals", 0)
        n_bear = meta.get("n_bearish_signals", 0)

        parts.append(f"Direction: {direction} | Confidence: {confidence:.1%} | Agreement: {agreement:.1%}")
        parts.append(f"Regime: {regime} | Volatility: {vol_state}")
        if n_bull or n_bear:
            parts.append(f"Signal vote: {n_bull} bullish vs {n_bear} bearish")

        comp = meta.get("component_scores", {})
        if comp:
            dominant = max(comp, key=lambda k: abs(comp[k]))
            parts.append(f"Dominant factor: {dominant} ({comp[dominant]:+.3f})")

        return " | ".join(parts)

    def _derive_pine_params(
        self,
        direction: str,
        confidence: float,
        regime: str,
        vol_state: str,
        atr: float,
        component_scores: dict,
    ) -> dict:
        """
        Derive Pine Script parameters from strategy context.
        These drive the Pine generator to produce regime/volatility-appropriate code.
        """
        # Adjust indicator parameters based on regime
        if "Bear" in regime or vol_state in ("HIGH", "EXTREME"):
            ema_fast, ema_slow = 10, 30
            rsi_overbought, rsi_oversold = 60, 40
        elif "Bull" in regime:
            ema_fast, ema_slow = 20, 50
            rsi_overbought, rsi_oversold = 70, 30
        else:
            ema_fast, ema_slow = 20, 50
            rsi_overbought, rsi_oversold = 70, 30

        # Volatility → ATR multiplier for stops
        vol_map = {"LOW": 1.5, "NORMAL": 2.0, "HIGH": 2.5, "EXTREME": 3.0, "UNKNOWN": 2.0}
        atr_mult = vol_map.get(vol_state, 2.0)

        # Score threshold based on confidence
        long_threshold = max(3, round(confidence * 5))
        short_threshold = max(3, round(confidence * 5))

        # Dominant component → which proxy gets extra weight in Pine
        dominant_proxy = "momentum"
        if component_scores:
            best_k = max(component_scores, key=lambda k: abs(component_scores[k]))
            proxy_map = {
                "volatility": "volatility",
                "options_flow": "volume",
                "transformer": "momentum",
                "swarm": "volume",
                "sentiment": "momentum",
            }
            dominant_proxy = proxy_map.get(best_k, "momentum")

        return {
            "direction": direction,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "rsi_len": 14,
            "rsi_overbought": rsi_overbought,
            "rsi_oversold": rsi_oversold,
            "atr_len": 14,
            "atr_mult": atr_mult,
            "vol_threshold": 1.5 if vol_state == "HIGH" else 1.3,
            "long_score_threshold": long_threshold,
            "short_score_threshold": short_threshold,
            "dominant_proxy": dominant_proxy,
            "regime": regime,
            "vol_state": vol_state,
            "confidence": round(confidence, 4),
        }
