"""
Pine Script v5 generator — converts strategy signals into valid, tradeable Pine Script.

Key constraint: Pine cannot run LSTM, Transformers, Swarm, or Monte Carlo.
Solution: Convert ML signals into proxy indicator logic that approximates the same decisions.

Proxy map:
  Transformer (TFT/MC) -> EMA crossover + RSI alignment (trend + momentum)
  Swarm consensus -> Volume surge + price action (crowd behavior proxy)
  GARCH vol forecast -> ATR expansion/contraction signal
  HMM regime -> EMA structure (bull/bear via slope and spacing)
  Options flow -> Volume z-score + open interest proxy
  Sentiment -> Higher timeframe trend alignment
"""

import textwrap
from dataclasses import dataclass
from typing import Any


@dataclass
class PineOutput:
    code: str
    version: str = "5"
    strategy_name: str = ""
    params: dict = None

    def to_dict(self) -> dict:
        return {
            "pine_code": self.code,
            "version": self.version,
            "strategy_name": self.strategy_name,
            "parameters": self.params or {},
        }


class PineScriptGenerator:
    """
    Generates Pine Script v5 strategy code from a StrategySignal dict.

    The generated script is:
    - Regime-aware (EMA structure as HMM proxy)
    - Volatility-adjusted (ATR-based stops scaled by vol state)
    - Options-informed (volume surge as flow proxy)
    - Multi-factor scored (composite long_score / short_score gate)
    - Fully parameterized via input() calls for optimization
    """

    def generate(self, strategy: dict) -> PineOutput:
        """
        Generate Pine Script v5 from a strategy dict.

        strategy dict must contain at minimum:
          ticker, direction, pine_params (dict with ema_fast, ema_slow, etc.)

        Optional: entry_price, stop_loss, take_profit, position_size_pct
        """
        ticker = strategy.get("ticker", "UNKNOWN")
        direction = strategy.get("direction", "FLAT")
        pine_params = strategy.get("pine_params", {})
        confidence = strategy.get("confidence", 0.5)
        regime = strategy.get("regime", "Neutral")
        vol_state = strategy.get("volatility_state", "NORMAL")
        component_scores = strategy.get("component_signals", {})

        strategy_name = f"QuantEngine: {ticker}"

        # Extract Pine parameters with defaults
        p = self._extract_params(pine_params)

        # Build script sections
        header = self._build_header(strategy_name, p)
        inputs = self._build_inputs(p)
        indicators = self._build_indicators(p)
        regime_proxy = self._build_regime_proxy(p)
        vol_proxy = self._build_vol_proxy(p)
        transformer_proxy = self._build_transformer_proxy(p, component_scores)
        swarm_proxy = self._build_swarm_proxy(p, component_scores)
        options_proxy = self._build_options_proxy(p, component_scores)
        fusion = self._build_fusion(p, direction)
        entries = self._build_entries(p, direction, strategy)
        exits = self._build_exits(p, direction, strategy)
        plots = self._build_plots(p, regime)
        alerts = self._build_alerts()

        code = "\n".join([
            header, inputs, indicators, regime_proxy, vol_proxy,
            transformer_proxy, swarm_proxy, options_proxy,
            fusion, entries, exits, plots, alerts,
        ])

        return PineOutput(
            code=code,
            version="5",
            strategy_name=strategy_name,
            params=p,
        )

    # ─── Section Builders ────────────────────────────────────────────────────

    def _build_header(self, name: str, p: dict) -> str:
        max_bars = 500
        return textwrap.dedent(f"""\
            //@version=5
            strategy("{name}",
                     overlay=true,
                     default_qty_type=strategy.percent_of_equity,
                     default_qty_value={p['position_size_pct']},
                     max_bars_back={max_bars},
                     pyramiding=0,
                     commission_type=strategy.commission.percent,
                     commission_value=0.05)
        """)

    def _build_inputs(self, p: dict) -> str:
        return textwrap.dedent(f"""\
            // ═══════════════════════════════════════════════
            // INPUTS — Optimize these in Strategy Tester
            // ═══════════════════════════════════════════════
            rsi_len         = input.int({p['rsi_len']},       "RSI Length",         minval=2,  maxval=50,  group="Indicators")
            ema_fast        = input.int({p['ema_fast']},       "Fast EMA",           minval=5,  maxval=100, group="Indicators")
            ema_slow        = input.int({p['ema_slow']},       "Slow EMA",           minval=20, maxval=300, group="Indicators")
            atr_len         = input.int({p['atr_len']},       "ATR Length",         minval=5,  maxval=30,  group="Risk")
            atr_mult        = input.float({p['atr_mult']},    "ATR Stop Multiplier",step=0.1,  minval=0.5, maxval=5.0, group="Risk")
            tp_mult         = input.float({p['tp_mult']},     "ATR TP Multiplier",  step=0.1,  minval=1.0, maxval=10.0, group="Risk")
            vol_thresh      = input.float({p['vol_threshold']},"Volume Threshold",   step=0.1,  minval=1.0, maxval=4.0, group="Filters")
            rsi_ob          = input.int({p['rsi_overbought']}, "RSI Overbought",     minval=55, maxval=90,  group="Filters")
            rsi_os          = input.int({p['rsi_oversold']},  "RSI Oversold",       minval=10, maxval=45,  group="Filters")
            long_thresh     = input.int({p['long_score_threshold']}, "Long Score Threshold",  minval=1, maxval=7, group="Signal")
            short_thresh    = input.int({p['short_score_threshold']}, "Short Score Threshold", minval=1, maxval=7, group="Signal")
            enable_long     = input.bool({str(p['enable_long']).lower()},  "Enable Long",  group="Direction")
            enable_short    = input.bool({str(p['enable_short']).lower()}, "Enable Short", group="Direction")
        """)

    def _build_indicators(self, p: dict) -> str:
        return textwrap.dedent("""\
            // ═══════════════════════════════════════════════
            // CORE INDICATORS
            // ═══════════════════════════════════════════════
            rsi         = ta.rsi(close, rsi_len)
            ema_f       = ta.ema(close, ema_fast)
            ema_s       = ta.ema(close, ema_slow)
            ema_200     = ta.ema(close, 200)
            atr         = ta.atr(atr_len)
            vol_sma     = ta.sma(volume, 20)
            vol_ratio   = volume / (vol_sma + 1e-10)

            // MACD for momentum confirmation
            [macd_line, macd_sig, macd_hist] = ta.macd(close, 12, 26, 9)

            // Bollinger Band width (volatility proxy)
            [bb_mid, bb_upper, bb_lower] = ta.bb(close, 20, 2.0)
            bb_width    = (bb_upper - bb_lower) / (bb_mid + 1e-10)
            bb_width_ma = ta.sma(bb_width, 20)
        """)

    def _build_regime_proxy(self, p: dict) -> str:
        return textwrap.dedent("""\
            // ═══════════════════════════════════════════════
            // REGIME PROXY  (HMM -> EMA structure)
            // HMM "Bull" maps to: price > ema_slow AND ema_fast > ema_slow AND ema_slow rising
            // HMM "Bear" maps to: price < ema_slow AND ema_fast < ema_slow AND ema_slow falling
            // ═══════════════════════════════════════════════
            ema_s_slope = ema_s - ema_s[5]
            bull_regime = close > ema_s and ema_f > ema_s and ema_s_slope > 0
            bear_regime = close < ema_s and ema_f < ema_s and ema_s_slope < 0
            neutral_regime = not bull_regime and not bear_regime
        """)

    def _build_vol_proxy(self, p: dict) -> str:
        return textwrap.dedent("""\
            // ═══════════════════════════════════════════════
            // VOLATILITY PROXY  (GARCH -> ATR dynamics)
            // GARCH "high vol" -> ATR expanding above its own MA
            // GARCH "low vol"  -> ATR contracting below its own MA
            // ═══════════════════════════════════════════════
            atr_ma          = ta.sma(atr, 20)
            vol_expansion   = atr > atr_ma * 1.2
            vol_contraction = atr < atr_ma * 0.8
            vol_normal      = not vol_expansion and not vol_contraction

            // Hurst rough-vol proxy: high BB width -> rough/trending vol
            rough_vol_regime = bb_width > bb_width_ma * 1.15
        """)

    def _build_transformer_proxy(self, p: dict, scores: dict) -> str:
        trans_score = scores.get("transformer", 0.0)
        direction_bias = "> 55" if trans_score >= 0 else "< 45"
        macd_bias = "> 0" if trans_score >= 0 else "< 0"

        return textwrap.dedent(f"""\
            // ═══════════════════════════════════════════════
            // TRANSFORMER PROXY  (TFT + MC -> momentum alignment)
            // TFT h5 bullish forecast -> close > ema_fast AND rsi aligns
            // MC pct_bull > 0.55 -> majority paths above entry
            // Combined into: momentum structure in [rsi_os, rsi_ob] range
            // Transformer score: {trans_score:+.4f}
            // ═══════════════════════════════════════════════
            momentum_bull = close > ema_f and rsi {direction_bias} and rsi < rsi_ob and macd_hist {macd_bias}
            momentum_bear = close < ema_f and rsi {"< 45" if trans_score >= 0 else "> 55"} and rsi > rsi_os and macd_hist {"< 0" if trans_score >= 0 else "> 0"}
        """)

    def _build_swarm_proxy(self, p: dict, scores: dict) -> str:
        swarm_score = scores.get("swarm", 0.0)
        return textwrap.dedent(f"""\
            // ═══════════════════════════════════════════════
            // SWARM PROXY  (Bio-agent consensus -> crowd behavior)
            // Swarm BUY consensus -> volume surge + price above open + ema alignment
            // ACO pheromone trail -> volume persistence
            // Swarm score: {swarm_score:+.4f}
            // ═══════════════════════════════════════════════
            vol_surge       = vol_ratio > vol_thresh
            swarm_bull      = vol_surge and close > open and close > ema_f and ta.rising(volume, 2)
            swarm_bear      = vol_surge and close < open and close < ema_f and ta.falling(volume, 2) == false and ta.rising(volume, 2)
            swarm_neutral   = not swarm_bull and not swarm_bear
        """)

    def _build_options_proxy(self, p: dict, scores: dict) -> str:
        opt_score = scores.get("options_flow", 0.0)
        return textwrap.dedent(f"""\
            // ═══════════════════════════════════════════════
            // OPTIONS FLOW PROXY  (Flow + GEX -> volume/price breakouts)
            // Put/Call ratio bullish -> calls accumulating -> price breaks above resistance
            // Negative GEX -> market maker short gamma -> amplified moves
            // Flow score: {opt_score:+.4f}
            // ═══════════════════════════════════════════════
            options_breakout_bull = vol_surge and close > high[1] and close > bb_upper
            options_breakout_bear = vol_surge and close < low[1] and close < bb_lower
            // Gamma squeeze proxy: wide BB + volume surge
            gamma_squeeze = bb_width > bb_width_ma * 1.3 and vol_surge
        """)

    def _build_fusion(self, p: dict, direction: str) -> str:
        enable_long = str(p["enable_long"]).lower()
        enable_short = str(p["enable_short"]).lower()

        return textwrap.dedent(f"""\
            // ═══════════════════════════════════════════════
            // MULTI-FACTOR FUSION  (Composite scoring gate)
            // Each factor contributes 1-2 points to long/short score.
            // Trade only when score >= threshold (set by confidence level).
            // Direction bias: {direction}
            // ═══════════════════════════════════════════════
            long_score = (bull_regime     ? 2 : 0) +
                         (momentum_bull   ? 2 : 0) +
                         (swarm_bull      ? 1 : 0) +
                         (options_breakout_bull ? 1 : 0) +
                         (not vol_expansion    ? 1 : 0)

            short_score = (bear_regime    ? 2 : 0) +
                          (momentum_bear  ? 2 : 0) +
                          (swarm_bear     ? 1 : 0) +
                          (options_breakout_bear ? 1 : 0) +
                          (not vol_expansion     ? 1 : 0)

            longCondition  = enable_long  and long_score  >= long_thresh  and not vol_expansion
            shortCondition = enable_short and short_score >= short_thresh and not vol_expansion
        """)

    def _build_entries(self, p: dict, direction: str, strategy: dict) -> str:
        entry_comment = ""
        if direction == "LONG":
            entry_comment = "// Directional bias: LONG — reduce short threshold or disable"
        elif direction == "SHORT":
            entry_comment = "// Directional bias: SHORT — reduce long threshold or disable"
        else:
            entry_comment = "// Directional bias: FLAT — conservative sizing"

        return textwrap.dedent(f"""\
            // ═══════════════════════════════════════════════
            // ENTRIES
            // {entry_comment}
            // ═══════════════════════════════════════════════
            if longCondition
                strategy.entry("Long", strategy.long)

            if shortCondition
                strategy.entry("Short", strategy.short)
        """)

    def _build_exits(self, p: dict, direction: str, strategy: dict) -> str:
        atr_mult = p["atr_mult"]
        tp_mult = p["tp_mult"]
        return textwrap.dedent(f"""\
            // ═══════════════════════════════════════════════
            // EXITS — ATR-based dynamic stop + take profit
            // ATR stop multiplier: {atr_mult}x | TP multiplier: {tp_mult}x
            // ═══════════════════════════════════════════════
            long_entry_price  = strategy.opentrades.entry_price(0)
            short_entry_price = strategy.opentrades.entry_price(0)

            long_stop  = long_entry_price  - atr * atr_mult
            long_tp    = long_entry_price  + atr * tp_mult
            short_stop = short_entry_price + atr * atr_mult
            short_tp   = short_entry_price - atr * tp_mult

            strategy.exit("Long Exit",  from_entry="Long",  stop=long_stop,  limit=long_tp)
            strategy.exit("Short Exit", from_entry="Short", stop=short_stop, limit=short_tp)

            // Signal-based exit override (regime flip)
            if strategy.position_size > 0 and bear_regime
                strategy.close("Long", comment="regime_flip")
            if strategy.position_size < 0 and bull_regime
                strategy.close("Short", comment="regime_flip")
        """)

    def _build_plots(self, p: dict, regime: str) -> str:
        return textwrap.dedent("""\
            // ═══════════════════════════════════════════════
            // PLOTS
            // ═══════════════════════════════════════════════
            plot(ema_f,   "Fast EMA",    color=color.blue,   linewidth=1)
            plot(ema_s,   "Slow EMA",    color=color.orange, linewidth=2)
            plot(ema_200, "EMA 200",     color=color.gray,   linewidth=1)
            plot(bb_upper, "BB Upper",   color=color.new(color.purple, 70))
            plot(bb_lower, "BB Lower",   color=color.new(color.purple, 70))

            // Regime background
            bgcolor(bull_regime   ? color.new(color.green,  92) : na, title="Bull Regime")
            bgcolor(bear_regime   ? color.new(color.red,    92) : na, title="Bear Regime")
            bgcolor(vol_expansion ? color.new(color.yellow, 92) : na, title="High Volatility")

            // Entry/Exit markers
            plotshape(longCondition,  "Long Signal",  shape.triangleup,   location.belowbar, color.green, size=size.small)
            plotshape(shortCondition, "Short Signal", shape.triangledown, location.abovebar, color.red,   size=size.small)

            // RSI panel (separate pane)
            rsi_plot = plot(rsi, "RSI", display=display.pane, color=color.purple)
            hline(70, "Overbought", color=color.red,   linestyle=hline.style_dashed)
            hline(30, "Oversold",   color=color.green, linestyle=hline.style_dashed)
            hline(50, "Midline",    color=color.gray,  linestyle=hline.style_dotted)
        """)

    def _build_alerts(self) -> str:
        return textwrap.dedent("""\
            // ═══════════════════════════════════════════════
            // ALERTS
            // ═══════════════════════════════════════════════
            alertcondition(longCondition,  "QuantEngine Long",  "LONG signal: {{ticker}} @ {{close}}")
            alertcondition(shortCondition, "QuantEngine Short", "SHORT signal: {{ticker}} @ {{close}}")
            alertcondition(vol_expansion,  "High Volatility",   "Vol expansion: {{ticker}} ATR rising")
            alertcondition(gamma_squeeze,  "Gamma Squeeze",     "Gamma squeeze setup: {{ticker}}")
        """)

    # ─── Param Extraction ────────────────────────────────────────────────────

    def _extract_params(self, pine_params: dict) -> dict:
        direction = pine_params.get("direction", "FLAT")
        return {
            "rsi_len": int(pine_params.get("rsi_len", 14)),
            "ema_fast": int(pine_params.get("ema_fast", 20)),
            "ema_slow": int(pine_params.get("ema_slow", 50)),
            "atr_len": int(pine_params.get("atr_len", 14)),
            "atr_mult": float(pine_params.get("atr_mult", 2.0)),
            "tp_mult": float(pine_params.get("atr_mult", 2.0)) * 1.5,
            "vol_threshold": float(pine_params.get("vol_threshold", 1.5)),
            "rsi_overbought": int(pine_params.get("rsi_overbought", 70)),
            "rsi_oversold": int(pine_params.get("rsi_oversold", 30)),
            "long_score_threshold": int(pine_params.get("long_score_threshold", 4)),
            "short_score_threshold": int(pine_params.get("short_score_threshold", 4)),
            "position_size_pct": float(pine_params.get("confidence", 0.5)) * 10.0,
            "enable_long": direction in ("LONG", "FLAT"),
            "enable_short": direction in ("SHORT", "FLAT"),
            "dominant_proxy": pine_params.get("dominant_proxy", "momentum"),
            "regime": pine_params.get("regime", "Neutral"),
            "vol_state": pine_params.get("vol_state", "NORMAL"),
            "confidence": float(pine_params.get("confidence", 0.5)),
        }
