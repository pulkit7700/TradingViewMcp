"""
Prebuilt pipeline definitions — five production pipelines.

Each pipeline = ordered chain of async steps.
Steps share state via PipelineContext.
All heavy computation runs in ThreadPoolExecutor to avoid blocking the event loop.
"""

import sys
import os
import asyncio
import logging
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from Tools.data_handler import MarketDataHandler
from Tools.volatility_forecast import GARCHForecaster
from Tools.rough_vol import RoughBergomiModel, HurstEstimator
from Tools.ml_volatility import IVPredictor
from Tools.regime_detection import RegimeDetector
from Tools.options_flow import OptionsFlowAnalyzer, compute_gamma_exposure, identify_max_pain
from Tools.greeks import GreeksCalculator
from Tools.pricing import PricingEngine, BlackScholesModel
from Tools.tft_predictor import TFTPredictor
from Tools.transformer_mc import TransformerMCModel
from Tools.sentiment_engine import SentimentEngine
from Tools.swarm_base import SwarmState
from Tools.swarm_agents import (
    BoidsMomentumAgent, VicsekConsensusAgent, ACOPathAgent,
    PSOOptimizerAgent, LeaderFollowerAgent, TopologicalAgent,
)
from Tools.swarm_aggregator import SignalAggregator, ConfidenceScorer
from Tools.swarm_market_data import (
    fetch_price_data, compute_swarm_features, get_vix_level,
    compute_atr, compute_support_resistance, compute_trade_levels,
)

from .engine import Pipeline, PipelineContext, PipelineStep

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=8)
_data_handler = MarketDataHandler()

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _run_sync(fn, *args, **kwargs):
    """Wrap sync call for use in run_in_executor."""
    return fn(*args, **kwargs)


async def _async(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, lambda: fn(*args, **kwargs))


def _scalar(val):
    """Convert numpy scalar to Python float safely."""
    try:
        return float(val)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE 1 — VOLATILITY STRATEGY
# tv_data → GARCH → rough_vol → HMM regime → ML IV → signal → strategy
# ═══════════════════════════════════════════════════════════════════════════════

async def _vol_step_fetch(ctx: PipelineContext) -> PipelineContext:
    ticker = ctx.ticker
    period = ctx.config.get("data", {}).get("default_period", "2y")
    df = await _async(_data_handler.fetch_history, ticker, period, "1d")
    price = await _async(_data_handler.get_current_price, ticker)
    ctx.put("ohlcv", df)
    ctx.put("price", price)
    ctx.put("returns", df["Close"].pct_change().dropna().values)
    return ctx


async def _vol_step_garch(ctx: PipelineContext) -> PipelineContext:
    returns = ctx.get("returns")
    cfg = ctx.config.get("models", {}).get("garch", {})
    horizon = cfg.get("forecast_horizon", 30)
    confidence = cfg.get("confidence", 0.95)

    def _fit():
        forecaster = GARCHForecaster(returns)
        results = forecaster.fit_all()
        best = forecaster.select_best(results)
        forecaster.forecast(best, horizon=horizon, confidence=confidence)
        # Annualize conditional vol for compare_to_iv
        cond_vol_daily = _scalar(best.conditional_vol[-1]) if best.conditional_vol is not None else None
        garch_ann_vol = cond_vol_daily * float(np.sqrt(252)) if cond_vol_daily else None
        # Use 20-day realized vol as current_iv proxy (no options chain needed)
        hv20 = float(np.std(returns[-20:]) * np.sqrt(252)) if len(returns) >= 20 else None
        signal = (
            forecaster.compare_to_iv(garch_ann_vol, hv20)
            if garch_ann_vol and hv20
            else {"signal": "FAIR", "strength": "WEAK", "premium_pct": 0.0}
        )
        fv = best.forecast_vol if best.forecast_vol is not None else []
        fu = best.forecast_upper if best.forecast_upper is not None else []
        fl = best.forecast_lower if best.forecast_lower is not None else []
        return {
            "model_type": best.model_type,
            "aic": _scalar(best.aic),
            "forecast_vol": [_scalar(v) for v in fv],
            "forecast_upper": [_scalar(v) for v in fu],
            "forecast_lower": [_scalar(v) for v in fl],
            "iv_signal": signal,
            "current_conditional_vol": cond_vol_daily,
            "garch_annualized_vol": garch_ann_vol,
            "hv20": hv20,
        }

    garch_out = await _async(_fit)
    ctx.result("garch", garch_out)
    ctx.put("garch", garch_out)
    return ctx


async def _vol_step_rough_vol(ctx: PipelineContext) -> PipelineContext:
    ohlcv = ctx.get("ohlcv")
    price = ctx.get("price")
    cfg = ctx.config.get("models", {}).get("rough_vol", {})
    returns = ctx.get("returns")

    def _fit():
        estimator = HurstEstimator(returns)
        H_var = estimator.estimate_variogram()
        H_rs = estimator.estimate_rs()
        H = float(np.nanmean([H_var, H_rs]))
        interp = estimator.interpret(H)
        model = RoughBergomiModel(
            H=H,
            eta=cfg.get("eta", 1.9),
            rho=cfg.get("rho", -0.7),
            n_paths=cfg.get("n_paths", 500),
            n_steps=cfg.get("n_steps", 60),
        )
        # Use ATR as proxy for xi_0 (spot variance)
        atr_pct = _scalar(ohlcv["High"].values[-1] - ohlcv["Low"].values[-1]) / (price + 1e-6)
        xi_0 = max(atr_pct ** 2 * 252, 0.04)
        rf = _data_handler.get_risk_free_rate()
        result = model.price(price, price * 0.97, 30 / 252, rf, xi_0)
        return {
            "H": H,
            "H_interpretation": interp,
            "rough_vol_call": _scalar(result.call_price),
            "rough_vol_put": _scalar(result.put_price),
            "xi_0_annualized": float(np.sqrt(xi_0 * 252)),
        }

    rv_out = await _async(_fit)
    ctx.result("rough_vol", rv_out)
    ctx.put("rough_vol", rv_out)
    return ctx


async def _vol_step_regime(ctx: PipelineContext) -> PipelineContext:
    returns = ctx.get("returns")

    def _detect():
        # Convert numpy array to pandas Series for RegimeDetector (expects .dropna() method)
        returns_series = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        detector = RegimeDetector(returns_series, n_regimes=3, seed=42)
        res = detector.fit()
        # regime_stats is a list[RegimeStats], convert to dict keyed by regime_id
        regime_stats_dict = {}
        for stat in res.regime_stats:
            regime_stats_dict[stat.regime_id] = {
                "label": stat.label,
                "mean_daily_return": _scalar(stat.mean_daily_return),
                "std_daily_return": _scalar(stat.std_daily_return),
                "annualized_return": _scalar(stat.annualized_return),
                "annualized_vol": _scalar(stat.annualized_vol),
                "sharpe_ratio": _scalar(stat.sharpe_ratio),
                "avg_duration_days": _scalar(stat.avg_duration_days),
                "pct_time_in_regime": _scalar(stat.pct_time_in_regime),
            }
        return {
            "current_regime": res.current_regime,
            "regime_stats": regime_stats_dict,
            "aic": _scalar(res.aic),
            "bic": _scalar(res.bic),
        }

    regime_out = await _async(_detect)
    ctx.result("regime", regime_out)
    ctx.put("regime", regime_out)
    return ctx


async def _vol_step_ml_iv(ctx: PipelineContext) -> PipelineContext:
    ohlcv = ctx.get("ohlcv")
    price = ctx.get("price")

    def _predict():
        predictor = IVPredictor()
        predictor.train(ohlcv)
        pred = predictor.predict(
            S=price,
            K=price,
            dte=30,
            option_type="call",
            hist_prices=ohlcv,
            current_iv=None,
        )
        return {
            "predicted_iv": _scalar(pred.predicted_iv),
            "rf_iv": _scalar(pred.rf_iv),
            "xgb_iv": _scalar(pred.xgb_iv),
            "signal": pred.signal,
            "signal_strength": pred.signal_strength,
            "ci_low": _scalar(pred.confidence_interval[0]) if pred.confidence_interval else None,
            "ci_high": _scalar(pred.confidence_interval[1]) if pred.confidence_interval else None,
        }

    ml_out = await _async(_predict)
    ctx.result("ml_iv", ml_out)
    ctx.put("ml_iv", ml_out)
    return ctx


async def _vol_step_signal(ctx: PipelineContext) -> PipelineContext:
    """Fuse vol inputs into a directional signal."""
    garch = ctx.get("garch", {})
    regime = ctx.get("regime", {})
    ml_iv = ctx.get("ml_iv", {})
    rough = ctx.get("rough_vol", {})

    regime_id = regime.get("current_regime", 0)
    regime_stats = regime.get("regime_stats", {})
    regime_label = regime_stats.get(regime_id, {}).get("label", "Neutral") if isinstance(regime_stats, dict) else "Neutral"

    iv_signal = garch.get("iv_signal", {}).get("signal", "FAIR")
    ml_signal = ml_iv.get("signal", "FAIR")
    H = rough.get("H", 0.5)

    # Direction score: +1 = bullish (vol selling), -1 = bearish (vol buying)
    score = 0.0
    if "Bull" in regime_label:
        score += 1.0
    elif "Bear" in regime_label:
        score -= 1.0

    # High vol → mean reversion expected → slight bullish
    if iv_signal == "OVERPRICED":
        score += 0.5  # sell vol
    elif iv_signal == "UNDERPRICED":
        score -= 0.5  # buy vol / hedge

    # ML IV signal
    if ml_signal == "OVERPRICED":
        score += 0.3
    elif ml_signal == "UNDERPRICED":
        score -= 0.3

    # Rough vol: H < 0.5 (rough) → mean-reverting vol → trending market
    if H < 0.3:
        score += 0.2  # rough vol → momentum persists

    direction = "LONG" if score > 0.3 else "SHORT" if score < -0.3 else "FLAT"
    confidence = min(abs(score) / 2.0, 1.0)

    ctx.result("signal", {
        "direction": direction,
        "confidence": round(confidence, 3),
        "score": round(score, 3),
        "contributing_factors": {
            "regime": regime_label,
            "iv_signal": iv_signal,
            "ml_signal": ml_signal,
            "hurst_H": round(H, 4),
        },
    })
    return ctx


def build_volatility_pipeline() -> Pipeline:
    return Pipeline(
        name="volatility",
        steps=[
            PipelineStep("fetch_data", _vol_step_fetch),
            PipelineStep("garch_forecast", _vol_step_garch),
            PipelineStep("rough_vol", _vol_step_rough_vol, optional=True),
            PipelineStep("regime_detection", _vol_step_regime),
            PipelineStep("ml_iv_prediction", _vol_step_ml_iv, optional=True),
            PipelineStep("signal_fusion", _vol_step_signal),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE 2 — OPTIONS FLOW STRATEGY
# tv_data → options_chain → flow_analysis → greeks → pricing → gamma_exp → signal
# ═══════════════════════════════════════════════════════════════════════════════

async def _opt_step_fetch(ctx: PipelineContext) -> PipelineContext:
    ticker = ctx.ticker
    period = ctx.config.get("data", {}).get("default_period", "2y")
    df, price, chain = await asyncio.gather(
        _async(_data_handler.fetch_history, ticker, period, "1d"),
        _async(_data_handler.get_current_price, ticker),
        _async(_data_handler.get_options_chain, ticker),
    )
    ctx.put("ohlcv", df)
    ctx.put("price", price)
    ctx.put("options_chain", chain)
    ctx.put("returns", df["Close"].pct_change().dropna().values)
    return ctx


async def _opt_step_flow(ctx: PipelineContext) -> PipelineContext:
    chain = ctx.get("options_chain")
    price = ctx.get("price")

    # No options chain available (non-F&O stock) — return neutral defaults
    if chain is None or (hasattr(chain, 'empty') and chain.empty):
        neutral = {"pcr_volume": 1.0, "pcr_oi": 1.0, "smart_money_score": 0.0,
                   "net_flow_score": 0.0, "n_unusual": 0, "max_pain": price, "net_gex": 0.0}
        ctx.result("options_flow", neutral)
        ctx.put("flow", neutral)
        return ctx

    def _analyze():
        calls = chain[chain["option_type"] == "call"] if "option_type" in chain.columns else chain.iloc[:len(chain)//2]
        puts = chain[chain["option_type"] == "put"] if "option_type" in chain.columns else chain.iloc[len(chain)//2:]

        analyzer = OptionsFlowAnalyzer(calls, puts, price, ctx.ticker)
        summary = analyzer.analyze()
        flow_score = analyzer.net_flow_score()

        gex_df = compute_gamma_exposure(calls, puts, price)
        max_pain = identify_max_pain(calls, puts)

        return {
            "pcr_volume": _scalar(summary.pcr_volume),
            "pcr_oi": _scalar(summary.pcr_oi),
            "smart_money_score": _scalar(summary.smart_money_score),
            "net_flow_score": _scalar(flow_score),
            "n_unusual": len(summary.unusual_activity),
            "max_pain": _scalar(max_pain),
            "net_gex": float(gex_df["net_gex"].sum()) if not gex_df.empty and "net_gex" in gex_df.columns else 0.0,
        }

    flow_out = await _async(_analyze)
    ctx.result("options_flow", flow_out)
    ctx.put("flow", flow_out)
    return ctx


async def _opt_step_greeks(ctx: PipelineContext) -> PipelineContext:
    price = ctx.get("price")
    ohlcv = ctx.get("ohlcv")
    returns = ctx.get("returns")
    rf = await _async(_data_handler.get_risk_free_rate)

    def _calc():
        hv = float(np.std(returns[-20:]) * np.sqrt(252)) if len(returns) >= 20 else 0.20
        greeks = GreeksCalculator.all_greeks(
            S=price, K=price, T=30/252, r=rf, sigma=hv, option_type="call", q=0.0
        )
        return {
            "delta": _scalar(greeks.delta),
            "gamma": _scalar(greeks.gamma),
            "theta": _scalar(greeks.theta),
            "vega": _scalar(greeks.vega),
            "sigma_used": round(hv, 4),
        }

    greeks_out = await _async(_calc)
    ctx.result("greeks", greeks_out)
    ctx.put("greeks", greeks_out)
    return ctx


async def _opt_step_pricing(ctx: PipelineContext) -> PipelineContext:
    price = ctx.get("price")
    ohlcv = ctx.get("ohlcv")
    returns = ctx.get("returns")
    rf = await _async(_data_handler.get_risk_free_rate)

    def _price():
        hv = float(np.std(returns[-20:]) * np.sqrt(252)) if len(returns) >= 20 else 0.20
        engine = PricingEngine()
        all_prices = engine.price_all(
            S=price, K=price, T=30/252, r=rf, sigma=hv, q=0.0,
            american=False, include_heston=False, include_jump_diffusion=False,
        )
        return {
            model: {
                "price": _scalar(res.price),
                "delta": _scalar(res.delta) if res.delta is not None else None,
            }
            for model, res in all_prices.items()
        }

    pricing_out = await _async(_price)
    ctx.result("pricing", pricing_out)
    ctx.put("pricing", pricing_out)
    return ctx


async def _opt_step_signal(ctx: PipelineContext) -> PipelineContext:
    flow = ctx.get("flow", {})
    greeks = ctx.get("greeks", {})

    net_flow = flow.get("net_flow_score", 0.0)
    pcr = flow.get("pcr_volume", 1.0)
    smart_money = flow.get("smart_money_score", 0.0)
    net_gex = flow.get("net_gex", 0.0)

    # PCR < 0.7 → call-heavy → bullish; PCR > 1.3 → put-heavy → bearish
    pcr_signal = 1.0 if pcr < 0.7 else (-1.0 if pcr > 1.3 else 0.0)
    # net_gex > 0 → dealers long gamma → dampens moves (mean reversion)
    gex_signal = -0.3 if net_gex > 0 else 0.3  # negative GEX → dealers short → amplifies

    score = net_flow * 0.4 + pcr_signal * 0.3 + gex_signal * 0.2 + smart_money * 0.1
    direction = "LONG" if score > 0.2 else "SHORT" if score < -0.2 else "FLAT"
    confidence = min(abs(score), 1.0)

    ctx.result("signal", {
        "direction": direction,
        "confidence": round(confidence, 3),
        "score": round(score, 3),
        "pcr_interpretation": "Bullish" if pcr < 0.7 else "Bearish" if pcr > 1.3 else "Neutral",
        "gamma_regime": "Positive GEX (dampening)" if net_gex > 0 else "Negative GEX (amplifying)",
    })
    return ctx


def build_options_flow_pipeline() -> Pipeline:
    return Pipeline(
        name="options_flow",
        steps=[
            PipelineStep("fetch_data", _opt_step_fetch),
            PipelineStep("flow_analysis", _opt_step_flow),
            PipelineStep("greeks_computation", _opt_step_greeks),
            PipelineStep("multi_model_pricing", _opt_step_pricing, optional=True),
            PipelineStep("signal_fusion", _opt_step_signal),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE 3 — TRANSFORMER PREDICTION
# tv_data → TFT quantile forecast → Transformer MC paths → probabilistic signal
# ═══════════════════════════════════════════════════════════════════════════════

async def _trans_step_fetch(ctx: PipelineContext) -> PipelineContext:
    ticker = ctx.ticker
    period = ctx.config.get("data", {}).get("default_period", "2y")
    df = await _async(_data_handler.fetch_history, ticker, period, "1d")
    price = await _async(_data_handler.get_current_price, ticker)
    ctx.put("ohlcv", df)
    ctx.put("price", price)
    return ctx


async def _trans_step_tft(ctx: PipelineContext) -> PipelineContext:
    ohlcv = ctx.get("ohlcv")
    cfg = ctx.config.get("models", {}).get("tft", {})

    def _run():
        model = TFTPredictor(
            horizons=cfg.get("horizons", [1, 5, 10, 20]),
            quantiles=cfg.get("quantiles", [0.10, 0.50, 0.90]),
        )
        model.train(ohlcv)
        pred = model.predict(ohlcv, ctx.ticker)
        pf = pred.price_forecast
        vf = pred.vol_forecast
        return {
            "price_q10": {h: _scalar(v) for h, v in zip(pf.horizons, pf.q10)},
            "price_q50": {h: _scalar(v) for h, v in zip(pf.horizons, pf.q50)},
            "price_q90": {h: _scalar(v) for h, v in zip(pf.horizons, pf.q90)},
            "vol_q50": {h: _scalar(v) for h, v in zip(vf.horizons, vf.q50)},
            "attention_proxy": {k: _scalar(v) for k, v in pred.attention_proxy.items()},
        }

    tft_out = await _async(_run)
    ctx.result("tft_forecast", tft_out)
    ctx.put("tft", tft_out)
    return ctx


async def _trans_step_mc(ctx: PipelineContext) -> PipelineContext:
    ohlcv = ctx.get("ohlcv")
    price = ctx.get("price")
    cfg = ctx.config.get("models", {}).get("transformer_mc", {})
    rf = await _async(_data_handler.get_risk_free_rate)

    def _run():
        model = TransformerMCModel(
            n_paths=cfg.get("n_paths", 2000),
            n_steps=cfg.get("n_steps", 60),
            context_window=cfg.get("context_window", 60),
            d_model=cfg.get("d_model", 64),
            n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("n_layers", 3),
            n_epochs=cfg.get("n_epochs", 100),
        )
        model.fit(ohlcv["Close"].values)
        res = model.price(S=price, K=price, T=30/252, r=rf, q=0.0)
        var_cvar = model.var_cvar(0.95)

        terminal = res.terminal_prices
        pct_above = float(np.mean(terminal > price)) if terminal is not None else 0.5

        return {
            "call_price": _scalar(res.call_price),
            "put_price": _scalar(res.put_price),
            "pct_paths_bullish": round(pct_above, 4),
            "device_used": res.device_used,
            "var_95": _scalar(var_cvar.get("var", 0.0)),
            "cvar_95": _scalar(var_cvar.get("cvar", 0.0)),
        }

    mc_out = await _async(_run)
    ctx.result("transformer_mc", mc_out)
    ctx.put("mc", mc_out)
    return ctx


async def _trans_step_signal(ctx: PipelineContext) -> PipelineContext:
    tft = ctx.get("tft", {})
    mc = ctx.get("mc", {})
    price = ctx.get("price", 100.0)

    # TFT: compare q50 horizon-5 to current price
    h5_q50 = tft.get("price_q50", {}).get(5, price)
    h5_q10 = tft.get("price_q10", {}).get(5, price)
    h5_q90 = tft.get("price_q90", {}).get(5, price)

    tft_direction = (h5_q50 - price) / (price + 1e-6)
    tft_uncertainty = (h5_q90 - h5_q10) / (price + 1e-6)

    # MC: fraction of paths ending above current price
    pct_bull = mc.get("pct_paths_bullish", 0.5)
    mc_score = (pct_bull - 0.5) * 2.0  # [-1, +1]

    score = tft_direction * 0.5 + mc_score * 0.5
    confidence = max(0.0, 1.0 - tft_uncertainty)

    direction = "LONG" if score > 0.01 else "SHORT" if score < -0.01 else "FLAT"

    ctx.result("signal", {
        "direction": direction,
        "confidence": round(min(confidence, 1.0), 3),
        "score": round(score, 4),
        "tft_h5_forecast": round(h5_q50, 2),
        "mc_pct_bullish": round(pct_bull, 4),
        "forecast_uncertainty": round(tft_uncertainty, 4),
    })
    return ctx


def build_transformer_pipeline() -> Pipeline:
    return Pipeline(
        name="transformer",
        steps=[
            PipelineStep("fetch_data", _trans_step_fetch),
            PipelineStep("tft_quantile_forecast", _trans_step_tft),
            PipelineStep("transformer_mc_simulation", _trans_step_mc, optional=True),
            PipelineStep("probabilistic_signal", _trans_step_signal),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE 4 — SWARM INTELLIGENCE
# market_data → agents → swarm_engine → aggregator → consensus signal
# ═══════════════════════════════════════════════════════════════════════════════

async def _swarm_step_fetch(ctx: PipelineContext) -> PipelineContext:
    ticker = ctx.ticker

    def _fetch():
        df = fetch_price_data(ticker, period="1y", interval="1d")
        features = compute_swarm_features(df)
        vix = get_vix_level()
        atr_series = compute_atr(df)
        sr = compute_support_resistance(df)
        return df, features, vix, atr_series, sr

    df, features, vix, atr_series, sr = await _async(_fetch)
    price = float(df["Close"].iloc[-1]) if not df.empty else 100.0

    ctx.put("ohlcv", df)
    ctx.put("swarm_features", features)
    ctx.put("vix", vix)
    ctx.put("price", price)
    ctx.put("atr", float(atr_series.iloc[-1]) if not atr_series.empty else 0.0)
    ctx.put("support_resistance", sr)
    return ctx


async def _swarm_step_run_agents(ctx: PipelineContext) -> PipelineContext:
    ticker = ctx.ticker
    features = ctx.get("swarm_features")
    vix = ctx.get("vix", 15.0)
    price = ctx.get("price")

    market_data = {
        "ticker": ticker,
        "price": price,
        "features": features,
        "vix": vix,
        "options_flow": {"put_call_ratio": 1.0},
    }

    def _run():
        state = SwarmState(tickers=[ticker], n_assets=1)
        state.initialize([ticker])

        agents = [
            BoidsMomentumAgent(ticker),
            VicsekConsensusAgent(ticker),
            PSOOptimizerAgent(ticker),
            LeaderFollowerAgent(ticker),
            TopologicalAgent(ticker),
        ]
        # Add ACO ants
        n_ants = ctx.config.get("swarm", {}).get("n_ants", 3)
        for i in range(n_ants):
            agents.append(ACOPathAgent(ticker, ant_id=i))

        signals = []
        for agent in agents:
            try:
                sig = agent.compute(market_data, state)
                signals.append(sig)
                state.agent_signals[agent.__class__.__name__] = sig
            except Exception as e:
                logger.debug("Agent %s failed: %s", agent.__class__.__name__, e)

        return signals, state

    signals, state = await _async(_run)
    ctx.put("agent_signals", signals)
    ctx.put("swarm_state", state)
    return ctx


async def _swarm_step_aggregate(ctx: PipelineContext) -> PipelineContext:
    signals = ctx.get("agent_signals", [])
    price = ctx.get("price")

    def _agg():
        agg = SignalAggregator(config={})
        result = agg.aggregate(signals, ctx.ticker, price, entry_price=None)
        return result

    agg_result = await _async(_agg)
    atr = ctx.get("atr", 0.0)
    sr = ctx.get("support_resistance", {})

    def _trade_plan():
        ohlcv = ctx.get("ohlcv")
        direction = 1 if agg_result.direction > 0 else -1
        return compute_trade_levels(ohlcv, direction)

    trade_plan = await _async(_trade_plan)

    ctx.result("swarm_aggregation", {
        "action": agg_result.action,
        "direction": round(float(agg_result.direction), 4),
        "confidence": round(float(agg_result.confidence), 4),
        "strength": round(float(agg_result.strength), 4),
        "zone": agg_result.zone,
        "n_agents": len(signals),
    })
    ctx.result("trade_plan", {
        "entry": _scalar(trade_plan.get("entry", price)),
        "stop_loss": _scalar(trade_plan.get("stop", price * 0.97)),
        "take_profit": _scalar(trade_plan.get("target", price * 1.05)),
        "risk_reward": _scalar(trade_plan.get("rr", 1.5)),
        "hold_days": trade_plan.get("hold_days", 5),
        "entry_type": trade_plan.get("entry_type", "market"),
        "support": sr.get("support", []),
        "resistance": sr.get("resistance", []),
    })

    ctx.result("signal", {
        "direction": "LONG" if agg_result.direction > 0.1 else "SHORT" if agg_result.direction < -0.1 else "FLAT",
        "confidence": round(float(agg_result.confidence), 4),
        "score": round(float(agg_result.direction), 4),
        "action": agg_result.action,
        "zone": agg_result.zone,
    })
    return ctx


def build_swarm_pipeline() -> Pipeline:
    return Pipeline(
        name="swarm",
        steps=[
            PipelineStep("fetch_swarm_data", _swarm_step_fetch),
            PipelineStep("run_agents", _swarm_step_run_agents),
            PipelineStep("aggregate_consensus", _swarm_step_aggregate),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE 5 — MULTI-FACTOR FUSION (MOST IMPORTANT)
# volatility + options_flow + sentiment + regime + transformer + swarm
# → meta-signal → strategy engine
# ═══════════════════════════════════════════════════════════════════════════════

async def _fusion_step_fetch(ctx: PipelineContext) -> PipelineContext:
    ticker = ctx.ticker
    period = ctx.config.get("data", {}).get("default_period", "2y")

    df, price, chain, info = await asyncio.gather(
        _async(_data_handler.fetch_history, ticker, period, "1d"),
        _async(_data_handler.get_current_price, ticker),
        _async(_data_handler.get_options_chain, ticker),
        _async(_data_handler.get_ticker_info, ticker),
    )

    returns = df["Close"].pct_change().dropna().values
    ctx.put("ohlcv", df)
    ctx.put("price", price)
    ctx.put("options_chain", chain)
    ctx.put("ticker_info", info)
    ctx.put("returns", returns)
    return ctx


async def _fusion_step_vol_regime(ctx: PipelineContext) -> PipelineContext:
    """Run GARCH + HMM concurrently."""
    returns = ctx.get("returns")

    def _garch():
        f = GARCHForecaster(returns)
        results = f.fit_all()
        best = f.select_best(results)
        f.forecast(best, horizon=10)
        vol_now = _scalar(best.conditional_vol[-1]) if best.conditional_vol is not None else None
        vol_fwd = _scalar(best.forecast_vol[0]) if best.forecast_vol is not None and len(best.forecast_vol) > 0 else None
        garch_ann = vol_now * float(np.sqrt(252)) if vol_now else None
        hv20 = float(np.std(returns[-20:]) * np.sqrt(252)) if len(returns) >= 20 else None
        sig = (
            f.compare_to_iv(garch_ann, hv20)
            if garch_ann and hv20
            else {"signal": "FAIR", "strength": "WEAK", "premium_pct": 0.0}
        )
        return {"iv_signal": sig, "current_vol": vol_now, "forecast_vol": vol_fwd}

    def _regime():
        returns_series = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        d = RegimeDetector(returns_series, n_regimes=3, seed=42)
        r = d.fit()
        # regime_stats is a list[RegimeStats], convert to dict keyed by regime_id
        regime_stats_dict = {}
        for stat in r.regime_stats:
            regime_stats_dict[stat.regime_id] = {
                "label": stat.label,
                "mean_daily_return": _scalar(stat.mean_daily_return),
                "std_daily_return": _scalar(stat.std_daily_return),
                "annualized_return": _scalar(stat.annualized_return),
                "annualized_vol": _scalar(stat.annualized_vol),
                "sharpe_ratio": _scalar(stat.sharpe_ratio),
                "avg_duration_days": _scalar(stat.avg_duration_days),
                "pct_time_in_regime": _scalar(stat.pct_time_in_regime),
            }
        return {"current_regime": r.current_regime, "regime_stats": regime_stats_dict}

    garch_out, regime_out = await asyncio.gather(
        _async(_garch), _async(_regime)
    )
    ctx.put("garch", garch_out)
    ctx.put("regime", regime_out)
    return ctx


async def _fusion_step_options_sentiment(ctx: PipelineContext) -> PipelineContext:
    """Run options flow + sentiment concurrently."""
    chain = ctx.get("options_chain")
    price = ctx.get("price")
    info = ctx.get("ticker_info", {})

    def _flow():
        # Guard: non-F&O stocks have no options chain
        if chain is None or (hasattr(chain, 'empty') and chain.empty):
            return {"net_flow": 0.0, "pcr": 1.0, "smart_money": 0.0}
        calls = chain[chain["option_type"] == "call"] if "option_type" in chain.columns else chain.iloc[:len(chain)//2]
        puts = chain[chain["option_type"] == "put"] if "option_type" in chain.columns else chain.iloc[len(chain)//2:]
        analyzer = OptionsFlowAnalyzer(calls, puts, price, ctx.ticker)
        summary = analyzer.analyze()
        return {
            "net_flow": _scalar(analyzer.net_flow_score()),
            "pcr": _scalar(summary.pcr_volume),
            "smart_money": _scalar(summary.smart_money_score),
        }

    def _sentiment():
        engine = SentimentEngine(use_vader_fallback=True)
        news = info.get("news", [])
        if not news:
            return {"signal": "NEUTRAL", "score": 0.0, "strength": "WEAK"}
        parsed = engine.parse_yfinance_news(news)
        scored = engine.score_texts([n.get("text", "") for n in parsed if n.get("text")])
        if not scored:
            return {"signal": "NEUTRAL", "score": 0.0, "strength": "WEAK"}
        avg_score = float(np.mean([s["score"] for s in scored]))
        signal = "BULLISH" if avg_score > 0.15 else "BEARISH" if avg_score < -0.15 else "NEUTRAL"
        return {"signal": signal, "score": round(avg_score, 4), "strength": "STRONG" if abs(avg_score) > 0.3 else "MODERATE"}

    flow_out, sent_out = await asyncio.gather(
        _async(_flow), _async(_sentiment)
    )
    ctx.put("flow", flow_out)
    ctx.put("sentiment", sent_out)
    return ctx


async def _fusion_step_transformer(ctx: PipelineContext) -> PipelineContext:
    """TFT + MC — lighter versions for fusion speed."""
    ohlcv = ctx.get("ohlcv")
    price = ctx.get("price")
    rf = await _async(_data_handler.get_risk_free_rate)

    def _run():
        tft = TFTPredictor(horizons=[1, 5, 10], quantiles=[0.10, 0.50, 0.90])
        tft.train(ohlcv)
        pred = tft.predict(ohlcv, ctx.ticker)
        h5_q50 = _scalar(pred.price_forecast.q50.get(5, price))
        tft_score = (h5_q50 - price) / (price + 1e-6)

        mc = TransformerMCModel(n_paths=1000, n_steps=30, n_epochs=50)
        mc.fit(ohlcv["Close"].values)
        mc_res = mc.price(S=price, K=price, T=10/252, r=rf, q=0.0)
        terminal = mc_res.terminal_prices
        pct_bull = float(np.mean(terminal > price)) if terminal is not None else 0.5
        mc_score = (pct_bull - 0.5) * 2.0

        return {
            "tft_score": round(tft_score, 4),
            "mc_score": round(mc_score, 4),
            "pct_bull": round(pct_bull, 4),
            "h5_forecast": round(h5_q50, 2),
        }

    trans_out = await _async(_run)
    ctx.put("transformer", trans_out)
    return ctx


async def _fusion_step_swarm(ctx: PipelineContext) -> PipelineContext:
    """Swarm intelligence sub-run."""
    ticker = ctx.ticker
    ohlcv = ctx.get("ohlcv")
    price = ctx.get("price")

    def _run():
        df_sw = fetch_price_data(ticker, period="1y", interval="1d")
        features = compute_swarm_features(df_sw)
        vix = get_vix_level()
        market_data = {"ticker": ticker, "price": price, "features": features, "vix": vix}

        state = SwarmState(tickers=[ticker], n_assets=1)
        state.initialize([ticker])

        agents = [
            BoidsMomentumAgent(ticker),
            VicsekConsensusAgent(ticker),
            LeaderFollowerAgent(ticker),
            TopologicalAgent(ticker),
        ]
        for i in range(3):
            agents.append(ACOPathAgent(ticker, ant_id=i))

        signals = []
        for ag in agents:
            try:
                signals.append(ag.compute(market_data, state))
            except Exception:
                pass

        if not signals:
            return {"direction": 0.0, "confidence": 0.0, "action": "HOLD"}

        agg = SignalAggregator(config={})
        result = agg.aggregate(signals, ticker, price, entry_price=None)
        return {
            "direction": round(float(result.direction), 4),
            "confidence": round(float(result.confidence), 4),
            "action": result.action,
        }

    swarm_out = await _async(_run)
    ctx.put("swarm", swarm_out)
    return ctx


async def _fusion_step_meta_signal(ctx: PipelineContext) -> PipelineContext:
    """
    Fuse all signals into a meta-signal with dynamic weighting.
    Resolves conflicts by regime-gating and confidence weighting.
    """
    cfg_weights = ctx.config.get("strategy", {}).get("fusion_weights", {})
    w_vol = cfg_weights.get("volatility", 0.20)
    w_opt = cfg_weights.get("options_flow", 0.20)
    w_trans = cfg_weights.get("transformer", 0.25)
    w_swarm = cfg_weights.get("swarm", 0.20)
    w_sent = cfg_weights.get("sentiment", 0.15)

    garch = ctx.get("garch", {})
    regime = ctx.get("regime", {})
    flow = ctx.get("flow", {})
    sentiment = ctx.get("sentiment", {})
    transformer = ctx.get("transformer", {})
    swarm = ctx.get("swarm", {})

    current_regime = regime.get("current_regime", "Neutral")

    # Volatility signal: OVERPRICED vol → sell vol → slight bull; UNDERPRICED → hedge
    iv_sig = garch.get("iv_signal", {}).get("signal", "FAIR")
    vol_score = 0.5 if iv_sig == "OVERPRICED" else -0.5 if iv_sig == "UNDERPRICED" else 0.0
    if "Bear" in current_regime:
        vol_score -= 0.5
    elif "Bull" in current_regime:
        vol_score += 0.5

    # Options flow: [-1, +1] from net_flow_score
    opt_score = flow.get("net_flow", 0.0)
    pcr = flow.get("pcr", 1.0)
    opt_score += (0.3 if pcr < 0.7 else -0.3 if pcr > 1.3 else 0.0)
    opt_score = max(-1.0, min(1.0, opt_score))

    # Transformer: combined TFT + MC
    trans_score = transformer.get("tft_score", 0.0) * 5.0 + transformer.get("mc_score", 0.0) * 0.5
    trans_score = max(-1.0, min(1.0, trans_score))

    # Swarm: direction is already [-1, +1]
    swarm_score = swarm.get("direction", 0.0)

    # Sentiment: convert to [-1, +1]
    sent_signal = sentiment.get("signal", "NEUTRAL")
    sent_score = 1.0 if sent_signal == "BULLISH" else -1.0 if sent_signal == "BEARISH" else 0.0

    # Weighted meta-score
    meta_score = (
        vol_score * w_vol +
        opt_score * w_opt +
        trans_score * w_trans +
        swarm_score * w_swarm +
        sent_score * w_sent
    )

    # Conflict detection: count agreements vs disagreements
    scores = [vol_score, opt_score, trans_score, swarm_score, sent_score]
    n_bullish = sum(1 for s in scores if s > 0.1)
    n_bearish = sum(1 for s in scores if s < -0.1)
    agreement = max(n_bullish, n_bearish) / len(scores)

    direction = "LONG" if meta_score > 0.15 else "SHORT" if meta_score < -0.15 else "FLAT"
    confidence = min(abs(meta_score) * agreement, 1.0)

    ctx.result("meta_signal", {
        "direction": direction,
        "confidence": round(confidence, 4),
        "meta_score": round(meta_score, 4),
        "agreement": round(agreement, 4),
        "component_scores": {
            "volatility": round(vol_score, 4),
            "options_flow": round(opt_score, 4),
            "transformer": round(trans_score, 4),
            "swarm": round(swarm_score, 4),
            "sentiment": round(sent_score, 4),
        },
        "regime": current_regime,
        "n_bullish_signals": n_bullish,
        "n_bearish_signals": n_bearish,
    })
    ctx.result("signal", {
        "direction": direction,
        "confidence": round(confidence, 4),
        "score": round(meta_score, 4),
    })
    return ctx


async def _fusion_step_risk_sizing(ctx: PipelineContext) -> PipelineContext:
    """Compute position size + stop/TP using ATR and risk config."""
    ohlcv = ctx.get("ohlcv")
    price = ctx.get("price")
    meta = ctx.results.get("meta_signal", {})
    direction = meta.get("direction", "FLAT")
    risk_cfg = ctx.config.get("risk", {})

    def _size():
        close = ohlcv["Close"].values
        high = ohlcv["High"].values
        low = ohlcv["Low"].values
        # ATR-14
        tr = np.maximum(high[1:] - low[1:],
               np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        atr = float(tr[-14:].mean()) if len(tr) >= 14 else float(tr.mean())

        atr_stop = risk_cfg.get("atr_stop_mult", 2.0)
        atr_tp = risk_cfg.get("atr_tp_mult", 3.0)
        max_pos = risk_cfg.get("max_position_pct", 0.10)

        stop_loss = price - atr * atr_stop if direction == "LONG" else price + atr * atr_stop
        take_profit = price + atr * atr_tp if direction == "LONG" else price - atr * atr_tp

        risk_per_share = abs(price - stop_loss)
        rr_ratio = abs(take_profit - price) / (risk_per_share + 1e-6)

        # Kelly fraction capped at max_pos
        confidence = meta.get("confidence", 0.5)
        win_rate = 0.5 + confidence * 0.2
        kelly = max(0.0, (win_rate * rr_ratio - (1 - win_rate)) / (rr_ratio + 1e-6))
        position_size_pct = min(kelly * 0.5, max_pos)  # half-Kelly

        return {
            "entry_price": round(price, 4),
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "atr": round(atr, 4),
            "risk_reward": round(rr_ratio, 4),
            "position_size_pct": round(position_size_pct * 100, 2),
            "kelly_fraction": round(kelly, 4),
        }

    sizing = await _async(_size)
    ctx.result("position_sizing", sizing)
    return ctx


def build_multi_factor_pipeline() -> Pipeline:
    return Pipeline(
        name="multi_factor",
        steps=[
            PipelineStep("fetch_all_data", _fusion_step_fetch),
            PipelineStep("vol_and_regime", _fusion_step_vol_regime),
            PipelineStep("options_and_sentiment", _fusion_step_options_sentiment),
            PipelineStep("transformer_prediction", _fusion_step_transformer, optional=True),
            PipelineStep("swarm_consensus", _fusion_step_swarm, optional=True),
            PipelineStep("meta_signal_fusion", _fusion_step_meta_signal),
            PipelineStep("risk_and_sizing", _fusion_step_risk_sizing),
        ],
    )
