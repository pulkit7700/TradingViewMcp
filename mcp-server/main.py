"""
TradingView Quant MCP Server — production entry point.

Exposes a full quantitative trading engine via the Model Context Protocol.
All heavy computation runs async in ThreadPoolExecutor to avoid blocking.

Run:
    python main.py

Or via MCP client (Claude Desktop, etc.) using stdio transport.
"""

import sys
import os
import asyncio
import logging
import json
from typing import Any

import yaml

# ─── Path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(ROOT, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, REPO_ROOT)

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("tv_mcp")

# ─── Config ───────────────────────────────────────────────────────────────────
def _load_config() -> dict:
    cfg_path = os.path.join(ROOT, "config.yaml")
    try:
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("config.yaml not found, using defaults")
        return {}

CONFIG = _load_config()

# ─── Core imports (after path setup) ─────────────────────────────────────────
from mcp.server.fastmcp import FastMCP

from connectors.tv_connector import TradingViewConnector
from pipelines.engine import PipelineEngine
from pipelines.prebuilt import (
    build_volatility_pipeline,
    build_options_flow_pipeline,
    build_transformer_pipeline,
    build_swarm_pipeline,
    build_multi_factor_pipeline,
)
from strategies.strategy_engine import StrategyEngine
from risk.risk_engine import RiskEngine
from pine.generator import PineScriptGenerator
from backtest.backtester import Backtester

# ─── Component initialization ─────────────────────────────────────────────────
mcp = FastMCP(
    CONFIG.get("server", {}).get("name", "TradingViewMCP"),
    instructions="""
    You are connected to a production quantitative trading engine.
    This server provides institutional-grade analysis via 5 research pipelines:
    1. volatility — GARCH + Rough Vol + HMM regime + ML IV prediction
    2. options_flow — Flow analysis + Greeks + GEX + Multi-model pricing
    3. transformer — TFT quantile forecasts + Monte Carlo path simulation
    4. swarm — 6 bio-inspired agents (Boids, ACO, Vicsek, PSO, Leader, Topological)
    5. multi_factor — All of the above fused into a meta-signal with dynamic weights

    Always run multi_factor pipeline for final trading decisions.
    Use individual pipelines for deep analysis of specific factors.
    Generate Pine Script for any strategy to backtest on TradingView.
    """,
)

connector = TradingViewConnector(CONFIG)
pipeline_engine = PipelineEngine(CONFIG)
strategy_engine = StrategyEngine(CONFIG)
risk_engine = RiskEngine(CONFIG)
pine_generator = PineScriptGenerator()
backtester = Backtester(CONFIG)

# Register all pipelines
for _pipe in [
    build_volatility_pipeline(),
    build_options_flow_pipeline(),
    build_transformer_pipeline(),
    build_swarm_pipeline(),
    build_multi_factor_pipeline(),
]:
    pipeline_engine.register(_pipe)

logger.info("All 5 pipelines registered: %s", pipeline_engine.list_pipelines())


# ═══════════════════════════════════════════════════════════════════════════════
# MCP TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def fetch_market_data(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
) -> dict:
    """
    Fetch live market data for a ticker.
    Tries TradingView Desktop first (if running), falls back to yfinance.
    Returns OHLCV summary, current price, and pre-computed indicators.

    Args:
        ticker: Stock ticker (e.g. "RELIANCE.NS", "AAPL", "TCS.NS")
        period: Data period ("1mo", "3mo", "6mo", "1y", "2y", "5y")
        interval: Bar interval ("1d", "1wk", "1mo")
    """
    data = await connector.get_chart_data(ticker, period, interval)

    return {
        "ticker": data.ticker,
        "timeframe": data.timeframe,
        "source": data.source,
        "current_price": round(data.current_price, 4),
        "n_bars": len(data.ohlcv),
        "date_range": {
            "start": str(data.ohlcv.index[0]) if not data.ohlcv.empty else None,
            "end": str(data.ohlcv.index[-1]) if not data.ohlcv.empty else None,
        },
        "latest_bar": {
            "open": round(float(data.ohlcv["Open"].iloc[-1]), 4),
            "high": round(float(data.ohlcv["High"].iloc[-1]), 4),
            "low": round(float(data.ohlcv["Low"].iloc[-1]), 4),
            "close": round(float(data.ohlcv["Close"].iloc[-1]), 4),
            "volume": int(data.ohlcv["Volume"].iloc[-1]),
        } if not data.ohlcv.empty else {},
        "indicators": {
            k: (round(float(v[-1]), 6) if v is not None and len(v) > 0 else None)
            for k, v in data.indicators.items()
            if k in ("ema_20", "ema_50", "rsi_14", "atr_14", "hv_20", "hv_60")
        },
        "info": {
            k: data.info.get(k)
            for k in ("sector", "industry", "marketCap", "beta", "shortName")
            if k in data.info
        },
    }


@mcp.tool()
async def run_pipeline(
    ticker: str,
    pipeline_name: str,
) -> dict:
    """
    Run a named research pipeline on a ticker.

    Available pipelines:
      - volatility: GARCH + Rough Vol + HMM + ML IV
      - options_flow: Flow analysis + Greeks + GEX + Pricing
      - transformer: TFT quantile + Monte Carlo simulation
      - swarm: 6 bio-inspired trading agents + consensus
      - multi_factor: All pipelines fused (use this for final decisions)

    Args:
        ticker: Stock ticker (e.g. "TCS.NS", "INFY.NS")
        pipeline_name: One of the pipeline names above
    """
    result = await pipeline_engine.run(pipeline_name, ticker)
    return _clean_output(result)


@mcp.tool()
async def run_volatility_analysis(ticker: str) -> dict:
    """
    Deep volatility analysis: GARCH family forecasting, Rough Bergomi model,
    Hurst exponent estimation, ML-based IV prediction, and HMM regime detection.

    Returns: GARCH forecast, rough vol parameters, regime state, ML IV signal.

    Args:
        ticker: Stock ticker
    """
    result = await pipeline_engine.run("volatility", ticker)
    return _clean_output(result)


@mcp.tool()
async def run_options_flow_analysis(ticker: str) -> dict:
    """
    Options flow analysis: Put/Call ratio, unusual activity detection,
    smart money score, gamma exposure, max pain, multi-model pricing (BS/Binomial/MC).

    Args:
        ticker: Stock ticker with listed options (e.g. "RELIANCE.NS", "AAPL")
    """
    result = await pipeline_engine.run("options_flow", ticker)
    return _clean_output(result)


@mcp.tool()
async def run_transformer_analysis(ticker: str) -> dict:
    """
    ML price forecasting: TFT-based quantile regression forest (10th/50th/90th
    percentile forecasts for 1/5/10/20 day horizons) + Transformer Monte Carlo
    path simulation for options pricing and directional probability.

    Args:
        ticker: Stock ticker
    """
    result = await pipeline_engine.run("transformer", ticker)
    return _clean_output(result)


@mcp.tool()
async def run_swarm_analysis(ticker: str) -> dict:
    """
    Swarm intelligence analysis using 6 bio-inspired agent types:
    - Boids (momentum alignment)
    - Vicsek (consensus with noise injection)
    - ACO (pheromone-based path finding)
    - PSO (particle swarm optimization of trading params)
    - Leader-Follower (NSE market leaders as guides)
    - Topological (7-neighbor correlation network)

    Returns: consensus direction, confidence, zone (ZOR/ZOO/ZOA), trade plan.

    Args:
        ticker: NSE stock ticker (e.g. "TCS.NS", "HDFCBANK.NS")
    """
    result = await pipeline_engine.run("swarm", ticker)
    return _clean_output(result)


@mcp.tool()
async def run_multi_factor_analysis(ticker: str) -> dict:
    """
    Full multi-factor fusion — the most important pipeline.

    Combines: GARCH volatility + HMM regime + options flow + sentiment
    + TFT forecasts + Monte Carlo + swarm consensus → weighted meta-signal.

    Applies: regime gating, volatility adjustment, conflict resolution,
    dynamic weighting, risk sizing (Kelly fraction), and produces
    a fully parameterized StrategySignal.

    Args:
        ticker: Stock ticker
    """
    result = await pipeline_engine.run("multi_factor", ticker)
    return _clean_output(result)


@mcp.tool()
async def build_strategy(ticker: str, pipeline: str = "multi_factor") -> dict:
    """
    Run a pipeline and convert its output into an actionable StrategySignal.

    The StrategySignal includes:
    - direction: LONG / SHORT / FLAT
    - confidence: 0–1 probability
    - entry_price, stop_loss, take_profit
    - position_size_pct: % of portfolio (Kelly-adjusted)
    - risk_reward ratio
    - regime and volatility state
    - rationale string
    - pine_params: ready for Pine Script generation

    Args:
        ticker: Stock ticker
        pipeline: Pipeline to run (default: multi_factor)
    """
    pipeline_result = await pipeline_engine.run(pipeline, ticker)
    signal = strategy_engine.build_signal(ticker, pipeline_result)
    return signal.to_dict()


@mcp.tool()
async def generate_pine_script(strategy: str) -> dict:
    """
    Convert a strategy signal JSON into valid Pine Script v5 code.

    The generated script includes:
    - Regime proxy (HMM → EMA structure)
    - Volatility proxy (GARCH → ATR dynamics)
    - Transformer proxy (TFT/MC → momentum alignment)
    - Swarm proxy (agent consensus → volume + price action)
    - Options flow proxy (GEX → volume breakout)
    - Multi-factor scoring gate (composite long_score / short_score)
    - ATR-based dynamic stops and take-profits
    - Alert conditions for all entry signals
    - Optimizable inputs for Strategy Tester

    Args:
        strategy: JSON string of strategy dict (output of build_strategy tool)
    """
    if isinstance(strategy, str):
        try:
            strategy_dict = json.loads(strategy)
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {e}"}
    else:
        strategy_dict = strategy

    pine_output = pine_generator.generate(strategy_dict)
    return pine_output.to_dict()


@mcp.tool()
async def run_full_workflow(
    ticker: str,
    include_pine: bool = True,
    include_backtest: bool = True,
) -> dict:
    """
    Complete end-to-end workflow:
    1. Run multi_factor pipeline
    2. Build StrategySignal (with risk gating)
    3. Apply risk engine (VaR, stress test)
    4. Generate Pine Script
    5. Backtest Pine strategy on historical data

    This is the primary tool for getting a complete trading decision.

    Args:
        ticker: Stock ticker
        include_pine: Generate Pine Script (default: True)
        include_backtest: Run vectorized backtest (default: True)
    """
    import numpy as np
    from Tools.data_handler import MarketDataHandler

    # Step 1: Multi-factor pipeline
    logger.info("Full workflow: %s — step 1/5 pipeline", ticker)
    pipeline_result = await pipeline_engine.run("multi_factor", ticker)

    # Step 2: Strategy signal
    logger.info("Full workflow: %s — step 2/5 strategy", ticker)
    signal = strategy_engine.build_signal(ticker, pipeline_result)
    signal_dict = signal.to_dict()

    # Step 3: Risk engine
    logger.info("Full workflow: %s — step 3/5 risk", ticker)
    risk_result = None
    try:
        handler = MarketDataHandler()
        df = handler.fetch_history(ticker, period="2y", interval="1d")
        returns = df["Close"].pct_change().dropna().values if not df.empty else np.array([])
        price = signal.entry_price or 100.0

        if len(returns) > 20:
            risk_filter = await risk_engine.evaluate(
                returns=returns,
                price=price,
                direction=signal.direction,
                entry=signal.entry_price,
                stop=signal.stop_loss,
                take_profit=signal.take_profit,
                position_pct=signal.position_size_pct,
            )
            # Apply risk adjustment to signal
            signal_dict["position_size_pct"] = risk_filter.adjusted_position_pct
            risk_result = {
                "passed": risk_filter.passed,
                "reason": risk_filter.reason,
                "var_95_pct": round(risk_filter.var_95 * 100, 2),
                "cvar_95_pct": round(risk_filter.cvar_95 * 100, 2),
                "max_drawdown_est_pct": round(risk_filter.max_drawdown_estimate * 100, 2),
                "adjusted_position_pct": risk_filter.adjusted_position_pct,
            }
    except Exception as e:
        logger.warning("Risk engine failed: %s", e)
        risk_result = {"error": str(e)}

    # Step 4: Pine Script
    logger.info("Full workflow: %s — step 4/5 pine", ticker)
    pine_result = None
    if include_pine:
        try:
            pine_output = pine_generator.generate(signal_dict)
            pine_result = pine_output.to_dict()
        except Exception as e:
            logger.warning("Pine generation failed: %s", e)
            pine_result = {"error": str(e)}

    # Step 5: Backtest
    logger.info("Full workflow: %s — step 5/5 backtest", ticker)
    backtest_result = None
    if include_backtest and signal.pine_params:
        try:
            bt = await backtester.run(
                ticker=ticker,
                strategy_name=f"QuantEngine_{ticker}",
                pine_params=signal.pine_params,
                period="2y",
            )
            backtest_result = bt.to_dict()
        except Exception as e:
            logger.warning("Backtest failed: %s", e)
            backtest_result = {"error": str(e)}

    return {
        "ticker": ticker,
        "strategy": signal_dict,
        "risk": risk_result,
        "pine": pine_result,
        "backtest": backtest_result,
        "pipeline_summary": {
            "pipeline": pipeline_result.get("pipeline"),
            "errors": pipeline_result.get("errors", {}),
            "elapsed_seconds": round(pipeline_result.get("meta", {}).get("total_elapsed", 0), 2),
        },
    }


@mcp.tool()
async def get_market_state(ticker: str) -> dict:
    """
    Get current market state: HMM regime, volatility level, sentiment signal.
    Faster than full pipeline — good for quick market context.

    Returns: regime (Bull/Bear/Neutral/Crash), vol state, sentiment, VIX level.

    Args:
        ticker: Stock ticker
    """
    import numpy as np
    from Tools.data_handler import MarketDataHandler
    from Tools.regime_detection import RegimeDetector
    from Tools.volatility_forecast import GARCHForecaster
    from Tools.sentiment_engine import SentimentEngine
    from Tools.swarm_market_data import get_vix_level

    loop = asyncio.get_event_loop()
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=3)

    def _compute():
        handler = MarketDataHandler()
        df = handler.fetch_history(ticker, period="2y", interval="1d")
        if df.empty:
            return {"error": "no_data"}

        returns = df["Close"].pct_change().dropna().values
        price = float(df["Close"].iloc[-1])

        # Regime
        try:
            detector = RegimeDetector(returns, n_regimes=3, seed=42)
            regime_result = detector.fit()
            regime = regime_result.current_regime
        except Exception:
            regime = "Unknown"

        # Volatility
        try:
            forecaster = GARCHForecaster(returns)
            results = forecaster.fit_all()
            best = forecaster.select_best(results)
            current_vol = float(best.conditional_vol[-1]) if best.conditional_vol is not None else None
            vol_ann = current_vol * np.sqrt(252) if current_vol and current_vol < 0.05 else current_vol
            if vol_ann:
                vol_state = "EXTREME" if vol_ann > 0.40 else "HIGH" if vol_ann > 0.25 else "NORMAL" if vol_ann > 0.15 else "LOW"
            else:
                vol_state = "UNKNOWN"
        except Exception:
            vol_state = "UNKNOWN"
            vol_ann = None

        # VIX
        try:
            vix = get_vix_level()
        except Exception:
            vix = None

        # Sentiment (quick VADER pass on ticker info)
        try:
            info = handler.get_ticker_info(ticker)
            engine = SentimentEngine(use_vader_fallback=True)
            news = info.get("news", [])
            if news:
                parsed = engine.parse_yfinance_news(news[:5])
                scored = engine.score_texts([n.get("text", "") for n in parsed if n.get("text")])
                sent_score = float(np.mean([s["score"] for s in scored])) if scored else 0.0
                sentiment = "BULLISH" if sent_score > 0.15 else "BEARISH" if sent_score < -0.15 else "NEUTRAL"
            else:
                sentiment = "NEUTRAL"
                sent_score = 0.0
        except Exception:
            sentiment = "NEUTRAL"
            sent_score = 0.0

        return {
            "ticker": ticker,
            "current_price": round(price, 4),
            "regime": regime,
            "volatility_state": vol_state,
            "annualized_vol": round(vol_ann * 100, 2) if vol_ann else None,
            "vix_level": round(float(vix), 2) if vix else None,
            "sentiment": sentiment,
            "sentiment_score": round(sent_score, 4),
        }

    return await loop.run_in_executor(executor, _compute)


@mcp.tool()
async def run_backtest(
    ticker: str,
    strategy_json: str,
    period: str = "2y",
) -> dict:
    """
    Run vectorized backtest of a strategy on historical data.

    Computes: Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor,
              Calmar Ratio, Total Return, equity curve.

    Args:
        ticker: Stock ticker
        strategy_json: JSON string of strategy dict (output of build_strategy)
        period: Backtest period ("1y", "2y", "5y")
    """
    if isinstance(strategy_json, str):
        try:
            strategy_dict = json.loads(strategy_json)
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {e}"}
    else:
        strategy_dict = strategy_json

    pine_params = strategy_dict.get("pine_params", {})
    if not pine_params:
        return {"error": "strategy must contain pine_params (run build_strategy first)"}

    result = await backtester.run(
        ticker=ticker,
        strategy_name=strategy_dict.get("pipeline", "custom"),
        pine_params=pine_params,
        period=period,
    )
    return result.to_dict()


@mcp.tool()
async def scan_tickers(
    tickers: list[str],
    pipeline: str = "multi_factor",
) -> dict:
    """
    Run pipeline analysis on multiple tickers concurrently.
    Returns ranked results sorted by confidence score.

    Args:
        tickers: List of stock tickers (max 10)
        pipeline: Pipeline to run on each ticker
    """
    tickers = tickers[:10]  # cap at 10
    results = await pipeline_engine.run_multi(pipeline, tickers)

    ranked = []
    for ticker, res in results.items():
        if "error" not in res:
            signal = res.get("results", {}).get("signal", {})
            meta = res.get("results", {}).get("meta_signal", signal)
            ranked.append({
                "ticker": ticker,
                "direction": meta.get("direction", "FLAT"),
                "confidence": meta.get("confidence", 0.0),
                "score": meta.get("meta_score", meta.get("score", 0.0)),
                "regime": res.get("results", {}).get("regime", {}).get("current_regime", "Unknown"),
            })
        else:
            ranked.append({"ticker": ticker, "error": res["error"]})

    ranked_valid = [r for r in ranked if "error" not in r]
    ranked_valid.sort(key=lambda x: abs(x["confidence"]), reverse=True)
    ranked_errors = [r for r in ranked if "error" in r]

    return {
        "pipeline": pipeline,
        "n_tickers": len(tickers),
        "ranked_signals": ranked_valid,
        "errors": ranked_errors,
    }


@mcp.tool()
async def list_pipelines() -> dict:
    """List all available pipelines and their descriptions."""
    descriptions = {
        "volatility": "GARCH family + Rough Bergomi + Hurst + ML IV + HMM regime",
        "options_flow": "Options flow analysis + Greeks + GEX + multi-model pricing",
        "transformer": "TFT quantile forecast (10/50/90th pct) + Transformer MC paths",
        "swarm": "6 bio-inspired agents: Boids, Vicsek, ACO, PSO, Leader, Topological",
        "multi_factor": "Full fusion of all above + sentiment → meta-signal + risk sizing",
    }
    return {
        "available_pipelines": pipeline_engine.list_pipelines(),
        "descriptions": descriptions,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RESOURCES
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.resource("config://server")
def get_server_config() -> str:
    """Server configuration."""
    return json.dumps(CONFIG, indent=2)


@mcp.resource("config://pipelines")
def get_pipeline_config() -> str:
    """Pipeline configuration and weights."""
    return json.dumps({
        "pipelines": pipeline_engine.list_pipelines(),
        "fusion_weights": CONFIG.get("strategy", {}).get("fusion_weights", {}),
        "risk_params": CONFIG.get("risk", {}),
    }, indent=2)


# ─── Utilities ───────────────────────────────────────────────────────────────

def _clean_output(obj: Any) -> Any:
    """Recursively convert numpy types to Python primitives for JSON serialization."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _clean_output(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_output(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_clean_output(v) for v in obj.tolist()]
    elif hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    return obj


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(
        "Starting %s v%s",
        CONFIG.get("server", {}).get("name", "TradingViewMCP"),
        CONFIG.get("server", {}).get("version", "1.0.0"),
    )
    mcp.run()
