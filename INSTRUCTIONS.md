# TradingView Quant MCP Server — Setup & Usage

## Architecture

```
TradingViewMcp/
├── Tools/                    ← 24 quantitative modules (your edge)
└── mcp-server/
    ├── main.py               ← MCP server entry point
    ├── config.yaml           ← All tunable parameters
    ├── requirements.txt      ← Dependencies
    ├── connectors/
    │   └── tv_connector.py   ← TradingView Desktop + yfinance adapter
    ├── pipelines/
    │   ├── engine.py         ← Async pipeline executor + cache
    │   └── prebuilt.py       ← 5 production pipelines
    ├── strategies/
    │   └── strategy_engine.py ← Signal builder with regime/vol gating
    ├── risk/
    │   └── risk_engine.py    ← VaR / stress test / position sizing
    ├── pine/
    │   └── generator.py      ← Pine Script v5 code generator
    └── backtest/
        └── backtester.py     ← Vectorized historical backtest
```

## Install

```bash
cd mcp-server
pip install -r requirements.txt
```

Optional (improves sentiment quality):
```bash
pip install transformers torch  # FinBERT sentiment
```

## Run (standalone test)

```bash
cd mcp-server
python main.py
```

## Connect to Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "tradingview-quant": {
      "command": "python",
      "args": ["E:/Pulkit Documents/Data Science/PRACTICE/GitHub Repos/TradingViewMcp/mcp-server/main.py"],
      "env": {}
    }
  }
}
```

Windows path for Claude Desktop config:
`%APPDATA%\Claude\claude_desktop_config.json`

## MCP Tools

| Tool | Description |
|------|-------------|
| `fetch_market_data` | Live OHLCV + indicators (TV Desktop or yfinance) |
| `run_pipeline` | Run any named pipeline |
| `run_volatility_analysis` | GARCH + Rough Vol + HMM + ML IV |
| `run_options_flow_analysis` | Flow + Greeks + GEX + pricing |
| `run_transformer_analysis` | TFT quantile + Monte Carlo |
| `run_swarm_analysis` | 6 bio-agent consensus |
| `run_multi_factor_analysis` | Full fusion (use this) |
| `build_strategy` | Pipeline → StrategySignal |
| `generate_pine_script` | Strategy → Pine Script v5 |
| `run_full_workflow` | All steps end-to-end |
| `get_market_state` | Quick regime + vol + sentiment |
| `run_backtest` | Vectorized historical backtest |
| `scan_tickers` | Multi-ticker ranked scan |
| `list_pipelines` | Show available pipelines |

## Pipeline Flow

```
TradingView / yfinance
        ↓
   tv_connector.py
        ↓
┌───────────────────────────────────────┐
│           pipeline_engine.py          │
│  ┌─────────┐  ┌────────────┐         │
│  │volatility│  │options_flow│  ...    │
│  └─────────┘  └────────────┘         │
│         ↓            ↓               │
│      ┌──────────────────┐            │
│      │  multi_factor    │            │
│      │  (meta-signal)   │            │
│      └──────────────────┘            │
└───────────────────────────────────────┘
        ↓
  strategy_engine.py  (regime gate + vol adjust)
        ↓
  risk_engine.py       (VaR + stress test)
        ↓
  pine/generator.py    (Pine Script v5)
        ↓
  backtester.py        (Sharpe / DD / Win Rate)
```

## Example Usage (via Claude)

```
"Run full workflow on TCS.NS"
→ Calls run_full_workflow("TCS.NS")
→ Returns: strategy signal + risk metrics + Pine Script + backtest results

"Scan RELIANCE.NS, INFY.NS, HDFCBANK.NS"
→ Calls scan_tickers(["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS"])
→ Returns ranked by confidence

"Generate Pine Script for this strategy: {..."
→ Calls generate_pine_script(strategy_json)
→ Returns valid Pine v5 code to paste in TradingView
```

## TradingView Desktop Integration

Launch TV Desktop with debug port enabled:
```
"C:\Users\...\TradingView.exe" --remote-debugging-port=9222
```

The connector auto-detects this. If TV is open with a chart, it reads the
active symbol and resolution. Data is still fetched via yfinance for reliability.

## Pine Script Proxy Logic

Since Pine cannot run ML models, the generator converts them to indicator proxies:

| ML Component | Pine Proxy |
|--------------|------------|
| HMM regime (Bull/Bear) | EMA structure + slope |
| GARCH high vol | ATR > ATR-SMA × 1.2 |
| Rough vol (H < 0.3) | BB width expansion |
| TFT bullish forecast | close > ema_fast AND rsi ∈ [30,70] |
| Monte Carlo pct_bull | momentum alignment (MACD hist) |
| Swarm consensus | Volume surge + price > open |
| Options flow bullish | Volume + breakout above high[1] |
| Sentiment bullish | Momentum confirmation |

## Key Config Options (config.yaml)

```yaml
strategy:
  fusion_weights:
    volatility: 0.20    # Increase if vol regime is primary driver
    options_flow: 0.20  # Increase for derivatives-heavy stocks
    transformer: 0.25   # Primary directional signal
    swarm: 0.20         # Bio-agent consensus
    sentiment: 0.15     # News-driven stocks

risk:
  min_rr_ratio: 1.5     # Minimum risk:reward to take trade
  max_var_pct: 0.05     # Max 5% daily VaR
  atr_stop_mult: 2.0    # Stop = entry ± 2×ATR
  atr_tp_mult: 3.0      # TP = entry ± 3×ATR
```

## Performance Notes

- First run per ticker: ~30–120s (model training)
- Cached runs: ~1–5s (600s TTL)
- Transformer MC with n_paths=2000: ~15–30s
- Swarm with all 6 agents: ~10–20s
- Reduce `n_paths`, `n_epochs`, `n_ants` in config.yaml for faster runs
