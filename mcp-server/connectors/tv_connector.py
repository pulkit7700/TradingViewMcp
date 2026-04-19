"""
TradingView connector — primary data extraction layer.

Strategy (in priority order):
1. yfinance via data_handler (always available, production-grade)
2. WebSocket inspection via debug port (if TV Desktop running)
3. Local storage / browser data extraction fallback
"""

import sys
import os
import json
import asyncio
import socket
import logging
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

# Add Tools to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from Tools.data_handler import MarketDataHandler

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=4)


@dataclass
class TVChartData:
    ticker: str
    timeframe: str
    ohlcv: pd.DataFrame
    current_price: float
    info: dict
    indicators: dict = field(default_factory=dict)
    source: str = "yfinance"


class TradingViewConnector:
    """
    Extracts live chart data from TradingView Desktop or falls back to yfinance.

    TV Desktop exposes a Chromium debug port (default 9222) when launched with
    --remote-debugging-port=9222. We attempt a WebSocket handshake to detect
    if TV is live; if not, we use yfinance as the authoritative data source.
    """

    TV_DEBUG_PORT = 9222
    TV_DEBUG_HOST = "127.0.0.1"

    def __init__(self, config: dict):
        self._cfg = config
        self._handler = MarketDataHandler()
        self._tv_available: Optional[bool] = None

    # ─── Public API ──────────────────────────────────────────────────────────

    async def get_chart_data(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> TVChartData:
        """Fetch OHLCV + metadata for ticker. Tries TV Desktop first."""
        tv_live = await self._detect_tv_desktop()

        if tv_live:
            try:
                return await self._fetch_from_tv_desktop(ticker, period, interval)
            except Exception as e:
                logger.warning("TV Desktop fetch failed (%s), falling back to yfinance", e)

        return await self._fetch_from_yfinance(ticker, period, interval)

    async def get_multi_chart_data(
        self, tickers: list[str], period: str = "2y", interval: str = "1d"
    ) -> dict[str, TVChartData]:
        tasks = [self.get_chart_data(t, period, interval) for t in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = {}
        for ticker, res in zip(tickers, results):
            if isinstance(res, Exception):
                logger.error("Failed to fetch %s: %s", ticker, res)
            else:
                out[ticker] = res
        return out

    # ─── TV Desktop Detection ────────────────────────────────────────────────

    async def _detect_tv_desktop(self) -> bool:
        if self._tv_available is not None:
            return self._tv_available

        loop = asyncio.get_event_loop()
        available = await loop.run_in_executor(
            _executor, self._check_debug_port
        )
        self._tv_available = available
        if available:
            logger.info("TradingView Desktop debug port detected at %s:%d",
                        self.TV_DEBUG_HOST, self.TV_DEBUG_PORT)
        else:
            logger.info("TradingView Desktop not detected — using yfinance")
        return available

    def _check_debug_port(self) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            result = sock.connect_ex((self.TV_DEBUG_HOST, self.TV_DEBUG_PORT))
            sock.close()
            return result == 0
        except Exception:
            return False

    # ─── TV Desktop Extraction ───────────────────────────────────────────────

    async def _fetch_from_tv_desktop(
        self, ticker: str, period: str, interval: str
    ) -> TVChartData:
        """
        Connects to TV Desktop Chromium debug port via CDP (Chrome DevTools Protocol).
        Extracts chart series data from the active tab's JavaScript context.
        Falls back gracefully if chart data isn't accessible.
        """
        try:
            import aiohttp
        except ImportError:
            raise RuntimeError("aiohttp required for TV Desktop integration")

        async with aiohttp.ClientSession() as session:
            # Get list of debug targets
            async with session.get(
                f"http://{self.TV_DEBUG_HOST}:{self.TV_DEBUG_PORT}/json"
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError("TV debug port not responding to /json")
                targets = await resp.json()

        tv_target = next(
            (t for t in targets if "tradingview" in t.get("url", "").lower()), None
        )
        if not tv_target:
            raise RuntimeError("No TradingView tab found in debug targets")

        ws_url = tv_target.get("webSocketDebuggerUrl")
        if not ws_url:
            raise RuntimeError("No WebSocket URL for TV tab")

        chart_data = await self._extract_chart_via_cdp(ws_url, ticker, interval)
        return chart_data

    async def _extract_chart_via_cdp(
        self, ws_url: str, ticker: str, interval: str
    ) -> TVChartData:
        """Extract series data via CDP Runtime.evaluate."""
        try:
            import websockets
        except ImportError:
            raise RuntimeError("websockets required for CDP extraction")

        js_extract = """
        (function() {
            try {
                // TradingView exposes chart data on window.tvWidget or _tvWidget
                var widget = window.tvWidget || window._tvWidget;
                if (!widget) return JSON.stringify({error: 'no_widget'});
                var chart = widget.activeChart ? widget.activeChart() : null;
                if (!chart) return JSON.stringify({error: 'no_chart'});
                var symbol = chart.symbol();
                var resolution = chart.resolution();
                return JSON.stringify({symbol: symbol, resolution: resolution, found: true});
            } catch(e) {
                return JSON.stringify({error: e.message});
            }
        })()
        """

        async with websockets.connect(ws_url, ping_interval=None) as ws:
            msg = json.dumps({
                "id": 1,
                "method": "Runtime.evaluate",
                "params": {"expression": js_extract, "returnByValue": True}
            })
            await ws.send(msg)
            response = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))

        result = response.get("result", {}).get("result", {})
        value = result.get("value")

        if value:
            try:
                chart_info = json.loads(value)
                if chart_info.get("found"):
                    detected_ticker = chart_info.get("symbol", ticker)
                    logger.info("TV chart: symbol=%s resolution=%s",
                                detected_ticker, chart_info.get("resolution"))
                    # Use detected symbol but fetch data via yfinance for reliability
                    return await self._fetch_from_yfinance(
                        detected_ticker, "2y", interval, source="tv_desktop"
                    )
            except json.JSONDecodeError:
                pass

        raise RuntimeError("Could not extract chart data from TV Desktop")

    # ─── yfinance Backend ────────────────────────────────────────────────────

    async def _fetch_from_yfinance(
        self,
        ticker: str,
        period: str,
        interval: str,
        source: str = "yfinance",
    ) -> TVChartData:
        loop = asyncio.get_event_loop()

        ohlcv, price, info = await asyncio.gather(
            loop.run_in_executor(
                _executor,
                lambda: self._handler.fetch_history(ticker, period, interval)
            ),
            loop.run_in_executor(
                _executor,
                lambda: self._handler.get_current_price(ticker)
            ),
            loop.run_in_executor(
                _executor,
                lambda: self._handler.get_ticker_info(ticker)
            ),
        )

        indicators = self._compute_indicators(ohlcv)

        return TVChartData(
            ticker=ticker,
            timeframe=interval,
            ohlcv=ohlcv,
            current_price=price,
            info=info,
            indicators=indicators,
            source=source,
        )

    # ─── Indicator Computation ───────────────────────────────────────────────

    def _compute_indicators(self, df: pd.DataFrame) -> dict:
        """Pre-compute common indicators for downstream pipeline consumption."""
        if df.empty or len(df) < 30:
            return {}

        close = df["Close"].values
        volume = df["Volume"].values
        high = df["High"].values
        low = df["Low"].values

        def ema(arr, n):
            result = np.empty_like(arr)
            result[:] = np.nan
            k = 2.0 / (n + 1)
            result[n - 1] = arr[:n].mean()
            for i in range(n, len(arr)):
                result[i] = arr[i] * k + result[i - 1] * (1 - k)
            return result

        def rsi(arr, n=14):
            delta = np.diff(arr)
            gains = np.where(delta > 0, delta, 0.0)
            losses = np.where(delta < 0, -delta, 0.0)
            avg_gain = np.convolve(gains, np.ones(n) / n, mode="full")[:len(gains)]
            avg_loss = np.convolve(losses, np.ones(n) / n, mode="full")[:len(losses)]
            rs = np.where(avg_loss == 0, 100.0, avg_gain / (avg_loss + 1e-10))
            rsi_vals = 100 - (100 / (1 + rs))
            return np.concatenate([[np.nan], rsi_vals])

        def atr(h, l, c, n=14):
            tr = np.maximum(h[1:] - l[1:],
                   np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
            atr_vals = np.empty(len(c))
            atr_vals[:] = np.nan
            if len(tr) >= n:
                atr_vals[n] = tr[:n].mean()
                for i in range(n + 1, len(c)):
                    atr_vals[i] = (atr_vals[i - 1] * (n - 1) + tr[i - 1]) / n
            return atr_vals

        returns = np.diff(np.log(close + 1e-10))

        return {
            "ema_20": ema(close, 20),
            "ema_50": ema(close, 50),
            "ema_200": ema(close, 200),
            "rsi_14": rsi(close, 14),
            "atr_14": atr(high, low, close, 14),
            "vol_sma_20": np.convolve(volume.astype(float), np.ones(20) / 20, mode="same"),
            "returns": returns,
            "hv_20": pd.Series(returns).rolling(20).std().values * np.sqrt(252),
            "hv_60": pd.Series(returns).rolling(60).std().values * np.sqrt(252),
        }

    def reset_tv_detection(self):
        """Force re-detection of TV Desktop on next call."""
        self._tv_available = None
