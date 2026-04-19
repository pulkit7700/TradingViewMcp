"""
Backtesting module — vectorized backtest on historical data.

Computes: Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor,
          Total Return, Calmar Ratio, equity curve.
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

from Tools.data_handler import MarketDataHandler

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2)
_handler = MarketDataHandler()


@dataclass
class BacktestResult:
    ticker: str
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    n_trades: int
    calmar_ratio: float
    equity_curve: list[float]
    trade_returns: list[float]
    metrics: dict

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "strategy": self.strategy_name,
            "total_return_pct": round(self.total_return * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "win_rate_pct": round(self.win_rate * 100, 2),
            "profit_factor": round(self.profit_factor, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "n_trades": self.n_trades,
            "equity_curve": [round(v, 4) for v in self.equity_curve[-200:]],  # last 200 points
            "metrics": self.metrics,
        }


class Backtester:
    """
    Vectorized backtester for strategy signals.

    Takes a StrategySignal + historical OHLCV data and simulates
    entry/exit on the strategy rules, using ATR-based stops.

    Signal-following logic (simplified for Pine parity):
    - Long entry: ema_fast > ema_slow AND rsi in [30, 70] AND volume surge
    - Short entry: ema_fast < ema_slow AND rsi in [30, 70] AND volume surge
    - Exit: ATR-based stop/TP OR signal reversal
    """

    def __init__(self, config: dict):
        self._cfg = config

    async def run(
        self,
        ticker: str,
        strategy_name: str,
        pine_params: dict,
        period: str = "2y",
    ) -> BacktestResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: self._run_sync(ticker, strategy_name, pine_params, period)
        )

    def _run_sync(
        self,
        ticker: str,
        strategy_name: str,
        pine_params: dict,
        period: str,
    ) -> BacktestResult:
        df = _handler.fetch_history(ticker, period=period, interval="1d")
        if df.empty or len(df) < 60:
            return self._empty_result(ticker, strategy_name)

        close = df["Close"].values.astype(float)
        high = df["High"].values.astype(float)
        low = df["Low"].values.astype(float)
        volume = df["Volume"].values.astype(float)

        ema_fast_n = pine_params.get("ema_fast", 20)
        ema_slow_n = pine_params.get("ema_slow", 50)
        rsi_len = pine_params.get("rsi_len", 14)
        atr_len = pine_params.get("atr_len", 14)
        atr_mult = pine_params.get("atr_mult", 2.0)
        vol_thresh = pine_params.get("vol_threshold", 1.5)
        direction = pine_params.get("direction", "LONG")

        # Compute indicators
        ema_f = self._ema(close, ema_fast_n)
        ema_s = self._ema(close, ema_slow_n)
        rsi = self._rsi(close, rsi_len)
        atr = self._atr(high, low, close, atr_len)
        vol_sma = self._sma(volume, 20)

        # Generate signals
        n = len(close)
        entries = np.zeros(n, dtype=int)
        exits = np.zeros(n, dtype=int)

        for i in range(max(ema_slow_n, rsi_len, atr_len) + 1, n):
            vol_surge = volume[i] > vol_sma[i] * vol_thresh
            rsi_ok = 30 < rsi[i] < 70
            long_cond = ema_f[i] > ema_s[i] and rsi_ok and vol_surge and close[i] > ema_f[i]
            short_cond = ema_f[i] < ema_s[i] and rsi_ok and vol_surge and close[i] < ema_f[i]

            if direction in ("LONG", "FLAT"):
                entries[i] = 1 if long_cond else 0
            if direction in ("SHORT", "FLAT"):
                entries[i] = -1 if short_cond else entries[i]

        # Simulate trades
        trade_returns = []
        equity = 1.0
        equity_curve = [1.0]
        in_trade = 0
        trade_entry = 0.0
        trade_stop = 0.0
        trade_tp = 0.0

        for i in range(1, n):
            if in_trade == 0:
                if entries[i] != 0:
                    in_trade = entries[i]
                    trade_entry = close[i]
                    trade_stop = close[i] - atr[i] * atr_mult * in_trade
                    trade_tp = close[i] + atr[i] * atr_mult * 2 * in_trade
            else:
                # Check stop/TP
                exit_price = None
                if in_trade == 1:
                    if low[i] <= trade_stop:
                        exit_price = trade_stop
                    elif high[i] >= trade_tp:
                        exit_price = trade_tp
                    elif entries[i] == -1:
                        exit_price = close[i]
                elif in_trade == -1:
                    if high[i] >= trade_stop:
                        exit_price = trade_stop
                    elif low[i] <= trade_tp:
                        exit_price = trade_tp
                    elif entries[i] == 1:
                        exit_price = close[i]

                if exit_price is not None:
                    ret = (exit_price - trade_entry) / trade_entry * in_trade
                    trade_returns.append(ret)
                    equity *= (1 + ret)
                    in_trade = 0

            equity_curve.append(equity)

        if not trade_returns:
            return self._empty_result(ticker, strategy_name)

        trade_returns = np.array(trade_returns)
        total_return = equity - 1.0

        # Metrics
        daily_eq = np.array(equity_curve)
        daily_ret = np.diff(daily_eq) / daily_eq[:-1]

        sharpe = float(np.mean(daily_ret) / (np.std(daily_ret) + 1e-10) * np.sqrt(252))

        roll_max = np.maximum.accumulate(daily_eq)
        drawdown = (daily_eq - roll_max) / (roll_max + 1e-10)
        max_dd = float(abs(drawdown.min()))
        calmar = total_return / (max_dd + 1e-6)

        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        win_rate = len(wins) / len(trade_returns)
        profit_factor = wins.sum() / (abs(losses.sum()) + 1e-10)

        return BacktestResult(
            ticker=ticker,
            strategy_name=strategy_name,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=float(profit_factor),
            n_trades=len(trade_returns),
            calmar_ratio=float(calmar),
            equity_curve=equity_curve,
            trade_returns=list(trade_returns),
            metrics={
                "avg_trade_return": float(np.mean(trade_returns)),
                "best_trade": float(np.max(trade_returns)),
                "worst_trade": float(np.min(trade_returns)),
                "avg_winner": float(np.mean(wins)) if len(wins) > 0 else 0.0,
                "avg_loser": float(np.mean(losses)) if len(losses) > 0 else 0.0,
            },
        )

    def _empty_result(self, ticker: str, strategy_name: str) -> BacktestResult:
        return BacktestResult(
            ticker=ticker,
            strategy_name=strategy_name,
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            n_trades=0,
            calmar_ratio=0.0,
            equity_curve=[1.0],
            trade_returns=[],
            metrics={},
        )

    # ─── Indicator helpers ───────────────────────────────────────────────────

    @staticmethod
    def _ema(arr: np.ndarray, n: int) -> np.ndarray:
        result = np.full(len(arr), np.nan)
        if len(arr) < n:
            return result
        k = 2.0 / (n + 1)
        result[n - 1] = arr[:n].mean()
        for i in range(n, len(arr)):
            result[i] = arr[i] * k + result[i - 1] * (1 - k)
        return result

    @staticmethod
    def _sma(arr: np.ndarray, n: int) -> np.ndarray:
        result = np.full(len(arr), np.nan)
        for i in range(n - 1, len(arr)):
            result[i] = arr[i - n + 1:i + 1].mean()
        return result

    @staticmethod
    def _rsi(arr: np.ndarray, n: int = 14) -> np.ndarray:
        delta = np.diff(arr)
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        result = np.full(len(arr), 50.0)
        if len(gains) < n:
            return result
        avg_g = gains[:n].mean()
        avg_l = losses[:n].mean()
        for i in range(n, len(delta)):
            avg_g = (avg_g * (n - 1) + gains[i]) / n
            avg_l = (avg_l * (n - 1) + losses[i]) / n
            rs = avg_g / (avg_l + 1e-10)
            result[i + 1] = 100 - (100 / (1 + rs))
        return result

    @staticmethod
    def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
        result = np.full(len(close), np.nan)
        if len(close) < n + 1:
            return result
        tr = np.maximum(high[1:] - low[1:],
               np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        result[n] = tr[:n].mean()
        for i in range(n + 1, len(close)):
            result[i] = (result[i - 1] * (n - 1) + tr[i - 1]) / n
        return result
