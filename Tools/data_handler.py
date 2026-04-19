"""
Market Data Handler
--------------------
Fetches historical price data and options chains via yfinance.
Computes historical/realised volatility and risk-free rate proxy.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Any, Optional, Tuple

from .persistent_cache import PersistentTTLCache, market_cache_settings


class MarketDataHandler:
    """Wrapper around yfinance with caching and volatility utilities."""

    def __init__(
        self,
        cache_ttl_minutes: Optional[int] = None,
        intraday_cache_ttl_minutes: Optional[int] = None,
        metadata_ttl_minutes: Optional[int] = None,
        options_ttl_minutes: Optional[int] = None,
        persistent_cache_dir: Optional[str | Path] = None,
        use_disk_cache: Optional[bool] = None,
        config: Optional[dict] = None,
    ):
        settings = market_cache_settings(config)
        if persistent_cache_dir is not None:
            settings["dir"] = Path(persistent_cache_dir)
        if use_disk_cache is not None:
            settings["enabled"] = bool(use_disk_cache)

        self._cache: dict = {}
        self._cache_ttl = pd.Timedelta(minutes=cache_ttl_minutes if cache_ttl_minutes is not None else settings["history_ttl_minutes"])
        self._intraday_ttl = pd.Timedelta(
            minutes=intraday_cache_ttl_minutes if intraday_cache_ttl_minutes is not None else settings["intraday_ttl_minutes"]
        )
        self._metadata_ttl = pd.Timedelta(
            minutes=metadata_ttl_minutes if metadata_ttl_minutes is not None else settings["metadata_ttl_minutes"]
        )
        self._options_ttl = pd.Timedelta(
            minutes=options_ttl_minutes if options_ttl_minutes is not None else settings["options_ttl_minutes"]
        )
        self._disk_cache = PersistentTTLCache(settings["dir"]) if settings["enabled"] else None

    @staticmethod
    def _normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        normalized = df.copy()
        normalized.index = pd.to_datetime(normalized.index)
        if getattr(normalized.index, "tz", None) is not None:
            normalized.index = normalized.index.tz_localize(None)
        return normalized[~normalized.index.duplicated(keep="last")].sort_index()

    @staticmethod
    def _disk_key(*parts: Any) -> str:
        return "::".join(map(str, parts))

    def _read_cache(self, key: tuple, ttl: pd.Timedelta):
        if key in self._cache:
            value, ts = self._cache[key]
            if pd.Timestamp.now() - ts < ttl:
                return value

        if self._disk_cache is not None:
            value = self._disk_cache.get(
                self._disk_key(*key),
                ttl_seconds=max(int(ttl.total_seconds()), 0),
            )
            if value is not None:
                self._cache[key] = (value, pd.Timestamp.now())
                return value

        return None

    def _write_cache(self, key: tuple, value: Any) -> None:
        self._cache[key] = (value, pd.Timestamp.now())
        if self._disk_cache is not None:
            self._disk_cache.set(self._disk_key(*key), value)

    def cache_stats(self) -> dict:
        disk_stats = self._disk_cache.stats() if self._disk_cache is not None else {
            "enabled": False,
            "path": None,
            "entry_count": 0,
            "size_bytes": 0,
            "size_mb": 0.0,
        }
        disk_stats["memory_entries"] = len(self._cache)
        return disk_stats

    def clear_cache(self, persist: bool = True) -> None:
        self._cache.clear()
        if persist and self._disk_cache is not None:
            self._disk_cache.clear()

    # ── price data ────────────────────────────────────────────────────────────

    def fetch_history(
        self, ticker: str, period: str = "2y", interval: str = "1d"
    ) -> pd.DataFrame:
        key = ("history", ticker.upper(), period, interval)
        cached = self._read_cache(key, self._cache_ttl)
        if isinstance(cached, pd.DataFrame):
            return cached

        try:
            t = yf.Ticker(ticker)
            df = self._normalize_price_frame(
                t.history(period=period, interval=interval, auto_adjust=True)
            )
            self._write_cache(key, df)
            return df
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch data for {ticker}: {exc}")

    def get_current_price(self, ticker: str) -> float:
        df = self.fetch_history(ticker, period="5d", interval="1d")
        if df.empty:
            raise RuntimeError(f"No data for {ticker}")
        # Use last *non-NaN* close — yfinance often appends a partial/empty row
        # for the current trading day on NSE/BSE tickers
        close_clean = df["Close"].dropna()
        if close_clean.empty:
            raise RuntimeError(f"No valid close prices for {ticker}")
        return float(close_clean.iloc[-1])

    def get_ticker_info(self, ticker: str) -> dict:
        key = ("info", ticker.upper())
        cached = self._read_cache(key, self._metadata_ttl)
        if isinstance(cached, dict):
            return dict(cached)

        try:
            t = yf.Ticker(ticker)
            info = t.info
            result = {
                "name": info.get("longName", ticker),
                "sector": info.get("sector", "N/A"),
                "currency": info.get("currency", "USD"),
                "dividend_yield": info.get("dividendYield", 0.0) or 0.0,
                "beta": info.get("beta", None),
                "market_cap": info.get("marketCap", None),
            }
            self._write_cache(key, result)
            return result
        except Exception:
            return {"name": ticker, "sector": "N/A", "currency": "USD",
                    "dividend_yield": 0.0, "beta": None, "market_cap": None}

    # ── volatility ────────────────────────────────────────────────────────────

    def historical_volatility(
        self, ticker: str, window: int = 30, period: str = "2y"
    ) -> pd.Series:
        """Annualised rolling historical volatility (close-to-close log returns)."""
        df = self.fetch_history(ticker, period=period)
        log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        hvol = log_ret.rolling(window).std() * np.sqrt(252)
        return hvol.dropna()

    def current_hv(self, ticker: str, window: int = 30) -> float:
        hv = self.historical_volatility(ticker, window)
        return float(hv.iloc[-1]) if not hv.empty else 0.20

    # ── options chain ─────────────────────────────────────────────────────────

    def get_options_chain(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Returns a combined call+put DataFrame from the nearest expiry available.
        Columns: strike, lastPrice, bid, ask, impliedVolatility, volume, openInterest,
                 inTheMoney, optionType, expiration
        """
        key = ("options_chain", ticker.upper())
        cached = self._read_cache(key, self._options_ttl)
        if isinstance(cached, pd.DataFrame):
            return cached

        try:
            t = yf.Ticker(ticker)
            expirations = t.options
            if not expirations:
                return None

            frames = []
            for exp in expirations[:6]:   # nearest 6 expiries
                try:
                    chain = t.option_chain(exp)
                    calls = chain.calls.copy()
                    calls["optionType"] = "call"
                    calls["expiration"] = exp

                    puts = chain.puts.copy()
                    puts["optionType"] = "put"
                    puts["expiration"] = exp

                    frames.extend([calls, puts])
                except Exception:
                    continue

            if not frames:
                return None

            combined = pd.concat(frames, ignore_index=True)
            combined = combined[combined["impliedVolatility"] > 0].copy()
            combined["expiration"] = pd.to_datetime(combined["expiration"])
            self._write_cache(key, combined)
            return combined
        except Exception:
            return None

    def implied_vol_surface(
        self, ticker: str, S: float
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Returns (strikes, expirations_in_days, iv_matrix) for surface plotting.
        iv_matrix shape: (n_strikes, n_expiries)
        """
        chain = self.get_options_chain(ticker)
        if chain is None:
            return None

        today = pd.Timestamp.today()
        chain["dte"] = (chain["expiration"] - today).dt.days
        chain = chain[(chain["dte"] > 0) & (chain["optionType"] == "call")]
        chain = chain[(chain["impliedVolatility"] > 0.01) & (chain["impliedVolatility"] < 5.0)]

        if chain.empty:
            return None

        # Filter to ±40% moneyness
        chain = chain[
            (chain["strike"] >= S * 0.60) & (chain["strike"] <= S * 1.40)
        ]

        pivot = chain.pivot_table(
            values="impliedVolatility", index="strike", columns="dte", aggfunc="mean"
        )
        pivot = pivot.dropna(thresh=2)
        if pivot.empty:
            return None

        strikes = pivot.index.values.astype(float)
        dtes = pivot.columns.values.astype(float)
        iv_matrix = pivot.values

        return strikes, dtes, iv_matrix

    def implied_vol_surface_history(
        self,
        hist_prices: pd.DataFrame,
        n_frames: int = 60,
        hv_window: int = 21,
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Build a time-series of IV surfaces from historical closing prices.

        Uses a SVI-inspired parametric model:
          IV(k, T) = atm_vol * sqrt(1 + skew*k + kurtosis*k^2) * sqrt(term_adj)
        where k = log(K/S) is log-moneyness, calibrated each day to the
        rolling historical volatility (21-day window by default).

        Returns
        -------
        strikes_grid : np.ndarray  (n_strikes,)  — moneyness labels (K/S ratio)
        dtes_grid    : np.ndarray  (n_dtes,)     — days to expiry
        frames       : list of (date_str, iv_matrix)
                       iv_matrix shape: (n_strikes, n_dtes), values 0..1
        """
        closes = hist_prices["Close"].dropna()
        if len(closes) < hv_window + 10:
            return None

        log_rets = np.log(closes / closes.shift(1)).dropna()

        # Rolling annualised realised vol
        rv = log_rets.rolling(hv_window).std() * np.sqrt(252)
        rv = rv.dropna()

        # Fixed grid: moneyness (K/S) and DTE
        moneyness = np.linspace(0.80, 1.20, 21)   # ±20% around spot
        dtes_grid = np.array([7, 14, 21, 30, 45, 60, 90, 120, 180], dtype=float)

        # Sample at most n_frames evenly-spaced dates
        step  = max(1, len(rv) // n_frames)
        dates = rv.index[::step][-n_frames:]

        frames = []
        for date in dates:
            atm  = float(rv.loc[date])
            if np.isnan(atm) or atm <= 0:
                continue

            # SVI-lite: equity skew ≈ -0.3/sqrt(T), smile curvature ≈ 0.1
            iv_mat = np.zeros((len(moneyness), len(dtes_grid)))
            for j, dte in enumerate(dtes_grid):
                T = max(dte / 252, 1e-4)
                # Term structure: slight upward slope (VIX mean-reversion)
                term_scale = 1.0 + 0.08 * np.log(max(T * 4, 1))
                for i, m in enumerate(moneyness):
                    k = np.log(m)           # log-moneyness
                    skew     = -0.25 / np.sqrt(T)   # equity skew
                    smile    = 0.10 / T             # smile curvature (flatter for long DTE)
                    iv_mat[i, j] = atm * term_scale * np.sqrt(
                        max(1.0 + skew * k + smile * k ** 2, 0.05)
                    )
                    iv_mat[i, j] = np.clip(iv_mat[i, j], 0.01, 2.0)

            frames.append((date.strftime("%Y-%m-%d"), iv_mat))

        if not frames:
            return None

        return moneyness, dtes_grid, frames

    # ── risk-free rate ────────────────────────────────────────────────────────

    @staticmethod
    def get_risk_free_rate() -> float:
        """
        Attempt to fetch 3-month US T-bill yield from yfinance (^IRX).
        Falls back to 4.5% if unavailable.
        """
        try:
            irx = yf.Ticker("^IRX")
            df = irx.history(period="5d")
            if not df.empty:
                rate = float(df["Close"].iloc[-1]) / 100.0
                return rate
        except Exception:
            pass
        return 0.045  # fallback

    # ── intraday data ─────────────────────────────────────────────────────────

    def fetch_intraday(
        self, ticker: str, interval: str = "5m", period: str = "5d"
    ) -> pd.DataFrame:
        """
        Fetch intraday OHLCV data.
        interval options: '1m','2m','5m','15m','30m','60m','90m','1h'
        period options: '1d','2d','5d','7d','1mo'
        """
        key = ("intraday", ticker.upper(), interval, period)
        cached = self._read_cache(key, self._intraday_ttl)
        if isinstance(cached, pd.DataFrame):
            return cached
        try:
            t = yf.Ticker(ticker)
            df = self._normalize_price_frame(
                t.history(period=period, interval=interval, auto_adjust=True)
            )
            self._write_cache(key, df)
            return df
        except Exception as exc:
            raise RuntimeError(f"Failed intraday fetch for {ticker}: {exc}")

    # ── volatility cone ───────────────────────────────────────────────────────

    def volatility_cone(
        self, ticker: str, windows: list = None, period: str = "3y"
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with vol percentiles for each rolling window.
        Rows = windows, Cols = [min, 10th, 25th, median, 75th, 90th, max, current]
        """
        if windows is None:
            windows = [5, 10, 20, 30, 60, 90, 120, 180, 252]
        df = self.fetch_history(ticker, period=period)
        log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()

        rows = []
        for w in windows:
            if len(log_ret) < w + 5:
                continue
            rv = log_ret.rolling(w).std() * np.sqrt(252)
            rv = rv.dropna()
            rows.append({
                "Window": w,
                "Min":    float(rv.min()),
                "10th":   float(rv.quantile(0.10)),
                "25th":   float(rv.quantile(0.25)),
                "Median": float(rv.median()),
                "75th":   float(rv.quantile(0.75)),
                "90th":   float(rv.quantile(0.90)),
                "Max":    float(rv.max()),
                "Current": float(rv.iloc[-1]),
            })
        return pd.DataFrame(rows)

    def multi_window_hv(
        self, ticker: str, windows: list = None, period: str = "2y"
    ) -> pd.DataFrame:
        """Returns DataFrame with rolling HV series for multiple windows."""
        if windows is None:
            windows = [10, 20, 30, 60, 90]
        df = self.fetch_history(ticker, period=period)
        log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()

        result = pd.DataFrame(index=df.index)
        result["Close"] = df["Close"]
        for w in windows:
            result[f"HV{w}"] = log_ret.rolling(w).std() * np.sqrt(252)
        return result.dropna(how="all")

    # ── returns / stats ───────────────────────────────────────────────────────

    @staticmethod
    def compute_returns(prices: pd.Series) -> pd.Series:
        return np.log(prices / prices.shift(1)).dropna()

    @staticmethod
    def annualised_stats(ticker_df: pd.DataFrame) -> dict:
        log_ret = np.log(ticker_df["Close"] / ticker_df["Close"].shift(1)).dropna()
        mean_ret = float(log_ret.mean() * 252)
        vol = float(log_ret.std() * np.sqrt(252))
        sharpe = mean_ret / vol if vol > 0 else np.nan
        skew = float(log_ret.skew())
        kurt = float(log_ret.kurtosis())
        max_dd = float((ticker_df["Close"] / ticker_df["Close"].cummax() - 1).min())
        return {
            "annual_return": mean_ret, "annual_vol": vol,
            "sharpe": sharpe, "skewness": skew,
            "kurtosis": kurt, "max_drawdown": max_dd,
        }
