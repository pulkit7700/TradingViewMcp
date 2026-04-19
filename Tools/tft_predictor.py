"""
tft_predictor.py
================
Multi-horizon quantile forecaster for stock price and volatility.

Primary: Quantile Regression Forest (sklearn) — production-ready, no GPU needed.
Architecture: Inspired by Temporal Fusion Transformer (Lim et al., 2021) feature
              engineering with variable selection and multi-horizon targets.

Outputs quantile forecasts (10th, 50th, 90th percentile) at horizons
1, 5, 10, 20 trading days ahead, with feature importance from the forest.

Dependencies: scikit-learn >= 1.3, numpy, pandas, ta (technical indicators)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
TRADING_DAYS   = 252
HORIZONS       = [1, 5, 10, 20]    # forecast horizons in trading days
QUANTILES      = [0.10, 0.50, 0.90]
N_ESTIMATORS   = 300
MAX_DEPTH      = 12
MIN_SAMPLES    = 5
LOOKBACK_DAYS  = 60                 # input window for features
MIN_HISTORY    = 120                # minimum price history to train


# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class QuantileForecast:
    """Multi-horizon quantile forecast for a single target."""
    target: str                     # "return" or "volatility"
    horizons: list[int]
    q10: np.ndarray                 # 10th percentile per horizon
    q50: np.ndarray                 # median (point forecast) per horizon
    q90: np.ndarray                 # 90th percentile per horizon
    feature_importance: dict[str, float]
    train_metrics: dict             # mae, rmse per horizon on validation
    last_price: float
    # Derived price forecasts (from return forecasts)
    price_q10: Optional[np.ndarray] = None
    price_q50: Optional[np.ndarray] = None
    price_q90: Optional[np.ndarray] = None


@dataclass
class TFTPrediction:
    """Combined price + volatility multi-horizon forecast."""
    ticker: str
    last_price: float
    last_vol: float                  # realized vol (annualized)
    horizons: list[int]
    price_forecast: QuantileForecast
    vol_forecast: QuantileForecast
    attention_proxy: dict[str, float]   # feature importance as "attention" analogue
    forecast_date: str


# ── Technical indicator builder ───────────────────────────────────────────────
class FeatureBuilder:
    """
    Builds a rich feature matrix from OHLCV price history.

    Inspired by TFT variable selection — includes:
    - Price momentum (returns at multiple horizons)
    - Volatility features (rolling HV at multiple windows)
    - Technical indicators (RSI, MACD, Bollinger, ATR)
    - Calendar features (day of week, month)
    - Trend features (price relative to moving averages)
    """

    BASE_FEATURES = [
        "ret_1d", "ret_5d", "ret_10d", "ret_20d",
        "hv_10", "hv_20", "hv_30", "hv_60",
        "hv_ratio_10_30", "hv_trend",
        "rsi_14", "macd_signal",
        "bb_width", "bb_position",
        "price_vs_ma20", "price_vs_ma50",
        "volume_ratio",
        "skew_20d", "kurt_20d",
        "day_of_week", "month",
        "days_since_high_52w", "drawdown_from_high",
    ]

    @classmethod
    def build(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build feature matrix from OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: Open, High, Low, Close, Volume.
            Index should be DatetimeIndex, sorted ascending.

        Returns
        -------
        pd.DataFrame
            One row per day, columns = feature names.
            Rows with NaN are dropped.
        """
        df = df.copy().sort_index()
        close = df["Close"]
        volume = df["Volume"] if "Volume" in df.columns else pd.Series(1.0, index=df.index)

        log_ret = np.log(close / close.shift(1))

        features = pd.DataFrame(index=df.index)

        # ── Return momentum ───────────────────────────────────────────────
        features["ret_1d"]  = log_ret
        features["ret_5d"]  = np.log(close / close.shift(5))
        features["ret_10d"] = np.log(close / close.shift(10))
        features["ret_20d"] = np.log(close / close.shift(20))

        # ── Realised volatility ───────────────────────────────────────────
        def _rv(n):
            return log_ret.rolling(n).std() * np.sqrt(TRADING_DAYS)

        features["hv_10"] = _rv(10)
        features["hv_20"] = _rv(20)
        features["hv_30"] = _rv(30)
        features["hv_60"] = _rv(60)
        features["hv_ratio_10_30"] = features["hv_10"] / features["hv_30"].clip(1e-6)
        features["hv_trend"] = features["hv_10"] - features["hv_30"]  # vol term structure slope

        # ── RSI (14) ──────────────────────────────────────────────────────
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.clip(1e-6)
        features["rsi_14"] = 100 - (100 / (1 + rs))

        # ── MACD signal ───────────────────────────────────────────────────
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        features["macd_signal"] = (macd - signal_line) / close.clip(1e-6)  # normalised

        # ── Bollinger Bands ───────────────────────────────────────────────
        ma20 = close.rolling(20).mean()
        sd20 = close.rolling(20).std()
        bb_upper = ma20 + 2 * sd20
        bb_lower = ma20 - 2 * sd20
        features["bb_width"]    = (bb_upper - bb_lower) / ma20.clip(1e-6)
        features["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-6)

        # ── Trend (price vs MAs) ──────────────────────────────────────────
        ma50 = close.rolling(50).mean()
        features["price_vs_ma20"] = close / ma20.clip(1e-6) - 1
        features["price_vs_ma50"] = close / ma50.clip(1e-6) - 1

        # ── Volume ratio ──────────────────────────────────────────────────
        vol_ma20 = volume.rolling(20).mean()
        features["volume_ratio"] = volume / vol_ma20.clip(1e-6)

        # ── Higher moments ────────────────────────────────────────────────
        features["skew_20d"] = log_ret.rolling(20).skew()
        features["kurt_20d"] = log_ret.rolling(20).kurt()

        # ── Calendar ──────────────────────────────────────────────────────
        if hasattr(df.index, "dayofweek"):
            features["day_of_week"] = df.index.dayofweek / 4.0   # normalise 0-1
            features["month"]       = df.index.month / 12.0
        else:
            features["day_of_week"] = 0.0
            features["month"]       = 0.0

        # ── 52-week high / drawdown ───────────────────────────────────────
        high_52w = close.rolling(252).max()
        features["days_since_high_52w"] = (
            (close == high_52w).astype(int)
            .replace(0, np.nan)
            .ffill()
            .fillna(252)
            .apply(lambda x: 0 if x == 1 else x) / 252
        )
        features["drawdown_from_high"] = close / high_52w.clip(1e-6) - 1

        return features.dropna()

    @classmethod
    def build_targets(cls, df: pd.DataFrame,
                      horizons: list[int]) -> pd.DataFrame:
        """
        Build forward return and volatility targets.

        Returns DataFrame with columns:
        ret_h{h} and vol_h{h} for each horizon h.
        """
        close = df["Close"].sort_index()
        log_ret = np.log(close / close.shift(1))
        targets = pd.DataFrame(index=close.index)
        for h in horizons:
            # Forward log return over h days
            targets[f"ret_h{h}"] = np.log(close.shift(-h) / close)
            # Forward realised vol over h days
            targets[f"vol_h{h}"] = (
                log_ret.shift(-h)
                .rolling(h)
                .std()
                .shift(-h)
                * np.sqrt(TRADING_DAYS)
            )
        return targets


# ── Quantile Regression Forest wrapper ───────────────────────────────────────
class _QuantileForest:
    """
    Quantile Regression Forest using scikit-learn's RandomForestRegressor.

    Extracts leaf-node predictions from all trees to compute quantiles,
    giving proper distributional uncertainty estimates.
    """

    def __init__(self, n_estimators: int = N_ESTIMATORS,
                 max_depth: int = MAX_DEPTH,
                 min_samples_leaf: int = MIN_SAMPLES,
                 seed: int = 42):
        from sklearn.ensemble import RandomForestRegressor
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
        )
        self._leaf_values: Optional[np.ndarray] = None   # (n_trees, n_train)
        self._y_train: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X, y)
        # Store training leaf indices for quantile prediction
        self._leaf_indices_train = self._model.apply(X)  # (n_train, n_trees)
        self._y_train = y.values

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point prediction (mean)."""
        return self._model.predict(X)

    def predict_quantiles(self, X: pd.DataFrame,
                          quantiles: list[float] = QUANTILES) -> np.ndarray:
        """
        Predict quantiles via leaf-node empirical distribution.

        Returns array (n_samples, n_quantiles).
        """
        leaf_pred = self._model.apply(X)           # (n_pred, n_trees)
        leaf_train = self._leaf_indices_train      # (n_train, n_trees)
        y_train = self._y_train

        n_pred = X.shape[0]
        n_q = len(quantiles)
        result = np.zeros((n_pred, n_q))

        for i in range(n_pred):
            # For each tree, find training samples in same leaf
            leaf_samples = []
            for t in range(self._model.n_estimators):
                mask = leaf_train[:, t] == leaf_pred[i, t]
                leaf_samples.extend(y_train[mask].tolist())
            if leaf_samples:
                result[i] = np.quantile(leaf_samples, quantiles)
            else:
                pt = self._model.predict(X.iloc[[i]])[0]
                result[i] = [pt] * n_q

        return result

    @property
    def feature_importances_(self):
        return self._model.feature_importances_


# ── Main predictor ────────────────────────────────────────────────────────────
class TFTPredictor:
    """
    Multi-horizon stock price & volatility quantile forecaster.

    Architecture inspired by Temporal Fusion Transformer:
    - Variable selection via feature importance
    - Multi-horizon outputs (1, 5, 10, 20 days)
    - Quantile predictions (10th, 50th, 90th percentile)
    - Interpretable "attention" weights (feature importance proxy)

    Parameters
    ----------
    horizons : list[int]
        Forecast horizons in trading days.
    quantiles : list[float]
        Quantile levels to predict.
    seed : int
        Random seed.
    """

    def __init__(self, horizons: list[int] = None,
                 quantiles: list[float] = None,
                 seed: int = 42):
        self.horizons  = horizons  or HORIZONS
        self.quantiles = quantiles or QUANTILES
        self.seed = seed
        self._models_ret: dict[int, _QuantileForest] = {}   # horizon → model (returns)
        self._models_vol: dict[int, _QuantileForest] = {}   # horizon → model (vol)
        self._features: Optional[list[str]] = None
        self._trained = False
        self._val_metrics: dict = {}

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> dict:
        """
        Train one QRF per horizon per target (returns + vol).

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with DatetimeIndex, at least MIN_HISTORY rows.

        Returns
        -------
        dict
            Validation metrics per horizon.
        """
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        if len(df) < MIN_HISTORY:
            raise ValueError(
                f"Need at least {MIN_HISTORY} trading days of history, got {len(df)}.")

        X_full = FeatureBuilder.build(df)
        y_full = FeatureBuilder.build_targets(df, self.horizons)

        # Align indices
        common = X_full.index.intersection(y_full.index)
        X = X_full.loc[common]
        y = y_full.loc[common]
        self._features = X.columns.tolist()

        metrics = {}
        val_size = max(20, len(X) // 5)   # 20% hold-out (time-series split)

        X_train = X.iloc[:-val_size]
        X_val   = X.iloc[-val_size:]

        for h in self.horizons:
            ret_col = f"ret_h{h}"
            vol_col = f"vol_h{h}"

            # Drop NaN targets
            mask_r = y[ret_col].dropna().index.intersection(X_train.index)
            mask_rv = y[ret_col].dropna().index.intersection(X_val.index)

            if len(mask_r) < MIN_SAMPLES:
                continue

            # Return model
            _qrf_r = _QuantileForest(seed=self.seed)
            _qrf_r.fit(X_train.loc[mask_r], y[ret_col].loc[mask_r])
            self._models_ret[h] = _qrf_r

            # Volatility model
            mask_v = y[vol_col].dropna().index.intersection(X_train.index)
            if len(mask_v) >= MIN_SAMPLES:
                _qrf_v = _QuantileForest(seed=self.seed)
                _qrf_v.fit(X_train.loc[mask_v], y[vol_col].loc[mask_v])
                self._models_vol[h] = _qrf_v

            # Validation
            if len(mask_rv) > 0:
                val_y_r = y[ret_col].loc[mask_rv]
                val_pred_r = _qrf_r.predict(X_val.loc[mask_rv])
                mae  = float(mean_absolute_error(val_y_r, val_pred_r))
                rmse = float(np.sqrt(mean_squared_error(val_y_r, val_pred_r)))
                metrics[h] = {"mae": mae, "rmse": rmse}

        self._trained = True
        self._val_metrics = metrics
        return metrics

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame, ticker: str = "") -> TFTPrediction:
        """
        Generate multi-horizon quantile forecasts from the most recent data point.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame. Uses last row as the "current" observation.
        ticker : str
            Ticker symbol (for labelling only).

        Returns
        -------
        TFTPrediction
        """
        if not self._trained:
            self.train(df)

        X_full = FeatureBuilder.build(df)
        if X_full.empty:
            raise ValueError("Feature building produced an empty DataFrame.")

        X_last = X_full.iloc[[-1]]   # most recent observation
        last_price = float(df["Close"].iloc[-1])

        log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        last_vol = float(log_ret.tail(20).std() * np.sqrt(TRADING_DAYS))

        # ── Return forecast ───────────────────────────────────────────────
        ret_q = {q: [] for q in self.quantiles}
        for h in self.horizons:
            if h in self._models_ret:
                qpreds = self._models_ret[h].predict_quantiles(X_last, self.quantiles)[0]
                for i, q in enumerate(self.quantiles):
                    ret_q[q].append(float(qpreds[i]))
            else:
                for q in self.quantiles:
                    ret_q[q].append(0.0)

        ret_q10 = np.array(ret_q[0.10])
        ret_q50 = np.array(ret_q[0.50])
        ret_q90 = np.array(ret_q[0.90])

        # Convert log returns to price forecasts
        price_q10 = last_price * np.exp(ret_q10)
        price_q50 = last_price * np.exp(ret_q50)
        price_q90 = last_price * np.exp(ret_q90)

        # Feature importance (from h=5 return model as representative)
        _rep_h = 5 if 5 in self._models_ret else self.horizons[0]
        fi_vals = self._models_ret[_rep_h]._model.feature_importances_
        fi_dict = dict(zip(self._features, fi_vals.tolist()))
        fi_sorted = dict(sorted(fi_dict.items(), key=lambda x: x[1], reverse=True))

        price_fc = QuantileForecast(
            target="return",
            horizons=self.horizons,
            q10=ret_q10, q50=ret_q50, q90=ret_q90,
            feature_importance=fi_sorted,
            train_metrics=self._val_metrics,
            last_price=last_price,
            price_q10=price_q10,
            price_q50=price_q50,
            price_q90=price_q90,
        )

        # ── Volatility forecast ───────────────────────────────────────────
        vol_q = {q: [] for q in self.quantiles}
        for h in self.horizons:
            if h in self._models_vol:
                qpreds = self._models_vol[h].predict_quantiles(X_last, self.quantiles)[0]
                for i, q in enumerate(self.quantiles):
                    vol_q[q].append(max(0.01, float(qpreds[i])))
            else:
                for q in self.quantiles:
                    vol_q[q].append(last_vol)

        vol_fc = QuantileForecast(
            target="volatility",
            horizons=self.horizons,
            q10=np.array(vol_q[0.10]),
            q50=np.array(vol_q[0.50]),
            q90=np.array(vol_q[0.90]),
            feature_importance=fi_sorted,
            train_metrics=self._val_metrics,
            last_price=last_price,
        )

        from datetime import datetime, timezone
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

        return TFTPrediction(
            ticker=ticker,
            last_price=last_price,
            last_vol=last_vol,
            horizons=self.horizons,
            price_forecast=price_fc,
            vol_forecast=vol_fc,
            attention_proxy=fi_sorted,
            forecast_date=ts,
        )

    @property
    def is_trained(self) -> bool:
        return self._trained


# ── Standalone helpers ────────────────────────────────────────────────────────
def fan_chart_data(forecast: QuantileForecast,
                   use_prices: bool = True) -> dict:
    """
    Build data arrays for a multi-horizon fan chart.

    Returns
    -------
    dict with keys: horizons, q10, q50, q90, label, last_value.
    """
    if use_prices and forecast.price_q50 is not None:
        return {
            "horizons": forecast.horizons,
            "q10": forecast.price_q10.tolist(),
            "q50": forecast.price_q50.tolist(),
            "q90": forecast.price_q90.tolist(),
            "label": "Price ($)",
            "last_value": forecast.last_price,
        }
    return {
        "horizons": forecast.horizons,
        "q10": (forecast.q10 * 100).tolist(),
        "q50": (forecast.q50 * 100).tolist(),
        "q90": (forecast.q90 * 100).tolist(),
        "label": "Volatility (%)" if forecast.target == "volatility" else "Return (%)",
        "last_value": None,
    }


def attention_heatmap_data(prediction: TFTPrediction,
                            top_n: int = 10) -> dict:
    """
    Return top-N features and their importance scores for heatmap display.
    """
    fi = prediction.attention_proxy
    top_items = list(fi.items())[:top_n]
    return {
        "features": [k for k, _ in top_items],
        "importance": [v for _, v in top_items],
    }
