"""
ml_volatility.py
================
Machine-learning-based implied volatility predictor.

Uses a Random Forest + XGBoost ensemble trained on synthetically generated
options data (bootstrapped from historical returns) to predict implied
volatility and generate OVERPRICED / FAIR / UNDERPRICED signals.

Dependencies: scikit-learn >= 1.3, xgboost >= 2.0, numpy, pandas
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

# ── Constants ─────────────────────────────────────────────────────────────────
TRADING_DAYS = 252
SIGNAL_THRESHOLD_STRONG = 0.25    # ±25% IV deviation = strong signal
SIGNAL_THRESHOLD_MODERATE = 0.10  # ±10% IV deviation = moderate signal
MIN_TRAINING_SAMPLES = 200
RF_N_ESTIMATORS = 200
XGB_N_ROUNDS = 300
XGB_LEARNING_RATE = 0.05
ENSEMBLE_RF_WEIGHT = 0.45
ENSEMBLE_XGB_WEIGHT = 0.55


# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class MLPrediction:
    """Output of the IVPredictor."""
    predicted_iv: float                 # ensemble IV prediction
    rf_iv: float                        # Random Forest prediction
    xgb_iv: float                       # XGBoost prediction
    confidence_interval: tuple[float, float]   # 10th–90th pct from RF leaf variance
    current_iv: float                   # current market IV for comparison
    iv_premium: float                   # current_iv - predicted_iv
    iv_premium_pct: float               # relative premium (%)
    signal: str                         # "OVERPRICED", "FAIR", "UNDERPRICED"
    signal_strength: str                # "STRONG", "MODERATE", "WEAK"
    feature_importance: dict[str, float]
    model_metrics: dict                 # rmse, mae, r2 on validation set


@dataclass
class TrainingResult:
    """Diagnostics from the training run."""
    n_samples: int
    features: list[str]
    rf_metrics: dict
    xgb_metrics: dict
    ensemble_metrics: dict
    val_predictions: np.ndarray
    val_actuals: np.ndarray


# ── Feature engineering ───────────────────────────────────────────────────────
class IVFeatureBuilder:
    """Builds ML feature matrix for IV prediction."""

    FEATURE_NAMES = [
        "moneyness",          # S / K
        "log_moneyness",      # ln(S/K)
        "dte_norm",           # DTE / 365
        "sqrt_dte",           # sqrt(DTE / 365)
        "hv10",               # 10-day historical vol
        "hv20",               # 20-day historical vol
        "hv30",               # 30-day historical vol
        "hv60",               # 60-day historical vol
        "hv_ratio_10_30",     # hv10 / hv30 (short-term vs medium)
        "hv_ratio_30_60",     # hv30 / hv60
        "skewness_20d",       # 20-day return skewness
        "kurtosis_20d",       # 20-day return kurtosis
        "abs_log_moneyness",  # |ln(S/K)|
        "otm_flag",           # 1 if OTM, 0 if ITM
        "is_call",            # 1 = call, 0 = put
    ]

    @classmethod
    def build(cls, S: float, K: float, dte: int, option_type: str,
              hist_prices: pd.Series) -> pd.DataFrame:
        """
        Build a single-row feature DataFrame.

        Parameters
        ----------
        S : float
            Current spot price.
        K : float
            Strike price.
        dte : int
            Days to expiry.
        option_type : str
            'call' or 'put'.
        hist_prices : pd.Series
            Historical closing prices (at least 60 days).
        """
        T = max(dte / 365.0, 1 / 365)
        log_rets = np.log(hist_prices / hist_prices.shift(1)).dropna()

        def _rolling_vol(n: int) -> float:
            if len(log_rets) < n:
                return float(log_rets.std() * np.sqrt(TRADING_DAYS))
            return float(log_rets.tail(n).std() * np.sqrt(TRADING_DAYS))

        hv10 = _rolling_vol(10)
        hv20 = _rolling_vol(20)
        hv30 = _rolling_vol(30)
        hv60 = _rolling_vol(60)

        tail_20 = log_rets.tail(20)
        skew_20 = float(tail_20.skew()) if len(tail_20) > 3 else 0.0
        kurt_20 = float(tail_20.kurtosis()) if len(tail_20) > 3 else 0.0

        moneyness = S / K
        log_m = np.log(moneyness)
        is_call = 1.0 if option_type.lower() == "call" else 0.0
        # OTM: call OTM if S<K, put OTM if S>K
        otm = 1.0 if (is_call and S < K) or (not is_call and S > K) else 0.0

        row = {
            "moneyness":        moneyness,
            "log_moneyness":    log_m,
            "dte_norm":         T,
            "sqrt_dte":         np.sqrt(T),
            "hv10":             hv10,
            "hv20":             hv20,
            "hv30":             hv30,
            "hv60":             hv60,
            "hv_ratio_10_30":   hv10 / max(hv30, 1e-6),
            "hv_ratio_30_60":   hv30 / max(hv60, 1e-6),
            "skewness_20d":     skew_20,
            "kurtosis_20d":     kurt_20,
            "abs_log_moneyness": abs(log_m),
            "otm_flag":         otm,
            "is_call":          is_call,
        }
        return pd.DataFrame([row])

    @classmethod
    def generate_synthetic(cls, hist_prices: pd.Series,
                            n_samples: int = 2000,
                            seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic training data by bootstrapping historical returns.

        For each synthetic sample:
        1. Pick random moneyness and DTE
        2. Compute historical vol from a random window
        3. Add realistic IV = HV + noise + smile adjustment
        4. Build feature row

        Returns (X, y) where y = implied volatility target.
        """
        rng = np.random.default_rng(seed)
        log_rets = np.log(hist_prices / hist_prices.shift(1)).dropna().values

        if len(log_rets) < 30:
            raise ValueError("Need at least 30 price observations for synthetic training.")

        rows, targets = [], []
        for _ in range(n_samples):
            # Random option parameters
            log_m    = rng.uniform(-0.25, 0.25)
            moneyness = np.exp(log_m)
            K         = 100.0
            S         = K * moneyness
            dte       = int(rng.integers(5, 365))
            T         = dte / 365.0
            is_call   = rng.random() > 0.5
            otm       = (is_call and S < K) or (not is_call and S > K)

            # Historical vols from random window
            n60  = min(60, len(log_rets))
            n30  = min(30, len(log_rets))
            n20  = min(20, len(log_rets))
            n10  = min(10, len(log_rets))
            start = rng.integers(0, max(1, len(log_rets) - n60))
            window = log_rets[start: start + n60]

            hv10 = float(window[-n10:].std() * np.sqrt(TRADING_DAYS)) if len(window) >= n10 else 0.20
            hv20 = float(window[-n20:].std() * np.sqrt(TRADING_DAYS)) if len(window) >= n20 else 0.20
            hv30 = float(window[-n30:].std() * np.sqrt(TRADING_DAYS)) if len(window) >= n30 else 0.20
            hv60 = float(window.std() * np.sqrt(TRADING_DAYS)) if len(window) >= 10 else 0.20
            skew = float(pd.Series(window[-n20:]).skew()) if len(window) >= n20 else 0.0
            kurt = float(pd.Series(window[-n20:]).kurtosis()) if len(window) >= n20 else 0.0

            # Synthetic IV = HV30 + noise + smile term + term structure
            smile_adj   = 0.03 * abs(log_m)           # vol smile: higher IV away from ATM
            skew_adj    = -0.02 * log_m * np.sign(-0.7)  # negative skew typical
            term_adj    = 0.01 * (1 - np.sqrt(T))     # short-dated vol premium
            noise       = rng.normal(0, 0.015)         # market noise
            iv = max(0.05, hv30 + smile_adj + skew_adj + term_adj + noise)

            row = {
                "moneyness":         moneyness,
                "log_moneyness":     log_m,
                "dte_norm":          T,
                "sqrt_dte":          np.sqrt(T),
                "hv10":              hv10,
                "hv20":              hv20,
                "hv30":              hv30,
                "hv60":              hv60,
                "hv_ratio_10_30":    hv10 / max(hv30, 1e-6),
                "hv_ratio_30_60":    hv30 / max(hv60, 1e-6),
                "skewness_20d":      skew,
                "kurtosis_20d":      kurt,
                "abs_log_moneyness": abs(log_m),
                "otm_flag":          1.0 if otm else 0.0,
                "is_call":           1.0 if is_call else 0.0,
            }
            rows.append(row)
            targets.append(iv)

        X = pd.DataFrame(rows)
        y = pd.Series(targets, name="implied_vol")
        return X, y


# ── Main predictor class ──────────────────────────────────────────────────────
class IVPredictor:
    """
    Ensemble IV predictor: Random Forest + XGBoost.

    Usage
    -----
    predictor = IVPredictor()
    predictor.train(hist_prices)          # builds synthetic training data
    result = predictor.predict(S, K, dte, option_type, hist_prices, current_iv)
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rf = None
        self._xgb = None
        self._trained = False
        self._feature_names = IVFeatureBuilder.FEATURE_NAMES
        self._training_result: Optional[TrainingResult] = None

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, hist_prices: pd.Series,
              n_samples: int = 2000) -> TrainingResult:
        """
        Train on synthetically generated options data.

        Parameters
        ----------
        hist_prices : pd.Series
            Historical closing prices used to bootstrap returns.
        n_samples : int
            Number of synthetic training samples.
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import xgboost as xgb

        X, y = IVFeatureBuilder.generate_synthetic(hist_prices, n_samples, self.seed)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.seed)

        # Random Forest
        self._rf = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=12,
            min_samples_leaf=3,
            random_state=self.seed,
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._rf.fit(X_train, y_train)

        # XGBoost
        self._xgb = xgb.XGBRegressor(
            n_estimators=XGB_N_ROUNDS,
            learning_rate=XGB_LEARNING_RATE,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.seed,
            verbosity=0,
            tree_method="hist",
        )
        self._xgb.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)

        self._trained = True

        # Validation metrics
        def _metrics(y_true, y_pred):
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae  = float(mean_absolute_error(y_true, y_pred))
            r2   = float(r2_score(y_true, y_pred))
            da   = float(np.mean(np.sign(y_pred - y_true.mean()) ==
                                  np.sign(y_true - y_true.mean())))
            return {"rmse": rmse, "mae": mae, "r2": r2, "directional_acc": da}

        rf_pred  = self._rf.predict(X_val)
        xgb_pred = self._xgb.predict(X_val)
        ens_pred = ENSEMBLE_RF_WEIGHT * rf_pred + ENSEMBLE_XGB_WEIGHT * xgb_pred

        self._training_result = TrainingResult(
            n_samples=len(X_train),
            features=self._feature_names,
            rf_metrics=_metrics(y_val, rf_pred),
            xgb_metrics=_metrics(y_val, xgb_pred),
            ensemble_metrics=_metrics(y_val, ens_pred),
            val_predictions=ens_pred,
            val_actuals=y_val.values,
        )
        return self._training_result

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict(self, S: float, K: float, dte: int, option_type: str,
                hist_prices: pd.Series, current_iv: float) -> MLPrediction:
        """
        Predict implied volatility and generate trading signal.

        Parameters
        ----------
        S : float  — current spot price
        K : float  — strike price
        dte : int  — days to expiry
        option_type : str  — 'call' or 'put'
        hist_prices : pd.Series  — historical closing prices
        current_iv : float  — current market implied volatility (decimal)
        """
        if not self._trained:
            self.train(hist_prices)

        X = IVFeatureBuilder.build(S, K, dte, option_type, hist_prices)

        rf_pred  = float(self._rf.predict(X)[0])
        xgb_pred = float(self._xgb.predict(X)[0])
        ensemble = ENSEMBLE_RF_WEIGHT * rf_pred + ENSEMBLE_XGB_WEIGHT * xgb_pred
        ensemble = max(0.01, ensemble)

        # Confidence interval via RF leaf variance
        rf_trees = self._rf.estimators_
        tree_preds = np.array([t.predict(X)[0] for t in rf_trees])
        ci_low  = float(np.percentile(tree_preds, 10))
        ci_high = float(np.percentile(tree_preds, 90))

        # Signal
        iv_premium     = current_iv - ensemble
        iv_premium_pct = iv_premium / max(ensemble, 1e-6)

        if abs(iv_premium_pct) >= SIGNAL_THRESHOLD_STRONG:
            strength = "STRONG"
        elif abs(iv_premium_pct) >= SIGNAL_THRESHOLD_MODERATE:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        if iv_premium_pct >= SIGNAL_THRESHOLD_MODERATE:
            signal = "OVERPRICED"
        elif iv_premium_pct <= -SIGNAL_THRESHOLD_MODERATE:
            signal = "UNDERPRICED"
        else:
            signal = "FAIR"

        # Feature importance (RF — more interpretable)
        importance = dict(zip(
            self._feature_names,
            self._rf.feature_importances_.tolist()
        ))

        metrics = self._training_result.ensemble_metrics if self._training_result else {}

        return MLPrediction(
            predicted_iv=ensemble,
            rf_iv=rf_pred,
            xgb_iv=xgb_pred,
            confidence_interval=(ci_low, ci_high),
            current_iv=current_iv,
            iv_premium=iv_premium,
            iv_premium_pct=iv_premium_pct * 100,
            signal=signal,
            signal_strength=strength,
            feature_importance=importance,
            model_metrics=metrics,
        )

    # ── Backtest ──────────────────────────────────────────────────────────────
    def walk_forward_report(self) -> pd.DataFrame:
        """
        Return validation set predicted vs actual as a DataFrame for plotting.
        """
        if self._training_result is None:
            return pd.DataFrame()
        tr = self._training_result
        return pd.DataFrame({
            "actual_iv":    tr.val_actuals,
            "predicted_iv": tr.val_predictions,
            "error":        tr.val_predictions - tr.val_actuals,
        })

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def training_result(self) -> Optional[TrainingResult]:
        return self._training_result
