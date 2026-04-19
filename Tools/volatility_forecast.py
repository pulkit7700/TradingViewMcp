"""
GARCH Volatility Forecasting Engine
-------------------------------------
Fits GARCH, GJR-GARCH, and EGARCH models to a log-returns series,
generates multi-step volatility forecasts with confidence bands, and
provides comparison utilities against implied volatility.

Requires: arch >= 5.0  (pip install arch)
"""

from __future__ import annotations
import warnings

try:
    from arch import arch_model
    from arch.univariate.base import ARCHModelResult
except ImportError as _arch_err:
    raise ImportError(
        "The 'arch' package is required for GARCH forecasting. "
        "Install it with:  pip install arch"
    ) from _arch_err

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TRADING_DAYS: int = 252          # annualisation factor
MIN_OBSERVATIONS: int = 30       # minimum returns needed to fit a model
SIM_PATHS: int = 500             # number of Monte Carlo paths for conf. bands
RNG_SEED: int = 42               # reproducible random state

# IV comparison thresholds (percentage premium / discount)
THRESHOLD_STRONG: float = 0.35   # ±35 % → STRONG signal
THRESHOLD_MODERATE: float = 0.20 # ±20 % → MODERATE signal


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GARCHResult:
    """Container for a fitted GARCH-family model and its forecasts."""

    model_type: str                          # "GARCH", "GJR-GARCH", "EGARCH"
    params: dict                             # omega, alpha, beta, gamma (if applicable)
    aic: float
    bic: float
    log_likelihood: float
    conditional_vol: np.ndarray             # annualised in-sample conditional vol series
    forecast_vol: np.ndarray                # annualised vol forecast (length = horizon_days)
    forecast_lower: np.ndarray              # lower confidence band
    forecast_upper: np.ndarray              # upper confidence band
    horizon_days: int
    returns: np.ndarray                     # input returns (daily, not annualised)
    fitted_variance: np.ndarray             # in-sample fitted variance (daily)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class GARCHForecaster:
    """
    Fits GARCH-family models to a daily log-return series and produces
    annualised volatility forecasts with simulation-based confidence bands.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns (not annualised). Index should be date-like.
    """

    def __init__(self, returns: pd.Series) -> None:
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
        self._returns: pd.Series = returns.dropna()
        self._rng = np.random.default_rng(RNG_SEED)

    # ── private helpers ───────────────────────────────────────────────────────

    def _check_min_obs(self) -> None:
        """Raise ValueError if the series is too short to fit a GARCH model."""
        if len(self._returns) < MIN_OBSERVATIONS:
            raise ValueError(
                f"At least {MIN_OBSERVATIONS} observations are required; "
                f"got {len(self._returns)}."
            )

    @staticmethod
    def _annualise_vol(variance: np.ndarray) -> np.ndarray:
        """Convert daily variance to annualised volatility."""
        return np.sqrt(TRADING_DAYS * np.maximum(variance, 0.0))

    def _extract_result(
        self,
        fit: "ARCHModelResult",
        model_type: str,
    ) -> GARCHResult:
        """
        Build a GARCHResult from an arch ModelResult object.
        Forecast arrays are initialised as empty; populate via :meth:`forecast`.
        """
        params = dict(fit.params)
        cond_var = fit.conditional_volatility ** 2  # arch returns vol, not var
        fitted_var = cond_var.values if hasattr(cond_var, "values") else cond_var

        return GARCHResult(
            model_type=model_type,
            params=params,
            aic=float(fit.aic),
            bic=float(fit.bic),
            log_likelihood=float(fit.loglikelihood),
            conditional_vol=self._annualise_vol(fitted_var),
            forecast_vol=np.array([]),
            forecast_lower=np.array([]),
            forecast_upper=np.array([]),
            horizon_days=0,
            returns=self._returns.values,
            fitted_variance=fitted_var,
        )

    # ── model fitting ─────────────────────────────────────────────────────────

    def fit_garch(self, p: int = 1, q: int = 1) -> Optional[GARCHResult]:
        """
        Fit a standard GARCH(p, q) model with Normal innovations.

        Parameters
        ----------
        p : int
            ARCH order (lagged squared returns). Default 1.
        q : int
            GARCH order (lagged conditional variances). Default 1.

        Returns
        -------
        GARCHResult or None
            None if the model fails to converge.
        """
        self._check_min_obs()
        try:
            am = arch_model(
                self._returns * 100,  # arch works better on percentage returns
                vol="Garch",
                p=p,
                q=q,
                dist="normal",
                rescale=False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = am.fit(disp="off", show_warning=False)
            return self._extract_result(fit, "GARCH")
        except Exception:
            return None

    def fit_gjr_garch(self, p: int = 1, o: int = 1, q: int = 1) -> Optional[GARCHResult]:
        """
        Fit a GJR-GARCH(p, o, q) model (Glosten-Jagannathan-Runkle).

        The asymmetric 'o' terms capture the leverage effect: negative shocks
        tend to increase volatility more than positive shocks of equal magnitude.

        Parameters
        ----------
        p : int
            ARCH order. Default 1.
        o : int
            Asymmetric term order. Default 1.
        q : int
            GARCH order. Default 1.

        Returns
        -------
        GARCHResult or None
            None if the model fails to converge.
        """
        self._check_min_obs()
        try:
            am = arch_model(
                self._returns * 100,
                vol="Garch",
                p=p,
                o=o,
                q=q,
                dist="normal",
                rescale=False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = am.fit(disp="off", show_warning=False)
            return self._extract_result(fit, "GJR-GARCH")
        except Exception:
            return None

    def fit_egarch(self, p: int = 1, o: int = 1, q: int = 1) -> Optional[GARCHResult]:
        """
        Fit an EGARCH(p, o, q) model (Exponential GARCH).

        Models the log of conditional variance, so positivity constraints on
        parameters are not required.  Naturally captures asymmetric responses.

        Parameters
        ----------
        p : int
            ARCH order. Default 1.
        o : int
            Asymmetric term order. Default 1.
        q : int
            GARCH order. Default 1.

        Returns
        -------
        GARCHResult or None
            None if the model fails to converge.
        """
        self._check_min_obs()
        try:
            am = arch_model(
                self._returns * 100,
                vol="EGARCH",
                p=p,
                o=o,
                q=q,
                dist="normal",
                rescale=False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = am.fit(disp="off", show_warning=False)
            return self._extract_result(fit, "EGARCH")
        except Exception:
            return None

    def fit_all(self) -> dict[str, GARCHResult]:
        """
        Fit all three model types (GARCH, GJR-GARCH, EGARCH) and return a
        dictionary keyed by model_type string.

        Convergence failures are silently skipped; the returned dict may
        therefore contain fewer than three entries.

        Returns
        -------
        dict[str, GARCHResult]
            Keys are "GARCH", "GJR-GARCH", "EGARCH".
        """
        results: dict[str, GARCHResult] = {}
        for name, result in [
            ("GARCH",     self.fit_garch()),
            ("GJR-GARCH", self.fit_gjr_garch()),
            ("EGARCH",    self.fit_egarch()),
        ]:
            if result is not None:
                results[name] = result
        return results

    # ── forecasting ───────────────────────────────────────────────────────────

    def forecast(
        self,
        result: GARCHResult,
        horizon: int = 30,
        confidence: float = 0.95,
    ) -> GARCHResult:
        """
        Generate an h-step-ahead annualised volatility forecast with
        simulation-based confidence bands.

        For a GARCH(1,1) the closed-form multi-step forecast is:

            sigma²_{t+h} = omega/(1-alpha-beta)
                           + (alpha+beta)^(h-1) * (sigma²_{t+1} - LR_var)

        For all models, SIM_PATHS Monte Carlo paths are simulated from the
        last fitted conditional variance to derive percentile confidence bands.

        Parameters
        ----------
        result : GARCHResult
            A fitted result produced by :meth:`fit_garch`, :meth:`fit_gjr_garch`,
            or :meth:`fit_egarch`.
        horizon : int
            Number of trading days to forecast. Default 30.
        confidence : float
            Confidence level for the bands (e.g. 0.95 → 2.5 % / 97.5 % quantiles).

        Returns
        -------
        GARCHResult
            The same object, updated in-place with forecast arrays.
        """
        p = result.params
        omega = p.get("omega", p.get("Const", 1e-6))
        alpha = p.get("alpha[1]", 0.05)
        beta  = p.get("beta[1]",  0.90)
        gamma = p.get("gamma[1]", 0.0)   # GJR asymmetry term

        persistence = alpha + beta + 0.5 * gamma  # GJR persistence
        persistence = np.clip(persistence, 0.0, 0.9999)

        # Long-run (unconditional) daily variance
        lr_var = omega / max(1.0 - persistence, 1e-8)

        # Last fitted daily variance (rescale from pct² back to decimal²)
        last_var = float(result.fitted_variance[-1]) / (100.0 ** 2)
        lr_var_dec = lr_var / (100.0 ** 2)

        # --- Closed-form point forecast (GARCH(1,1) style) ---
        h_array = np.arange(1, horizon + 1)
        mean_var_daily = (
            lr_var_dec
            + (persistence ** (h_array - 1)) * (last_var - lr_var_dec)
        )
        mean_var_daily = np.maximum(mean_var_daily, 0.0)
        point_forecast = self._annualise_vol(mean_var_daily)

        # --- Simulation-based confidence bands ---
        alpha_tail = (1.0 - confidence) / 2.0
        sim_terminal = self._simulate_paths(
            last_var=last_var,
            omega=lr_var_dec * (1.0 - persistence),
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            horizon=horizon,
        )  # shape (SIM_PATHS, horizon)

        annualised_sim = self._annualise_vol(sim_terminal)
        lower = np.quantile(annualised_sim, alpha_tail, axis=0)
        upper = np.quantile(annualised_sim, 1.0 - alpha_tail, axis=0)

        result.forecast_vol   = point_forecast
        result.forecast_lower = lower
        result.forecast_upper = upper
        result.horizon_days   = horizon

        return result

    def _simulate_paths(
        self,
        last_var: float,
        omega: float,
        alpha: float,
        beta: float,
        gamma: float,
        horizon: int,
    ) -> np.ndarray:
        """
        Simulate SIM_PATHS GARCH paths of daily variance for `horizon` steps.

        Returns
        -------
        np.ndarray
            Shape (SIM_PATHS, horizon). Values are daily variances in decimal².
        """
        rng = self._rng
        paths = np.empty((SIM_PATHS, horizon))
        current_var = np.full(SIM_PATHS, max(last_var, 1e-10))

        for h in range(horizon):
            z = rng.standard_normal(SIM_PATHS)
            shock = z * np.sqrt(current_var)
            # Asymmetric indicator (negative shock → 1)
            indicator = (shock < 0).astype(float)
            next_var = (
                omega
                + alpha * shock ** 2
                + gamma * shock ** 2 * indicator
                + beta * current_var
            )
            next_var = np.maximum(next_var, 0.0)
            paths[:, h] = next_var
            current_var = next_var

        return paths

    # ── IV comparison ─────────────────────────────────────────────────────────

    def compare_to_iv(
        self,
        garch_annualized_vol: float,
        current_iv: float,
    ) -> dict:
        """
        Compare a GARCH-derived annualised volatility estimate to a current
        implied volatility (IV), and generate a trading signal.

        Signal logic
        ------------
        Premium > +35 % → OVERPRICED / STRONG
        Premium > +20 % → OVERPRICED / MODERATE
        Premium >   0 % → OVERPRICED / WEAK
        Premium < -35 % → UNDERPRICED / STRONG
        Premium < -20 % → UNDERPRICED / MODERATE
        Premium <   0 % → UNDERPRICED / WEAK
        Otherwise       → FAIR / WEAK

        Parameters
        ----------
        garch_annualized_vol : float
            Annualised volatility from a GARCH model (decimal, e.g. 0.22).
        current_iv : float
            Market implied volatility (decimal, e.g. 0.28).

        Returns
        -------
        dict with keys: garch_vol, current_iv, iv_premium, iv_premium_pct,
                        signal, signal_strength
        """
        if garch_annualized_vol <= 0:
            return {
                "garch_vol": garch_annualized_vol,
                "current_iv": current_iv,
                "iv_premium": float("nan"),
                "iv_premium_pct": float("nan"),
                "signal": "UNKNOWN",
                "signal_strength": "WEAK",
            }

        iv_premium = current_iv - garch_annualized_vol
        iv_premium_pct = (current_iv / garch_annualized_vol - 1.0) * 100.0

        abs_pct = abs(iv_premium_pct) / 100.0   # back to fraction for thresholds

        if iv_premium_pct > 0:
            signal = "OVERPRICED"
        elif iv_premium_pct < 0:
            signal = "UNDERPRICED"
        else:
            signal = "FAIR"

        if abs_pct >= THRESHOLD_STRONG:
            strength = "STRONG"
        elif abs_pct >= THRESHOLD_MODERATE:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        if signal == "FAIR":
            strength = "WEAK"

        return {
            "garch_vol":       garch_annualized_vol,
            "current_iv":      current_iv,
            "iv_premium":      iv_premium,
            "iv_premium_pct":  iv_premium_pct,
            "signal":          signal,
            "signal_strength": strength,
        }

    # ── model selection ───────────────────────────────────────────────────────

    def select_best(self, results: dict[str, GARCHResult]) -> GARCHResult:
        """
        Select the best-fitting model by lowest AIC.

        Parameters
        ----------
        results : dict[str, GARCHResult]
            Dictionary as returned by :meth:`fit_all`.

        Returns
        -------
        GARCHResult
            The model with the minimum AIC value.

        Raises
        ------
        ValueError
            If `results` is empty.
        """
        if not results:
            raise ValueError("No fitted results provided to select_best.")
        return min(results.values(), key=lambda r: r.aic)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def garch_forecast_chart_data(
    result: GARCHResult,
    current_iv: Optional[float] = None,
) -> dict:
    """
    Assemble data arrays for building a Plotly volatility forecast chart.

    The returned dictionary contains both the historical conditional volatility
    series and forward-looking forecast arrays, ready for direct use as trace
    data in ``go.Scatter`` / ``go.Figure`` calls.

    Parameters
    ----------
    result : GARCHResult
        A fitted (and ideally forecasted) GARCHResult.
    current_iv : float, optional
        Current implied volatility to overlay as a horizontal reference line.

    Returns
    -------
    dict with keys:
        - ``conditional_vol``  : np.ndarray — historical annualised GARCH vol
        - ``forecast_vol``     : np.ndarray — point forecast
        - ``forecast_lower``   : np.ndarray — lower confidence band
        - ``forecast_upper``   : np.ndarray — upper confidence band
        - ``current_iv_line``  : float or None
        - ``days``             : np.ndarray — day indices for forecast x-axis
                                 (1-based, length = horizon_days)
    """
    return {
        "conditional_vol":  result.conditional_vol,
        "forecast_vol":     result.forecast_vol,
        "forecast_lower":   result.forecast_lower,
        "forecast_upper":   result.forecast_upper,
        "current_iv_line":  current_iv,
        "days":             np.arange(1, result.horizon_days + 1),
    }


def garch_model_comparison_table(
    results: dict[str, GARCHResult],
) -> pd.DataFrame:
    """
    Summarise multiple fitted GARCH models in a comparison DataFrame.

    Parameters
    ----------
    results : dict[str, GARCHResult]
        Dictionary as returned by :meth:`GARCHForecaster.fit_all`.

    Returns
    -------
    pd.DataFrame
        Columns: Model, AIC, BIC, Log-Likelihood, Persistence, LR Volatility.

        - **Persistence** = alpha + beta (+ 0.5*gamma for GJR-GARCH).
          Values close to 1 indicate highly persistent volatility clusters.
        - **LR Volatility** = annualised long-run (unconditional) volatility
          derived from the fitted parameters (decimal, e.g. 0.20 = 20 %).
    """
    rows = []
    for model_type, res in results.items():
        p = res.params
        omega = p.get("omega", p.get("Const", np.nan))
        alpha = p.get("alpha[1]", 0.0)
        beta  = p.get("beta[1]",  0.0)
        gamma = p.get("gamma[1]", 0.0)

        persistence = alpha + beta + 0.5 * gamma

        # Long-run annualised vol (parameters are in percentage² units)
        denom = 1.0 - persistence
        if denom > 1e-8 and not np.isnan(omega):
            lr_var_pct2 = omega / denom           # percentage² units
            lr_var_dec  = lr_var_pct2 / (100.0 ** 2)
            lr_vol = float(np.sqrt(TRADING_DAYS * lr_var_dec))
        else:
            lr_vol = float("nan")

        rows.append({
            "Model":           model_type,
            "AIC":             round(res.aic, 2),
            "BIC":             round(res.bic, 2),
            "Log-Likelihood":  round(res.log_likelihood, 2),
            "Persistence":     round(float(persistence), 4),
            "LR Volatility":   round(lr_vol, 4) if not np.isnan(lr_vol) else float("nan"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("AIC").reset_index(drop=True)
    return df
