"""
Risk Analytics Engine
----------------------
Value-at-Risk (VaR / CVaR), stress testing, scenario P&L matrices,
and portfolio-level Greek aggregation for option books.

Methods
-------
- Historical VaR / CVaR
- Parametric (Gaussian) VaR
- Cornish-Fisher VaR (skewness / kurtosis adjusted)
- Monte Carlo VaR
- Scenario stress testing via Taylor expansion
- 2-D P&L matrix over spot × vol shocks
- Portfolio-level Greek roll-up across multi-leg positions
- Rolling VaR, max drawdown, and return statistics
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VaRResult:
    """Container for all four VaR estimates plus distributional statistics."""

    confidence: float
    """Confidence level, e.g. 0.95 means 95% VaR."""

    horizon_days: int
    """Holding-period horizon in calendar days."""

    var_historical: float
    """Historical simulation VaR (reported as a positive loss figure)."""

    var_parametric: float
    """Parametric Gaussian VaR (positive loss)."""

    var_cornish_fisher: float
    """Cornish-Fisher VaR adjusted for skewness and excess kurtosis (positive loss)."""

    var_monte_carlo: float
    """Monte Carlo simulated VaR (positive loss)."""

    cvar_historical: float
    """Expected Shortfall (CVaR) from historical simulation (positive loss)."""

    cvar_monte_carlo: float
    """Expected Shortfall (CVaR) from Monte Carlo simulation (positive loss)."""

    returns_used: np.ndarray
    """The daily log-return series used for estimation."""

    skewness: float
    """Skewness of the return series."""

    kurtosis: float
    """Excess kurtosis of the return series."""


@dataclass
class StressResult:
    """P&L impact of a single stress scenario decomposed by Greek."""

    scenario_name: str
    """Human-readable scenario label."""

    spot_shock_pct: float
    """Fractional spot shock applied, e.g. -0.10 = -10%."""

    vol_shock_pct: float
    """Absolute vol shock, e.g. 0.05 = +5 vol points."""

    rate_shock_pct: float
    """Absolute rate shock, e.g. 0.01 = +1%."""

    pnl_change: float
    """Estimated total P&L change from Taylor expansion."""

    delta_contribution: float
    """First-order spot contribution: delta * dS."""

    gamma_contribution: float
    """Second-order spot contribution: 0.5 * gamma * dS^2."""

    vega_contribution: float
    """Vol contribution: vega * (d_sigma * 100)."""

    theta_contribution: float
    """Time-decay contribution: theta * time_days."""

    rho_contribution: float
    """Rate contribution: rho * (dr * 100)."""

    pct_change: float
    """pnl_change expressed as a fraction of the current option value."""


@dataclass
class PortfolioGreeks:
    """Aggregated Greeks and dollar sensitivities for a multi-leg option book."""

    positions: list[dict]
    """Original list of position dicts supplied by the caller."""

    net_delta: float
    """Book delta in contract units."""

    net_gamma: float
    """Book gamma in contract units."""

    net_theta: float
    """Book theta ($ per calendar day)."""

    net_vega: float
    """Book vega ($ per 1% move in implied vol)."""

    net_rho: float
    """Book rho ($ per 1% move in risk-free rate)."""

    dollar_delta: float
    """Net delta * spot = approximate $ change for a $1 move in spot."""

    dollar_gamma: float
    """Net gamma * spot^2 / 100 — standard dollar-gamma convention."""

    dollar_vega: float
    """Net vega — already expressed as $ per 1% vol move."""

    dollar_theta: float
    """Net theta — already expressed as $ per calendar day."""

    greeks_breakdown: pd.DataFrame
    """Per-position DataFrame with columns: strike, dte, type, contracts,
    multiplier, delta, gamma, theta, vega, rho, pos_delta, pos_gamma,
    pos_theta, pos_vega, pos_rho."""


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL GREEK HELPERS (self-contained, no circular imports)
# ─────────────────────────────────────────────────────────────────────────────

def _d1d2(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> tuple[float, float]:
    """Compute Black-Scholes d1 and d2."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan, np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def _bs_delta(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0, option_type: str = "call"
) -> float:
    d1, _ = _d1d2(S, K, T, r, sigma, q)
    if np.isnan(d1):
        return 0.0
    if option_type == "call":
        return float(np.exp(-q * T) * norm.cdf(d1))
    return float(-np.exp(-q * T) * norm.cdf(-d1))


def _bs_gamma(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> float:
    d1, _ = _d1d2(S, K, T, r, sigma, q)
    if np.isnan(d1) or S <= 0 or sigma <= 0 or T <= 0:
        return 0.0
    return float(np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def _bs_theta(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0, option_type: str = "call"
) -> float:
    """Theta per calendar day (negative means time decay)."""
    if T <= 0:
        return 0.0
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    if np.isnan(d1):
        return 0.0
    common = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        val = (common
               - r * K * np.exp(-r * T) * norm.cdf(d2)
               + q * S * np.exp(-q * T) * norm.cdf(d1))
    else:
        val = (common
               + r * K * np.exp(-r * T) * norm.cdf(-d2)
               - q * S * np.exp(-q * T) * norm.cdf(-d1))
    return float(val / 365.0)


def _bs_vega(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> float:
    """Vega per 1% move in implied vol."""
    if T <= 0:
        return 0.0
    d1, _ = _d1d2(S, K, T, r, sigma, q)
    if np.isnan(d1):
        return 0.0
    return float(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) * 0.01)


def _bs_rho(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0, option_type: str = "call"
) -> float:
    """Rho per 1% move in risk-free rate."""
    if T <= 0:
        return 0.0
    _, d2 = _d1d2(S, K, T, r, sigma, q)
    if np.isnan(d2):
        return 0.0
    if option_type == "call":
        return float(K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01)
    return float(-K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class RiskAnalytics:
    """
    Advanced risk analytics engine for options and equity portfolios.

    Parameters
    ----------
    returns : pd.Series
        Daily log-returns with a DatetimeIndex.  Must contain at least 30
        observations for meaningful VaR estimates.
    spot : float, optional
        Current underlying price used for dollar-Greek calculations.
    """

    _DEFAULT_STRESS_SCENARIOS: list[dict] = [
        {"name": "Black Monday (-20%)", "spot": -0.20, "vol":  0.50, "rate": -0.01, "time": 0},
        {"name": "Crash (-15%)",        "spot": -0.15, "vol":  0.35, "rate": -0.005,"time": 0},
        {"name": "Correction (-10%)",   "spot": -0.10, "vol":  0.25, "rate":  0.0,  "time": 0},
        {"name": "Mild Decline (-5%)",  "spot": -0.05, "vol":  0.10, "rate":  0.0,  "time": 0},
        {"name": "Flat",                "spot":  0.0,  "vol":  0.0,  "rate":  0.0,  "time": 0},
        {"name": "Mild Rally (+5%)",    "spot":  0.05, "vol": -0.05, "rate":  0.0,  "time": 0},
        {"name": "Strong Rally (+10%)", "spot":  0.10, "vol": -0.10, "rate":  0.005,"time": 0},
        {"name": "Euphoria (+20%)",     "spot":  0.20, "vol": -0.15, "rate":  0.01, "time": 0},
    ]

    def __init__(self, returns: pd.Series, spot: Optional[float] = None) -> None:
        if not isinstance(returns, pd.Series):
            raise TypeError("returns must be a pandas Series.")
        if returns.empty:
            raise ValueError("returns series is empty — cannot compute risk metrics.")
        if len(returns) < 5:
            raise ValueError(
                f"returns series has only {len(returns)} observations; "
                "at least 5 are required."
            )

        self._returns: pd.Series = returns.dropna().astype(float)
        self.spot: Optional[float] = spot

        if len(self._returns) < len(returns):
            warnings.warn(
                f"{len(returns) - len(self._returns)} NaN values were dropped "
                "from the returns series.",
                stacklevel=2,
            )

    # ── public properties ─────────────────────────────────────────────────────

    @property
    def returns(self) -> pd.Series:
        """Clean daily log-return series."""
        return self._returns

    # ── VaR computation ───────────────────────────────────────────────────────

    def compute_var(
        self,
        confidence: float = 0.95,
        horizon_days: int = 1,
        mc_paths: int = 10_000,
        mc_seed: int = 42,
    ) -> VaRResult:
        """
        Compute VaR at the given confidence level using four methods.

        Parameters
        ----------
        confidence : float
            Confidence level in [0, 1), e.g. 0.95 for 95% VaR.
        horizon_days : int
            Holding-period horizon in trading days (square-root scaling applied).
        mc_paths : int
            Number of Monte Carlo paths for the MC VaR estimate.
        mc_seed : int
            Random seed for reproducibility of MC paths.

        Returns
        -------
        VaRResult
        """
        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be in (0, 1); got {confidence}.")
        if horizon_days < 1:
            raise ValueError(f"horizon_days must be >= 1; got {horizon_days}.")

        r = self._returns.values
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1))
        alpha = 1.0 - confidence          # tail probability
        sqrt_h = np.sqrt(horizon_days)

        # ── 1. Historical ──────────────────────────────────────────────────────
        var_hist = float(-np.percentile(r, alpha * 100))
        tail_mask = r <= -var_hist
        if tail_mask.sum() > 0:
            cvar_hist = float(-np.mean(r[tail_mask]))
        else:
            cvar_hist = var_hist          # fallback when no observations in tail

        # Scale to horizon
        var_hist_h   = var_hist * sqrt_h
        cvar_hist_h  = cvar_hist * sqrt_h

        # ── 2. Parametric (Gaussian) ───────────────────────────────────────────
        z = norm.ppf(alpha)               # e.g. -1.645 at 95%
        var_param = float(-(mu + sigma * z) * sqrt_h)

        # ── 3. Cornish-Fisher ─────────────────────────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            S = float(skew(r))
            K = float(kurtosis(r, fisher=True))   # excess kurtosis

        z_cf = (z
                + (z ** 2 - 1) * S / 6
                + (z ** 3 - 3 * z) * K / 24
                - (2 * z ** 3 - 5 * z) * S ** 2 / 36)
        var_cf = float(-(mu + sigma * z_cf) * sqrt_h)

        # ── 4. Monte Carlo ────────────────────────────────────────────────────
        rng = np.random.default_rng(mc_seed)
        sim = rng.normal(mu * horizon_days, sigma * sqrt_h, size=mc_paths)
        var_mc  = float(-np.percentile(sim, alpha * 100))
        tail_mc = sim[sim <= -var_mc]
        cvar_mc = float(-np.mean(tail_mc)) if len(tail_mc) > 0 else var_mc

        return VaRResult(
            confidence=confidence,
            horizon_days=horizon_days,
            var_historical=max(var_hist_h, 0.0),
            var_parametric=max(var_param, 0.0),
            var_cornish_fisher=max(var_cf, 0.0),
            var_monte_carlo=max(var_mc, 0.0),
            cvar_historical=max(cvar_hist_h, 0.0),
            cvar_monte_carlo=max(cvar_mc, 0.0),
            returns_used=r.copy(),
            skewness=S,
            kurtosis=K,
        )

    # ── Stress testing ────────────────────────────────────────────────────────

    def stress_test(
        self,
        option_value: float,
        delta: float,
        gamma: float,
        vega: float,
        theta: float,
        rho: float,
        spot: float,
        scenarios: Optional[list[dict]] = None,
    ) -> list[StressResult]:
        """
        Apply stress scenarios to an option position via Taylor expansion.

        The P&L approximation is:
            dV ≈ delta*dS + 0.5*gamma*dS² + vega*(d_sigma*100)
                 + theta*time_days + rho*(dr*100)

        Parameters
        ----------
        option_value : float
            Current mark-to-market value of the option / position.
        delta : float
            Position delta (units: $ per $1 move in spot).
        gamma : float
            Position gamma.
        vega : float
            Position vega per 1% move in implied vol.
        theta : float
            Position theta per calendar day.
        rho : float
            Position rho per 1% move in risk-free rate.
        spot : float
            Current spot price.
        scenarios : list[dict], optional
            Each dict must have keys: ``name``, ``spot`` (fractional shock),
            ``vol`` (absolute vol shock), ``rate`` (absolute rate shock),
            ``time`` (days elapsed — usually 0 for instant shocks).
            Defaults to the eight canonical scenarios defined on the class.

        Returns
        -------
        list[StressResult]
        """
        if option_value == 0:
            warnings.warn("option_value is zero; pct_change will be undefined (set to NaN).")

        if scenarios is None:
            scenarios = self._DEFAULT_STRESS_SCENARIOS

        results: list[StressResult] = []
        for sc in scenarios:
            spot_shock  = float(sc.get("spot",  0.0))
            vol_shock   = float(sc.get("vol",   0.0))
            rate_shock  = float(sc.get("rate",  0.0))
            time_days   = float(sc.get("time",  0.0))
            name        = str(sc.get("name",    "Unnamed"))

            dS = spot * spot_shock

            delta_contrib = delta * dS
            gamma_contrib = 0.5 * gamma * dS ** 2
            vega_contrib  = vega * (vol_shock * 100)
            theta_contrib = theta * time_days
            rho_contrib   = rho * (rate_shock * 100)

            pnl = delta_contrib + gamma_contrib + vega_contrib + theta_contrib + rho_contrib
            pct = pnl / option_value if option_value != 0 else np.nan

            results.append(StressResult(
                scenario_name=name,
                spot_shock_pct=spot_shock,
                vol_shock_pct=vol_shock,
                rate_shock_pct=rate_shock,
                pnl_change=pnl,
                delta_contribution=delta_contrib,
                gamma_contribution=gamma_contrib,
                vega_contribution=vega_contrib,
                theta_contribution=theta_contrib,
                rho_contribution=rho_contrib,
                pct_change=pct,
            ))

        return results

    # ── Scenario P&L matrix ───────────────────────────────────────────────────

    def scenario_pnl_matrix(
        self,
        option_value: float,
        delta: float,
        gamma: float,
        vega: float,
        spot: float,
        spot_shocks: Optional[list] = None,
        vol_shocks: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Build a 2-D P&L matrix over a grid of spot and vol shocks.

        Uses the second-order Taylor approximation (delta + gamma for spot,
        vega for vol).  Theta and rho are excluded because all shocks are
        assumed to be instantaneous.

        Parameters
        ----------
        option_value : float
            Current option mark-to-market (used for labelling only).
        delta : float
            Position delta.
        gamma : float
            Position gamma.
        vega : float
            Position vega per 1% vol move.
        spot : float
            Current spot price.
        spot_shocks : list of float, optional
            Fractional spot shocks (rows).  Defaults to
            [-0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20].
        vol_shocks : list of float, optional
            Absolute vol shocks (columns).  Defaults to
            [-0.10, -0.05, 0, 0.05, 0.10, 0.20, 0.30, 0.50].

        Returns
        -------
        pd.DataFrame
            Rows = spot shocks (formatted as strings, e.g. "-20%"),
            Columns = vol shocks (e.g. "+5% vol"),
            Values = estimated P&L change in dollars.
        """
        if spot_shocks is None:
            spot_shocks = [-0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20]
        if vol_shocks is None:
            vol_shocks = [-0.10, -0.05, 0, 0.05, 0.10, 0.20, 0.30, 0.50]

        row_labels = [f"{s*100:+.0f}%" for s in spot_shocks]
        col_labels = [f"{v*100:+.0f}% vol" for v in vol_shocks]

        data = np.zeros((len(spot_shocks), len(vol_shocks)))
        for i, s_shock in enumerate(spot_shocks):
            dS = spot * s_shock
            spot_pnl = delta * dS + 0.5 * gamma * dS ** 2
            for j, v_shock in enumerate(vol_shocks):
                vol_pnl = vega * (v_shock * 100)
                data[i, j] = spot_pnl + vol_pnl

        return pd.DataFrame(data, index=row_labels, columns=col_labels)

    # ── Portfolio Greeks aggregation ──────────────────────────────────────────

    def compute_portfolio_greeks(
        self,
        positions: list[dict],
        spot: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> PortfolioGreeks:
        """
        Aggregate Black-Scholes Greeks across a multi-leg option book.

        Each position dict must contain:

        =========  =======  =====================================================
        Key        Type     Description
        =========  =======  =====================================================
        strike     float    Option strike price
        dte        int      Days to expiry
        option_type str     ``'call'`` or ``'put'``
        contracts  int      Number of contracts (positive = long, negative = short)
        multiplier int      Contract multiplier (e.g. 100 for US equity options)
        =========  =======  =====================================================

        Parameters
        ----------
        positions : list[dict]
            List of position specifications (see above).
        spot : float
            Current spot price.
        r : float
            Annualised risk-free rate.
        sigma : float
            Annualised implied volatility (same for all legs; can be overridden
            per position by adding a ``sigma`` key to the dict).
        q : float
            Continuous dividend yield.

        Returns
        -------
        PortfolioGreeks
        """
        if not positions:
            raise ValueError("positions list is empty.")

        rows: list[dict] = []
        for pos in positions:
            K     = float(pos["strike"])
            dte   = int(pos["dte"])
            otype = str(pos.get("option_type", "call")).lower()
            n     = float(pos.get("contracts", 1))
            mult  = float(pos.get("multiplier", 100))
            sig   = float(pos.get("sigma", sigma))
            T     = max(dte, 1) / 365.0

            d  = _bs_delta(spot, K, T, r, sig, q, otype)
            g  = _bs_gamma(spot, K, T, r, sig, q)
            th = _bs_theta(spot, K, T, r, sig, q, otype)
            ve = _bs_vega(spot, K, T, r, sig, q)
            rh = _bs_rho(spot, K, T, r, sig, q, otype)

            scale = n * mult
            rows.append({
                "strike":      K,
                "dte":         dte,
                "type":        otype,
                "contracts":   int(n),
                "multiplier":  int(mult),
                "delta":       d,
                "gamma":       g,
                "theta":       th,
                "vega":        ve,
                "rho":         rh,
                "pos_delta":   d  * scale,
                "pos_gamma":   g  * scale,
                "pos_theta":   th * scale,
                "pos_vega":    ve * scale,
                "pos_rho":     rh * scale,
            })

        df = pd.DataFrame(rows)

        net_delta = float(df["pos_delta"].sum())
        net_gamma = float(df["pos_gamma"].sum())
        net_theta = float(df["pos_theta"].sum())
        net_vega  = float(df["pos_vega"].sum())
        net_rho   = float(df["pos_rho"].sum())

        dollar_delta = net_delta * spot
        dollar_gamma = net_gamma * spot ** 2 / 100.0
        dollar_vega  = net_vega   # already $/1% vol
        dollar_theta = net_theta  # already $/day

        return PortfolioGreeks(
            positions=positions,
            net_delta=net_delta,
            net_gamma=net_gamma,
            net_theta=net_theta,
            net_vega=net_vega,
            net_rho=net_rho,
            dollar_delta=dollar_delta,
            dollar_gamma=dollar_gamma,
            dollar_vega=dollar_vega,
            dollar_theta=dollar_theta,
            greeks_breakdown=df,
        )

    # ── Rolling VaR ──────────────────────────────────────────────────────────

    def rolling_var(
        self,
        confidence: float = 0.95,
        window: int = 252,
    ) -> pd.Series:
        """
        Compute rolling historical VaR over a sliding window.

        Parameters
        ----------
        confidence : float
            VaR confidence level.
        window : int
            Rolling window length in observations.

        Returns
        -------
        pd.Series
            VaR time-series aligned with the input returns index.
            Values are positive (loss convention).
        """
        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be in (0, 1); got {confidence}.")
        if window < 2:
            raise ValueError(f"window must be >= 2; got {window}.")

        alpha = (1.0 - confidence) * 100

        def _hist_var(x: np.ndarray) -> float:
            return float(-np.percentile(x, alpha))

        rolling = (
            self._returns
            .rolling(window=window, min_periods=max(2, window // 4))
            .apply(_hist_var, raw=True)
        )
        return rolling

    # ── Max drawdown ─────────────────────────────────────────────────────────

    def max_drawdown(self) -> float:
        """
        Compute the maximum drawdown from the cumulative return series.

        Returns
        -------
        float
            Maximum drawdown as a positive fraction (e.g. 0.35 = 35% loss).
        """
        cum_ret = (1.0 + self._returns).cumprod()
        rolling_max = cum_ret.cummax()
        drawdowns = (cum_ret - rolling_max) / rolling_max
        return float(-drawdowns.min())

    # ── Return statistics ─────────────────────────────────────────────────────

    def return_stats(self) -> dict:
        """
        Compute a comprehensive set of return statistics.

        Returns
        -------
        dict with keys:
            annualized_return, annualized_vol, sharpe_ratio,
            skewness, kurtosis, max_drawdown,
            var_95, cvar_95,
            positive_days_pct, avg_win, avg_loss, win_loss_ratio
        """
        r = self._returns.values
        n = len(r)

        ann_ret = float(np.mean(r) * 252)
        ann_vol = float(np.std(r, ddof=1) * np.sqrt(252))
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sk = float(skew(r))
            ku = float(kurtosis(r, fisher=True))

        mdd = self.max_drawdown()

        var_95  = float(-np.percentile(r, 5.0))
        tail    = r[r <= -var_95]
        cvar_95 = float(-np.mean(tail)) if len(tail) > 0 else var_95

        positive_mask  = r > 0
        negative_mask  = r < 0
        pos_pct        = float(positive_mask.sum() / n) if n > 0 else np.nan
        avg_win        = float(np.mean(r[positive_mask])) if positive_mask.sum() > 0 else np.nan
        avg_loss       = float(np.mean(r[negative_mask])) if negative_mask.sum() > 0 else np.nan
        win_loss_ratio = (
            abs(avg_win / avg_loss)
            if (avg_win is not np.nan and avg_loss not in (np.nan, 0))
            else np.nan
        )

        return {
            "annualized_return":   ann_ret,
            "annualized_vol":      ann_vol,
            "sharpe_ratio":        sharpe,
            "skewness":            sk,
            "kurtosis":            ku,
            "max_drawdown":        mdd,
            "var_95":              var_95,
            "cvar_95":             cvar_95,
            "positive_days_pct":   pos_pct,
            "avg_win":             avg_win,
            "avg_loss":            avg_loss,
            "win_loss_ratio":      win_loss_ratio,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY CHART DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def var_comparison_chart_data(var_result: VaRResult) -> dict:
    """
    Build data for a bar chart comparing the four VaR estimates.

    Returns
    -------
    dict with keys ``methods``, ``values``, ``colors``.
        - ``methods`` : list of method name strings
        - ``values``  : list of VaR figures (positive losses)
        - ``colors``  : list of hex colour strings (one per bar)
    """
    methods = [
        "Historical",
        "Parametric",
        "Cornish-Fisher",
        "Monte Carlo",
    ]
    values = [
        var_result.var_historical,
        var_result.var_parametric,
        var_result.var_cornish_fisher,
        var_result.var_monte_carlo,
    ]
    colors = ["#EF553B", "#636EFA", "#00CC96", "#AB63FA"]
    return {"methods": methods, "values": values, "colors": colors}


def loss_distribution_chart_data(var_result: VaRResult) -> dict:
    """
    Build histogram data for the return distribution annotated with
    VaR and CVaR marker positions.

    Returns
    -------
    dict with keys:
        - ``returns``       : np.ndarray — raw return series
        - ``var_historical``: float — VaR marker x-position (negative sign
          for plotting on the return axis)
        - ``cvar_historical``: float — CVaR marker x-position
        - ``var_mc``        : float
        - ``cvar_mc``       : float
        - ``confidence``    : float
        - ``horizon_days``  : int
    """
    return {
        "returns":         var_result.returns_used,
        "var_historical":  -var_result.var_historical,
        "cvar_historical": -var_result.cvar_historical,
        "var_mc":          -var_result.var_monte_carlo,
        "cvar_mc":         -var_result.cvar_monte_carlo,
        "confidence":       var_result.confidence,
        "horizon_days":     var_result.horizon_days,
    }


def scenario_matrix_chart_data(matrix_df: pd.DataFrame) -> dict:
    """
    Build data for a Plotly heatmap from the scenario P&L matrix.

    Returns
    -------
    dict with keys:
        - ``x``     : list — column labels (vol shocks)
        - ``y``     : list — row labels (spot shocks)
        - ``z``     : list of lists — P&L values (row-major)
        - ``zmin``  : float — symmetric colour-scale minimum
        - ``zmax``  : float — symmetric colour-scale maximum
    """
    z     = matrix_df.values.tolist()
    abs_max = float(np.abs(matrix_df.values).max())
    return {
        "x":    list(matrix_df.columns),
        "y":    list(matrix_df.index),
        "z":    z,
        "zmin": -abs_max,
        "zmax":  abs_max,
    }
