"""
regime_detection.py
===================
Hidden Markov Model-based market regime detection.

Identifies hidden market states (Bull / Neutral / Bear) from price returns
using a Gaussian HMM. Provides regime-conditional statistics, transition
probabilities, and current-regime probability.

Dependencies: hmmlearn >= 0.3.0, numpy, pandas, scipy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

# ── Constants ─────────────────────────────────────────────────────────────────
TRADING_DAYS = 252
MIN_OBSERVATIONS = 60        # minimum returns needed to fit HMM
MAX_REGIMES = 5
DEFAULT_REGIMES = 3
DEFAULT_LABELS = {2: ["Bear", "Bull"],
                  3: ["Bear", "Neutral", "Bull"],
                  4: ["Bear", "Stress", "Neutral", "Bull"],
                  5: ["Crash", "Bear", "Neutral", "Bull", "Rally"]}


# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class RegimeStats:
    """Statistics for a single regime."""
    regime_id: int
    label: str
    mean_daily_return: float        # mean daily log return
    std_daily_return: float         # std of daily returns
    annualized_return: float        # mean_daily * 252
    annualized_vol: float           # std_daily * sqrt(252)
    sharpe_ratio: float             # annualized return / annualized vol
    avg_duration_days: float        # average number of consecutive days
    pct_time_in_regime: float       # fraction of total history in this regime
    color: str                      # display color


@dataclass
class RegimeResult:
    """Full output of regime detection."""
    n_regimes: int
    state_sequence: np.ndarray          # Viterbi-decoded states (length T)
    state_probabilities: np.ndarray     # posterior probs (T, n_regimes)
    regime_stats: list[RegimeStats]
    current_regime: int                 # index of most likely current regime
    current_regime_label: str
    current_probs: np.ndarray           # probability vector for last observation
    transition_matrix: np.ndarray       # (n_regimes, n_regimes)
    aic: float
    bic: float
    log_likelihood: float
    returns: np.ndarray                 # input returns used for fitting
    dates: Optional[pd.DatetimeIndex]   # index of returns series
    labels: list[str]                   # regime label list


# ── Main class ────────────────────────────────────────────────────────────────
class RegimeDetector:
    """
    Gaussian HMM-based market regime detector.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns (not annualized).
    n_regimes : int
        Number of hidden states (2-5). Default 3.
    n_iter : int
        EM algorithm iterations. Default 200.
    covariance_type : str
        HMM covariance type. 'full' or 'diag'. Default 'full'.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, returns: pd.Series, n_regimes: int = DEFAULT_REGIMES,
                 n_iter: int = 200, covariance_type: str = "full", seed: int = 42):
        if len(returns) < MIN_OBSERVATIONS:
            raise ValueError(
                f"Need at least {MIN_OBSERVATIONS} return observations, got {len(returns)}.")
        if not (2 <= n_regimes <= MAX_REGIMES):
            raise ValueError(f"n_regimes must be between 2 and {MAX_REGIMES}.")

        self.returns = returns.dropna()
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.seed = seed
        self._model = None

    # ── Core fitting ──────────────────────────────────────────────────────────
    def fit(self) -> RegimeResult:
        """
        Fit Gaussian HMM and decode regime sequence.

        Returns
        -------
        RegimeResult
            Complete regime analysis including Viterbi states,
            posterior probabilities, per-regime statistics, and
            model information criteria.
        """
        try:
            from hmmlearn import hmm as _hmm
        except ImportError:
            raise ImportError("Install hmmlearn: pip install hmmlearn")

        X = self.returns.values.reshape(-1, 1)

        # Fit model — retry with different seeds if convergence fails
        best_model = None
        best_ll = -np.inf
        for attempt_seed in [self.seed, self.seed + 1, self.seed + 7]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model = _hmm.GaussianHMM(
                        n_components=self.n_regimes,
                        covariance_type=self.covariance_type,
                        n_iter=self.n_iter,
                        random_state=attempt_seed,
                        tol=1e-4,
                    )
                    model.fit(X)
                    ll = model.score(X)
                    if ll > best_ll:
                        best_ll = ll
                        best_model = model
                except Exception:
                    continue

        if best_model is None:
            raise RuntimeError("HMM failed to converge. Try fewer regimes or more data.")

        self._model = best_model

        # Decode states
        state_seq = best_model.predict(X)
        state_probs = best_model.predict_proba(X)

        # Re-order states by mean return (lowest = Bear, highest = Bull)
        means = best_model.means_.flatten()
        order = np.argsort(means)          # ascending: Bear → Bull
        remap = {old: new for new, old in enumerate(order)}
        state_seq_sorted = np.array([remap[s] for s in state_seq])
        state_probs_sorted = state_probs[:, order]

        # Transition matrix (re-ordered)
        A = best_model.transmat_[np.ix_(order, order)]

        # Labels
        labels = DEFAULT_LABELS.get(self.n_regimes,
                                    [f"Regime {i}" for i in range(self.n_regimes)])

        # Per-regime statistics
        regime_stats = self._compute_regime_stats(state_seq_sorted, labels)

        # Model selection criteria
        n_params = self._count_params(best_model)
        T = len(X)
        aic = -2 * best_ll + 2 * n_params
        bic = -2 * best_ll + np.log(T) * n_params

        current_regime = int(state_seq_sorted[-1])
        current_probs = state_probs_sorted[-1]

        dates = self.returns.index if isinstance(self.returns.index, pd.DatetimeIndex) else None

        return RegimeResult(
            n_regimes=self.n_regimes,
            state_sequence=state_seq_sorted,
            state_probabilities=state_probs_sorted,
            regime_stats=regime_stats,
            current_regime=current_regime,
            current_regime_label=labels[current_regime],
            current_probs=current_probs,
            transition_matrix=A,
            aic=aic,
            bic=bic,
            log_likelihood=best_ll,
            returns=self.returns.values,
            dates=dates,
            labels=labels,
        )

    # ── Regime statistics ─────────────────────────────────────────────────────
    def _compute_regime_stats(self, state_seq: np.ndarray,
                               labels: list[str]) -> list[RegimeStats]:
        """Compute per-regime return, vol, Sharpe, duration statistics."""
        rets = self.returns.values
        colors = ["#F43F5E", "#F59E0B", "#00D4AA", "#6366F1", "#818CF8"]
        stats = []
        T = len(state_seq)

        for k in range(self.n_regimes):
            mask = state_seq == k
            k_rets = rets[mask]

            if len(k_rets) == 0:
                mu_d, sig_d = 0.0, 1e-6
            else:
                mu_d = float(np.mean(k_rets))
                sig_d = float(np.std(k_rets)) if len(k_rets) > 1 else 1e-6

            ann_ret = mu_d * TRADING_DAYS
            ann_vol = sig_d * np.sqrt(TRADING_DAYS)
            sharpe = ann_ret / max(ann_vol, 1e-6)

            # Average duration: count consecutive runs
            durations = []
            run = 0
            for s in state_seq:
                if s == k:
                    run += 1
                elif run > 0:
                    durations.append(run)
                    run = 0
            if run > 0:
                durations.append(run)
            avg_dur = float(np.mean(durations)) if durations else 0.0

            pct_time = float(np.sum(mask)) / T

            stats.append(RegimeStats(
                regime_id=k,
                label=labels[k],
                mean_daily_return=mu_d,
                std_daily_return=sig_d,
                annualized_return=ann_ret,
                annualized_vol=ann_vol,
                sharpe_ratio=sharpe,
                avg_duration_days=avg_dur,
                pct_time_in_regime=pct_time,
                color=colors[k % len(colors)],
            ))
        return stats

    def _count_params(self, model) -> int:
        """Count free parameters for AIC/BIC."""
        k = self.n_regimes
        # transition matrix: k*(k-1), means: k, variances: k (diag) or k*(k+1)/2 (full)
        cov_params = k if self.covariance_type == "diag" else k
        return k * (k - 1) + k + cov_params

    # ── BIC-based regime count selection ─────────────────────────────────────
    def select_n_regimes(self, max_k: int = 5) -> dict:
        """
        Fit HMMs with 2..max_k regimes and return BIC scores.

        Returns
        -------
        dict with keys: 'best_k', 'bic_scores' (list), 'results' (list of RegimeResult)
        """
        bic_scores = []
        results = []
        for k in range(2, min(max_k, MAX_REGIMES) + 1):
            try:
                detector = RegimeDetector(
                    self.returns, n_regimes=k,
                    n_iter=self.n_iter, seed=self.seed
                )
                res = detector.fit()
                bic_scores.append(res.bic)
                results.append(res)
            except Exception:
                bic_scores.append(np.inf)
                results.append(None)

        best_idx = int(np.argmin(bic_scores))
        return {
            "best_k": best_idx + 2,
            "bic_scores": bic_scores,
            "results": results,
        }


# ── Standalone helpers ────────────────────────────────────────────────────────
def regime_overlay_data(result: RegimeResult,
                         prices: pd.Series) -> pd.DataFrame:
    """
    Build a DataFrame aligning prices with regime labels for chart overlays.

    Returns DataFrame with columns: date, price, regime, regime_label, color.
    """
    n = min(len(prices), len(result.state_sequence))
    idx = prices.index[-n:] if hasattr(prices.index, "__len__") else range(n)
    df = pd.DataFrame({
        "date": idx,
        "price": prices.values[-n:],
        "regime": result.state_sequence[-n:],
        "regime_label": [result.labels[r] for r in result.state_sequence[-n:]],
        "color": [result.regime_stats[r].color for r in result.state_sequence[-n:]],
    })
    return df


def regime_stats_table(result: RegimeResult) -> pd.DataFrame:
    """Return a formatted DataFrame of per-regime statistics for display."""
    rows = []
    for rs in result.regime_stats:
        rows.append({
            "Regime": rs.label,
            "Ann. Return": f"{rs.annualized_return * 100:+.1f}%",
            "Ann. Vol": f"{rs.annualized_vol * 100:.1f}%",
            "Sharpe": f"{rs.sharpe_ratio:.2f}",
            "Avg Duration": f"{rs.avg_duration_days:.1f}d",
            "% Time": f"{rs.pct_time_in_regime * 100:.1f}%",
        })
    return pd.DataFrame(rows)


def current_regime_summary(result: RegimeResult) -> dict:
    """
    Return a concise summary of the current market regime.

    Returns
    -------
    dict with: label, probability, annualized_vol, annualized_return,
               color, regime_id, transition_probs
    """
    k = result.current_regime
    rs = result.regime_stats[k]
    next_probs = result.transition_matrix[k]          # transition row for current state
    return {
        "label": rs.label,
        "probability": float(result.current_probs[k]),
        "annualized_return": rs.annualized_return,
        "annualized_vol": rs.annualized_vol,
        "sharpe": rs.sharpe_ratio,
        "color": rs.color,
        "regime_id": k,
        "transition_probs": {result.labels[j]: float(next_probs[j])
                              for j in range(result.n_regimes)},
    }
