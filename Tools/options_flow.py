"""
Options Flow & Unusual Activity Detector
-----------------------------------------
Analyses options chain data to surface unusual volume/OI activity,
compute put-call ratios, gamma exposure, max pain, and smart-money signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UnusualActivity:
    strike: float
    expiration: str
    option_type: str          # 'call' or 'put'
    volume: int
    open_interest: int
    volume_oi_ratio: float    # volume / OI
    z_score: float            # how unusual vs other strikes at the same expiry
    implied_vol: float
    last_price: float
    bid: float
    ask: float
    spread_pct: float         # (ask - bid) / mid
    moneyness: str            # "ITM", "ATM", "OTM"
    signal: str               # "BULLISH", "BEARISH", "NEUTRAL"
    smart_money_score: float  # 0 to 1


@dataclass
class FlowSummary:
    ticker: str
    spot: float
    pcr_volume: float
    pcr_oi: float
    total_call_volume: int
    total_put_volume: int
    total_call_oi: int
    total_put_oi: int
    net_flow_bias: str            # "BULLISH", "BEARISH", "NEUTRAL"
    net_flow_score: float         # -1 (very bearish) to +1 (very bullish)
    unusual_activity: list        # list[UnusualActivity]
    smart_money_score: float      # aggregate -1 to +1
    volume_by_strike: pd.DataFrame
    oi_by_strike: pd.DataFrame
    flow_heatmap: pd.DataFrame
    most_active_calls: pd.DataFrame
    most_active_puts: pd.DataFrame
    timestamp: str


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _sanitise(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN volumes with 0 and NaN open interest with 1."""
    df = df.copy()
    for col in ["volume", "openInterest"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0).clip(lower=0)
    if "openInterest" in df.columns:
        df["openInterest"] = df["openInterest"].fillna(1).clip(lower=1)
    for col in ["lastPrice", "bid", "ask", "impliedVolatility"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "strike" in df.columns:
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df = df.dropna(subset=["strike"])
    if "expiration" in df.columns:
        df["expiration"] = df["expiration"].astype(str)
    return df


def _moneyness(strike: float, spot: float, option_type: str) -> str:
    """Classify moneyness relative to spot."""
    ratio = abs(spot - strike) / spot
    if ratio < 0.02:
        return "ATM"
    if option_type == "call":
        return "ITM" if spot > strike else "OTM"
    # put
    return "ITM" if spot < strike else "OTM"


def _spread_pct(bid: float, ask: float) -> float:
    """Percentage spread relative to midpoint; 0 when both sides are zero."""
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return 0.0
    return (ask - bid) / mid


def _utc_now_str() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class OptionsFlowAnalyzer:
    """
    Detects unusual options flow and summarises market-maker / smart-money signals.

    Parameters
    ----------
    calls_df : pd.DataFrame
        Raw calls chain from yfinance with columns:
        strike, lastPrice, bid, ask, volume, openInterest, impliedVolatility,
        expiration (string).
    puts_df : pd.DataFrame
        Same structure for puts.
    spot : float
        Current underlying price.
    ticker : str
        Ticker symbol for labelling purposes.
    """

    def __init__(
        self,
        calls_df: pd.DataFrame,
        puts_df: pd.DataFrame,
        spot: float,
        ticker: str = "",
    ) -> None:
        self.spot = float(spot)
        self.ticker = ticker
        self.calls = _sanitise(calls_df) if calls_df is not None and len(calls_df) > 0 else pd.DataFrame()
        self.puts = _sanitise(puts_df) if puts_df is not None and len(puts_df) > 0 else pd.DataFrame()

    # ── put-call ratio ────────────────────────────────────────────────────────

    def compute_pcr(self) -> dict:
        """
        Compute put-call ratios by volume and open interest.

        Returns a dict with pcr_volume, pcr_oi, totals, and an interpretation
        string (contrarian sentiment view).

        PCR thresholds (contrarian reading):
            > 1.5  -> Extreme Fear  (contrarian bullish)
            1.0-1.5 -> Fear
            0.7-1.0 -> Neutral
            0.5-0.7 -> Greed
            < 0.5  -> Extreme Greed (contrarian bearish)
        """
        call_vol = int(self.calls["volume"].sum()) if "volume" in self.calls.columns else 0
        put_vol  = int(self.puts["volume"].sum())  if "volume" in self.puts.columns  else 0
        call_oi  = int(self.calls["openInterest"].sum()) if "openInterest" in self.calls.columns else 0
        put_oi   = int(self.puts["openInterest"].sum())  if "openInterest" in self.puts.columns  else 0

        pcr_vol = put_vol / max(call_vol, 1)
        pcr_oi  = put_oi  / max(call_oi,  1)

        if pcr_vol > 1.5:
            interpretation = "Extreme Fear"
        elif pcr_vol > 1.0:
            interpretation = "Fear"
        elif pcr_vol > 0.7:
            interpretation = "Neutral"
        elif pcr_vol > 0.5:
            interpretation = "Greed"
        else:
            interpretation = "Extreme Greed"

        return {
            "pcr_volume":     round(pcr_vol, 4),
            "pcr_oi":         round(pcr_oi,  4),
            "total_call_vol": call_vol,
            "total_put_vol":  put_vol,
            "total_call_oi":  call_oi,
            "total_put_oi":   put_oi,
            "interpretation": interpretation,
        }

    # ── unusual activity detection ────────────────────────────────────────────

    def detect_unusual_activity(
        self,
        z_threshold: float = 2.0,
        min_volume: int = 100,
    ) -> list:
        """
        Identify options with abnormally high volume relative to open interest.

        Algorithm per expiry:
        1. vol_oi_ratio = volume / openInterest
        2. z-score within the expiry group
        3. Flag if z > z_threshold AND volume > min_volume
        4. Classify moneyness, signal, and compute smart-money score
        5. Return top 20 sorted by z-score descending
        """
        results: list[UnusualActivity] = []
        spot = self.spot

        for opt_type, df in [("call", self.calls), ("put", self.puts)]:
            if len(df) == 0:
                continue

            df = df.copy()
            df["vol_oi_ratio"] = df["volume"] / df["openInterest"].clip(lower=1)

            for exp, grp in df.groupby("expiration"):
                if len(grp) < 2:
                    continue

                mean_ratio = grp["vol_oi_ratio"].mean()
                std_ratio  = grp["vol_oi_ratio"].std()
                if std_ratio == 0 or np.isnan(std_ratio):
                    continue

                for _, row in grp.iterrows():
                    vol  = int(row["volume"])
                    if vol < min_volume:
                        continue

                    z = (row["vol_oi_ratio"] - mean_ratio) / std_ratio
                    if z < z_threshold:
                        continue

                    strike    = float(row["strike"])
                    oi        = int(row["openInterest"])
                    iv        = float(row["impliedVolatility"])
                    last      = float(row["lastPrice"])
                    bid       = float(row["bid"])
                    ask       = float(row["ask"])
                    sp_pct    = _spread_pct(bid, ask)
                    money     = _moneyness(strike, spot, opt_type)
                    vol_oi_r  = float(row["vol_oi_ratio"])

                    # Signal determination
                    if money == "OTM" and opt_type == "call":
                        signal = "BULLISH"
                    elif money == "OTM" and opt_type == "put":
                        signal = "BEARISH"
                    else:
                        signal = "NEUTRAL"

                    # Smart-money score
                    otm_factor = 1.0 if money == "OTM" else (0.5 if money == "ATM" else 0.2)
                    sms = min(1.0, vol_oi_r * (1.0 / max(sp_pct, 0.01)) * otm_factor)

                    results.append(UnusualActivity(
                        strike=strike,
                        expiration=str(exp),
                        option_type=opt_type,
                        volume=vol,
                        open_interest=oi,
                        volume_oi_ratio=round(vol_oi_r, 4),
                        z_score=round(float(z), 4),
                        implied_vol=round(iv, 4),
                        last_price=round(last, 4),
                        bid=round(bid, 4),
                        ask=round(ask, 4),
                        spread_pct=round(sp_pct, 4),
                        moneyness=money,
                        signal=signal,
                        smart_money_score=round(sms, 4),
                    ))

        results.sort(key=lambda x: x.z_score, reverse=True)
        return results[:20]

    # ── smart money aggregate ─────────────────────────────────────────────────

    def compute_smart_money_score(self) -> float:
        """
        Aggregate smart-money directional indicator in [-1, +1].
        Positive -> smart money positioned bullishly (unusual call activity).
        Negative -> smart money positioned bearishly (unusual put activity).
        Returns 0.0 when no unusual activity is detected.
        """
        unusual = self.detect_unusual_activity()
        if not unusual:
            return 0.0

        call_scores = [u.smart_money_score for u in unusual if u.option_type == "call"]
        put_scores  = [u.smart_money_score for u in unusual if u.option_type == "put"]

        call_total = sum(call_scores)
        put_total  = sum(put_scores)
        denom = call_total + put_total

        if denom == 0:
            return 0.0

        # Ranges from -1 (all puts) to +1 (all calls)
        raw = (call_total - put_total) / denom
        return round(float(np.clip(raw, -1.0, 1.0)), 4)

    # ── volume / OI profiles ──────────────────────────────────────────────────

    def build_volume_by_strike(self) -> pd.DataFrame:
        """
        Aggregate call and put volumes per strike filtered to ±30% of spot.

        Returns DataFrame with columns:
            strike, call_volume, put_volume, net_volume
        sorted by strike ascending.
        """
        lo, hi = self.spot * 0.70, self.spot * 1.30
        empty = pd.DataFrame(columns=["strike", "call_volume", "put_volume", "net_volume"])

        def _agg(df: pd.DataFrame) -> pd.Series:
            if len(df) == 0:
                return pd.Series(dtype=float)
            filtered = df[(df["strike"] >= lo) & (df["strike"] <= hi)]
            return filtered.groupby("strike")["volume"].sum()

        call_vol = _agg(self.calls).rename("call_volume")
        put_vol  = _agg(self.puts).rename("put_volume")

        merged = pd.DataFrame({"call_volume": call_vol, "put_volume": put_vol}).fillna(0)
        if merged.empty:
            return empty

        merged = merged.reset_index()
        merged.columns = ["strike", "call_volume", "put_volume"]
        merged["net_volume"] = merged["call_volume"] - merged["put_volume"]
        merged = merged.sort_values("strike").reset_index(drop=True)
        merged[["call_volume", "put_volume", "net_volume"]] = (
            merged[["call_volume", "put_volume", "net_volume"]].astype(int)
        )
        return merged

    def build_oi_by_strike(self) -> pd.DataFrame:
        """
        Aggregate call and put open interest per strike filtered to ±30% of spot.

        Returns DataFrame with columns:
            strike, call_oi, put_oi
        sorted by strike ascending.
        """
        lo, hi = self.spot * 0.70, self.spot * 1.30
        empty = pd.DataFrame(columns=["strike", "call_oi", "put_oi"])

        def _agg(df: pd.DataFrame) -> pd.Series:
            if len(df) == 0:
                return pd.Series(dtype=float)
            filtered = df[(df["strike"] >= lo) & (df["strike"] <= hi)]
            return filtered.groupby("strike")["openInterest"].sum()

        call_oi = _agg(self.calls).rename("call_oi")
        put_oi  = _agg(self.puts).rename("put_oi")

        merged = pd.DataFrame({"call_oi": call_oi, "put_oi": put_oi}).fillna(0)
        if merged.empty:
            return empty

        merged = merged.reset_index()
        merged.columns = ["strike", "call_oi", "put_oi"]
        merged = merged.sort_values("strike").reset_index(drop=True)
        merged[["call_oi", "put_oi"]] = merged[["call_oi", "put_oi"]].astype(int)
        return merged

    # ── flow heatmap ──────────────────────────────────────────────────────────

    def build_flow_heatmap(self) -> pd.DataFrame:
        """
        Pivot table of net flow (call_volume - put_volume) per strike/expiry.

        Rows   = strikes within ±20% of spot
        Columns = expiration strings
        Values  = net volume (positive = bullish bias)
        Missing cells filled with 0.
        """
        lo, hi = self.spot * 0.80, self.spot * 1.20
        empty = pd.DataFrame()

        def _filter(df: pd.DataFrame, opt: str) -> pd.DataFrame:
            if len(df) == 0:
                return pd.DataFrame()
            out = df[(df["strike"] >= lo) & (df["strike"] <= hi)].copy()
            out["_type"] = opt
            return out[["strike", "expiration", "volume", "_type"]]

        calls_f = _filter(self.calls, "call")
        puts_f  = _filter(self.puts,  "put")

        if calls_f.empty and puts_f.empty:
            return empty

        combined = pd.concat([calls_f, puts_f], ignore_index=True)
        combined["signed_vol"] = combined.apply(
            lambda r: r["volume"] if r["_type"] == "call" else -r["volume"], axis=1
        )

        try:
            pivot = combined.pivot_table(
                index="strike",
                columns="expiration",
                values="signed_vol",
                aggfunc="sum",
            ).fillna(0)
            pivot.columns.name = None
            pivot.index.name = "strike"
        except Exception:
            return empty

        return pivot

    # ── most active ───────────────────────────────────────────────────────────

    def get_most_active(self, top_n: int = 5) -> tuple:
        """
        Return (top_calls, top_puts) DataFrames sorted by volume descending.

        Columns: strike, expiration, volume, openInterest, impliedVolatility,
                 lastPrice, bid, ask
        """
        _cols = ["strike", "expiration", "volume", "openInterest",
                 "impliedVolatility", "lastPrice", "bid", "ask"]

        def _top(df: pd.DataFrame) -> pd.DataFrame:
            if len(df) == 0:
                return pd.DataFrame(columns=_cols)
            avail = [c for c in _cols if c in df.columns]
            out = df[avail].sort_values("volume", ascending=False).head(top_n)
            return out.reset_index(drop=True)

        return _top(self.calls), _top(self.puts)

    # ── net flow score ────────────────────────────────────────────────────────

    def net_flow_score(self) -> float:
        """
        Composite directional score in [-1, +1].

        Components (equal-weighted thirds):
        1. PCR factor     — inverted PCR mapped to [-1, +1]; low PCR => bullish => +1
        2. Unusual bias   — fraction of unusual call z-scores vs put z-scores
        3. Volume ratio   — net call volume as fraction of total volume
        """
        pcr_data = self.compute_pcr()
        pcr = pcr_data["pcr_volume"]

        # PCR factor: PCR=1 -> 0, PCR=0 -> +1, PCR=2 -> -1 (clamped)
        pcr_factor = float(np.clip(1.0 - pcr, -1.0, 1.0))

        # Unusual activity bias
        unusual = self.detect_unusual_activity()
        if unusual:
            call_z_sum = sum(u.z_score for u in unusual if u.option_type == "call")
            put_z_sum  = sum(u.z_score for u in unusual if u.option_type == "put")
            total_z = call_z_sum + put_z_sum
            unusual_factor = (call_z_sum - put_z_sum) / max(total_z, 1e-9)
        else:
            unusual_factor = 0.0

        # Volume ratio factor
        call_vol = pcr_data["total_call_vol"]
        put_vol  = pcr_data["total_put_vol"]
        total_vol = call_vol + put_vol
        if total_vol > 0:
            vol_ratio_factor = float(np.clip((call_vol - put_vol) / total_vol, -1.0, 1.0))
        else:
            vol_ratio_factor = 0.0

        score = (pcr_factor + unusual_factor + vol_ratio_factor) / 3.0
        return round(float(np.clip(score, -1.0, 1.0)), 4)

    # ── full analysis ─────────────────────────────────────────────────────────

    def analyze(self) -> FlowSummary:
        """Run the complete flow analysis and return a FlowSummary."""
        pcr_data        = self.compute_pcr()
        unusual         = self.detect_unusual_activity()
        sms             = self.compute_smart_money_score()
        score           = self.net_flow_score()
        vol_by_strike   = self.build_volume_by_strike()
        oi_by_strike    = self.build_oi_by_strike()
        heatmap         = self.build_flow_heatmap()
        top_calls, top_puts = self.get_most_active(top_n=5)

        if score > 0.15:
            bias = "BULLISH"
        elif score < -0.15:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        return FlowSummary(
            ticker=self.ticker,
            spot=self.spot,
            pcr_volume=pcr_data["pcr_volume"],
            pcr_oi=pcr_data["pcr_oi"],
            total_call_volume=pcr_data["total_call_vol"],
            total_put_volume=pcr_data["total_put_vol"],
            total_call_oi=pcr_data["total_call_oi"],
            total_put_oi=pcr_data["total_put_oi"],
            net_flow_bias=bias,
            net_flow_score=score,
            unusual_activity=unusual,
            smart_money_score=sms,
            volume_by_strike=vol_by_strike,
            oi_by_strike=oi_by_strike,
            flow_heatmap=heatmap,
            most_active_calls=top_calls,
            most_active_puts=top_puts,
            timestamp=_utc_now_str(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_gamma_exposure(
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
    spot: float,
    multiplier: int = 100,
) -> pd.DataFrame:
    """
    Compute Dealer Gamma Exposure (GEX) per strike.

    GEX formula:
        call_gex =  gamma * OI * multiplier * spot^2 * 0.01
        put_gex  = -gamma * OI * multiplier * spot^2 * 0.01

    When gamma is not directly available it is approximated from the Black-Scholes
    formula using the contract's implied volatility and a rough DTE estimate
    (default: 30 days / 365).

    Returns
    -------
    pd.DataFrame with columns: strike, call_gex, put_gex, net_gex
        sorted by strike ascending.
    """
    empty = pd.DataFrame(columns=["strike", "call_gex", "put_gex", "net_gex"])

    calls_s = _sanitise(calls_df) if calls_df is not None and len(calls_df) > 0 else pd.DataFrame()
    puts_s  = _sanitise(puts_df)  if puts_df  is not None and len(puts_df)  > 0 else pd.DataFrame()

    if calls_s.empty and puts_s.empty:
        return empty

    S = float(spot)
    spot_sq = S ** 2

    def _bs_gamma(row: pd.Series, T: float = 30 / 365) -> float:
        """Approximate BS gamma from IV."""
        sigma = float(row.get("impliedVolatility", 0.0))
        K     = float(row.get("strike", S))
        if sigma <= 0 or T <= 0 or K <= 0:
            return 0.0
        try:
            from scipy.stats import norm as _norm
            d1 = (np.log(S / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            return float(_norm.pdf(d1) / (S * sigma * np.sqrt(T)))
        except Exception:
            return 0.0

    def _gex_series(df: pd.DataFrame, sign: float) -> pd.DataFrame:
        if len(df) == 0:
            return pd.DataFrame(columns=["strike", "gex"])
        df = df.copy()
        if "gamma" in df.columns:
            df["gamma"] = pd.to_numeric(df["gamma"], errors="coerce").fillna(0.0)
        else:
            df["gamma"] = df.apply(_bs_gamma, axis=1)
        df["gex"] = sign * df["gamma"] * df["openInterest"] * multiplier * spot_sq * 0.01
        return df.groupby("strike")["gex"].sum().reset_index()

    call_gex = _gex_series(calls_s, sign=1.0).rename(columns={"gex": "call_gex"})
    put_gex  = _gex_series(puts_s,  sign=-1.0).rename(columns={"gex": "put_gex"})

    merged = pd.merge(call_gex, put_gex, on="strike", how="outer").fillna(0.0)
    if merged.empty:
        return empty

    merged["net_gex"] = merged["call_gex"] + merged["put_gex"]
    merged = merged.sort_values("strike").reset_index(drop=True)
    return merged


def identify_max_pain(
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
) -> float:
    """
    Compute the max-pain strike price.

    Max pain is the strike at which aggregate option holder losses (i.e.,
    option writer gains) are maximised — equivalently, the strike that
    minimises total intrinsic value of all outstanding options at expiry.

    For each candidate expiry price P (= each strike in the chain):
        pain(P) = sum_{calls} max(P - K, 0) * call_OI
                + sum_{puts}  max(K - P, 0) * put_OI

    Returns the argmin of pain over all strikes.
    Returns NaN if inputs are empty or invalid.
    """
    calls_s = _sanitise(calls_df) if calls_df is not None and len(calls_df) > 0 else pd.DataFrame()
    puts_s  = _sanitise(puts_df)  if puts_df  is not None and len(puts_df)  > 0 else pd.DataFrame()

    if calls_s.empty and puts_s.empty:
        return float("nan")

    all_strikes = pd.concat(
        [calls_s["strike"], puts_s["strike"]] if not calls_s.empty and not puts_s.empty
        else ([calls_s["strike"]] if not calls_s.empty else [puts_s["strike"]])
    ).dropna().unique()
    all_strikes = np.sort(all_strikes)

    if len(all_strikes) == 0:
        return float("nan")

    call_strikes = calls_s["strike"].values if not calls_s.empty else np.array([])
    call_oi      = calls_s["openInterest"].values if not calls_s.empty else np.array([])
    put_strikes  = puts_s["strike"].values  if not puts_s.empty  else np.array([])
    put_oi       = puts_s["openInterest"].values  if not puts_s.empty  else np.array([])

    min_pain  = float("inf")
    max_pain_strike = all_strikes[0]

    for p in all_strikes:
        call_pain = float(np.sum(np.maximum(p - call_strikes, 0.0) * call_oi)) if len(call_strikes) > 0 else 0.0
        put_pain  = float(np.sum(np.maximum(put_strikes - p,  0.0) * put_oi))  if len(put_strikes)  > 0 else 0.0
        total_pain = call_pain + put_pain
        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = float(p)

    return max_pain_strike
