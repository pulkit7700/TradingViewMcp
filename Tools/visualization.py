"""
Plotly-based Visualiser
------------------------
All charts used in the Streamlit app.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List

TEMPLATE = "simple_white"
COLOR_CALL    = "#00B894"
COLOR_PUT     = "#E84393"
COLOR_NEUTRAL = "#5B5BD6"
COLOR_GRID = [
    "#5B5BD6", "#E84393", "#00B894", "#7C3AED", "#D97706",
    "#0EA5E9", "#F43F5E", "#059669", "#9333EA", "#B45309"
]

# Shared layout defaults for a clean, minimal light look
_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248,250,255,0.85)",
    font=dict(family="Inter, JetBrains Mono, sans-serif", color="#334155", size=11),
    xaxis=dict(
        gridcolor="rgba(99,102,241,0.10)", zeroline=False,
        linecolor="rgba(99,102,241,0.14)", tickfont=dict(size=10, color="#64748B"),
    ),
    yaxis=dict(
        gridcolor="rgba(99,102,241,0.10)", zeroline=False,
        linecolor="rgba(99,102,241,0.14)", tickfont=dict(size=10, color="#64748B"),
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(99,102,241,0.15)",
        borderwidth=1, font=dict(size=10, color="#334155"),
    ),
    margin=dict(l=48, r=16, t=44, b=36),
    title_font=dict(size=13, color="#0F172A", family="Inter, sans-serif"),
)


def _apply_clean_layout(fig: go.Figure, **kwargs) -> go.Figure:
    """Apply shared minimal layout to any figure."""
    update = dict(_LAYOUT_DEFAULTS)
    # propagate axis defaults for subplots (xaxis2, yaxis2, …)
    for k, v in list(update.items()):
        if k in ("xaxis", "yaxis"):
            for i in range(2, 9):
                update[f"{k}{i}"] = v
    update.update(kwargs)
    fig.update_layout(**update)
    return fig


class Visualizer:

    # ── Monte Carlo paths ────────────────────────────────────────────────────

    @staticmethod
    def plot_mc_paths(
        paths: np.ndarray,
        S0: float,
        percentiles: List[float] = [5, 25, 50, 75, 95],
        n_display: int = 50,
        K: Optional[float] = None,
    ) -> go.Figure:
        n_steps, n_paths = paths.shape

        # Downsample time axis to max 100 points to keep payload small
        max_t_points = 100
        if n_steps > max_t_points:
            t_idx = np.linspace(0, n_steps - 1, max_t_points, dtype=int)
            paths_plot = paths[t_idx, :]
        else:
            t_idx = np.arange(n_steps)
            paths_plot = paths
        t_arr = t_idx / (n_steps - 1)

        fig = go.Figure()

        # Show sample paths (thin, transparent) — capped at 30 to reduce payload
        n_show = min(n_display, n_paths, 30)
        idx = np.random.choice(n_paths, n_show, replace=False)
        for i in idx:
            fig.add_trace(go.Scatter(
                x=t_arr, y=paths_plot[:, i].tolist(),
                mode="lines", line=dict(width=0.5, color="rgba(100,149,237,0.25)"),
                showlegend=False, hoverinfo="skip",
            ))

        # Percentile bands (computed on full paths, plotted on downsampled axis)
        pct_data = np.percentile(paths_plot, percentiles, axis=1)
        colors_pct = ["#F43F5E", "#F59E0B", COLOR_CALL, "#F59E0B", "#F43F5E"]
        names_pct = [f"{p}th pct" for p in percentiles]
        for i, (p, name, c) in enumerate(zip(pct_data, names_pct, colors_pct)):
            dash = "dash" if i in (0, 4) else ("dot" if i in (1, 3) else "solid")
            width = 2 if i == 2 else 1
            fig.add_trace(go.Scatter(
                x=t_arr, y=p, mode="lines",
                line=dict(color=c, width=width, dash=dash),
                name=name,
            ))

        # Strike line
        if K is not None:
            fig.add_hline(y=K, line_dash="longdash", line_color="rgba(140,158,200,0.55)",
                          opacity=0.6, annotation_text=f"Strike K={K:.2f}")

        # Initial price
        fig.add_hline(y=S0, line_dash="dot", line_color="#F59E0B",
                      opacity=0.8, annotation_text=f"S₀={S0:.2f}")

        _apply_clean_layout(fig,
            title="Monte Carlo Price Path Simulation",
            xaxis_title="Time (fraction of expiry)",
            yaxis_title="Price",
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        bgcolor="rgba(13,21,48,0.8)", bordercolor="rgba(99,102,241,0.2)",
                        borderwidth=1, font=dict(size=10)),
        )
        return fig

    # ── Terminal price distribution ──────────────────────────────────────────

    @staticmethod
    def plot_terminal_distribution(
        terminal: np.ndarray,
        K: float,
        S0: float,
        call_price: float,
        put_price: float,
    ) -> go.Figure:
        # Downsample terminal array — histogram only needs ~5000 representative points
        if len(terminal) > 5000:
            terminal = np.random.choice(terminal, 5000, replace=False)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Terminal Price Distribution", "Cumulative Distribution (CDF)"])

        # Histogram
        fig.add_trace(go.Histogram(
            x=terminal.tolist(), nbinsx=80, histnorm="probability density",
            marker_color=COLOR_NEUTRAL, opacity=0.7, name="Density",
        ), row=1, col=1)

        # Strike and S0 lines
        for val, label, col in [(K, f"K={K:.2f}", "white"), (S0, f"S₀={S0:.2f}", "yellow")]:
            fig.add_vline(x=val, line_dash="dash", line_color=col,
                          annotation_text=label, row=1, col=1)

        # CDF — downsample to 500 points for the line chart
        sorted_t = np.sort(terminal)
        cdf_full = np.arange(1, len(sorted_t) + 1) / len(sorted_t)
        cdf_idx  = np.linspace(0, len(sorted_t) - 1, min(500, len(sorted_t)), dtype=int)
        fig.add_trace(go.Scatter(
            x=sorted_t[cdf_idx].tolist(), y=cdf_full[cdf_idx].tolist(),
            mode="lines", line=dict(color=COLOR_CALL, width=2), name="CDF",
        ), row=1, col=2)
        fig.add_vline(x=K, line_dash="dash", line_color="rgba(140,158,200,0.55)",
                      annotation_text=f"K={K:.2f}", row=1, col=2)

        # Add annotations for call/put prices
        fig.add_annotation(
            text=f"Call: {call_price:.4f}  |  Put: {put_price:.4f}",
            xref="paper", yref="paper", x=0.02, y=0.97,
            showarrow=False,
            font=dict(color="#0F172A", size=11, family="JetBrains Mono, monospace"),
            bgcolor="rgba(13,21,48,0.90)", bordercolor="rgba(99,102,241,0.35)", borderwidth=1,
            borderpad=6,
        )

        _apply_clean_layout(fig, height=430, showlegend=False)
        return fig

    # ── MC convergence ───────────────────────────────────────────────────────

    @staticmethod
    def plot_mc_convergence(
        paths: np.ndarray, K: float, r: float, T: float
    ) -> go.Figure:
        n_paths = paths.shape[1]
        terminal = paths[-1]
        checkpoints = np.geomspace(10, n_paths, 50).astype(int)
        call_prices, put_prices, ci_upper, ci_lower = [], [], [], []

        for n in checkpoints:
            t = terminal[:n]
            disc = np.exp(-r * T)
            c = disc * np.maximum(t - K, 0).mean()
            p = disc * np.maximum(K - t, 0).mean()
            se = disc * np.maximum(t - K, 0).std() / np.sqrt(n)
            call_prices.append(c)
            put_prices.append(p)
            ci_upper.append(c + 1.96 * se)
            ci_lower.append(c - 1.96 * se)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=checkpoints, y=call_prices,
            mode="lines", line=dict(color=COLOR_CALL), name="Call Price",
        ))
        fig.add_trace(go.Scatter(
            x=checkpoints, y=ci_upper,
            mode="lines", line=dict(color=COLOR_CALL, dash="dot", width=1),
            name="95% CI Upper", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=checkpoints, y=ci_lower,
            mode="lines", line=dict(color=COLOR_CALL, dash="dot", width=1),
            fill="tonexty", fillcolor="rgba(0,212,170,0.08)",
            name="95% CI", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=checkpoints, y=put_prices,
            mode="lines", line=dict(color=COLOR_PUT), name="Put Price",
        ))

        _apply_clean_layout(fig,
            title="MC Option Price Convergence",
            xaxis_title="Number of Paths (log scale)",
            yaxis_title="Option Price",
            xaxis_type="log",
            height=400,
        )
        return fig

    # ── Greeks profile (vs spot) ─────────────────────────────────────────────

    @staticmethod
    def plot_greeks_vs_spot(profiles: dict, option_type: str = "call") -> go.Figure:
        S_arr = profiles["spot"]
        prefix = option_type
        greek_list = ["delta", "gamma", "theta", "vega", "rho"]
        greek_labels = ["Delta", "Gamma", "Theta (per day)", "Vega (per 1% vol)", "Rho (per 1% rate)"]

        fig = make_subplots(rows=2, cols=3,
                            subplot_titles=greek_labels + [""],
                            shared_xaxes=False)

        positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
        colors = [COLOR_CALL if option_type == "call" else COLOR_PUT] * 5

        for (row, col), greek, label, color in zip(positions, greek_list, greek_labels, colors):
            key = f"{prefix}_{greek}"
            if key in profiles:
                fig.add_trace(go.Scatter(
                    x=S_arr, y=profiles[key], mode="lines",
                    line=dict(color=color, width=2), name=label, showlegend=False,
                ), row=row, col=col)
                fig.update_xaxes(title_text="Spot Price", row=row, col=col)
                fig.update_yaxes(title_text=label, row=row, col=col)

        _apply_clean_layout(fig,
            title=f"{option_type.capitalize()} Greeks vs Spot Price",
            height=550,
        )
        return fig

    @staticmethod
    def plot_greeks_vs_vol(profiles: dict, option_type: str = "call") -> go.Figure:
        vol_arr = profiles["vol"] * 100   # pct
        prefix = option_type
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=["Delta", "Gamma", "Theta", "Vega"])

        pairs = [("delta", 1, 1), ("gamma", 1, 2), ("theta", 2, 1), ("vega", 2, 2)]
        color = COLOR_CALL if option_type == "call" else COLOR_PUT
        for (g, row, col) in pairs:
            key = f"{prefix}_{g}"
            if key in profiles:
                fig.add_trace(go.Scatter(
                    x=vol_arr, y=profiles[key], mode="lines",
                    line=dict(color=color, width=2), showlegend=False,
                ), row=row, col=col)
                fig.update_xaxes(title_text="Implied Volatility (%)", row=row, col=col)

        _apply_clean_layout(fig,
            title=f"{option_type.capitalize()} Greeks vs Implied Volatility",
            height=480,
        )
        return fig

    @staticmethod
    def plot_greeks_vs_time(profiles: dict, option_type: str = "call") -> go.Figure:
        dte_arr = profiles["dte"]
        prefix = option_type
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=["Delta", "Gamma", "Theta", "Vega"])

        pairs = [("delta", 1, 1), ("gamma", 1, 2), ("theta", 2, 1), ("vega", 2, 2)]
        color = COLOR_CALL if option_type == "call" else COLOR_PUT
        for (g, row, col) in pairs:
            key = f"{prefix}_{g}"
            if key in profiles:
                fig.add_trace(go.Scatter(
                    x=dte_arr, y=profiles[key], mode="lines",
                    line=dict(color=color, width=2), showlegend=False,
                ), row=row, col=col)
                fig.update_xaxes(title_text="Days to Expiry", row=row, col=col)

        _apply_clean_layout(fig,
            title=f"{option_type.capitalize()} Greeks vs Time to Expiry",
            height=480,
        )
        return fig

    # ── Strategy P&L ─────────────────────────────────────────────────────────

    @staticmethod
    def plot_strategy_pnl(
        strategy,
        S_range: tuple,
        current_S: float,
        n: int = 300,
    ) -> go.Figure:
        S_arr = np.linspace(S_range[0], S_range[1], n)
        pnl = strategy.pnl_at_expiry(S_arr)

        # Colour above/below zero
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=S_arr, y=np.where(pnl >= 0, pnl, None),
            mode="lines", fill="tozeroy",
            line=dict(color=COLOR_CALL, width=2),
            fillcolor="rgba(0,212,170,0.12)", name="Profit",
        ))
        fig.add_trace(go.Scatter(
            x=S_arr, y=np.where(pnl < 0, pnl, None),
            mode="lines", fill="tozeroy",
            line=dict(color=COLOR_PUT, width=2),
            fillcolor="rgba(244,63,94,0.12)", name="Loss",
        ))
        fig.add_vline(x=current_S, line_dash="dot", line_color="#F59E0B",
                      annotation_text=f"Current S={current_S:.2f}")

        # Strike annotations for each leg
        for leg in strategy.legs:
            fig.add_vline(x=leg.strike, line_dash="dash", line_color="rgba(140,158,200,0.5)",
                          opacity=0.5, annotation_text=f"K={leg.strike:.0f}")

        fig.add_hline(y=0, line_dash="solid", line_color="rgba(140,158,200,0.3)", opacity=1)

        _apply_clean_layout(fig,
            title=f"P&L at Expiry: {strategy.name}",
            xaxis_title="Underlying Price at Expiry",
            yaxis_title="Profit / Loss ($)",
            height=440,
            legend=dict(orientation="h", y=1.01,
                        bgcolor="rgba(13,21,48,0.8)", bordercolor="rgba(99,102,241,0.2)",
                        borderwidth=1, font=dict(size=10)),
        )
        return fig

    # ── IV Surface ───────────────────────────────────────────────────────────

    @staticmethod
    def plot_iv_surface(
        strikes: np.ndarray,
        dtes: np.ndarray,
        iv_matrix: np.ndarray,
    ) -> go.Figure:
        """
        Static / single-snapshot IV surface (used as fallback when no
        historical series is available).  Kept for backward compatibility.
        """
        _iv_pct = np.where(np.isnan(iv_matrix), np.nanmean(iv_matrix), iv_matrix) * 100
        fig = go.Figure(data=[go.Surface(
            x=dtes, y=strikes, z=_iv_pct,
            colorscale="Viridis",
            colorbar=dict(title="IV (%)", tickfont=dict(color="#334155")),
            hovertemplate="DTE: %{x:.0f}d<br>Strike: %{y:.2f}<br>IV: %{z:.1f}%<extra></extra>",
        )])
        fig.update_layout(
            title=dict(text="Implied Volatility Surface",
                       font=dict(size=13, color="#0F172A", family="Inter, sans-serif")),
            scene=dict(
                xaxis_title="Days to Expiry", yaxis_title="Strike", zaxis_title="IV (%)",
                xaxis=dict(gridcolor="rgba(99,102,241,0.12)",
                           backgroundcolor="rgba(240,244,255,0.85)"),
                yaxis=dict(gridcolor="rgba(99,102,241,0.12)",
                           backgroundcolor="rgba(240,244,255,0.85)"),
                zaxis=dict(gridcolor="rgba(99,102,241,0.12)",
                           backgroundcolor="rgba(240,244,255,0.70)"),
                bgcolor="rgba(248,250,255,0.0)",
                camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
            ),
            paper_bgcolor="rgba(0,0,0,0)", template=TEMPLATE,
            height=520, margin=dict(l=0, r=0, t=44, b=20),
            font=dict(family="Inter, sans-serif", color="#334155", size=11),
        )
        return fig

    @staticmethod
    def plot_iv_surface_animated(
        moneyness: np.ndarray,
        dtes: np.ndarray,
        frames: list,           # [(date_str, iv_matrix), ...]
        spot: float = 1.0,
    ) -> go.Figure:
        """
        Animated IV surface showing how the surface evolves through time.

        Each frame is one trading day snapshot; the slider lets you scrub
        through history and the Play button animates the full sequence.

        Parameters
        ----------
        moneyness : (n_strikes,)  K/S ratios, e.g. [0.80 … 1.20]
        dtes      : (n_dtes,)     days to expiry
        frames    : [(date_str, iv_matrix), ...]  iv_matrix in decimal (0..1)
        spot      : current spot price (used for axis label only)
        """
        if not frames:
            return go.Figure()

        strike_labels = (moneyness * spot).round(2) if spot != 1.0 else moneyness

        # ── colour scale range fixed across all frames for consistent colour ──
        all_iv = np.concatenate([f[1] for f in frames])
        z_min  = float(np.nanpercentile(all_iv, 2))  * 100
        z_max  = float(np.nanpercentile(all_iv, 98)) * 100

        def _surface(iv_mat, date_str):
            iv_pct = np.clip(iv_mat * 100, z_min, z_max)
            return go.Surface(
                x=dtes,
                y=strike_labels,
                z=iv_pct,
                colorscale="Viridis",
                cmin=z_min, cmax=z_max,
                showscale=True,
                colorbar=dict(
                    title=dict(text="IV (%)", font=dict(color="#334155", size=11)),
                    tickfont=dict(color="#334155", size=10),
                    len=0.7, thickness=14,
                ),
                hovertemplate=(
                    f"<b>{date_str}</b><br>"
                    "DTE: %{x:.0f}d<br>"
                    "Strike: %{y:.2f}<br>"
                    "IV: %{z:.1f}%<extra></extra>"
                ),
                contours=dict(
                    z=dict(show=True, usecolormap=True,
                           highlightcolor="rgba(99,102,241,0.4)", project_z=True),
                ),
            )

        # First frame as initial data
        fig = go.Figure(data=[_surface(frames[0][1], frames[0][0])])

        # Build Plotly animation frames
        plotly_frames = []
        for date_str, iv_mat in frames:
            plotly_frames.append(go.Frame(
                data=[_surface(iv_mat, date_str)],
                name=date_str,
                layout=dict(
                    title=dict(
                        text=f"IV Surface — {date_str}",
                        font=dict(size=13, color="#0F172A", family="Inter, sans-serif"),
                    )
                ),
            ))
        fig.frames = plotly_frames

        # Slider steps — show every N-th date label to avoid crowding
        n_frames = len(plotly_frames)
        label_every = max(1, n_frames // 12)
        slider_steps = [
            dict(
                args=[[f.name], dict(
                    frame=dict(duration=160, redraw=True),
                    mode="immediate",
                    transition=dict(duration=80, easing="cubic-in-out"),
                )],
                label=f.name if i % label_every == 0 else "",
                method="animate",
            )
            for i, f in enumerate(plotly_frames)
        ]

        fig.update_layout(
            title=dict(
                text=f"IV Surface — {frames[0][0]}",
                font=dict(size=13, color="#0F172A", family="Inter, sans-serif"),
            ),
            scene=dict(
                xaxis_title="Days to Expiry",
                yaxis_title="Strike (K)" if spot == 1.0 else "Strike (₹)",
                zaxis_title="IV (%)",
                xaxis=dict(gridcolor="rgba(99,102,241,0.12)",
                           backgroundcolor="rgba(240,244,255,0.85)",
                           tickfont=dict(color="#334155")),
                yaxis=dict(gridcolor="rgba(99,102,241,0.12)",
                           backgroundcolor="rgba(240,244,255,0.85)",
                           tickfont=dict(color="#334155")),
                zaxis=dict(gridcolor="rgba(99,102,241,0.12)",
                           backgroundcolor="rgba(240,244,255,0.70)",
                           tickfont=dict(color="#334155"),
                           range=[z_min * 0.9, z_max * 1.05]),
                bgcolor="rgba(248,250,255,0.0)",
                camera=dict(eye=dict(x=1.7, y=1.4, z=0.85)),
                aspectmode="manual",
                aspectratio=dict(x=1.2, y=1.0, z=0.7),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,255,0.85)",
            template=TEMPLATE,
            height=600,
            margin=dict(l=0, r=0, t=50, b=120),
            font=dict(family="Inter, sans-serif", color="#334155", size=11),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                x=0.02, y=-0.12,
                xanchor="left", yanchor="top",
                direction="left",
                pad=dict(r=8, t=0),
                buttons=[
                    dict(
                        label="▶  Play",
                        method="animate",
                        args=[None, dict(
                            frame=dict(duration=160, redraw=True),
                            fromcurrent=True,
                            mode="immediate",
                            transition=dict(duration=80, easing="cubic-in-out"),
                        )],
                    ),
                    dict(
                        label="⏸  Pause",
                        method="animate",
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                        )],
                    ),
                ],
                font=dict(color="#0F172A", size=12),
                bgcolor="rgba(99,102,241,0.15)",
                bordercolor="rgba(99,102,241,0.5)",
                borderwidth=1,
            )],
            sliders=[dict(
                active=0,
                steps=slider_steps,
                x=0.02, len=0.96,
                xanchor="left", y=-0.06, yanchor="top",
                pad=dict(b=0, t=30),
                currentvalue=dict(
                    prefix="Date: ",
                    visible=True,
                    xanchor="center",
                    font=dict(color="#0F172A", size=12, family="Inter, sans-serif"),
                ),
                font=dict(color="#64748B", size=9),
                bgcolor="rgba(240,244,255,0.8)",
                activebgcolor="rgba(99,102,241,0.4)",
                bordercolor="rgba(99,102,241,0.25)",
                borderwidth=1,
                tickcolor="rgba(99,102,241,0.3)",
                minorticklen=0,
            )],
        )
        return fig

    @staticmethod
    def plot_vol_smile(
        strikes: np.ndarray,
        ivs: np.ndarray,
        S: float,
        title: str = "Volatility Smile",
    ) -> go.Figure:
        moneyness = strikes / S
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=strikes, y=ivs * 100, mode="lines+markers",
            line=dict(color=COLOR_NEUTRAL, width=2),
            marker=dict(size=6), name="IV",
        ))
        fig.add_vline(x=S, line_dash="dash", line_color="#F59E0B",
                      annotation_text=f"ATM S={S:.2f}")
        _apply_clean_layout(fig,
            title=title,
            xaxis_title="Strike Price",
            yaxis_title="Implied Volatility (%)",
            height=400,
        )
        return fig

    # ── Historical volatility ─────────────────────────────────────────────────

    @staticmethod
    def plot_historical_vol(
        hv_series: pd.Series,
        price_series: pd.Series,
        ticker: str,
    ) -> go.Figure:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=[f"{ticker} Close Price", "Historical Volatility (30d)"],
                            row_heights=[0.55, 0.45])

        fig.add_trace(go.Scatter(
            x=price_series.index, y=price_series.values,
            mode="lines", line=dict(color=COLOR_NEUTRAL), name="Close",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=hv_series.index, y=hv_series.values * 100,
            mode="lines", line=dict(color=COLOR_CALL), fill="tozeroy",
            fillcolor="rgba(0,212,170,0.08)", name="30d HV (%)",
        ), row=2, col=1)

        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="HV (%)", row=2, col=1)
        _apply_clean_layout(fig, height=520, showlegend=False,
                            title=f"{ticker} — Price & Historical Volatility")
        return fig

    # ── Model comparison bar chart ────────────────────────────────────────────

    @staticmethod
    def plot_model_comparison(results: dict) -> go.Figure:
        models = list(results.keys())
        calls = [r.call for r in results.values()]
        puts = [r.put for r in results.values()]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Call", x=models, y=calls,
            marker_color=COLOR_CALL, text=[f"{c:.4f}" for c in calls],
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            name="Put", x=models, y=puts,
            marker_color=COLOR_PUT, text=[f"{p:.4f}" for p in puts],
            textposition="outside",
        ))
        _apply_clean_layout(fig,
            title="Option Price by Model",
            xaxis_title="Model",
            yaxis_title="Price",
            barmode="group",
            height=380,
        )
        return fig

    # ── Put-Call parity gauge ─────────────────────────────────────────────────

    @staticmethod
    def plot_pcp_check(call: float, put: float, S: float, K: float,
                       r: float, T: float, q: float = 0.0) -> go.Figure:
        lhs = call - put
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        parity_error = abs(lhs - rhs)

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=parity_error,
            title={"text": "Put-Call Parity Error"},
            delta={"reference": 0.0, "relative": False},
            gauge={
                "axis": {"range": [0, max(0.5, parity_error * 2)]},
                "bar": {"color": COLOR_CALL if parity_error < 0.01 else COLOR_PUT},
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": 0.01,
                },
            },
        ))
        _apply_clean_layout(fig, height=250)
        return fig

    # ── Candlestick / intraday chart ──────────────────────────────────────────

    @staticmethod
    def plot_candlestick(df: pd.DataFrame, ticker: str, title: str = "") -> go.Figure:
        """OHLCV candlestick with volume bars."""
        # Limit to last 500 bars to keep payload small
        if len(df) > 500:
            df = df.iloc[-500:]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=[title or f"{ticker} Price", "Volume"],
                            row_heights=[0.72, 0.28])

        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing_line_color=COLOR_CALL,
            decreasing_line_color=COLOR_PUT,
            name="OHLC",
        ), row=1, col=1)

        colors = [COLOR_CALL if c >= o else COLOR_PUT
                  for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            marker_color=colors, opacity=0.6, name="Volume",
        ), row=2, col=1)

        fig.update_xaxes(rangeslider_visible=False)
        _apply_clean_layout(fig,
            height=520,
            showlegend=False,
            xaxis2_title="Date/Time",
            yaxis_title="Price",
            yaxis2_title="Volume",
        )
        return fig

    # ── Returns distribution ──────────────────────────────────────────────────

    @staticmethod
    def plot_returns_distribution(returns: pd.Series, ticker: str) -> go.Figure:
        from scipy.stats import norm as sp_norm
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Daily Log-Return Distribution", "Q-Q Plot vs Normal"])

        r = returns.dropna()
        # Cap at 10,000 points so the chart stays well under the size limit
        if len(r) > 10000:
            r = pd.Series(np.random.choice(r.values, 10000, replace=False))
        mu, sigma = float(r.mean()), float(r.std())

        # Histogram + normal fit
        fig.add_trace(go.Histogram(
            x=r, nbinsx=80, histnorm="probability density",
            marker_color=COLOR_NEUTRAL, opacity=0.7, name="Returns",
        ), row=1, col=1)

        x_range = np.linspace(r.min(), r.max(), 200)
        fig.add_trace(go.Scatter(
            x=x_range, y=sp_norm.pdf(x_range, mu, sigma),
            mode="lines", line=dict(color="#5B5BD6", width=1.5, dash="dot"), name="Normal fit",
        ), row=1, col=1)
        fig.add_vline(x=0, line_dash="dot", line_color="#F59E0B", row=1, col=1)

        # Q-Q plot
        sorted_r = np.sort(r)
        n = len(sorted_r)
        theoretical = sp_norm.ppf(np.linspace(0.01, 0.99, n), mu, sigma)
        fig.add_trace(go.Scatter(
            x=theoretical, y=sorted_r, mode="markers",
            marker=dict(color=COLOR_NEUTRAL, size=3, opacity=0.5), name="Q-Q",
            showlegend=False,
        ), row=1, col=2)
        diag = [min(theoretical.min(), sorted_r.min()), max(theoretical.max(), sorted_r.max())]
        fig.add_trace(go.Scatter(
            x=diag, y=diag, mode="lines",
            line=dict(color="rgba(140,158,200,0.45)", dash="dash", width=1),
            showlegend=False,
        ), row=1, col=2)

        fig.update_xaxes(title_text="Log Return", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Quantile", row=1, col=2)
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_yaxes(title_text="Observed Quantile", row=1, col=2)
        _apply_clean_layout(fig,
            title=f"{ticker} Return Distribution",
            height=420, showlegend=False,
        )
        return fig

    # ── Multi-window HV chart ─────────────────────────────────────────────────

    @staticmethod
    def plot_multi_hv(hv_df: pd.DataFrame, ticker: str) -> go.Figure:
        """Plot multiple rolling HV windows on one chart."""
        # Limit to last 500 rows
        if len(hv_df) > 500:
            hv_df = hv_df.iloc[-500:]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=[f"{ticker} Price", "Rolling Historical Volatility"],
                            row_heights=[0.45, 0.55])

        fig.add_trace(go.Scatter(
            x=hv_df.index, y=hv_df["Close"], mode="lines",
            line=dict(color=COLOR_NEUTRAL, width=1.5), name="Close",
        ), row=1, col=1)

        hv_cols = [c for c in hv_df.columns if c.startswith("HV")]
        for i, col in enumerate(hv_cols):
            fig.add_trace(go.Scatter(
                x=hv_df.index, y=hv_df[col] * 100,
                mode="lines", line=dict(width=1.5, color=COLOR_GRID[i % len(COLOR_GRID)]),
                name=col,
            ), row=2, col=1)

        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        _apply_clean_layout(fig,
            height=540,
            legend=dict(orientation="h", y=1.02,
                        bgcolor="rgba(13,21,48,0.8)", bordercolor="rgba(99,102,241,0.2)",
                        borderwidth=1, font=dict(size=10)),
        )
        return fig

    # ── Volatility cone ───────────────────────────────────────────────────────

    @staticmethod
    def plot_vol_cone(cone_df: pd.DataFrame, current_iv: float = None) -> go.Figure:
        """Shaded volatility cone chart."""
        fig = go.Figure()

        windows = cone_df["Window"].tolist()

        # Shaded bands
        fig.add_trace(go.Scatter(
            x=windows, y=(cone_df["90th"] * 100).tolist(),
            mode="lines", line=dict(width=0), showlegend=False,
            name="90th", fillcolor="rgba(99,102,241,0.09)",
        ))
        fig.add_trace(go.Scatter(
            x=windows, y=(cone_df["10th"] * 100).tolist(),
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(99,102,241,0.09)", name="10-90th pct",
        ))
        fig.add_trace(go.Scatter(
            x=windows, y=(cone_df["75th"] * 100).tolist(),
            mode="lines", line=dict(width=0), showlegend=False,
            fillcolor="rgba(99,102,241,0.18)",
        ))
        fig.add_trace(go.Scatter(
            x=windows, y=(cone_df["25th"] * 100).tolist(),
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(99,102,241,0.18)", name="25-75th pct",
        ))

        # Median line
        fig.add_trace(go.Scatter(
            x=windows, y=(cone_df["Median"] * 100).tolist(),
            mode="lines+markers",
            line=dict(color=COLOR_NEUTRAL, width=2, dash="dot"),
            marker=dict(size=6), name="Median",
        ))

        # Current HV line
        fig.add_trace(go.Scatter(
            x=windows, y=(cone_df["Current"] * 100).tolist(),
            mode="lines+markers",
            line=dict(color=COLOR_CALL, width=2.5),
            marker=dict(size=8, symbol="circle"), name="Current HV",
        ))

        # Current IV horizontal line
        if current_iv is not None:
            fig.add_hline(
                y=current_iv * 100, line_dash="dash", line_color="#F59E0B",
                annotation_text=f"Current IV {current_iv*100:.1f}%",
                annotation_position="right",
            )

        _apply_clean_layout(fig,
            title="Volatility Cone",
            xaxis_title="Rolling Window (days)",
            yaxis_title="Annualised Volatility (%)",
            height=430,
            legend=dict(orientation="h", y=1.02,
                        bgcolor="rgba(13,21,48,0.8)", bordercolor="rgba(99,102,241,0.2)",
                        borderwidth=1, font=dict(size=10)),
        )
        return fig

    # ── HV vs IV comparison ───────────────────────────────────────────────────

    @staticmethod
    def plot_hv_iv(hv_series: pd.Series, iv_series: pd.Series, ticker: str) -> go.Figure:
        """Overlay HV and IV time series with IV premium shading."""
        fig = go.Figure()

        # IV
        fig.add_trace(go.Scatter(
            x=iv_series.index, y=iv_series.values * 100,
            mode="lines", line=dict(color=COLOR_PUT, width=1.5),
            name="Implied Vol (IV)", fill=None,
        ))
        # HV
        fig.add_trace(go.Scatter(
            x=hv_series.index, y=hv_series.values * 100,
            mode="lines", line=dict(color=COLOR_CALL, width=1.5),
            name="Historical Vol (HV 30d)",
        ))

        _apply_clean_layout(fig,
            title=f"{ticker} — Implied vs Historical Volatility",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            height=380,
            legend=dict(orientation="h", y=1.02,
                        bgcolor="rgba(13,21,48,0.8)", bordercolor="rgba(99,102,241,0.2)",
                        borderwidth=1, font=dict(size=10)),
        )
        return fig

    # ── Scenario analysis table chart ─────────────────────────────────────────

    @staticmethod
    def plot_scenario_table(scenarios: list) -> go.Figure:
        """Heatmap-style table of scenario P&L results."""
        df = pd.DataFrame(scenarios)
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df.columns),
                fill_color="rgba(240,244,255,0.90)",
                line_color="rgba(99,102,241,0.2)",
                font=dict(color="#64748B", size=11, family="Inter, sans-serif"),
                align="center",
            ),
            cells=dict(
                values=[df[c] for c in df.columns],
                fill_color=[
                    ["rgba(0,212,170,0.10)" if str(v).startswith("+") or
                     (isinstance(v, (int, float)) and v > 0) else
                     "rgba(244,63,94,0.10)" if str(v).startswith("-") or
                     (isinstance(v, (int, float)) and v < 0) else
                     "rgba(13,21,48,0.7)" for v in df[c]] for c in df.columns
                ],
                line_color="rgba(99,102,241,0.12)",
                font=dict(color="#0F172A", size=11, family="JetBrains Mono, monospace"),
                align="center",
            )
        )])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            font=dict(family="Inter, sans-serif"),
        )
        return fig

    # ── Payoff diagram (individual option) ───────────────────────────────────

    @staticmethod
    def plot_single_payoff(
        S0: float, K: float, premium: float,
        option_type: str, direction: str,
    ) -> go.Figure:
        S_arr = np.linspace(K * 0.60, K * 1.40, 300)
        sign = 1 if direction == "long" else -1
        if option_type == "call":
            payoff = sign * (np.maximum(S_arr - K, 0) - premium)
        else:
            payoff = sign * (np.maximum(K - S_arr, 0) - premium)

        color = COLOR_CALL if option_type == "call" else COLOR_PUT
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=S_arr, y=payoff, mode="lines",
            line=dict(color=color, width=2.5), name="P&L at Expiry",
        ))
        fig.add_hline(y=0, line_dash="solid", line_color="rgba(140,158,200,0.3)", opacity=1)
        fig.add_vline(x=S0, line_dash="dot", line_color="#F59E0B",
                      annotation_text=f"S₀={S0:.2f}")
        fig.add_vline(x=K, line_dash="dash", line_color="rgba(140,158,200,0.5)",
                      annotation_text=f"K={K:.2f}")
        _apply_clean_layout(fig,
            title=f"{direction.capitalize()} {option_type.capitalize()} P&L at Expiry",
            xaxis_title="Spot at Expiry",
            yaxis_title="P&L",
            height=380,
        )
        return fig
