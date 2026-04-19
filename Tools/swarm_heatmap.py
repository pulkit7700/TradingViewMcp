"""
Swarm Intelligence Trading — 3D Heatmap & Visualization
========================================================
Advanced quant-style 3D heatmaps showing:
  - Money flow across sectors/tickers over time
  - Pheromone trail visualization (ACO paths)
  - Force map (Katz 2011 method)
  - Swarm state dashboard data
Uses Plotly for interactive 3D surfaces.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource


def create_money_flow_heatmap_3d(
    features_cache: Dict[str, pd.DataFrame],
    tickers: List[str],
    window: int = 20,
    title: str = "3D Money Flow Heatmap"
) -> go.Figure:
    """
    3D Surface heatmap showing money flow across tickers animated over time.
    X, Y = Ticker Grid, Z = Pheromone strength
    Color = Net direction (buy=green, sell=red) using Seaborn diverging aesthetic
    """
    import numpy as np

    pheromone_data = []
    direction_data = []
    time_index = None
    valid_tickers = []

    for ticker in tickers:
        features = features_cache.get(ticker)
        if features is None or features.empty or len(features) < window:
            continue

        vol_z = features['volume_zscore'] if 'volume_zscore' in features.columns else pd.Series(0, index=features.index)
        mom = features['momentum_5d'] if 'momentum_5d' in features.columns else pd.Series(0, index=features.index)

        # Pheromone = smoothed |volume_z * momentum| — the "heat"
        pheromone = (vol_z.abs() * mom.abs() * 100).rolling(window, min_periods=1).mean()
        # Direction = smoothed momentum sign — the "color"
        direction = mom.rolling(window, min_periods=1).mean()

        # Take last 60 trading days for clean visualization
        n_points = min(len(pheromone), 60)
        pheromone_data.append(pheromone.iloc[-n_points:].values)
        direction_data.append(direction.iloc[-n_points:].values)
        valid_tickers.append(ticker)

        if time_index is None:
            time_index = features.index[-n_points:]

    if not pheromone_data or time_index is None:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for 3D heatmap", showarrow=False,
                          xref="paper", yref="paper", x=0.5, y=0.5,
                          font=dict(size=16, color='#334155'))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(248,250,255,0.85)')
        return fig

    # Build matrices — pad shorter arrays to max length
    max_len = max(len(d) for d in pheromone_data)
    n_tickers = len(valid_tickers)
    
    # Precompute time labels
    time_labels = [str(t.date()) if hasattr(t, 'date') else str(t) for t in time_index[-max_len:]]
    if len(time_labels) < max_len:
        time_labels = [f"t-{max_len-i}" for i in range(max_len)]

    z_matrix = np.zeros((n_tickers, max_len))
    dir_matrix = np.zeros((n_tickers, max_len))

    for i in range(n_tickers):
        p_data = pheromone_data[i]
        d_data = direction_data[i]
        pad = max_len - len(p_data)
        z_matrix[i, pad:] = p_data
        d_pad = max_len - len(d_data)
        dir_matrix[i, d_pad:] = d_data

    z_matrix = np.nan_to_num(z_matrix, nan=0.0)
    dir_matrix = np.nan_to_num(dir_matrix, nan=0.0)

    # Grid mapping
    G = int(np.ceil(np.sqrt(n_tickers)))

    def get_grid(t_idx):
        Z = np.full(G * G, np.nan)
        C = np.full(G * G, np.nan)
        Z[:n_tickers] = z_matrix[:, t_idx]
        C[:n_tickers] = dir_matrix[:, t_idx]
        return Z.reshape((G, G)), C.reshape((G, G))

    # Calculate global max for scale bounds using percentiles to ignore outliers
    z_max = np.nanpercentile(z_matrix, 98) if np.nanmax(z_matrix) > 0 else 1
    dir_max = np.nanpercentile(np.abs(dir_matrix), 95)
    if dir_max < 1e-3 or np.isnan(dir_max):
        dir_max = 1.0

    Z_init, C_init = get_grid(0)

    fig = go.Figure()

    # Primary surface — Harsh diverging colors to avoid pastel washout
    # Smallest values -> dark gray/black, Large values -> vivid red/green
    vivid_colorscale = [
        [0.0, '#FF0000'],    # Pure Red
        [0.35, '#202020'],   # Dark Gray transition
        [0.5, '#050505'],    # Near-black neutral
        [0.65, '#202020'],   # Dark Gray transition
        [1.0, '#00FF00'],    # Pure Green
    ]

    fig.add_trace(go.Surface(
        z=Z_init,
        surfacecolor=C_init,
        cmin=-dir_max,
        cmax=dir_max,
        colorscale=vivid_colorscale,
        colorbar=dict(
            title=dict(text="Buy/Sell", font=dict(color='#334155', size=12)),
            tickfont=dict(color='#64748B', size=10),
            x=1.02, len=0.6, y=0.5,
            thickness=15,
            outlinewidth=0,
            bgcolor='rgba(0,0,0,0)',
        ),
        opacity=1.0,  # Full opacity to avoid washing out against background
        lighting=dict(
            ambient=0.9,     # High ambient to show true colors
            diffuse=0.3,     # Low diffuse to avoid white bouncing light
            specular=0.02,   # Very low specular to remove shiny white patches
            roughness=0.9,
            fresnel=0.01,
        ),
        lightposition=dict(x=0, y=0, z=100),
        hovertemplate=(
            'Intensity: %{z:.2f}<br>'
            'Direction: %{surfacecolor:.4f}'
            '<extra></extra>'
        ),
    ))

    # Time animation frames
    frames = []
    for k in range(max_len):
        Z_k, C_k = get_grid(k)
        frames.append(go.Frame(
            data=[go.Surface(
                z=Z_k, 
                surfacecolor=C_k,
                cmin=-dir_max,
                cmax=dir_max,
                colorscale=vivid_colorscale  # Ensure frames keep colorscale
            )],
            name=str(k),
        ))
    fig.frames = frames

    fig.update_layout(
        title=dict(
            text=f"{title} | Animated Over Time",
            font=dict(color='#0F172A', size=16, family='Inter, sans-serif'),
            x=0.02, y=0.98,
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text='Grid X', font=dict(size=11, color='#334155')),
                showticklabels=False,
                gridcolor='rgba(99,102,241,0.08)',
                showbackground=True,
                backgroundcolor='rgba(240,244,255,0.90)',
            ),
            yaxis=dict(
                title=dict(text='Grid Y', font=dict(size=11, color='#334155')),
                showticklabels=False,
                gridcolor='rgba(99,102,241,0.08)',
                showbackground=True,
                backgroundcolor='rgba(240,244,255,0.90)',
            ),
            zaxis=dict(
                title=dict(text='Flow Intensity', font=dict(size=11, color='#334155')),
                range=[0, z_max * 1.05],
                tickfont=dict(size=9, color='#64748B'),
                gridcolor='rgba(99,102,241,0.10)',
                showbackground=True,
                backgroundcolor='rgba(240,244,255,0.90)',
            ),
            bgcolor='rgba(248,250,255,0.80)',
            camera=dict(
                eye=dict(x=1.6, y=-1.4, z=0.9),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.1),
            ),
            aspectratio=dict(x=1.2, y=1.2, z=0.4),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,250,255,0.85)',
        margin=dict(l=0, r=60, t=40, b=50),
        height=750,
        updatemenus=[
            dict(
                type='buttons',
                showactive=True,
                x=0.02, y=0.05,
                xanchor='left', yanchor='bottom',
                direction='right',
                pad=dict(r=10, t=10),
                buttons=[
                    dict(
                        label='\u25B6 Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=150, redraw=True),
                            fromcurrent=True,
                            mode='immediate',
                            transition=dict(duration=0),
                        )],
                    ),
                    dict(
                        label='\u23F8 Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                        )],
                    ),
                ],
                font=dict(color='#334155', size=12),
                bgcolor='rgba(99,102,241,0.35)',
                bordercolor='rgba(99,102,241,0.7)',
                borderwidth=1,
            ),
            dict(
                type='dropdown',
                showactive=True,
                x=0.8, y=0.05,
                xanchor='left', yanchor='bottom',
                buttons=[
                    dict(
                        label=time_labels[k],
                        method='animate',
                        args=[[str(k)], dict(frame=dict(duration=0, redraw=True), mode='immediate')]
                    ) for k in range(max_len)
                ],
                font=dict(color='#334155', size=12),
            )
        ]
    )

    return fig


def create_pheromone_network_map(
    pheromone_matrix: np.ndarray,
    tickers: List[str],
    threshold: float = 0.5,
    title: str = "ACO Pheromone Network"
) -> go.Figure:
    """
    Visualize pheromone trails as a network graph.
    Nodes = tickers, edges = pheromone strength between them.
    """
    n = len(tickers)
    if pheromone_matrix.shape[0] != n:
        fig = go.Figure()
        fig.add_annotation(text="Pheromone matrix size mismatch", showarrow=False)
        return fig

    # Circular layout
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)

    fig = go.Figure()

    # Draw edges (pheromone trails)
    max_pheromone = pheromone_matrix.max() + 1e-9
    for i in range(n):
        for j in range(i + 1, n):
            strength = pheromone_matrix[i, j]
            if strength > threshold:
                normalized = strength / max_pheromone
                fig.add_trace(go.Scatter(
                    x=[x_pos[i], x_pos[j]], y=[y_pos[i], y_pos[j]],
                    mode='lines',
                    line=dict(
                        width=max(0.5, normalized * 5),
                        color=f'rgba(0, 230, 118, {min(normalized + 0.2, 1.0)})',
                    ),
                    showlegend=False,
                    hoverinfo='text',
                    text=f"{tickers[i]} ↔ {tickers[j]}: {strength:.2f}",
                ))

    # Draw nodes
    node_sizes = [float(pheromone_matrix[i].sum()) for i in range(n)]
    max_size = max(node_sizes) + 1e-9
    node_sizes_scaled = [10 + (s / max_size) * 30 for s in node_sizes]

    fig.add_trace(go.Scatter(
        x=x_pos, y=y_pos,
        mode='markers+text',
        marker=dict(
            size=node_sizes_scaled,
            color=node_sizes,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Pheromone", tickfont=dict(color='#334155'),
                         title_font=dict(color='#334155')),
            line=dict(width=1, color='rgba(99,102,241,0.25)'),
        ),
        text=tickers,
        textposition='top center',
        textfont=dict(color='#334155', size=10),
        hoverinfo='text',
        hovertext=[f"{t}: total pheromone = {s:.2f}" for t, s in zip(tickers, node_sizes)],
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color='#0F172A', size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,250,255,0.85)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x'),
        height=600,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def create_force_map(
    features_cache: Dict[str, pd.DataFrame],
    tickers: List[str],
    window: int = 20,
    title: str = "Market Force Map (Katz 2011)"
) -> go.Figure:
    """
    Force map visualization (Katz et al. 2011 method).
    For each pair of assets:
      - X: relative position (spread in returns)
      - Y: relative direction (difference in momentum)
      - Arrow/Color: acceleration (rate of change)
    Shows where buying/selling pressure concentrates.
    """
    positions = []
    directions = []
    accelerations = []
    labels = []

    for ticker in tickers:
        features = features_cache.get(ticker)
        if features is None or features.empty:
            continue

        mom_5d = float(features['momentum_5d'].iloc[-1]) if 'momentum_5d' in features.columns else 0.0
        mom_20d = float(features['momentum_20d'].iloc[-1]) if 'momentum_20d' in features.columns else 0.0
        vol_z = float(features['volume_zscore'].iloc[-1]) if 'volume_zscore' in features.columns else 0.0
        rsi = float(features['rsi'].iloc[-1]) if 'rsi' in features.columns else 50.0

        positions.append(mom_20d * 100)         # longer-term position
        directions.append(mom_5d * 100)          # shorter-term direction
        accelerations.append(vol_z * mom_5d * 100)  # volume-weighted acceleration
        labels.append(ticker)

    if not positions:
        fig = go.Figure()
        fig.add_annotation(text="No data for force map", showarrow=False)
        return fig

    # Normalize accelerations for color
    acc_arr = np.array(accelerations)
    max_acc = max(abs(acc_arr).max(), 1e-9)
    acc_normalized = acc_arr / max_acc

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=positions, y=directions,
        mode='markers+text',
        marker=dict(
            size=[max(8, abs(a) * 30 + 8) for a in acc_normalized],
            color=accelerations,
            colorscale=[
                [0, '#FF1744'], [0.5, '#424242'], [1, '#00E676']
            ],
            showscale=True,
            colorbar=dict(title="Acceleration", tickfont=dict(color='#334155'),
                         title_font=dict(color='#334155')),
            line=dict(width=1, color='rgba(255,255,255,0.3)'),
        ),
        text=labels,
        textposition='top center',
        textfont=dict(color='#334155', size=9),
        hoverinfo='text',
        hovertext=[
            f"{l}<br>Position: {p:.2f}%<br>Direction: {d:.2f}%<br>Accel: {a:.2f}"
            for l, p, d, a in zip(labels, positions, directions, accelerations)
        ],
    ))

    # Add quadrant lines
    fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.2)', dash='dash'))
    fig.add_vline(x=0, line=dict(color='rgba(255,255,255,0.2)', dash='dash'))

    # Quadrant labels
    fig.add_annotation(x=max(positions) * 0.7, y=max(directions) * 0.7,
                      text="Momentum BUY", showarrow=False,
                      font=dict(color='#059669', size=12))
    fig.add_annotation(x=min(positions) * 0.7, y=min(directions) * 0.7,
                      text="Momentum SELL", showarrow=False,
                      font=dict(color='#E84393', size=12))
    fig.add_annotation(x=max(positions) * 0.7, y=min(directions) * 0.7,
                      text="Reversal SELL", showarrow=False,
                      font=dict(color='#D97706', size=12))
    fig.add_annotation(x=min(positions) * 0.7, y=max(directions) * 0.7,
                      text="Reversal BUY", showarrow=False,
                      font=dict(color='#5B5BD6', size=12))

    fig.update_layout(
        title=dict(text=title, font=dict(color='#0F172A', size=16)),
        xaxis=dict(title='20d Position (Return %)', gridcolor='rgba(99,102,241,0.10)',
                   color='#334155', zeroline=False),
        yaxis=dict(title='5d Direction (Momentum %)', gridcolor='rgba(99,102,241,0.10)',
                   color='#334155', zeroline=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,250,255,0.85)',
        height=600,
        margin=dict(l=60, r=20, t=50, b=60),
    )

    return fig


def create_sector_heatmap_2d(
    features_cache: Dict[str, pd.DataFrame],
    sector_map: Dict[str, str],
    title: str = "Sector Flow Heatmap"
) -> go.Figure:
    """
    2D heatmap showing sector-level momentum and volume flow.
    Rows = sectors, Columns = time windows (1d, 5d, 20d).
    """
    sectors = sorted(set(sector_map.values()))
    timeframes = ['1d', '5d', '20d']
    z_data = []
    text_data = []

    for sector in sectors:
        sector_tickers = [t for t, s in sector_map.items() if s == sector]
        row = []
        text_row = []
        for tf in timeframes:
            col_name = f'momentum_{tf}'
            momenta = []
            for ticker in sector_tickers:
                feat = features_cache.get(ticker)
                if feat is not None and not feat.empty and col_name in feat.columns:
                    momenta.append(float(feat[col_name].iloc[-1]))
            avg_mom = np.mean(momenta) * 100 if momenta else 0.0
            row.append(avg_mom)
            text_row.append(f"{avg_mom:+.2f}%")
        z_data.append(row)
        text_data.append(text_row)

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=timeframes,
        y=sectors,
        text=text_data,
        texttemplate="%{text}",
        textfont=dict(color='#334155', size=11),
        colorscale=[
            [0, '#B71C1C'], [0.25, '#E53935'], [0.5, '#212121'],
            [0.75, '#43A047'], [1, '#1B5E20']
        ],
        colorbar=dict(title="Momentum %", tickfont=dict(color='#334155'),
                     title_font=dict(color='#334155')),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color='#0F172A', size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,250,255,0.85)',
        xaxis=dict(color='#334155', title='Timeframe'),
        yaxis=dict(color='#334155', title='Sector'),
        height=500,
        margin=dict(l=120, r=20, t=50, b=60),
    )

    return fig


def create_swarm_dashboard_data(
    aggregated_signals: Dict[str, 'AggregatedSignal'],
    swarm_state: 'SwarmState',
) -> dict:
    """
    Prepare data for a swarm dashboard display.
    Returns structured data for Streamlit rendering.
    """
    signal_summary = {}
    for ticker, sig in aggregated_signals.items():
        signal_summary[ticker] = {
            'action': sig.action,
            'direction': sig.direction,
            'confidence': sig.confidence,
            'strength': sig.strength,
            'zone': sig.zone,
            'components': sig.component_signals,
        }

    # Pheromone stats
    pheromone_stats = {
        'mean': float(swarm_state.pheromone_matrix.mean()),
        'max': float(swarm_state.pheromone_matrix.max()),
        'std': float(swarm_state.pheromone_matrix.std()),
    }

    # Zone distribution
    zone_counts = {'ZOR': 0, 'ZOO': 0, 'ZOA': 0}
    for zone in swarm_state.zone_assignments.values():
        if zone in zone_counts:
            zone_counts[zone] += 1

    # Top leaders
    top_leaders = sorted(swarm_state.leader_scores.items(),
                        key=lambda x: x[1], reverse=True)[:5]

    return {
        'signals': signal_summary,
        'pheromone': pheromone_stats,
        'zones': zone_counts,
        'top_leaders': top_leaders,
        'n_active_tickers': len(aggregated_signals),
    }


def create_money_flow_heatmap_2d(
    features_cache: Dict[str, pd.DataFrame],
    tickers: List[str],
    window: int = 20,
    n_days: int = 60,
    title: str = "Money Flow Heatmap",
    max_tickers: int = 80,
) -> plt.Figure:
    """
    Seaborn-style 2D heatmap: rows = tickers, columns = recent trading days.
    Cell color = net momentum direction (RdYlGn diverging).
    Rows sorted by latest net direction (strongest buy on top).
    Returns a matplotlib Figure for st.pyplot().
    """
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    direction_rows = []
    valid_tickers = []

    for ticker in tickers:
        features = features_cache.get(ticker)
        if features is None or features.empty or len(features) < window:
            continue
        mom = features['momentum_5d'] if 'momentum_5d' in features.columns else pd.Series(
            0.0, index=features.index
        )
        smoothed = (mom.rolling(window, min_periods=1).mean() * 100).iloc[-n_days:]
        direction_rows.append(smoothed.values[-n_days:])
        valid_tickers.append(ticker)

    if not valid_tickers:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='#334155')
        ax.set_facecolor('#F8FAFF')
        fig.patch.set_alpha(0.0)
        return fig

    # Pad all rows to same length
    max_len = max(len(r) for r in direction_rows)
    matrix = np.full((len(valid_tickers), max_len), np.nan)
    for i, row in enumerate(direction_rows):
        matrix[i, max_len - len(row):] = row

    # Sort rows by last column (latest net direction)
    last_col = np.nanmean(matrix[:, -5:], axis=1)
    sort_idx = np.argsort(last_col)[::-1]  # strongest buy first
    matrix = matrix[sort_idx]
    sorted_tickers = [valid_tickers[i] for i in sort_idx]

    # Cap at max_tickers for readability
    if len(sorted_tickers) > max_tickers:
        # Keep top max_tickers/2 buys and top max_tickers/2 sells
        half = max_tickers // 2
        keep_idx = list(range(half)) + list(range(len(sorted_tickers) - half, len(sorted_tickers)))
        matrix = matrix[keep_idx]
        sorted_tickers = [sorted_tickers[i] for i in keep_idx]

    n_rows = len(sorted_tickers)
    n_cols = matrix.shape[1]

    # Color scale limits: symmetric, 98th percentile
    vmax = np.nanpercentile(np.abs(matrix), 98)
    if vmax < 0.1 or np.isnan(vmax):
        vmax = 1.0

    # Figure sizing: each row ~0.28 inches, min 6 max 28
    fig_height = max(6, min(28, n_rows * 0.28 + 2))
    fig_width = 16

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('#FAFBFF')
    ax.set_facecolor('#FAFBFF')

    if sns is not None:
        cmap = sns.diverging_palette(10, 133, s=85, l=45, as_cmap=True)
    else:
        cmap = plt.cm.RdYlGn

    im = ax.imshow(
        matrix,
        aspect='auto',
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation='nearest',
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.set_label('Momentum %', color='#334155', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='#334155', labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#334155')

    # Y axis: ticker labels
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(sorted_tickers, fontsize=max(5, min(9, 180 // n_rows)),
                       color='#1E293B', fontfamily='monospace')

    # X axis: date labels — show every ~10 days
    x_step = max(1, n_cols // 10)
    x_ticks = list(range(0, n_cols, x_step))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"t-{n_cols - x}" for x in x_ticks], fontsize=8, color='#475569')
    ax.set_xlabel("Trading Days (most recent = right)", fontsize=10, color='#475569')

    # Spine styling
    for spine in ax.spines.values():
        spine.set_edgecolor('#CBD5E1')
        spine.set_linewidth(0.8)

    ax.set_title(title, fontsize=13, color='#0F172A', pad=12,
                 fontweight='semibold', loc='left')

    # Horizontal separator: split buys (top) from sells (bottom)
    split_line = (last_col[sort_idx] > 0).sum()
    if 0 < split_line < n_rows:
        ax.axhline(split_line - 0.5, color='#94A3B8', linewidth=1.2, linestyle='--', alpha=0.7)
        ax.text(n_cols * 0.01, split_line - 1.2, 'BUY bias ▲',
                fontsize=8, color='#059669', va='bottom')
        ax.text(n_cols * 0.01, split_line + 0.3, 'SELL bias ▼',
                fontsize=8, color='#DC2626', va='top')

    fig.tight_layout(pad=1.5)
    return fig


def create_money_flow_heatmap_3d_seaborn(
    features_cache: Dict[str, pd.DataFrame],
    tickers: List[str],
    window: int = 20,
    n_days: int = 60,
    title: str = "3D Money Flow Heatmap",
    max_tickers: int = 50,
    elev: float = 28,
    azim: float = -55,
) -> plt.Figure:
    """
    3D surface heatmap using matplotlib Axes3D + seaborn RdYlGn palette.
    X = time (days), Y = ticker, Z = momentum intensity (|vol_z * mom| * 100).
    Face colors = net direction (green=buy, red=sell) via diverging RdYlGn colormap.
    Returns matplotlib Figure for st.pyplot().
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    try:
        import seaborn as sns
        cmap = sns.diverging_palette(10, 133, s=85, l=45, as_cmap=True)
    except ImportError:
        cmap = plt.cm.RdYlGn

    # ── Build data matrices ──────────────────────────────────────────────
    intensity_rows: List[np.ndarray] = []
    direction_rows: List[np.ndarray] = []
    valid_tickers: List[str] = []

    for ticker in tickers:
        feat = features_cache.get(ticker)
        if feat is None or feat.empty or len(feat) < window:
            continue
        vol_z = feat['volume_zscore'] if 'volume_zscore' in feat.columns else pd.Series(0.0, index=feat.index)
        mom   = feat['momentum_5d']  if 'momentum_5d'   in feat.columns else pd.Series(0.0, index=feat.index)

        intensity = (vol_z.abs() * mom.abs() * 100).rolling(window, min_periods=1).mean()
        direction = (mom.rolling(window, min_periods=1).mean() * 100)

        intensity_rows.append(intensity.iloc[-n_days:].values)
        direction_rows.append(direction.iloc[-n_days:].values)
        valid_tickers.append(ticker)

    if not valid_tickers:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                transform=ax.transAxes, fontsize=13, color='#475569')
        fig.patch.set_alpha(0.0)
        return fig

    # Pad to same length
    max_len = max(len(r) for r in intensity_rows)
    n_t = len(valid_tickers)
    Z_int = np.zeros((n_t, max_len))   # intensity
    Z_dir = np.zeros((n_t, max_len))   # direction
    for i, (intv, dirv) in enumerate(zip(intensity_rows, direction_rows)):
        Z_int[i, max_len - len(intv):] = intv
        Z_dir[i, max_len - len(dirv):] = dirv

    Z_int = np.nan_to_num(Z_int)
    Z_dir = np.nan_to_num(Z_dir)

    # Sort tickers by latest net direction (buy on top for visual clarity)
    last_dir = np.nanmean(Z_dir[:, -5:], axis=1)
    sort_idx = np.argsort(last_dir)[::-1]
    Z_int = Z_int[sort_idx]
    Z_dir = Z_dir[sort_idx]
    sorted_tickers = [valid_tickers[i] for i in sort_idx]

    # Cap tickers
    if n_t > max_tickers:
        half = max_tickers // 2
        keep = list(range(half)) + list(range(n_t - half, n_t))
        Z_int = Z_int[keep]
        Z_dir = Z_dir[keep]
        sorted_tickers = [sorted_tickers[i] for i in keep]
        n_t = len(sorted_tickers)

    # Meshgrid: X = time, Y = ticker index
    X = np.arange(max_len)          # time axis
    Y = np.arange(n_t)              # ticker axis
    XX, YY = np.meshgrid(X, Y)

    # Intensity: smooth Z for cleaner surface, clip outliers
    z_vmax = np.nanpercentile(Z_int, 97)
    if z_vmax < 0.01:
        z_vmax = 1.0
    Z_plot = np.clip(Z_int, 0, z_vmax)

    # Direction: normalize for colormap [-vmax_dir, +vmax_dir]
    dir_vmax = np.nanpercentile(np.abs(Z_dir), 97)
    if dir_vmax < 0.01:
        dir_vmax = 1.0
    norm = Normalize(vmin=-dir_vmax, vmax=dir_vmax)
    facecolors = cmap(norm(Z_dir))  # shape (n_t, max_len, 4) RGBA

    # ── Figure ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor('#F8FAFF')

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#F8FAFF')

    surf = ax.plot_surface(
        XX, YY, Z_plot,
        facecolors=facecolors,
        rstride=1, cstride=1,
        antialiased=True,
        shade=True,
        lightsource=LightSource(azdeg=225, altdeg=45),
    )

    # ── Axes styling ─────────────────────────────────────────────────────
    ax.view_init(elev=elev, azim=azim)

    # Ticker labels on Y axis (show max 20 to avoid crowding)
    tick_step = max(1, n_t // 20)
    y_tick_pos = list(range(0, n_t, tick_step))
    ax.set_yticks(y_tick_pos)
    ax.set_yticklabels(
        [sorted_tickers[i] for i in y_tick_pos],
        fontsize=7, color='#1E293B',
        ha='left', rotation=-15,
    )

    # X axis: day labels
    x_step = max(1, max_len // 8)
    x_ticks = list(range(0, max_len, x_step))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"t-{max_len - x}" for x in x_ticks], fontsize=7, color='#475569')

    # Z axis
    ax.set_zticks(np.linspace(0, z_vmax, 5))
    ax.tick_params(axis='z', labelsize=7, colors='#475569')

    ax.set_xlabel("Days (→ recent)", fontsize=9, color='#475569', labelpad=8)
    ax.set_ylabel("Ticker", fontsize=9, color='#475569', labelpad=10)
    ax.set_zlabel("Flow Intensity", fontsize=9, color='#475569', labelpad=6)

    # Pane colors — light glass look
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = True
        pane.set_facecolor('#EEF2FF')
        pane.set_edgecolor('#CBD5E1')
        pane.set_alpha(0.55)

    ax.grid(True, color='#CBD5E1', linewidth=0.4, linestyle='--', alpha=0.5)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.08, aspect=18)
    cbar.set_label('Net Direction %', fontsize=9, color='#334155')
    cbar.ax.tick_params(labelsize=7, colors='#475569')
    cbar.ax.set_yticks([round(-dir_vmax, 1), 0, round(dir_vmax, 1)])
    cbar.ax.set_yticklabels(['SELL', '0', 'BUY'], color='#334155', fontsize=8)

    fig.text(0.04, 0.97, title, fontsize=13, color='#0F172A',
             fontweight='semibold', va='top', transform=fig.transFigure)

    fig.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.08)
    return fig


# ── Seaborn-inspired diverging colorscale for Plotly ─────────────────────────
_SBN_RDYLGN = [
    [0.000, '#7F0000'],
    [0.120, '#C0392B'],
    [0.250, '#E74C3C'],
    [0.380, '#F1948A'],
    [0.465, '#FDEBD0'],
    [0.500, '#F5F5F5'],
    [0.535, '#D5F5E3'],
    [0.620, '#82E0AA'],
    [0.750, '#27AE60'],
    [0.880, '#1E8449'],
    [1.000, '#0B5345'],
]


def create_money_flow_heatmap_interactive(
    features_cache: Dict[str, pd.DataFrame],
    tickers: List[str],
    window: int = 20,
    total_days: int = 80,
    n_anim_frames: int = 45,
    max_tickers: int = 55,
    title: str = "3D Money Flow Heatmap",
) -> go.Figure:
    """
    Interactive Plotly 3D surface: fully rotatable, animated over time.

    X = day offset within rolling window (0 → window-1, i.e. older → recent)
    Y = ticker index (sorted strongest buy → sell)
    Z = flow intensity  |vol_z * momentum * 100|  (height of surface)
    Color = seaborn RdYlGn diverging  (green = buy, red = sell)

    Animation: rolling window slides through history — each frame advances
    the window by ~1 trading day so you see the surface morph over time.
    Play/Pause buttons + scrubber slider with smooth cubic-in-out transitions.
    Fully rotatable/zoomable via Plotly 3D scene (mouse drag).
    """
    # ── Build full-history data matrices ─────────────────────────────────────
    intensity_full: List[np.ndarray] = []
    direction_full: List[np.ndarray] = []
    valid_tickers: List[str] = []
    date_index: Optional[pd.Index] = None

    for ticker in tickers:
        feat = features_cache.get(ticker)
        if feat is None or feat.empty or len(feat) < window + 5:
            continue
        vol_z = feat['volume_zscore'] if 'volume_zscore' in feat.columns else pd.Series(0.0, index=feat.index)
        mom   = feat['momentum_5d']   if 'momentum_5d'   in feat.columns else pd.Series(0.0, index=feat.index)

        # Light smoothing (5-bar) to reduce noise on surface
        intens = (vol_z.abs() * mom.abs() * 100).rolling(5, min_periods=1).mean()
        direc  = (mom * 100).rolling(5, min_periods=1).mean()

        intensity_full.append(intens.values[-total_days:])
        direction_full.append(direc.values[-total_days:])
        valid_tickers.append(ticker)
        if date_index is None:
            date_index = feat.index[-total_days:]

    if not valid_tickers:
        fig = go.Figure()
        fig.add_annotation(text="No data available for heatmap",
                           showarrow=False, xref="paper", yref="paper",
                           x=0.5, y=0.5, font=dict(size=16, color='#334155'))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(248,250,255,0.85)', height=400)
        return fig

    # Pad all to same length
    actual_days = max(len(r) for r in intensity_full)
    n_t = len(valid_tickers)
    INT = np.zeros((n_t, actual_days))
    DIR = np.zeros((n_t, actual_days))
    for i, (iv, dv) in enumerate(zip(intensity_full, direction_full)):
        INT[i, actual_days - len(iv):] = iv
        DIR[i, actual_days - len(dv):] = dv

    INT = np.nan_to_num(INT)
    DIR = np.nan_to_num(DIR)

    # Sort rows: strongest buy on top
    last_dir = np.nanmean(DIR[:, -5:], axis=1)
    sidx = np.argsort(last_dir)[::-1]
    INT, DIR = INT[sidx], DIR[sidx]
    sorted_tickers = [valid_tickers[i] for i in sidx]

    # Cap ticker count for performance
    if n_t > max_tickers:
        half = max_tickers // 2
        keep = list(range(half)) + list(range(n_t - half, n_t))
        INT, DIR = INT[keep], DIR[keep]
        sorted_tickers = [sorted_tickers[i] for i in keep]
        n_t = len(sorted_tickers)

    # Global scale — consistent across all frames
    z_vmax = float(np.nanpercentile(INT, 97)) or 1.0
    dir_vmax = float(np.nanpercentile(np.abs(DIR), 97)) or 1.0

    # ── Animation frame starts ────────────────────────────────────────────────
    max_start = max(actual_days - window, 1)
    frame_starts = np.unique(
        np.linspace(0, max_start, min(n_anim_frames, max_start + 1), dtype=int)
    )

    # Date labels
    if date_index is not None and len(date_index) >= actual_days:
        dates = [str(d.date()) if hasattr(d, 'date') else str(d)
                 for d in date_index[-actual_days:]]
    else:
        dates = [f"t-{actual_days - i}" for i in range(actual_days)]

    x_vals = np.arange(window)   # constant X: day offset 0..window-1
    y_vals = np.arange(n_t)      # constant Y: ticker index

    def _get_slice(start: int) -> Tuple[np.ndarray, np.ndarray]:
        end = min(start + window, actual_days)
        w   = end - start
        Z   = INT[:, start:end]
        C   = DIR[:, start:end]
        if w < window:
            Z = np.pad(Z, ((0, 0), (0, window - w)))
            C = np.pad(C, ((0, 0), (0, window - w)))
        return np.clip(Z, 0, z_vmax), C

    Z0, C0 = _get_slice(frame_starts[0])

    # Y tick labels — show every Nth ticker to avoid crowding
    tick_step = max(1, n_t // 20)
    y_tick_vals = list(range(0, n_t, tick_step))
    y_tick_text = [sorted_tickers[i] for i in y_tick_vals]

    # ── Initial surface trace ─────────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=x_vals,
        y=y_vals,
        z=Z0,
        surfacecolor=C0,
        cmin=-dir_vmax,
        cmax=dir_vmax,
        colorscale=_SBN_RDYLGN,
        showscale=True,
        colorbar=dict(
            title=dict(text="Direction %", font=dict(color='#334155', size=11)),
            tickfont=dict(color='#475569', size=9),
            tickvals=[-dir_vmax, -dir_vmax * 0.5, 0, dir_vmax * 0.5, dir_vmax],
            ticktext=[
                f'SELL {dir_vmax:.1f}%', f'{-dir_vmax*0.5:.1f}%',
                '0', f'{dir_vmax*0.5:.1f}%', f'BUY {dir_vmax:.1f}%',
            ],
            x=1.01, len=0.65, y=0.5,
            thickness=14, outlinewidth=0,
        ),
        opacity=0.97,
        lighting=dict(
            ambient=0.72,
            diffuse=0.55,
            specular=0.04,
            roughness=0.82,
            fresnel=0.02,
        ),
        lightposition=dict(x=100, y=100, z=250),
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor='rgba(255,255,255,0.6)',
                project=dict(z=True),
            ),
        ),
        hovertemplate=(
            'Ticker idx: %{y}<br>'
            'Day +%{x}<br>'
            'Intensity: %{z:.2f}<br>'
            'Direction: %{surfacecolor:.2f}%'
            '<extra></extra>'
        ),
    ))

    # ── Animation frames ──────────────────────────────────────────────────────
    frames = []
    slider_steps = []
    for k, start in enumerate(frame_starts):
        Zk, Ck = _get_slice(int(start))
        label = dates[int(start)] if int(start) < len(dates) else f't-{actual_days - int(start)}'

        frames.append(go.Frame(
            data=[go.Surface(
                z=Zk,
                surfacecolor=Ck,
                cmin=-dir_vmax,
                cmax=dir_vmax,
                colorscale=_SBN_RDYLGN,
                contours=dict(z=dict(show=True, usecolormap=True, project=dict(z=True))),
            )],
            name=str(k),
        ))
        slider_steps.append(dict(
            args=[[str(k)], dict(
                frame=dict(duration=0, redraw=True),
                mode='immediate',
                transition=dict(duration=0),
            )],
            label=label,
            method='animate',
        ))

    fig.frames = frames

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"{title}  ·  rotate drag  ·  ▶ animate over time",
            font=dict(color='#0F172A', size=14, family='Inter, sans-serif'),
            x=0.02, y=0.99,
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text='Day offset (→ recent)', font=dict(size=10, color='#475569')),
                tickfont=dict(size=8, color='#64748B'),
                gridcolor='rgba(99,102,241,0.12)',
                showbackground=True,
                backgroundcolor='rgba(238,242,255,0.80)',
                zeroline=False,
            ),
            yaxis=dict(
                title=dict(text='Ticker  (buy↑ → sell↓)', font=dict(size=10, color='#475569')),
                tickvals=y_tick_vals,
                ticktext=y_tick_text,
                tickfont=dict(size=8, color='#334155', family='monospace'),
                gridcolor='rgba(99,102,241,0.12)',
                showbackground=True,
                backgroundcolor='rgba(238,242,255,0.80)',
                zeroline=False,
            ),
            zaxis=dict(
                title=dict(text='Flow Intensity', font=dict(size=10, color='#475569')),
                range=[0, z_vmax * 1.1],
                tickfont=dict(size=8, color='#64748B'),
                gridcolor='rgba(99,102,241,0.14)',
                showbackground=True,
                backgroundcolor='rgba(238,242,255,0.80)',
                zeroline=False,
            ),
            bgcolor='rgba(244,246,255,0.70)',
            camera=dict(
                eye=dict(x=1.55, y=-1.65, z=0.80),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.1),
            ),
            aspectratio=dict(x=1.5, y=1.5, z=0.42),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,250,255,0.85)',
        margin=dict(l=0, r=70, t=55, b=60),
        height=740,
        # Play / Pause buttons
        updatemenus=[dict(
            type='buttons',
            showactive=True,
            x=0.01, y=0.07,
            xanchor='left', yanchor='bottom',
            direction='right',
            pad=dict(r=12, t=10),
            buttons=[
                dict(
                    label='▶  Play',
                    method='animate',
                    args=[None, dict(
                        frame=dict(duration=180, redraw=True),
                        fromcurrent=True,
                        mode='immediate',
                        transition=dict(duration=90, easing='cubic-in-out'),
                    )],
                ),
                dict(
                    label='⏸  Pause',
                    method='animate',
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode='immediate',
                    )],
                ),
            ],
            font=dict(color='#1E293B', size=12),
            bgcolor='rgba(99,102,241,0.13)',
            bordercolor='rgba(99,102,241,0.45)',
            borderwidth=1,
        )],
        # Scrubber slider
        sliders=[dict(
            active=0,
            steps=slider_steps,
            x=0.0, y=0.03,
            len=1.0,
            xanchor='left', yanchor='bottom',
            pad=dict(b=10, t=55),
            currentvalue=dict(
                font=dict(size=11, color='#334155'),
                prefix='Window end: ',
                visible=True,
                xanchor='right',
            ),
            transition=dict(duration=90, easing='cubic-in-out'),
            bgcolor='rgba(99,102,241,0.07)',
            activebgcolor='rgba(99,102,241,0.30)',
            bordercolor='rgba(99,102,241,0.25)',
            font=dict(size=8, color='#64748B'),
            tickwidth=0,
            minorticklen=0,
        )],
    )

    return fig
