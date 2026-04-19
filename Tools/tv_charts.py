"""
TradingView Lightweight Charts — Streamlit HTML components
============================================================
Wraps the open-source TradingView Lightweight Charts v4 library
(https://github.com/tradingview/lightweight-charts) to render
professional, interactive financial charts inside Streamlit via
st.components.v1.html().

All functions return an (html_string, suggested_height) tuple.
Call with:
    html, h = tv_charts.candlestick(df, ...)
    st.components.v1.html(html, height=h, scrolling=False)

Chart types
-----------
  candlestick()   — OHLCV candlestick with volume pane + SMA overlays +
                    horizontal price levels (entry/stop/target/S-R)
  line_multi()    — Multiple line series on one pane (HV, IV, vol smile, etc.)
  area_line()     — Single area+line (price history, strategy P&L)
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from typing import Optional

# CDN — pinned to v4.2 for API stability
_LWC_CDN = "https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"

# ── Shared theme tokens ───────────────────────────────────────────────────────
_BG          = "#FFFFFF"
_BG_PANE     = "#F8FAFF"
_GRID        = "rgba(99,102,241,0.10)"
_BORDER      = "rgba(99,102,241,0.20)"
_TEXT        = "#334155"
_TEXT_DIM    = "#94A3B8"
_UP          = "#00B894"
_DOWN        = "#F43F5E"
_UP_FILL     = "rgba(0,184,148,0.15)"
_DOWN_FILL   = "rgba(244,63,94,0.15)"
_ACCENT      = "#6366F1"
_ACCENT2     = "#F59E0B"
_ACCENT3     = "#0EA5E9"
_ACCENT4     = "#A855F7"

# Palette for multiple lines
_PALETTE = [_ACCENT, _UP, _DOWN, _ACCENT2, _ACCENT3, _ACCENT4,
            "#EC4899", "#14B8A6", "#F97316"]


def _ts(idx) -> list[str]:
    """Convert DatetimeIndex / Index to list of 'YYYY-MM-DD' strings."""
    return [str(d)[:10] for d in idx]


def _chart_base_options(height: int = 400) -> dict:
    return {
        "width": 0,          # 0 = autosize to container
        "height": height,
        "layout": {
            "background": {"type": "solid", "color": _BG},
            "textColor": _TEXT,
            "fontFamily": "Inter, system-ui, sans-serif",
            "fontSize": 11,
        },
        "grid": {
            "vertLines": {"color": _GRID},
            "horzLines": {"color": _GRID},
        },
        "crosshair": {"mode": 1},   # Normal crosshair
        "rightPriceScale": {
            "borderColor": _BORDER,
            "scaleMargins": {"top": 0.12, "bottom": 0.08},
        },
        "timeScale": {
            "borderColor": _BORDER,
            "timeVisible": True,
            "secondsVisible": False,
            "rightOffset": 5,
            "barSpacing": 8,
            "minBarSpacing": 2,
            "fixLeftEdge": False,
            "fixRightEdge": False,
        },
        "handleScroll": True,
        "handleScale": True,
    }


def _html_wrapper(chart_js: str, height: int, title: str = "") -> str:
    """Wrap chart JS in a full self-contained HTML page."""
    title_html = (
        f'<div style="font-family:Inter,sans-serif;font-size:12px;'
        f'font-weight:600;color:{_TEXT};padding:6px 10px 4px;'
        f'letter-spacing:0.02em;">{title}</div>'
        if title else ""
    )
    total_h = height + (26 if title else 0)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: {_BG};
    overflow: hidden;
    font-family: Inter, system-ui, sans-serif;
  }}
  #wrapper {{
    width: 100%;
    background: {_BG};
    border: 1px solid {_BORDER};
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 16px rgba(91,91,214,0.07);
  }}
  #chart {{ width: 100%; }}
  .tv-tooltip {{
    position: absolute;
    background: rgba(255,255,255,0.97);
    border: 1px solid {_BORDER};
    border-radius: 6px;
    padding: 7px 10px;
    font-size: 11px;
    color: {_TEXT};
    pointer-events: none;
    z-index: 999;
    font-family: Inter, sans-serif;
    box-shadow: 0 4px 16px rgba(91,91,214,0.12);
    display: none;
  }}
</style>
<script src="{_LWC_CDN}"></script>
</head>
<body>
<div id="wrapper">
  {title_html}
  <div id="chart"></div>
  <div class="tv-tooltip" id="tooltip"></div>
</div>
<script>
{chart_js}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# 1. CANDLESTICK  with volume pane, SMA overlays, price-level lines
# ─────────────────────────────────────────────────────────────────────────────

def candlestick(
    df: pd.DataFrame,
    ticker: str = "",
    sma_windows: list[int] | None = None,
    hlines: list[dict] | None = None,
    height: int = 500,
    show_volume: bool = True,
) -> tuple[str, int]:
    """
    Full candlestick chart with volume pane.

    Parameters
    ----------
    df          : DataFrame with columns Open/High/Low/Close and optionally Volume.
                  Index must be DatetimeIndex.
    ticker      : Label shown in the top-left legend.
    sma_windows : e.g. [20, 50] — SMA lines overlaid on price pane.
    hlines      : List of dicts: {value, color, label, dash}
                  dash: 'solid'|'dashed'|'dotted'   (default 'dashed')
    height      : Total chart height in pixels.
    show_volume : Whether to add a volume histogram pane.
    """
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).tail(500)
    if df.empty:
        return "<p style='color:#64748B'>No data</p>", 60

    candle_data = [
        {"time": t, "open": round(float(o), 4), "high": round(float(h), 4),
         "low": round(float(l), 4), "close": round(float(c), 4)}
        for t, o, h, l, c in zip(
            _ts(df.index),
            df["Open"], df["High"], df["Low"], df["Close"],
        )
    ]

    vol_data = []
    if show_volume and "Volume" in df.columns:
        vol_data = [
            {"time": t, "value": float(v),
             "color": _UP_FILL if float(c) >= float(o) else _DOWN_FILL}
            for t, o, c, v in zip(
                _ts(df.index), df["Open"], df["Close"], df["Volume"]
            )
        ]

    sma_series = {}
    if sma_windows:
        for w in sma_windows:
            sma = df["Close"].rolling(w).mean().dropna()
            sma_series[w] = [
                {"time": t, "value": round(float(v), 4)}
                for t, v in zip(_ts(sma.index), sma.values)
            ]

    # Height split: if volume shown, price pane = 72%, vol pane = 28%
    price_h  = int(height * (0.72 if vol_data else 1.0))
    vol_h    = height - price_h if vol_data else 0

    opts = _chart_base_options(price_h)
    opts_json = json.dumps(opts)

    vol_opts = _chart_base_options(vol_h) if vol_data else {}
    if vol_data:
        vol_opts["rightPriceScale"]["scaleMargins"] = {"top": 0.1, "bottom": 0.0}
        vol_opts["timeScale"]["visible"] = True
    vol_opts_json = json.dumps(vol_opts) if vol_data else "{}"

    sma_colors = [_ACCENT, _ACCENT2, _ACCENT3, _ACCENT4]

    js = f"""
(function() {{
  const container = document.getElementById('chart');

  // ── Price chart ─────────────────────────────────────────────────────────
  const priceDiv = document.createElement('div');
  priceDiv.style.width = '100%';
  container.appendChild(priceDiv);

  const chart = LightweightCharts.createChart(priceDiv, {opts_json});

  const candleSeries = chart.addCandlestickSeries({{
    upColor:          '{_UP}',
    downColor:        '{_DOWN}',
    borderUpColor:    '{_UP}',
    borderDownColor:  '{_DOWN}',
    wickUpColor:      '{_UP}',
    wickDownColor:    '{_DOWN}',
  }});
  candleSeries.setData({json.dumps(candle_data)});

  // SMA overlays
  const smaColors = {json.dumps(sma_colors)};
  const smaWindows = {json.dumps(list(sma_series.keys()))};
  const smaData    = {json.dumps(list(sma_series.values()))};
  smaWindows.forEach(function(w, i) {{
    const s = chart.addLineSeries({{
      color:       smaColors[i % smaColors.length],
      lineWidth:   1.5,
      lineStyle:   LightweightCharts.LineStyle.Dashed,
      priceLineVisible: false,
      lastValueVisible: true,
      title: 'SMA' + w,
    }});
    s.setData(smaData[i]);
  }});

  // Horizontal price levels
  const hlines = {json.dumps(hlines or [])};
  hlines.forEach(function(hl) {{
    candleSeries.createPriceLine({{
      price:       hl.value,
      color:       hl.color || '{_ACCENT}',
      lineWidth:   1.5,
      lineStyle:   hl.dash === 'solid'  ? LightweightCharts.LineStyle.Solid  :
                   hl.dash === 'dotted' ? LightweightCharts.LineStyle.Dotted :
                                          LightweightCharts.LineStyle.Dashed,
      axisLabelVisible: true,
      title:       hl.label || '',
    }});
  }});

  chart.timeScale().fitContent();

  // ── Tooltip ─────────────────────────────────────────────────────────────
  const tooltip = document.getElementById('tooltip');
  chart.subscribeCrosshairMove(function(param) {{
    if (!param || !param.time || !param.seriesData) {{
      tooltip.style.display = 'none';
      return;
    }}
    const bar = param.seriesData.get(candleSeries);
    if (!bar) {{ tooltip.style.display = 'none'; return; }}
    tooltip.style.display = 'block';
    tooltip.style.left = Math.min(param.point.x + 14, priceDiv.clientWidth - 160) + 'px';
    tooltip.style.top  = (param.point.y - 80) + 'px';
    const chg = bar.close - bar.open;
    const pct = ((chg / bar.open) * 100).toFixed(2);
    const col = chg >= 0 ? '{_UP}' : '{_DOWN}';
    tooltip.innerHTML =
      '<b style="color:{_TEXT}">{ticker} — ' + param.time + '</b><br>' +
      'O: ' + bar.open.toFixed(2) + '  H: ' + bar.high.toFixed(2) + '<br>' +
      'L: ' + bar.low.toFixed(2)  + '  C: <b style="color:'+col+'">' + bar.close.toFixed(2) + '</b><br>' +
      '<span style="color:'+col+'">' + (chg>=0?'+':'') + chg.toFixed(2) + ' (' + (chg>=0?'+':'') + pct + '%)</span>';
  }});

  // ── Volume pane ─────────────────────────────────────────────────────────
  {f'''
  const volDiv = document.createElement('div');
  volDiv.style.width = '100%';
  volDiv.style.borderTop = '1px solid {_BORDER}';
  container.appendChild(volDiv);

  const volChart = LightweightCharts.createChart(volDiv, {vol_opts_json});
  const volSeries = volChart.addHistogramSeries({{
    priceFormat: {{ type: 'volume' }},
    priceScaleId: 'vol',
    color: 'rgba(99,102,241,0.25)',
  }});
  volSeries.priceScale().applyOptions({{
    scaleMargins: {{ top: 0.05, bottom: 0.0 }},
  }});
  volSeries.setData({json.dumps(vol_data)});
  volChart.timeScale().fitContent();

  // Keep time scales in sync
  chart.timeScale().subscribeVisibleLogicalRangeChange(function(range) {{
    if (range) volChart.timeScale().setVisibleLogicalRange(range);
  }});
  volChart.timeScale().subscribeVisibleLogicalRangeChange(function(range) {{
    if (range) chart.timeScale().setVisibleLogicalRange(range);
  }});
  ''' if vol_data else ''}

  // ── Resize observer ─────────────────────────────────────────────────────
  const ro = new ResizeObserver(function() {{
    const w = container.clientWidth;
    chart.applyOptions({{ width: w }});
    {f"if (typeof volChart !== 'undefined') volChart.applyOptions({{ width: w }});" if vol_data else ''}
  }});
  ro.observe(container);
}})();
"""

    total_h = height + 26
    return _html_wrapper(js, height, f"{ticker} — Price Chart"), total_h


# ─────────────────────────────────────────────────────────────────────────────
# 2. MULTI-LINE  (HV windows, HV vs IV, volatility comparisons)
# ─────────────────────────────────────────────────────────────────────────────

def line_multi(
    series: dict[str, pd.Series],
    title: str = "",
    y_suffix: str = "",
    y_precision: int = 2,
    height: int = 380,
    fill_first: bool = False,
    hlines: list[dict] | None = None,
) -> tuple[str, int]:
    """
    Multiple line series on a single pane.

    Parameters
    ----------
    series      : {label: pd.Series with DatetimeIndex}
    title       : Chart title shown above.
    y_suffix    : e.g. '%' appended to tooltip values.
    fill_first  : Fill area under the first series.
    hlines      : Horizontal reference lines [{value, color, label}].
    """
    opts = _chart_base_options(height)
    opts_json = json.dumps(opts)

    series_js_parts = []
    for i, (name, s) in enumerate(series.items()):
        s = s.dropna().tail(1000)
        data = [{"time": t, "value": round(float(v), 6)}
                for t, v in zip(_ts(s.index), s.values)]
        color = _PALETTE[i % len(_PALETTE)]
        fill  = (i == 0 and fill_first)
        series_js_parts.append(f"""
  {{
    const s{i} = chart.add{'Area' if fill else 'Line'}Series({{
      color:            '{color}',
      lineWidth:        {'2' if i == 0 else '1.5'},
      {'topColor: "' + color.replace(')', ',0.15)').replace('rgb(','rgba(') + '", bottomColor: "rgba(255,255,255,0)", lineColor: "' + color + '",' if fill else ''}
      priceLineVisible: false,
      lastValueVisible: true,
      title:            '{name}',
      priceFormat: {{ type: 'custom', formatter: function(p) {{
        return p.toFixed({y_precision}) + '{y_suffix}';
      }} }},
    }});
    s{i}.setData({json.dumps(data)});
  }}""")

    hlines_json = json.dumps(hlines or [])

    js = f"""
(function() {{
  const container = document.getElementById('chart');
  const chart = LightweightCharts.createChart(container, {opts_json});

  {''.join(series_js_parts)}

  // Horizontal reference lines
  const hlines = {hlines_json};
  if (hlines.length) {{
    const refSeries = chart.addLineSeries({{
      color: 'transparent', priceLineVisible: false, lastValueVisible: false,
    }});
    hlines.forEach(function(hl) {{
      refSeries.createPriceLine({{
        price: hl.value,
        color: hl.color || '{_ACCENT2}',
        lineWidth: 1.5,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: true,
        title: hl.label || '',
      }});
    }});
  }}

  chart.timeScale().fitContent();

  const ro = new ResizeObserver(function() {{
    chart.applyOptions({{ width: container.clientWidth }});
  }});
  ro.observe(container);
}})();
"""
    total_h = height + 26
    return _html_wrapper(js, height, title), total_h


# ─────────────────────────────────────────────────────────────────────────────
# 3. AREA / LINE  (single series — price history, equity curve)
# ─────────────────────────────────────────────────────────────────────────────

def area_line(
    series: pd.Series,
    title: str = "",
    color: str = _ACCENT,
    y_suffix: str = "",
    y_precision: int = 2,
    height: int = 300,
    hlines: list[dict] | None = None,
) -> tuple[str, int]:
    """Single area+line series."""
    s = series.dropna().tail(1000)
    data = [{"time": t, "value": round(float(v), 6)}
            for t, v in zip(_ts(s.index), s.values)]

    opts = _chart_base_options(height)
    opts_json = json.dumps(opts)
    hlines_json = json.dumps(hlines or [])

    # Derive a transparent fill from the hex color
    if color.startswith("#") and len(color) == 7:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        top_fill    = f"rgba({r},{g},{b},0.18)"
        bottom_fill = f"rgba({r},{g},{b},0.01)"
    else:
        top_fill    = "rgba(99,102,241,0.18)"
        bottom_fill = "rgba(99,102,241,0.01)"

    js = f"""
(function() {{
  const container = document.getElementById('chart');
  const chart = LightweightCharts.createChart(container, {opts_json});

  const series = chart.addAreaSeries({{
    lineColor:    '{color}',
    topColor:     '{top_fill}',
    bottomColor:  '{bottom_fill}',
    lineWidth:    2,
    priceLineVisible: false,
    lastValueVisible: true,
    priceFormat: {{ type: 'custom', formatter: function(p) {{
      return p.toFixed({y_precision}) + '{y_suffix}';
    }} }},
  }});
  series.setData({json.dumps(data)});

  const hlines = {hlines_json};
  if (hlines.length) {{
    hlines.forEach(function(hl) {{
      series.createPriceLine({{
        price: hl.value,
        color: hl.color || '{_ACCENT2}',
        lineWidth: 1.5,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: true,
        title: hl.label || '',
      }});
    }});
  }}

  chart.timeScale().fitContent();
  const ro = new ResizeObserver(function() {{
    chart.applyOptions({{ width: container.clientWidth }});
  }});
  ro.observe(container);
}})();
"""
    total_h = height + 26
    return _html_wrapper(js, height, title), total_h


# ─────────────────────────────────────────────────────────────────────────────
# 4. VOL SMILE  (strike vs IV — bar or line)
# ─────────────────────────────────────────────────────────────────────────────

def vol_smile(
    strikes: "np.ndarray",
    ivs: "np.ndarray",
    spot: float,
    title: str = "Volatility Smile",
    height: int = 320,
) -> tuple[str, int]:
    """
    Vol smile as a line chart with ATM vertical marker.
    Uses numeric x-axis (strikes) rather than time-series.
    """
    opts = _chart_base_options(height)
    # Override for non-time x-axis
    opts["timeScale"]["timeVisible"] = False
    opts_json = json.dumps(opts)

    # Encode strikes as pseudo-dates offset from epoch for LWC time axis
    # Lightweight Charts requires time, so we use a baseline + index trick
    data = [
        {"time": str(20000101 + i), "value": round(float(iv * 100), 3),
         "strike": round(float(k), 2)}
        for i, (k, iv) in enumerate(zip(strikes, ivs))
    ]
    atm_idx = int(np.argmin(np.abs(strikes - spot)))
    atm_time = str(20000101 + atm_idx)

    js = f"""
(function() {{
  const container = document.getElementById('chart');
  const chart = LightweightCharts.createChart(container, {opts_json});

  // Custom tick labels for strikes
  const strikeLabels = {json.dumps([round(float(k), 2) for k in strikes])};
  chart.applyOptions({{
    timeScale: {{
      tickMarkFormatter: function(time) {{
        const idx = parseInt(String(time).slice(4)) - 1;
        return strikeLabels[idx] !== undefined ? String(strikeLabels[idx]) : '';
      }},
    }},
  }});

  const smileSeries = chart.addAreaSeries({{
    lineColor:    '{_ACCENT}',
    topColor:     'rgba(99,102,241,0.15)',
    bottomColor:  'rgba(99,102,241,0.01)',
    lineWidth:    2,
    priceLineVisible: false,
    lastValueVisible: false,
    priceFormat: {{ type: 'custom', formatter: p => p.toFixed(1) + '%' }},
  }});
  smileSeries.setData({json.dumps(data)});

  // ATM marker
  smileSeries.createPriceLine({{
    price: {round(float(ivs[atm_idx] * 100), 2)},
    color: '{_ACCENT2}',
    lineWidth: 1.5,
    lineStyle: LightweightCharts.LineStyle.Dashed,
    axisLabelVisible: true,
    title: 'ATM',
  }});

  chart.timeScale().fitContent();
  const ro = new ResizeObserver(function() {{
    chart.applyOptions({{ width: container.clientWidth }});
  }});
  ro.observe(container);
}})();
"""
    total_h = height + 26
    return _html_wrapper(js, height, title), total_h
