"""
Utilities
---------
Config loading, formatting helpers, and small numerical utilities.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Any


def load_config(config_path: str = "config/config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)


def format_currency(value: float, decimals: int = 4) -> str:
    if np.isnan(value) or np.isinf(value):
        return "N/A"
    return f"${value:,.{decimals}f}"


def format_pct(value: float, decimals: int = 2) -> str:
    if np.isnan(value) or np.isinf(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def dte_to_years(dte: int) -> float:
    return max(dte, 1) / 365.0


def years_to_dte(T: float) -> int:
    return max(int(round(T * 365)), 1)


def spot_range(S: float, pct: float = 0.30) -> tuple:
    return S * (1 - pct), S * (1 + pct)


def safe_div(a: float, b: float, default: float = np.nan) -> float:
    return a / b if b != 0 else default
