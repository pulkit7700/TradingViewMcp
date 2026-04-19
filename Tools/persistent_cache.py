"""
Persistent disk-backed TTL cache for market data.

Designed for local app usage where we want yfinance responses to survive
Streamlit reruns and full app restarts.
"""

from __future__ import annotations

import hashlib
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from .utils import load_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "market_data"


def market_cache_settings(config: Optional[dict] = None) -> dict:
    """Resolve market-cache settings from config with sane defaults."""
    cfg = config if config is not None else load_config(str(PROJECT_ROOT / "config" / "config.yaml")) or {}
    cache_cfg = cfg.get("cache", {}) or {}

    cache_dir = Path(cache_cfg.get("dir", DEFAULT_CACHE_DIR))
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir

    return {
        "enabled": bool(cache_cfg.get("enabled", True)),
        "dir": cache_dir,
        "history_ttl_minutes": int(cache_cfg.get("history_ttl_minutes", 15)),
        "intraday_ttl_minutes": int(cache_cfg.get("intraday_ttl_minutes", 5)),
        "metadata_ttl_minutes": int(cache_cfg.get("metadata_ttl_minutes", 60)),
        "options_ttl_minutes": int(cache_cfg.get("options_ttl_minutes", 15)),
        "batch_ttl_minutes": int(cache_cfg.get("batch_ttl_minutes", 15)),
        "vix_ttl_minutes": int(cache_cfg.get("vix_ttl_minutes", 5)),
    }


class PersistentTTLCache:
    """Simple pickle-based persistent TTL cache."""

    def __init__(self, cache_dir: Optional[str | Path] = None):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.pkl"

    def get(self, key: str, ttl_seconds: Optional[int] = None) -> Any:
        path = self._path_for_key(key)
        if not path.exists():
            return None

        try:
            with path.open("rb") as fh:
                payload = pickle.load(fh)
        except Exception:
            path.unlink(missing_ok=True)
            return None

        created_at = float(payload.get("created_at", 0.0))
        if ttl_seconds is not None and ttl_seconds >= 0 and (time.time() - created_at) > ttl_seconds:
            path.unlink(missing_ok=True)
            return None

        return payload.get("value")

    def set(self, key: str, value: Any) -> None:
        path = self._path_for_key(key)
        payload = {
            "key": key,
            "created_at": time.time(),
            "value": value,
        }

        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.cache_dir, delete=False) as tmp:
                tmp_path = Path(tmp.name)
                pickle.dump(payload, tmp, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def delete(self, key: str) -> None:
        self._path_for_key(key).unlink(missing_ok=True)

    def clear(self) -> None:
        for path in self.cache_dir.glob("*.pkl"):
            path.unlink(missing_ok=True)

    def stats(self) -> dict:
        files = list(self.cache_dir.glob("*.pkl"))
        size_bytes = sum(p.stat().st_size for p in files if p.exists())
        return {
            "enabled": True,
            "path": str(self.cache_dir),
            "entry_count": len(files),
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
        }
