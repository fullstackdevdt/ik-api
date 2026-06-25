"""
Persistent JSON cache for historical OHLCV data.

Design
------
Historical bars never change once a candle closes, so we only ever need
to fetch the *delta* (bars since the last cached bar) on subsequent calls.

Cache layout
  cache/{symbol}_{bar_size}.json
  e.g. cache/tsla_1_day.json, cache/spy_4_hours.json

Initial fetch durations (per bar size)
  "1 day"   → 5 Y   (≈1 260 bars – enough for ML feature engineering)
  "4 hours" → 1 Y   (≈500 bars – IBKR max for 4h is ~1-2 Y)
  "1 hour"  → 6 M
  "30 mins" → 3 M
  "5 mins"  → 1 W

Usage
-----
    cache = DataCache(ib_client)
    daily_bars = cache.get("TSLA", "1 day")    # fetches 5Y on first call
    bars_4h    = cache.get("TSLA", "4 hours")  # fetches 1Y on first call
    # Subsequent calls return cached data + any new bars appended
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CACHE_DIR = "cache"

# How much history to pull on the very first fetch for each bar size
INITIAL_DURATIONS: Dict[str, str] = {
    "1 day":    "5 Y",
    "1 day,":   "5 Y",   # IBKR sometimes appends comma
    "4 hours":  "1 Y",
    "4 hour":   "1 Y",
    "1 hour":   "6 M",
    "30 mins":  "3 M",
    "5 mins":   "1 W",
}

# Minimum gap (days) that must have elapsed before we bother fetching a delta
MIN_STALE_DAYS: Dict[str, float] = {
    "1 day":   0.9,   # refresh once per calendar day
    "4 hours": 0.15,  # refresh if >≈3.5 h old (intra-day)
    "1 hour":  0.04,  # refresh if >≈1 h old
    "30 mins": 0.02,
    "5 mins":  0.003,
}


def _bar_to_dict(bar) -> Dict:
    """Convert an ib_insync BarData object to a plain dict."""
    d = bar.date
    if hasattr(d, "strftime"):
        date_str = d.strftime("%Y-%m-%d %H:%M:%S")
    else:
        date_str = str(d)
    return {
        "date":   date_str,
        "open":   bar.open,
        "high":   bar.high,
        "low":    bar.low,
        "close":  bar.close,
        "volume": bar.volume,
    }


def _parse_date(s: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y%m%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:len(fmt)], fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date format: {s!r}")


def _days_since(date_str: str) -> float:
    """Fractional days elapsed since date_str (positive = in the past)."""
    try:
        return (datetime.now() - _parse_date(date_str)).total_seconds() / 86400
    except Exception:
        return 999.0  # treat as very stale


def _delta_duration_str(days: float) -> str:
    """
    Build an IBKR durationStr that covers `days` of history with a small
    overlap buffer so boundary bars are never missed.
    """
    buffered = int(days) + 3   # always fetch a couple of extra days
    if buffered <= 365:
        return f"{buffered} D"
    return f"{buffered // 365 + 1} Y"


def _cache_key(bar_size: str) -> str:
    """Normalise bar_size to a safe filename component."""
    return bar_size.strip().lower().replace(" ", "_").replace(",", "")


class DataCache:
    """
    Manages persistent OHLCV data per symbol+bar_size.

    Parameters
    ----------
    ib_client:
        An IBKRClient instance (only needed when network fetches are required).
    cache_dir:
        Directory where JSON cache files are stored (created automatically).
    """

    def __init__(self, ib_client, cache_dir: str = CACHE_DIR) -> None:
        self.ib_client = ib_client
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    def cache_path(self, symbol: str, bar_size: str) -> str:
        key = _cache_key(bar_size)
        return os.path.join(self.cache_dir, f"{symbol.lower()}_{key}.json")

    def _load(self, path: str) -> Optional[Dict]:
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)

    def _save(
        self,
        path: str,
        symbol: str,
        bar_size: str,
        data: List[Dict],
        initial_duration: str,
    ) -> None:
        payload = {
            "symbol":           symbol.upper(),
            "bar_size":         bar_size,
            "initial_duration": initial_duration,
            "last_bar_date":    data[-1]["date"] if data else "",
            "data_points":      len(data),
            "updated_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data":             data,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(
            f"Cache saved: {symbol} {bar_size} — {len(data)} bars → {path}"
        )

    # ------------------------------------------------------------------
    # Merge helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge(existing: List[Dict], new_bars: List[Dict]) -> List[Dict]:
        """
        Combine existing cached bars with freshly fetched bars.
        Deduplicates on the 'date' field and sorts chronologically.
        """
        seen = {d["date"] for d in existing}
        appended = [b for b in new_bars if b["date"] not in seen]
        merged = existing + appended
        merged.sort(key=lambda d: d["date"])
        return merged

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(
        self,
        symbol: str,
        bar_size: str,
        force_refresh: bool = False,
    ) -> List[Dict]:
        """
        Return OHLCV data for (symbol, bar_size).

        Behaviour:
          - No cache file:     fetch full initial history, save, return.
          - Cache exists, not stale: return cached data immediately.
          - Cache exists, stale:    fetch delta, merge, re-save, return.
          - IBKR unavailable:       return cached data (if any) with a warning.
          - force_refresh=True:     always re-fetch the full initial history.
        """
        sym = symbol.upper()
        path = self.cache_path(sym, bar_size)
        initial_duration = INITIAL_DURATIONS.get(bar_size, "1 Y")
        min_stale = MIN_STALE_DAYS.get(bar_size, 0.9)

        cached = self._load(path)

        # ---- Full initial fetch ----
        if cached is None or force_refresh:
            logger.info(f"Cache MISS: {sym} {bar_size} — fetching {initial_duration}")
            try:
                bars = self.ib_client.get_historical_data(
                    symbol=sym,
                    duration=initial_duration,
                    bar_size=bar_size,
                    what_to_show="TRADES",
                )
                data = [_bar_to_dict(b) for b in (bars or [])]
            except Exception as e:
                logger.error(f"Initial fetch failed for {sym} {bar_size}: {e}")
                return cached["data"] if cached else []

            if data:
                self._save(path, sym, bar_size, data, initial_duration)
            return data

        # ---- Check staleness ----
        last_bar_date = cached.get("last_bar_date", "")
        days_old = _days_since(last_bar_date)

        if days_old < min_stale:
            logger.debug(f"Cache HIT: {sym} {bar_size} ({len(cached['data'])} bars, age {days_old:.2f}d)")
            return cached["data"]

        # ---- Delta fetch ----
        delta_duration = _delta_duration_str(days_old)
        logger.info(
            f"Cache STALE: {sym} {bar_size} — last bar {last_bar_date} "
            f"({days_old:.1f}d ago) — fetching delta {delta_duration}"
        )
        try:
            new_bars_raw = self.ib_client.get_historical_data(
                symbol=sym,
                duration=delta_duration,
                bar_size=bar_size,
                what_to_show="TRADES",
            )
            new_bars = [_bar_to_dict(b) for b in (new_bars_raw or [])]
        except Exception as e:
            logger.warning(
                f"Delta fetch failed for {sym} {bar_size}: {e}. "
                f"Returning stale cache ({len(cached['data'])} bars)."
            )
            return cached["data"]

        if not new_bars:
            logger.debug(f"No new bars for {sym} {bar_size} — market likely closed")
            return cached["data"]

        merged = self._merge(cached["data"], new_bars)
        self._save(path, sym, bar_size, merged, initial_duration)
        logger.info(
            f"Cache UPDATE: {sym} {bar_size} — "
            f"{len(merged) - len(cached['data'])} new bars, total {len(merged)}"
        )
        return merged

    def get_spy_vix(self) -> Dict:
        """
        Cached SPY (daily) + latest VIX.
        SPY uses the standard get() cache; VIX is fetched directly (small payload).
        """
        spy_data = self.get("SPY", "1 day")
        spy_closes = [d["close"] for d in spy_data]

        vix_close: Optional[float] = None
        try:
            ctx = self.ib_client.get_spy_vix_context()
            vix_close = ctx.get("vix_close")
        except Exception:
            logger.warning("VIX fetch failed; macro regime uses SPY-only signals")

        return {"spy_closes": spy_closes, "vix_close": vix_close}

    def warm(self, symbols: List[str], bar_sizes: Optional[List[str]] = None) -> Dict:
        """
        Pre-warm the cache for all (symbol, bar_size) combinations.
        Useful to call once before starting the bot so the first scan cycle is fast.

        Returns a summary dict: {symbol: {bar_size: data_points}}.
        """
        if bar_sizes is None:
            bar_sizes = ["1 day", "4 hours"]

        summary: Dict[str, Dict] = {}
        for sym in symbols:
            summary[sym] = {}
            for bs in bar_sizes:
                data = self.get(sym, bs, force_refresh=False)
                summary[sym][bs] = len(data)

        return summary

    def status(self) -> List[Dict]:
        """Return metadata for every cached file (useful for the /bot/cache_status endpoint)."""
        result = []
        if not os.path.exists(self.cache_dir):
            return result
        for fname in sorted(os.listdir(self.cache_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(self.cache_dir, fname)
            raw = self._load(fpath)
            if raw:
                result.append({
                    "file":           fname,
                    "symbol":         raw.get("symbol"),
                    "bar_size":       raw.get("bar_size"),
                    "data_points":    raw.get("data_points"),
                    "last_bar_date":  raw.get("last_bar_date"),
                    "updated_at":     raw.get("updated_at"),
                    "initial_duration": raw.get("initial_duration"),
                    "size_kb":        round(os.path.getsize(fpath) / 1024, 1),
                })
        return result
