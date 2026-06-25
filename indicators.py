"""
Technical indicator calculations for swing trading.
All functions accept List[float] arrays and return List[Optional[float]].
None values represent bars with insufficient history.
compute_all() augments each bar dict in a data list with all indicators.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------

def compute_sma(closes: List[float], period: int) -> List[Optional[float]]:
    """Simple Moving Average."""
    result: List[Optional[float]] = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        result[i] = round(sum(closes[i - period + 1 : i + 1]) / period, 6)
    return result


def compute_ema(closes: List[float], period: int) -> List[Optional[float]]:
    """Exponential Moving Average (seeded with first-period SMA)."""
    result: List[Optional[float]] = [None] * len(closes)
    if len(closes) < period:
        return result
    multiplier = 2.0 / (period + 1)
    ema = sum(closes[:period]) / period
    result[period - 1] = round(ema, 6)
    for i in range(period, len(closes)):
        ema = (closes[i] - ema) * multiplier + ema
        result[i] = round(ema, 6)
    return result


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def compute_rsi(closes: List[float], period: int = 14) -> List[Optional[float]]:
    """
    Wilder's RSI.
    Returns None for the first `period` bars (insufficient history).
    """
    result: List[Optional[float]] = [None] * len(closes)
    if len(closes) < period + 1:
        return result

    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    avg_gain = sum(max(c, 0.0) for c in changes[:period]) / period
    avg_loss = sum(abs(min(c, 0.0)) for c in changes[:period]) / period

    for i in range(period, len(closes)):
        if avg_loss == 0.0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = round(100.0 - (100.0 / (1.0 + rs)), 4)

        # Wilder smoothing for next iteration
        change = changes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(change, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + abs(min(change, 0.0))) / period

    return result


def compute_macd(
    closes: List[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Dict[str, List[Optional[float]]]:
    """
    MACD indicator.
    Returns {"macd": [...], "signal_line": [...], "histogram": [...]}.
    """
    ema_fast = compute_ema(closes, fast)
    ema_slow = compute_ema(closes, slow)

    macd_line: List[Optional[float]] = [
        round(f - s, 6) if f is not None and s is not None else None
        for f, s in zip(ema_fast, ema_slow)
    ]

    # Build signal EMA from the compacted (non-None) MACD values
    valid_pairs = [(i, v) for i, v in enumerate(macd_line) if v is not None]
    signal_line: List[Optional[float]] = [None] * len(closes)
    histogram: List[Optional[float]] = [None] * len(closes)

    if len(valid_pairs) >= signal:
        compact_values = [v for _, v in valid_pairs]
        sig_ema = compute_ema(compact_values, signal)
        for k, (orig_i, _) in enumerate(valid_pairs):
            if sig_ema[k] is not None:
                signal_line[orig_i] = sig_ema[k]
                if macd_line[orig_i] is not None:
                    histogram[orig_i] = round(macd_line[orig_i] - sig_ema[k], 6)

    return {"macd": macd_line, "signal_line": signal_line, "histogram": histogram}


def compute_bollinger(
    closes: List[float],
    period: int = 20,
    std_dev: float = 2.0,
) -> Dict[str, List[Optional[float]]]:
    """
    Bollinger Bands.
    Returns {"upper", "middle", "lower", "pct_b"}.
    pct_b = (close - lower) / (upper - lower); 0 = at lower band, 1 = at upper.
    """
    n = len(closes)
    upper: List[Optional[float]] = [None] * n
    middle: List[Optional[float]] = [None] * n
    lower: List[Optional[float]] = [None] * n
    pct_b: List[Optional[float]] = [None] * n

    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        sma = sum(window) / period
        std = math.sqrt(sum((p - sma) ** 2 for p in window) / period)
        u = sma + std_dev * std
        lo = sma - std_dev * std
        upper[i] = round(u, 6)
        middle[i] = round(sma, 6)
        lower[i] = round(lo, 6)
        band_width = u - lo
        pct_b[i] = round((closes[i] - lo) / band_width, 4) if band_width > 0 else 0.5

    return {"upper": upper, "middle": middle, "lower": lower, "pct_b": pct_b}


def compute_atr(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14,
) -> List[Optional[float]]:
    """
    Average True Range using Wilder's smoothing.
    """
    n = len(closes)
    result: List[Optional[float]] = [None] * n
    if n < period + 1:
        return result

    # True Range for each bar (first bar has no prior close)
    tr = [highs[0] - lows[0]]
    for i in range(1, n):
        tr.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))

    atr = sum(tr[:period]) / period
    result[period - 1] = round(atr, 6)
    for i in range(period, n):
        atr = (atr * (period - 1) + tr[i]) / period
        result[i] = round(atr, 6)

    return result


def compute_adx(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14,
) -> Dict[str, List[Optional[float]]]:
    """
    Average Directional Index (Wilder's method).
    Returns {"adx": [...], "plus_di": [...], "minus_di": [...]}.
    ADX > 25 indicates a trending market; < 20 indicates ranging.
    """
    n = len(closes)
    adx_out: List[Optional[float]] = [None] * n
    plus_di_out: List[Optional[float]] = [None] * n
    minus_di_out: List[Optional[float]] = [None] * n

    if n < period * 2 + 1:
        return {"adx": adx_out, "plus_di": plus_di_out, "minus_di": minus_di_out}

    # Build True Range, +DM, -DM arrays (length n-1, index j → closes index j+1)
    tr, pdm, mdm = [], [], []
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        dn = lows[i - 1] - lows[i]
        tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
        pdm.append(up if up > dn and up > 0 else 0.0)
        mdm.append(dn if dn > up and dn > 0 else 0.0)

    # Wilder-smooth from the first `period` bars
    s_tr = sum(tr[:period])
    s_pdm = sum(pdm[:period])
    s_mdm = sum(mdm[:period])

    dx_vals: List[tuple] = []  # (closes_index, dx_value)

    for j in range(period, len(tr)):
        s_tr = s_tr - s_tr / period + tr[j]
        s_pdm = s_pdm - s_pdm / period + pdm[j]
        s_mdm = s_mdm - s_mdm / period + mdm[j]

        pdi = 100.0 * s_pdm / s_tr if s_tr > 0 else 0.0
        mdi = 100.0 * s_mdm / s_tr if s_tr > 0 else 0.0

        ci = j + 1  # offset: tr index j maps to closes index j+1
        plus_di_out[ci] = round(pdi, 4)
        minus_di_out[ci] = round(mdi, 4)

        di_sum = pdi + mdi
        dx = 100.0 * abs(pdi - mdi) / di_sum if di_sum > 0 else 0.0
        dx_vals.append((ci, dx))

    # ADX = Wilder-smoothed DX
    if len(dx_vals) >= period:
        adx_val = sum(dv for _, dv in dx_vals[:period]) / period
        adx_out[dx_vals[period - 1][0]] = round(adx_val, 4)
        for k in range(period, len(dx_vals)):
            adx_val = (adx_val * (period - 1) + dx_vals[k][1]) / period
            adx_out[dx_vals[k][0]] = round(adx_val, 4)

    return {"adx": adx_out, "plus_di": plus_di_out, "minus_di": minus_di_out}


def compute_volume_ratio(volumes: List[float], period: int = 20) -> List[Optional[float]]:
    """Current volume divided by N-period average volume."""
    result: List[Optional[float]] = [None] * len(volumes)
    for i in range(period - 1, len(volumes)):
        avg = sum(volumes[i - period + 1 : i + 1]) / period
        result[i] = round(volumes[i] / avg, 4) if avg > 0 else None
    return result


# ---------------------------------------------------------------------------
# Convenience: augment a data list with all indicators
# ---------------------------------------------------------------------------

def compute_all(data: List[Dict]) -> List[Dict]:
    """
    Augments each bar dict with all computed indicator values.
    Returns a new list; original dicts are not mutated.
    Keys added: rsi_14, sma_20, sma_50, sma_200, macd, macd_signal,
                macd_histogram, bb_upper, bb_middle, bb_lower, bb_pct_b,
                atr_14, atr_pct, adx_14, plus_di, minus_di, volume_ratio.
    """
    if not data:
        return data

    closes = [d["close"] for d in data]
    highs = [d["high"] for d in data]
    lows = [d["low"] for d in data]
    volumes = [d["volume"] for d in data]

    rsi = compute_rsi(closes)
    sma_20 = compute_sma(closes, 20)
    sma_50 = compute_sma(closes, 50)
    sma_200 = compute_sma(closes, 200)
    macd_data = compute_macd(closes)
    bb_data = compute_bollinger(closes)
    atr = compute_atr(highs, lows, closes)
    adx_data = compute_adx(highs, lows, closes)
    vol_ratio = compute_volume_ratio(volumes)

    result = []
    for i, bar in enumerate(data):
        enriched = dict(bar)
        enriched["rsi_14"] = rsi[i]
        enriched["sma_20"] = sma_20[i]
        enriched["sma_50"] = sma_50[i]
        enriched["sma_200"] = sma_200[i]
        enriched["macd"] = macd_data["macd"][i]
        enriched["macd_signal"] = macd_data["signal_line"][i]
        enriched["macd_histogram"] = macd_data["histogram"][i]
        enriched["bb_upper"] = bb_data["upper"][i]
        enriched["bb_middle"] = bb_data["middle"][i]
        enriched["bb_lower"] = bb_data["lower"][i]
        enriched["bb_pct_b"] = bb_data["pct_b"][i]
        enriched["atr_14"] = atr[i]
        enriched["atr_pct"] = round(atr[i] / closes[i] * 100, 4) if atr[i] and closes[i] else None
        enriched["adx_14"] = adx_data["adx"][i]
        enriched["plus_di"] = adx_data["plus_di"][i]
        enriched["minus_di"] = adx_data["minus_di"][i]
        enriched["volume_ratio"] = vol_ratio[i]
        result.append(enriched)

    return result
