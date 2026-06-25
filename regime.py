"""
Market regime detection: classifies macro (SPY/VIX) and micro (per-stock) conditions.

Phase 1: Rule-based thresholds using RSI, MACD, SMA, ATR, VIX.
Phase 2 (future): Replace classify_macro / classify_micro with XGBoost/HMM models
                  trained on labels produced by this phase.

Macro labels: "bull", "caution", "bear", "neutral"
Micro labels: "bull", "bear", "oversold", "overbought", "neutral"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from indicators import compute_sma, compute_rsi, compute_macd

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------
MACRO_BULL = "bull"
MACRO_CAUTION = "caution"
MACRO_BEAR = "bear"
MACRO_NEUTRAL = "neutral"

MICRO_BULL = "bull"
MICRO_BEAR = "bear"
MICRO_OVERSOLD = "oversold"
MICRO_OVERBOUGHT = "overbought"
MICRO_NEUTRAL = "neutral"


@dataclass
class RegimeResult:
    macro: str                          # Macro label
    micro: str                          # Micro label
    confidence: float                   # 0.0 – 1.0 combined confidence
    raw_signals: Dict = field(default_factory=dict)
    trade_direction: str = "none"       # "long", "short", "none"
    position_size_multiplier: float = 0.0  # 0.0 (skip) → 1.0 (full size)


# ---------------------------------------------------------------------------
# Macro regime: SPY 50/200 SMA + VIX level
# ---------------------------------------------------------------------------

def classify_macro(
    spy_closes: List[float],
    vix_close: Optional[float],
) -> Tuple[str, float, Dict]:
    """
    Classify broad-market regime.

    Rules (macro wins in conflicts):
      Bull:    SPY > 200-SMA AND SPY > 50-SMA AND VIX < 20
      Caution: SPY between 50-SMA and 200-SMA, OR VIX 20-25
      Bear:    SPY < 200-SMA AND SPY < 50-SMA, OR VIX > 25

    Returns (label, confidence 0-1, signal dict).
    """
    signals: Dict = {}

    if len(spy_closes) < 50:
        signals["error"] = "Insufficient SPY data (need ≥50 bars)"
        return MACRO_NEUTRAL, 0.3, signals

    sma50_list = compute_sma(spy_closes, 50)
    sma200_list = compute_sma(spy_closes, 200) if len(spy_closes) >= 200 else [None] * len(spy_closes)

    last_close = spy_closes[-1]
    last_sma50 = sma50_list[-1]
    last_sma200 = sma200_list[-1]

    signals["spy_close"] = round(last_close, 2)
    signals["spy_sma_50"] = round(last_sma50, 2) if last_sma50 else None
    signals["spy_sma_200"] = round(last_sma200, 2) if last_sma200 else None
    signals["vix"] = vix_close

    above_50 = last_sma50 is not None and last_close > last_sma50
    above_200 = last_sma200 is not None and last_close > last_sma200
    vix_calm = vix_close is not None and vix_close < 20.0
    vix_elevated = vix_close is not None and 20.0 <= vix_close <= 25.0
    vix_panic = vix_close is not None and vix_close > 25.0

    signals.update({
        "above_sma_50": above_50,
        "above_sma_200": above_200,
        "vix_calm": vix_calm,
        "vix_elevated": vix_elevated,
        "vix_panic": vix_panic,
    })

    # ---- Bull ----
    if above_200 and above_50 and (vix_calm or vix_close is None):
        conf = 0.85 if vix_calm else 0.70
        return MACRO_BULL, conf, signals

    # ---- Bear ----
    if not above_200 and not above_50 and (vix_panic or vix_close is None):
        conf = 0.85 if vix_panic else 0.70
        return MACRO_BEAR, conf, signals

    if vix_panic and not above_200:
        return MACRO_BEAR, 0.80, signals

    # ---- Caution ----
    if above_200 and not above_50:
        return MACRO_CAUTION, 0.60, signals
    if above_50 and not above_200:
        return MACRO_CAUTION, 0.55, signals
    if vix_elevated:
        return MACRO_CAUTION, 0.55, signals

    return MACRO_NEUTRAL, 0.40, signals


# ---------------------------------------------------------------------------
# Micro regime: per-stock RSI, MACD, SMA
# ---------------------------------------------------------------------------

def classify_micro(
    stock_closes: List[float],
    stock_highs: Optional[List[float]] = None,
    stock_lows: Optional[List[float]] = None,
) -> Tuple[str, float, Dict]:
    """
    Classify individual stock regime using RSI + MACD histogram + SMA position.

    Returns (label, confidence 0-1, signal dict).
    """
    signals: Dict = {}

    if len(stock_closes) < 30:
        signals["error"] = "Insufficient data (need ≥30 bars)"
        return MICRO_NEUTRAL, 0.3, signals

    rsi_vals = compute_rsi(stock_closes)
    sma20_vals = compute_sma(stock_closes, 20)
    sma50_vals = compute_sma(stock_closes, 50)
    macd_data = compute_macd(stock_closes)

    # Pull last valid values
    last_rsi = next((v for v in reversed(rsi_vals) if v is not None), None)
    last_sma20 = next((v for v in reversed(sma20_vals) if v is not None), None)
    last_sma50 = next((v for v in reversed(sma50_vals) if v is not None), None)
    last_close = stock_closes[-1]

    hist_vals = [v for v in macd_data["histogram"] if v is not None]
    last_hist = hist_vals[-1] if hist_vals else None
    hist_rising = len(hist_vals) >= 2 and hist_vals[-1] > hist_vals[-2]

    signals.update({
        "rsi_14": round(last_rsi, 2) if last_rsi is not None else None,
        "close": round(last_close, 2),
        "sma_20": round(last_sma20, 2) if last_sma20 else None,
        "sma_50": round(last_sma50, 2) if last_sma50 else None,
        "macd_histogram": round(last_hist, 6) if last_hist is not None else None,
        "macd_hist_rising": hist_rising,
        "above_sma_20": last_sma20 is not None and last_close > last_sma20,
        "above_sma_50": last_sma50 is not None and last_close > last_sma50,
    })

    # Hard boundaries: overbought / oversold
    if last_rsi is not None and last_rsi > 70:
        return MICRO_OVERBOUGHT, 0.80, signals
    if last_rsi is not None and last_rsi < 30:
        return MICRO_OVERSOLD, 0.80, signals

    # Score-based bull / bear
    bull_score = 0.0
    bear_score = 0.0

    if last_rsi is not None:
        if last_rsi >= 50:
            bull_score += 1.0
        else:
            bear_score += 1.0
        if 45 <= last_rsi <= 65:   # "healthy" momentum zone
            bull_score += 0.5

    if last_hist is not None:
        if last_hist > 0:
            bull_score += 1.5 if hist_rising else 0.8
        else:
            bear_score += 1.5 if not hist_rising else 0.8

    if signals["above_sma_20"]:
        bull_score += 1.0
    else:
        bear_score += 1.0

    if signals["above_sma_50"]:
        bull_score += 0.5
    else:
        bear_score += 0.5

    total = bull_score + bear_score
    if total == 0:
        return MICRO_NEUTRAL, 0.40, signals

    if bull_score >= bear_score * 1.5:
        conf = min(0.90, 0.50 + (bull_score - bear_score) / total * 0.50)
        return MICRO_BULL, round(conf, 3), signals

    if bear_score >= bull_score * 1.5:
        conf = min(0.90, 0.50 + (bear_score - bull_score) / total * 0.50)
        return MICRO_BEAR, round(conf, 3), signals

    return MICRO_NEUTRAL, 0.40, signals


# ---------------------------------------------------------------------------
# Combined regime + trade gate
# ---------------------------------------------------------------------------

def get_regime(
    stock_closes: List[float],
    spy_closes: List[float],
    vix_close: Optional[float] = None,
    stock_highs: Optional[List[float]] = None,
    stock_lows: Optional[List[float]] = None,
    allow_short: bool = False,
) -> RegimeResult:
    """
    Full regime classification combining macro (SPY/VIX) and micro (stock).

    Returns RegimeResult with trade_direction and position_size_multiplier.

    Trade gate logic (macro takes priority):
      Bull macro + Bull micro   → LONG, full size
      Bull macro + Oversold     → LONG mean-reversion, 0.75x size
      Bull macro + Neutral      → LONG, 0.5x size
      Bull macro + Overbought   → skip (don't chase)
      Caution + Oversold/Bull   → LONG, 0.5x size
      Caution + else            → skip
      Bear macro + Bear/OB      → SHORT (if allow_short), else skip
      Bear macro + else         → skip (protect capital)
      Neutral + Bull/Oversold   → LONG, 0.5-0.75x size
    """
    macro, macro_conf, macro_sigs = classify_macro(spy_closes, vix_close)
    micro, micro_conf, micro_sigs = classify_micro(stock_closes, stock_highs, stock_lows)

    confidence = round(macro_conf * 0.5 + micro_conf * 0.5, 3)
    raw_signals = {"macro": macro_sigs, "micro": micro_sigs}

    direction = "none"
    size_mult = 0.0

    if macro == MACRO_BULL:
        if micro == MICRO_BULL:
            direction, size_mult = "long", 1.0
        elif micro == MICRO_OVERSOLD:
            direction, size_mult = "long", 0.75
        elif micro == MICRO_NEUTRAL:
            direction, size_mult = "long", 0.50
        elif micro == MICRO_OVERBOUGHT:
            direction, size_mult = "none", 0.0   # Don't chase extended moves
        elif micro == MICRO_BEAR:
            direction, size_mult = "none", 0.0   # Conflicting signals — wait

    elif macro == MACRO_CAUTION:
        if micro in (MICRO_OVERSOLD, MICRO_BULL):
            direction, size_mult = "long", 0.50
        # else: skip

    elif macro == MACRO_BEAR:
        if allow_short and micro in (MICRO_BEAR, MICRO_OVERBOUGHT):
            direction, size_mult = "short", (1.0 if micro == MICRO_BEAR else 0.75)
        # else: protect capital, no trade

    else:  # MACRO_NEUTRAL
        if micro == MICRO_BULL:
            direction, size_mult = "long", 0.75
        elif micro == MICRO_OVERSOLD:
            direction, size_mult = "long", 0.50

    return RegimeResult(
        macro=macro,
        micro=micro,
        confidence=confidence,
        raw_signals=raw_signals,
        trade_direction=direction,
        position_size_multiplier=size_mult,
    )
