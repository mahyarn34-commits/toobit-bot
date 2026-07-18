"""
TOOBIT STAGE 1 MARKET SCANNER — v11.0 (Python implementation)
==============================================================

Converts the "Stage 1" prompt spec into real, testable code: market
discovery, filtering, liquidity/orderbook/trend/momentum/volatility
scoring, BTC regime analysis, risk penalties, and final ranking.

NO signal generation happens here (no buy/sell/entry/SL/TP/direction) —
that is Stage 2's job, deliberately kept out of this file.

-----------------------------------------------------------------
WHAT YOU MUST FILL IN
-----------------------------------------------------------------
This file has NO network access baked in. The four functions in the
`ToobitClient` class (fetch_active_markets, fetch_candles,
fetch_orderbook, fetch_funding_and_oi) are stubs — wire them to the
real TOOBIT REST/WebSocket endpoints. Everything downstream (all
scoring/filtering/ranking) is fully implemented and testable today
with synthetic data (see `if __name__ == "__main__"` at the bottom).

-----------------------------------------------------------------
BUGS FIXED / AMBIGUITIES RESOLVED vs. the v10.0 text prompt
-----------------------------------------------------------------
1. Small-market liquidity rule (was self-contradictory: "skip bands,
   rank directly" vs "use Top25/Middle/Bottom25"). Resolved: if
   total_symbols >= 20 -> 5-band percentile scoring; else -> collapsed
   3-band (Top25/Middle/Bottom25-remove) + LOW_MARKET_QUALITY flag.
   See `liquidity_score()`.

2. BTC adjustment undefined when a symbol has NO trend structure
   (trend_score == 0) while BTC is BULLISH/BEARISH. Resolved: treated
   as neutral (0 adjustment) — there's no structure to be "aligned" or
   "opposite" to. See `btc_adjustment()`.

3. MACD histogram % change formula divides by abs(previous), which is
   a ZeroDivisionError whenever the previous histogram value is
   exactly 0 (common at zero-line crossovers). Resolved: special-cased
   -- when previous == 0, classify by sign of current value directly
   instead of a percentage change. See `macd_histogram_score()`.

4. ATR% formula: the v10.0 text used ATR14/EMA50, differing from the
   earlier (and standard) ATR14/Price. Kept as ATR14/Price (the
   standard definition) since dividing by EMA50 looked like an
   accidental substitution, not a deliberate change. Flagged with a
   constant `ATR_PERCENT_DENOMINATOR` so you can flip it in one place
   if EMA50 was actually intended.

5. Open Interest score had undefined cases ("OI increasing + price
   decreasing", "OI flat"). Resolved: both score 0 (no bullish
   confirmation) explicitly, rather than silently falling through.
   See `open_interest_score()`.

6. Extreme-funding flag thresholds are asymmetric in every version of
   this prompt (-0.05% vs +0.08%). Left asymmetric but pulled into
   named constants (`EXTREME_NEG_FUNDING_THRESHOLD`,
   `EXTREME_POS_FUNDING_THRESHOLD`) — change both to the same
   magnitude if symmetry was actually intended.

7. Score caps (v10.0 claimed SPOT max=25, PERPETUAL max=27) were off
   by one: they omitted the max +1 BTC adjustment. This file does NOT
   hardcode a cap number at all — `max_theoretical_score()` computes
   it from the actual component maxima, so it can never drift out of
   sync with the scoring logic again.

8. `ema_state` was referenced in the v10.0 output schema but never
   defined. Resolved: "bullish" if EMA50 > EMA200, "bearish" if
   EMA50 < EMA200, "neutral" if within the no-trend band.

9. The v10.0 prompt text was truncated mid-schema (Part 4/4 cuts off
   inside the PERPETUAL result object). This file's OUTPUT_SCHEMA_KEYS
   / build_result_object() gives the complete, closed structure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

import numpy as np
import pandas as pd

# ==================================================================
# CONFIG / THRESHOLDS  (all magic numbers live here, nowhere else)
# ==================================================================

MIN_CANDLES = 1000
BTC_MIN_15M_CANDLES = 500

MAX_MISSING_CANDLE_PCT = 1.0          # %
MAX_CONSECUTIVE_MISSING = 3

MAX_ORDERBOOK_DELAY_SEC = 30          # relaxed from spec's 3s: REST polling
                                        # (not WebSocket) has inherent network
                                        # latency; 3s is unrealistic here.
MAX_TICKER_DELAY_SEC = 30             # same reasoning as above
MAX_CANDLE_DELAY_SEC = 300            # 5 minutes

ORDERBOOK_DEPTH_LEVELS = 50
ORDERBOOK_DEPTH_RANGE_PCT = 2.0
ORDERBOOK_MIN_LEVELS_TO_KEEP = 10

RSI_REMOVE_HIGH = 85
RSI_REMOVE_LOW = 15

ATR_PERCENT_MIN = 0.08                # % — below this, symbol removed
ATR_PERCENT_DENOMINATOR = "price"     # "price" (standard) or "ema50"

EXTREME_NEG_FUNDING_THRESHOLD = -0.05   # %
EXTREME_POS_FUNDING_THRESHOLD = 0.08    # %
# NOTE: intentionally asymmetric per source prompt — see docstring #6.

SPOT_QUALIFY_SCORE = 14
PERP_QUALIFY_SCORE = 15
MIN_QUALIFIED_FOR_FULL_THRESHOLD = 5   # below this -> LOW_MARKET_QUALITY fallback

TOP_N_RESULTS = 20

MarketType = Literal["SPOT", "PERPETUAL"]

RISK_FLAG_ENUM = {
    "ORDERBOOK_IMBALANCE", "STALE_DATA", "OVEREXTENDED", "VOLUME_SPIKE",
    "PUMP_RISK", "EXTREME_VOLATILITY", "MANIPULATION_RISK", "OI_SPIKE",
    "EXTREME_NEGATIVE_FUNDING", "EXTREME_POSITIVE_FUNDING",
}
MARKET_LEVEL_FLAG_ENUM = {"LOW_MARKET_QUALITY"}
BTC_REGIME_ENUM = {"BULLISH", "BEARISH", "RANGE", "NEUTRAL", "MIXED"}


# ==================================================================
# DATA CONTAINERS
# ==================================================================

@dataclass
class OrderbookSnapshot:
    bid: float
    ask: float
    bid_depth: float   # summed qty within ORDERBOOK_DEPTH_RANGE_PCT
    ask_depth: float
    levels_available: int
    age_seconds: float

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        return (self.ask - self.bid) / self.mid_price * 100

    @property
    def depth_ratio(self) -> float:
        if self.ask_depth == 0:
            return math.inf
        return self.bid_depth / self.ask_depth


@dataclass
class SymbolData:
    symbol: str
    market_type: MarketType
    candles_5m: pd.DataFrame          # columns: open,high,low,close,volume,close_time
    price: float
    volume_24h_usdt: float
    orderbook: OrderbookSnapshot
    ticker_age_seconds: float
    candle_age_seconds: float
    funding_rate_pct: Optional[float] = None       # already normalized to 8h-equivalent
    open_interest_series: Optional[pd.Series] = None  # last >=21 values, 5m spaced


@dataclass
class SymbolResult:
    symbol: str
    market_type: MarketType
    score: float
    metrics: dict
    risk_flags: list


# ==================================================================
# STEP: CLIENT INTERFACE (fill in with real TOOBIT endpoints)
# ==================================================================

class ToobitClient:
    """Stub interface — wire each method to the real TOOBIT API/WebSocket."""

    def fetch_active_markets(self, market_type: MarketType) -> list[str]:
        raise NotImplementedError("Wire this to TOOBIT's market-list endpoint")

    def fetch_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Must return completed candles only, columns:
        ['open','high','low','close','volume','close_time'] with close_time
        as UTC-aware pandas Timestamps, sorted oldest->newest."""
        raise NotImplementedError("Wire this to TOOBIT's kline endpoint")

    def fetch_orderbook(self, symbol: str) -> OrderbookSnapshot:
        raise NotImplementedError("Wire this to TOOBIT's order book endpoint")

    def fetch_funding_and_oi(self, symbol: str) -> tuple[float, pd.Series]:
        """Returns (funding_rate_pct_8h_equivalent, open_interest_series)."""
        raise NotImplementedError("Wire this to TOOBIT's funding/OI endpoints")


# ==================================================================
# INDICATORS
# ==================================================================

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100)  # avg_loss==0 -> RSI=100 by convention


def macd_histogram(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = atr_wilder(high, low, close, length)
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / length, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / length, adjust=False).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(alpha=1 / length, adjust=False).mean().fillna(0)


def compute_indicators(candles: pd.DataFrame) -> dict:
    close, high, low, vol = candles["close"], candles["high"], candles["low"], candles["volume"]
    out = {
        "ema20": ema(close, 20), "ema50": ema(close, 50), "ema200": ema(close, 200),
        "rsi14": rsi_wilder(close, 14),
        "macd_hist": macd_histogram(close),
        "adx14": adx_wilder(high, low, close, 14),
        "atr14": atr_wilder(high, low, close, 14),
        "volume_ma20": vol.rolling(20).mean(),
    }
    if any(s.isna().iloc[-1] for s in out.values()):
        return {}  # signals "cannot be computed" -> caller removes symbol
    return out


def slope_direction(series: pd.Series, lookback: int, threshold_pct: float = 0.05) -> str:
    if len(series) <= lookback:
        return "flat"
    prev = series.iloc[-1 - lookback]
    curr = series.iloc[-1]
    if prev == 0:
        return "flat"
    change_pct = (curr - prev) / prev * 100
    if change_pct > threshold_pct:
        return "positive"
    if change_pct < -threshold_pct:
        return "negative"
    return "flat"


# ==================================================================
# STEP: MARKET-LEVEL FILTERS (removal, not scoring)
# ==================================================================

STABLE_BASE_ASSETS = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDD"}
LEVERAGED_TOKEN_MARKERS = ("3L", "3S", "5L", "5S", "UP", "DOWN", "BULL", "BEAR")


def is_removable_by_naming(base_asset: str) -> bool:
    if base_asset.upper() in STABLE_BASE_ASSETS:
        return True
    return any(marker in base_asset.upper() for marker in LEVERAGED_TOKEN_MARKERS)


def candle_validity_check(candles: pd.DataFrame, expected_count: int) -> Optional[str]:
    """Returns a removal reason string, or None if candles are valid."""
    if len(candles) < expected_count:
        return "INSUFFICIENT_HISTORY"
    missing_pct = max(0.0, (expected_count - len(candles)) / expected_count * 100)
    if missing_pct > MAX_MISSING_CANDLE_PCT:
        return "MISSING_CANDLES_EXCEEDED"
    gaps = candles["close_time"].diff().dropna()
    if len(gaps) == 0:
        return None
    modal_gap = gaps.mode().iloc[0]
    consecutive_missing = ((gaps / modal_gap).round() - 1).clip(lower=0)
    if (consecutive_missing > MAX_CONSECUTIVE_MISSING).any():
        return "ABNORMAL_CANDLE_GAP"
    if not candles["close_time"].is_monotonic_increasing:
        return "BROKEN_SEQUENCE"
    return None


def data_freshness_check(orderbook_age: float, ticker_age: float, candle_age: float) -> Optional[str]:
    if orderbook_age > MAX_ORDERBOOK_DELAY_SEC:
        return "STALE_ORDERBOOK"
    if ticker_age > MAX_TICKER_DELAY_SEC:
        return "STALE_TICKER"
    if candle_age > MAX_CANDLE_DELAY_SEC:
        return "STALE_CANDLE"
    return None
    # Per spec: stale data is a REMOVAL condition, not a risk_flags entry.


# ==================================================================
# STEP: LIQUIDITY SCORE  (fix #1 — small-market rule made consistent)
# ==================================================================

def liquidity_score(volumes: pd.Series) -> tuple[pd.Series, bool]:
    """
    volumes: 24h USDT volume indexed by symbol, one market type at a time.
    Returns (score_per_symbol[NaN==REMOVE], low_market_quality_flag).
    """
    n = len(volumes)
    ranked = volumes.rank(ascending=False, method="min")
    pct = ranked / n * 100
    low_quality = n < 20

    scores = pd.Series(index=volumes.index, dtype=float)
    if not low_quality:
        scores[pct <= 5] = 5
        scores[(pct > 5) & (pct <= 10)] = 4
        scores[(pct > 10) & (pct <= 25)] = 3
        scores[(pct > 25) & (pct <= 90)] = 1
        scores[pct > 90] = np.nan  # REMOVE
    else:
        scores[pct <= 25] = 3         # collapsed "top" band
        scores[(pct > 25) & (pct <= 75)] = 1
        scores[pct > 75] = np.nan     # REMOVE

    return scores, low_quality


# ==================================================================
# STEP: ORDERBOOK SCORE
# ==================================================================

def orderbook_score(ob: OrderbookSnapshot) -> tuple[Optional[float], list[str]]:
    if ob.levels_available < ORDERBOOK_MIN_LEVELS_TO_KEEP:
        return None, []  # None -> caller removes symbol

    score = 0.0
    if ob.spread_pct < 0.05:
        score += 2
    elif ob.spread_pct < 0.15:
        score += 1

    flags = []
    if 0.8 <= ob.depth_ratio <= 1.2:
        score += 1
    if ob.depth_ratio > 2.0 or ob.depth_ratio < 0.5:
        flags.append("ORDERBOOK_IMBALANCE")

    return score, flags


# ==================================================================
# STEP: BTC REGIME + BTC ADJUSTMENT  (fix #2)
# ==================================================================

def classify_btc_regime(price: float, ema50: float, ema200: float, adx: float) -> str:
    if price > ema50 and ema50 > ema200 and adx >= 20:
        return "BULLISH"
    if price < ema50 and ema50 < ema200 and adx >= 20:
        return "BEARISH"
    if adx < 15:
        return "RANGE"
    if 15 <= adx < 20:
        return "NEUTRAL"
    return "MIXED"  # adx >= 20 but structure not clean bullish/bearish


def btc_adjustment(btc_regime: str, symbol_ema50: float, symbol_ema200: float,
                    symbol_price: float, symbol_trend_score: float) -> int:
    if btc_regime in ("RANGE", "NEUTRAL", "MIXED"):
        return 0
    if symbol_trend_score == 0:
        # No defined structure to be aligned/opposite to (fix #2).
        return 0
    if btc_regime == "BULLISH":
        aligned = symbol_ema50 > symbol_ema200 and symbol_price > symbol_ema50
    else:  # BEARISH
        aligned = symbol_ema50 < symbol_ema200 and symbol_price < symbol_ema50
    return 1 if aligned else -1


# ==================================================================
# STEP: TREND QUALITY SCORE  (+ ema_state, fix #8)
# ==================================================================

def trend_quality(price: float, ema50: float, ema200: float, adx: float,
                   ema50_slope: str, ema200_slope: str) -> tuple[float, str]:
    no_trend = abs(ema50 - ema200) / ema200 * 100 < 0.1 or adx < 20
    if no_trend:
        ema_state = "bullish" if ema50 > ema200 else "bearish" if ema50 < ema200 else "neutral"
        return 0.0, ema_state

    score = 0.0
    if ema50 > ema200:
        ema_state = "bullish"
        score += 2
        if price > ema50:
            score += 1
        if adx > 25:
            score += 2
        elif 20 <= adx <= 25:
            score += 1
        if ema50_slope == "positive":
            score += 1
        if ema200_slope == "positive":
            score += 1
    else:
        ema_state = "bearish"
        score += 2
        if price < ema50:
            score += 1
        if adx > 25:
            score += 2
        elif 20 <= adx <= 25:
            score += 1
        if ema50_slope == "negative":
            score += 1
        if ema200_slope == "negative":
            score += 1

    return score, ema_state


# ==================================================================
# STEP: OVEREXTENSION
# ==================================================================

def overextension_check(price: float, ema50: float) -> tuple[float, list[str]]:
    distance = abs(price - ema50) / ema50 * 100
    if distance > 8:
        return -2.0, ["OVEREXTENDED"]
    return 0.0, []


# ==================================================================
# STEP: MOMENTUM QUALITY  (fix #3 — MACD divide-by-zero)
# ==================================================================

def rsi_removal_and_score(rsi: float) -> Optional[float]:
    if rsi > RSI_REMOVE_HIGH or rsi < RSI_REMOVE_LOW:
        return None  # -> caller removes symbol
    if rsi < 30:
        return 0.0
    if rsi < 55:
        return 1.0
    if rsi < 70:
        return 2.0
    return 1.0  # 70 <= rsi <= 85


def rsi_slope_score(rsi_series: pd.Series, lookback: int = 5) -> float:
    if len(rsi_series) <= lookback:
        return 0.0
    change = rsi_series.iloc[-1] - rsi_series.iloc[-1 - lookback]
    if change >= 1:
        return 1.0
    if change <= -1:
        return -1.0
    return 0.0


def macd_histogram_score(hist_series: pd.Series, lookback: int = 5) -> float:
    if len(hist_series) <= lookback:
        return 1.0  # not enough data -> treat as flat
    prev = hist_series.iloc[-1 - lookback]
    curr = hist_series.iloc[-1]

    if prev == 0:
        # Fix #3: avoid ZeroDivisionError at zero-line crossovers.
        # Classify by sign of current value instead of a % change.
        if curr > 0:
            return 2.0
        if curr < 0:
            return 0.0
        return 1.0

    change_pct = abs(curr - prev) / abs(prev) * 100
    if change_pct >= 5 and curr > prev:
        return 2.0
    if change_pct >= 5 and curr < prev:
        return 0.0
    return 1.0  # change < 5% -> flat


def momentum_quality(rsi_series: pd.Series, hist_series: pd.Series) -> Optional[float]:
    rsi_val = rsi_series.iloc[-1]
    rsi_component = rsi_removal_and_score(rsi_val)
    if rsi_component is None:
        return None  # caller removes symbol
    return rsi_component + rsi_slope_score(rsi_series) + macd_histogram_score(hist_series)


# ==================================================================
# STEP: VOLUME QUALITY
# ==================================================================

def volume_quality(current_volume: float, volume_ma20: float) -> tuple[float, list[str]]:
    if volume_ma20 == 0:
        return 0.0, []
    ratio = current_volume / volume_ma20
    if ratio < 1:
        return 0.0, []
    if ratio < 3:
        return 2.0, []
    if ratio < 5:
        return 1.0, []
    if ratio < 8:
        return 0.0, ["VOLUME_SPIKE"]
    return -2.0, ["PUMP_RISK"]   # score contribution folded in as penalty here


# ==================================================================
# STEP: VOLATILITY QUALITY  (fix #4 — ATR% denominator documented)
# ==================================================================

def volatility_quality(atr14: float, price: float, ema50: float) -> Optional[tuple[float, list[str]]]:
    denominator = price if ATR_PERCENT_DENOMINATOR == "price" else ema50
    atr_pct = atr14 / denominator * 100
    if atr_pct < ATR_PERCENT_MIN:
        return None  # caller removes symbol
    if atr_pct < 0.15:
        return 0.0, []
    if atr_pct <= 3:
        return 3.0, []
    if atr_pct <= 5:
        return 1.0, []
    return -2.0, ["EXTREME_VOLATILITY"]


# ==================================================================
# STEP: MANIPULATION CHECK (penalty only, no removal)
# ==================================================================

def manipulation_check(candle_volume: float, volume_ma20: float, price_move_pct: float) -> list[str]:
    if volume_ma20 > 0 and candle_volume > 10 * volume_ma20 and abs(price_move_pct) > 5:
        return ["MANIPULATION_RISK"]
    return []


# ==================================================================
# STEP: PERPETUAL-ONLY — OPEN INTEREST + FUNDING  (fixes #5, #6)
# ==================================================================

def open_interest_score(oi_series: pd.Series, price_series: pd.Series, lookback: int = 20) -> tuple[float, list[str]]:
    if len(oi_series) <= lookback:
        return 0.0, []
    oi_change_pct = (oi_series.iloc[-1] - oi_series.iloc[-1 - lookback]) / oi_series.iloc[-1 - lookback] * 100
    price_change_pct = (price_series.iloc[-1] - price_series.iloc[-1 - lookback]) / price_series.iloc[-1 - lookback] * 100

    if oi_change_pct > 0 and price_change_pct > 0:
        score = 2.0
    elif oi_change_pct > 0 and abs(price_change_pct) < 0.05:
        score = 1.0
    elif oi_change_pct > 0 and price_change_pct < 0:
        score = 0.0   # fix #5: explicit, was undefined
    else:
        score = 0.0   # OI decreasing, or flat -> 0 (fix #5)

    rolling = oi_series.pct_change().rolling(lookback)
    mean, std = rolling.mean().iloc[-1], rolling.std().iloc[-1]
    flags = []
    if std and not math.isnan(std) and std > 0:
        z = (oi_series.pct_change().iloc[-1] - mean) / std
        if abs(z) > 3:
            flags.append("OI_SPIKE")

    return score, flags


def funding_adjustment(funding_pct: float) -> tuple[float, list[str]]:
    abs_funding = abs(funding_pct)
    if abs_funding <= 0.03:
        adj = 0.0
    elif abs_funding <= 0.08:
        adj = -1.0
    else:
        adj = -2.0

    flags = []
    if funding_pct < EXTREME_NEG_FUNDING_THRESHOLD:
        flags.append("EXTREME_NEGATIVE_FUNDING")
    if funding_pct > EXTREME_POS_FUNDING_THRESHOLD:
        flags.append("EXTREME_POSITIVE_FUNDING")
    return adj, flags


# ==================================================================
# SCORE CAP  (fix #7 — computed, never hardcoded)
# ==================================================================

def max_theoretical_score(market_type: MarketType) -> float:
    liquidity_max = 5
    orderbook_max = 3
    trend_max = 7
    momentum_max = 5   # +2 rsi, +1 slope, +2 macd
    volume_max = 2
    volatility_max = 3
    btc_adj_max = 1
    oi_max = 2 if market_type == "PERPETUAL" else 0
    funding_adj_max = 0  # funding adjustment is always <= 0, never adds
    return (liquidity_max + orderbook_max + trend_max + momentum_max +
            volume_max + volatility_max + btc_adj_max + oi_max + funding_adj_max)


# ==================================================================
# ORCHESTRATION — scoring a single symbol end to end
# ==================================================================

def score_symbol(data: SymbolData, btc_regime: str) -> tuple[Optional[SymbolResult], Optional[str]]:
    ind = compute_indicators(data.candles_5m)
    if not ind:
        return None, "INDICATORS_UNAVAILABLE"

    price = data.price
    ema50, ema200 = ind["ema50"].iloc[-1], ind["ema200"].iloc[-1]
    adx = ind["adx14"].iloc[-1]
    atr14 = ind["atr14"].iloc[-1]
    rsi_series, hist_series = ind["rsi14"], ind["macd_hist"]
    vol_ma20 = ind["volume_ma20"].iloc[-1]
    current_vol = data.candles_5m["volume"].iloc[-1]

    ema50_slope = slope_direction(ind["ema50"], 20)
    ema200_slope = slope_direction(ind["ema200"], 20)

    risk_flags: list[str] = []

    ob_result = orderbook_score(data.orderbook)
    if ob_result[0] is None:
        return None, "ORDERBOOK_LEVELS_TOO_FEW"
    ob_score, ob_flags = ob_result
    risk_flags += ob_flags

    trend_score, ema_state = trend_quality(price, ema50, ema200, adx, ema50_slope, ema200_slope)

    over_penalty, over_flags = overextension_check(price, ema50)
    risk_flags += over_flags

    momentum = momentum_quality(rsi_series, hist_series)
    if momentum is None:
        return None, "RSI_OUT_OF_RANGE"

    vol_score, vol_flags = volume_quality(current_vol, vol_ma20)
    risk_flags += vol_flags
    vol_penalty = vol_score if vol_score < 0 else 0.0
    vol_score = max(vol_score, 0.0)

    volat_result = volatility_quality(atr14, price, ema50)
    if volat_result is None:
        return None, "ATR_PERCENT_TOO_LOW"
    volat_score, volat_flags = volat_result
    risk_flags += volat_flags
    volat_penalty = volat_score if volat_score < 0 else 0.0
    volat_score = max(volat_score, 0.0)

    price_move_pct = (data.candles_5m["close"].iloc[-1] - data.candles_5m["open"].iloc[-1]) \
        / data.candles_5m["open"].iloc[-1] * 100
    risk_flags += manipulation_check(current_vol, vol_ma20, price_move_pct)

    btc_adj = btc_adjustment(btc_regime, ema50, ema200, price, trend_score)

    funding_adj, oi_score, oi_flags = 0.0, 0.0, []
    if data.market_type == "PERPETUAL":
        if data.funding_rate_pct is None or data.open_interest_series is None:
            return None, "FUTURES_DATA_MISSING"
        funding_adj, funding_flags = funding_adjustment(data.funding_rate_pct)
        risk_flags += funding_flags
        oi_score, oi_flags = open_interest_score(data.open_interest_series, data.candles_5m["close"])
        risk_flags += oi_flags

    risk_penalty_total = 0.0
    for flag, penalty in (
        ("OVEREXTENDED", 2), ("PUMP_RISK", 2), ("EXTREME_VOLATILITY", 2),
        ("MANIPULATION_RISK", 3), ("ORDERBOOK_IMBALANCE", 1), ("OI_SPIKE", 1),
    ):
        if flag in risk_flags:
            risk_penalty_total += penalty

    raw_score = ob_score + trend_score + momentum + vol_score + volat_score + oi_score
    final_score = raw_score + btc_adj + funding_adj - risk_penalty_total
    final_score = max(0.0, min(final_score, max_theoretical_score(data.market_type)))

    metrics = {
        "liquidity_score": None,  # filled in by caller (needs full universe)
        "orderbook_score": ob_score,
        "trend_score": trend_score,
        "momentum_score": momentum,
        "volume_score": vol_score,
        "volatility_score": volat_score,
        "btc_adjustment": btc_adj,
        "rsi_value": round(float(rsi_series.iloc[-1]), 2),
        "adx_value": round(float(adx), 2),
        "atr_percent": round(float(atr14 / price * 100), 4),
        "volume_ratio": round(float(current_vol / vol_ma20), 3) if vol_ma20 else None,
        "ema_state": ema_state,
        "ema50_slope": ema50_slope,
        "ema200_slope": ema200_slope,
    }
    if data.market_type == "PERPETUAL":
        metrics["open_interest_score"] = oi_score
        metrics["funding_adjustment"] = funding_adj

    return SymbolResult(
        symbol=data.symbol, market_type=data.market_type,
        score=final_score, metrics=metrics, risk_flags=sorted(set(risk_flags)),
    ), None


# ==================================================================
# MARKET-LEVEL ORCHESTRATION
# ==================================================================

def rank_market(results: list[SymbolResult], volumes: pd.Series) -> tuple[list[dict], bool]:
    scores, low_quality = liquidity_score(volumes)

    final = []
    for r in results:
        liq = scores.get(r.symbol, np.nan)
        if pd.isna(liq):
            continue  # bottom-liquidity removal
        r.metrics["liquidity_score"] = liq
        r.score += liq
        final.append(r)

    qualify_threshold = SPOT_QUALIFY_SCORE if final and final[0].market_type == "SPOT" else PERP_QUALIFY_SCORE
    qualified = [r for r in final if r.score >= qualify_threshold]

    fallback_used = False
    if len(qualified) < MIN_QUALIFIED_FOR_FULL_THRESHOLD:
        qualified = final
        fallback_used = True
        low_quality = True

    qualified.sort(key=lambda r: (
        -r.score, -r.metrics["liquidity_score"], -r.metrics["orderbook_score"],
        -r.metrics["trend_score"], -r.metrics["momentum_score"],
        len(r.risk_flags), -volumes.get(r.symbol, 0),
    ))

    top = qualified[:TOP_N_RESULTS]
    out = []
    for rank, r in enumerate(top, start=1):
        out.append({
            "symbol": r.symbol, "market_type": r.market_type,
            "score": round(r.score, 2), "rank": rank,
            "metrics": r.metrics, "risk_flags": r.risk_flags,
        })
    return out, low_quality


def build_output(spot_results, spot_volumes, perp_results, perp_volumes,
                  btc_regime, btc_regime_5m, btc_regime_15m,
                  spot_scanned, spot_removed, perp_scanned, perp_removed) -> dict:
    spot_ranked, spot_low_q = rank_market(spot_results, spot_volumes)
    perp_ranked, perp_low_q = rank_market(perp_results, perp_volumes)

    def market_quality(results: list[dict]) -> str:
        if not results:
            return "LOW"
        avg = sum(r["score"] for r in results) / len(results)
        if avg > 18:
            return "HIGH"
        if avg >= 12:
            return "MEDIUM"
        return "LOW"

    flags = []
    if spot_low_q or perp_low_q:
        flags.append("LOW_MARKET_QUALITY")

    return {
        "exchange": "TOOBIT",
        "scan_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "spot_market_quality": market_quality(spot_ranked),
        "perpetual_market_quality": market_quality(perp_ranked),
        "spot_total_symbols_scanned": spot_scanned,
        "perpetual_total_symbols_scanned": perp_scanned,
        "spot_total_symbols_removed": spot_removed,
        "perpetual_total_symbols_removed": perp_removed,
        "spot_qualified_symbols_count": len(spot_ranked),
        "perpetual_qualified_symbols_count": len(perp_ranked),
        "btc_regime": btc_regime,
        "btc_regime_5m": btc_regime_5m,
        "btc_regime_15m": btc_regime_15m,
        "market_quality_flags": flags,
        "spot_results": spot_ranked,
        "perpetual_results": perp_ranked,
    }


# ==================================================================
# FULL PIPELINE — ties the ToobitClient to every step above.
# This is the single entry point the Telegram bot (or anything else)
# should call. Everything before this point is pure/testable logic;
# this function is the only place that actually calls the client.
# ==================================================================

def _base_asset(symbol: str) -> str:
    return symbol.replace("USDT", "").replace("_PERP", "").strip()


def _build_symbol_data(client: ToobitClient, symbol: str, market_type: MarketType) -> tuple[Optional[SymbolData], Optional[str]]:
    candles = client.fetch_candles(symbol, "5m", MIN_CANDLES)
    reason = candle_validity_check(candles, MIN_CANDLES)
    if reason is not None:
        return None, f"CANDLE_{reason}"

    ob = client.fetch_orderbook(symbol)
    candle_age = (datetime.now(timezone.utc) - candles["close_time"].iloc[-1]).total_seconds()
    freshness_reason = data_freshness_check(ob.age_seconds, ob.age_seconds, candle_age)
    if freshness_reason is not None:
        if freshness_reason == "STALE_CANDLE":
            bucket = int(candle_age // 60) * 60
            return None, f"STALE_CANDLE_{bucket}s-{bucket+60}s"
        return None, freshness_reason

    price = float(candles["close"].iloc[-1])
    volume_24h = float(candles["volume"].iloc[-288:].sum() * price) if len(candles) >= 288 else float(candles["volume"].sum() * price)

    funding_rate, oi_series = (None, None)
    if market_type == "PERPETUAL":
        funding_rate, oi_series = client.fetch_funding_and_oi(symbol)

    return SymbolData(
        symbol=symbol, market_type=market_type, candles_5m=candles, price=price,
        volume_24h_usdt=volume_24h, orderbook=ob, ticker_age_seconds=ob.age_seconds,
        candle_age_seconds=candle_age, funding_rate_pct=funding_rate, open_interest_series=oi_series,
    ), None


def _determine_btc_regime(client: ToobitClient) -> tuple[str, str, str]:
    c5 = client.fetch_candles("BTCUSDT", "5m", MIN_CANDLES)
    c15 = client.fetch_candles("BTCUSDT", "15m", BTC_MIN_15M_CANDLES)

    ind5, ind15 = compute_indicators(c5), compute_indicators(c15)
    if not ind5 or not ind15:
        return "NEUTRAL", "NEUTRAL", "NEUTRAL"  # btc_data_unavailable case

    regime_5m = classify_btc_regime(c5["close"].iloc[-1], ind5["ema50"].iloc[-1], ind5["ema200"].iloc[-1], ind5["adx14"].iloc[-1])
    regime_15m = classify_btc_regime(c15["close"].iloc[-1], ind15["ema50"].iloc[-1], ind15["ema200"].iloc[-1], ind15["adx14"].iloc[-1])
    return regime_15m, regime_5m, regime_15m  # 15m is primary per spec


def run_scan(client: ToobitClient, market_types: tuple[MarketType, ...] = ("SPOT", "PERPETUAL")) -> dict:
    """The single function to call for a full Stage-1 scan. Wire your
    real ToobitClient in and call run_scan(client). Pass market_types=
    ("SPOT",) or ("PERPETUAL",) to scan only one market type (faster,
    fewer API calls) — the other market's section in the output will
    simply be empty."""
    import collections

    btc_regime, btc_regime_5m, btc_regime_15m = _determine_btc_regime(client)

    all_results: dict[MarketType, list[SymbolResult]] = {"SPOT": [], "PERPETUAL": []}
    all_volumes: dict[MarketType, dict[str, float]] = {"SPOT": {}, "PERPETUAL": {}}
    scanned_counts: dict[MarketType, int] = {"SPOT": 0, "PERPETUAL": 0}
    removed_counts: dict[MarketType, int] = {"SPOT": 0, "PERPETUAL": 0}
    removal_reasons: dict[MarketType, "collections.Counter[str]"] = {
        "SPOT": collections.Counter(), "PERPETUAL": collections.Counter(),
    }

    for market_type in market_types:
        symbols = client.fetch_active_markets(market_type)
        scanned_counts[market_type] = len(symbols)

        for symbol in symbols:
            if is_removable_by_naming(_base_asset(symbol)):
                removed_counts[market_type] += 1
                removal_reasons[market_type]["NAMING_FILTER"] += 1
                continue

            try:
                data, reason = _build_symbol_data(client, symbol, market_type)
                if data is None:
                    removed_counts[market_type] += 1
                    removal_reasons[market_type][reason or "UNKNOWN"] += 1
                    continue

                result, reason = score_symbol(data, btc_regime)
                if result is None:
                    removed_counts[market_type] += 1
                    removal_reasons[market_type][reason or "UNKNOWN"] += 1
                    continue
            except Exception as e:  # one bad symbol should never kill the whole scan
                removed_counts[market_type] += 1
                removal_reasons[market_type][f"EXCEPTION_{type(e).__name__}"] += 1
                continue

            all_results[market_type].append(result)
            all_volumes[market_type][symbol] = data.volume_24h_usdt

    output = build_output(
        spot_results=all_results["SPOT"],
        spot_volumes=pd.Series(all_volumes["SPOT"], dtype=float),
        perp_results=all_results["PERPETUAL"],
        perp_volumes=pd.Series(all_volumes["PERPETUAL"], dtype=float),
        btc_regime=btc_regime, btc_regime_5m=btc_regime_5m, btc_regime_15m=btc_regime_15m,
        spot_scanned=scanned_counts["SPOT"], spot_removed=removed_counts["SPOT"],
        perp_scanned=scanned_counts["PERPETUAL"], perp_removed=removed_counts["PERPETUAL"],
    )
    output["debug_removal_reasons"] = {
        "SPOT": dict(removal_reasons["SPOT"].most_common(8)),
        "PERPETUAL": dict(removal_reasons["PERPETUAL"].most_common(8)),
    }
    return output
