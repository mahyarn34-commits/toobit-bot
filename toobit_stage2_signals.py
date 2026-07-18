"""
TOOBIT STAGE 2 — SIGNAL GENERATION  (v2)
==========================================
Turns Stage 1 scan results into actual LONG/SHORT signals with entry,
stop-loss, take-profit, and suggested position size. Pure logic, no
network calls — reused unchanged by the live bot and the backtester.

-----------------------------------------------------------------
FILTER PIPELINE (in order — any failure drops the signal)
-----------------------------------------------------------------
1. Blocking risk flags (manipulation, pump, extreme volatility, ...)
2. Score >= SCORE_THRESHOLD_PCT of Stage-1's theoretical max
3. Clear trend on the base timeframe (ema_state bullish/bearish)
4. Momentum alignment: RSI and the EMA50 slope must agree with the
   trend direction — catches "EMA still bullish but momentum has
   already turned" situations that ema_state alone misses.
5. Higher-timeframe confirmation (15m ema_state must agree, or be
   neutral) — cuts down on 5m noise trades.
6. BTC regime hard filter — if BTC has a clean BULLISH/BEARISH
   regime, only signals aligned with it pass. Most alts are too
   correlated to BTC to safely trade against it.
7. Risk:reward on TP1 >= MIN_RR
8. Per-symbol-per-direction cooldown

After all of that, generate_signals_from_scan() also caps the number
of signals sent per scan to MAX_SIGNALS_PER_SCAN (best score first),
so a wide qualifying market doesn't turn into a spam blast.

-----------------------------------------------------------------
POSITION SIZING
-----------------------------------------------------------------
If ACCOUNT_BALANCE_USDT is set (env var or passed in directly), each
signal includes a suggested position size: risk_amount = balance *
RISK_PCT_PER_TRADE / 100, position_size = risk_amount / (price - SL).
This is sizing math only — it does NOT place any order.

-----------------------------------------------------------------
INVALIDATION (for a future monitoring loop)
-----------------------------------------------------------------
check_invalidation() compares a live signal against a fresh Stage-1
result for the same symbol and flags it if the trend that justified
the signal has broken down. Wire this into a periodic job (e.g. every
15 min) if you want early-exit alerts before SL/TP is hit.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal, Optional

Direction = Literal["LONG", "SHORT"]

# ============================== CONFIG ==============================
SCORE_THRESHOLD_PCT = 0.70        # fraction of max_theoretical_score required
BLOCKING_FLAGS = {
    "MANIPULATION_RISK", "PUMP_RISK", "EXTREME_VOLATILITY",
    "STALE_DATA", "EXTREME_NEGATIVE_FUNDING", "EXTREME_POSITIVE_FUNDING",
}
SL_ATR_MULT = 1.2
TP1_ATR_MULT = 2.0
TP2_ATR_MULT = 3.5
MIN_RR = 1.5                      # minimum reward:risk on TP1, else signal dropped
COOLDOWN_SECONDS = 4 * 3600       # per symbol+direction

REQUIRE_MOMENTUM_ALIGNMENT = True   # RSI + EMA50 slope must agree with direction
REQUIRE_HTF_CONFIRMATION = True     # 15m ema_state must agree (or be neutral)
REQUIRE_BTC_ALIGNMENT = True        # hard-block trades against a clean BTC regime

MAX_SIGNALS_PER_SCAN = 5            # best-score-first cap, per call to generate_signals_from_scan

ACCOUNT_BALANCE_USDT = float(os.environ.get("ACCOUNT_BALANCE_USDT", "0") or 0)
RISK_PCT_PER_TRADE = float(os.environ.get("RISK_PCT_PER_TRADE", "1.0") or 1.0)
MARGIN_PCT_PER_TRADE = float(os.environ.get("MARGIN_PCT_PER_TRADE", "15.0") or 15.0)  # % of balance used as margin per trade
MAX_LEVERAGE_CAP = float(os.environ.get("MAX_LEVERAGE_CAP", "200") or 200)            # hard safety ceiling
# ======================================================================


@dataclass
class Signal:
    symbol: str
    market_type: str
    direction: Direction
    score: float
    score_pct: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward_1: float
    risk_reward_2: float
    atr_value: float
    risk_flags: list
    reasons: list
    position_size_units: Optional[float] = None
    risk_amount_usdt: Optional[float] = None
    margin_usdt: Optional[float] = None
    suggested_leverage: Optional[float] = None
    leverage_capped: bool = False


class CooldownTracker:
    """Keeps last-signal timestamps per (symbol, direction) in memory.
    For persistence across bot restarts, swap the dict for a small
    JSON/SQLite-backed store."""

    def __init__(self, cooldown_seconds: int = COOLDOWN_SECONDS):
        self.cooldown_seconds = cooldown_seconds
        self._last_seen: dict[str, float] = {}

    def is_blocked(self, key: str, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        last = self._last_seen.get(key)
        return last is not None and (now - last) < self.cooldown_seconds

    def mark(self, key: str, now: Optional[float] = None) -> None:
        self._last_seen[key] = now if now is not None else time.time()


def max_theoretical_score(market_type: str) -> float:
    base = 5 + 3 + 7 + 5 + 2 + 3 + 1
    return base + (2 if market_type == "PERPETUAL" else 0)


def _momentum_aligned(direction: Direction, metrics: dict) -> bool:
    rsi = metrics.get("rsi_value")
    ema50_slope = metrics.get("ema50_slope")
    if rsi is None or ema50_slope is None:
        return True  # missing data -> don't block on it
    if direction == "LONG":
        return rsi > 50 and ema50_slope != "negative"
    return rsi < 50 and ema50_slope != "positive"


def _htf_aligned(direction: Direction, higher_tf_trend: Optional[str]) -> bool:
    if higher_tf_trend is None or higher_tf_trend == "neutral":
        return True  # no HTF data, or genuinely no trend there -> don't block
    return (direction == "LONG" and higher_tf_trend == "bullish") or \
           (direction == "SHORT" and higher_tf_trend == "bearish")


def _btc_aligned(direction: Direction, btc_regime: Optional[str]) -> bool:
    if btc_regime not in ("BULLISH", "BEARISH"):
        return True  # RANGE/NEUTRAL/MIXED/unknown -> don't block
    return (direction == "LONG" and btc_regime == "BULLISH") or \
           (direction == "SHORT" and btc_regime == "BEARISH")


def generate_signal(
    result: dict,
    price: float,
    cooldown: Optional[CooldownTracker] = None,
    now: Optional[float] = None,
    higher_tf_trend: Optional[str] = None,
    btc_regime: Optional[str] = None,
    account_balance_usdt: Optional[float] = None,
    margin_pct_per_trade: Optional[float] = None,
    max_leverage_cap: Optional[float] = None,
) -> tuple[Optional[Signal], Optional[str]]:
    """
    result: one element from Stage-1's spot_results / perpetual_results.
    price:  current price (Stage 1 doesn't carry raw price in `metrics`).
    higher_tf_trend: ema_state ("bullish"/"bearish"/"neutral") computed
                      on a higher timeframe (e.g. 15m) for the same symbol.
    btc_regime: Stage-1's btc_regime string from the same scan.
    account_balance_usdt: overrides the ACCOUNT_BALANCE_USDT env default.

    Returns (Signal, None) or (None, reason_str).
    """
    symbol = result["symbol"]
    market_type = result["market_type"]
    metrics = result["metrics"]
    risk_flags = set(result["risk_flags"])

    blocked_flags = risk_flags & BLOCKING_FLAGS
    if blocked_flags:
        return None, f"BLOCKED_FLAG:{','.join(sorted(blocked_flags))}"

    max_score = max_theoretical_score(market_type)
    score_pct = result["score"] / max_score if max_score else 0
    if score_pct < SCORE_THRESHOLD_PCT:
        return None, "SCORE_BELOW_THRESHOLD"

    ema_state = metrics.get("ema_state")
    if ema_state == "bullish":
        direction: Direction = "LONG"
    elif ema_state == "bearish":
        direction = "SHORT"
    else:
        return None, "NO_CLEAR_TREND"

    if REQUIRE_MOMENTUM_ALIGNMENT and not _momentum_aligned(direction, metrics):
        return None, "MOMENTUM_NOT_ALIGNED"

    if REQUIRE_HTF_CONFIRMATION and not _htf_aligned(direction, higher_tf_trend):
        return None, "HTF_NOT_ALIGNED"

    if REQUIRE_BTC_ALIGNMENT and not _btc_aligned(direction, btc_regime):
        return None, "AGAINST_BTC_REGIME"

    atr_pct = metrics.get("atr_percent")
    if not atr_pct or atr_pct <= 0:
        return None, "NO_ATR_DATA"
    atr_value = atr_pct / 100 * price

    if direction == "LONG":
        stop_loss = price - SL_ATR_MULT * atr_value
        tp1 = price + TP1_ATR_MULT * atr_value
        tp2 = price + TP2_ATR_MULT * atr_value
    else:
        stop_loss = price + SL_ATR_MULT * atr_value
        tp1 = price - TP1_ATR_MULT * atr_value
        tp2 = price - TP2_ATR_MULT * atr_value

    risk = abs(price - stop_loss)
    if risk == 0:
        return None, "ZERO_RISK_DISTANCE"
    rr1 = abs(tp1 - price) / risk
    rr2 = abs(tp2 - price) / risk
    if rr1 < MIN_RR:
        return None, "RR_TOO_LOW"

    cooldown_key = f"{symbol}:{direction}"
    if cooldown is not None and cooldown.is_blocked(cooldown_key, now):
        return None, "COOLDOWN_ACTIVE"

    balance = account_balance_usdt if account_balance_usdt is not None else ACCOUNT_BALANCE_USDT
    margin_pct = margin_pct_per_trade if margin_pct_per_trade is not None else MARGIN_PCT_PER_TRADE
    lev_cap = max_leverage_cap if max_leverage_cap is not None else MAX_LEVERAGE_CAP

    position_size, risk_amount, margin_used, leverage, leverage_capped = None, None, None, None, False
    if balance and balance > 0:
        # Desired position size to risk exactly RISK_PCT_PER_TRADE of the
        # account if SL is hit.
        risk_amount = balance * RISK_PCT_PER_TRADE / 100
        position_size = risk_amount / risk

        margin_used = balance * margin_pct / 100
        notional = position_size * price
        required_leverage = notional / margin_used if margin_used > 0 else None

        if required_leverage is not None and required_leverage > lev_cap:
            # Reduce position size to fit inside the leverage cap instead
            # of ever exceeding it — this makes the ACTUAL risk taken
            # smaller than RISK_PCT_PER_TRADE, never larger.
            leverage_capped = True
            leverage = lev_cap
            notional = margin_used * lev_cap
            position_size = notional / price
            risk_amount = position_size * risk
        else:
            leverage = required_leverage

    reasons = [f"ema_state={ema_state}", f"score={result['score']}/{max_score}"]
    if metrics.get("adx_value") is not None:
        reasons.append(f"adx={metrics['adx_value']}")
    if higher_tf_trend:
        reasons.append(f"15m={higher_tf_trend}")
    if btc_regime:
        reasons.append(f"btc={btc_regime}")

    signal = Signal(
        symbol=symbol, market_type=market_type, direction=direction,
        score=result["score"], score_pct=round(score_pct, 3),
        entry_price=price, stop_loss=stop_loss,
        take_profit_1=tp1, take_profit_2=tp2,
        risk_reward_1=round(rr1, 2), risk_reward_2=round(rr2, 2),
        atr_value=atr_value, risk_flags=sorted(risk_flags), reasons=reasons,
        position_size_units=round(position_size, 6) if position_size else None,
        risk_amount_usdt=round(risk_amount, 2) if risk_amount else None,
        margin_usdt=round(margin_used, 2) if margin_used else None,
        suggested_leverage=round(leverage, 1) if leverage else None,
        leverage_capped=leverage_capped,
    )
    if cooldown is not None:
        cooldown.mark(cooldown_key, now)
    return signal, None


def generate_signals_from_scan(
    scan_output: dict,
    price_lookup: dict,
    cooldown: Optional[CooldownTracker] = None,
    now: Optional[float] = None,
    higher_tf_trend_lookup: Optional[dict] = None,
    account_balance_usdt: Optional[float] = None,
    margin_pct_per_trade: Optional[float] = None,
    max_leverage_cap: Optional[float] = None,
    max_signals: int = MAX_SIGNALS_PER_SCAN,
) -> list[Signal]:
    """Runs generate_signal() over every Stage-1 result in a run_scan()
    output dict, then caps the result to the top `max_signals` by score.

    price_lookup: {symbol: current_price}
    higher_tf_trend_lookup: {symbol: "bullish"/"bearish"/"neutral"} — optional
    """
    btc_regime = scan_output.get("btc_regime")
    higher_tf_trend_lookup = higher_tf_trend_lookup or {}

    candidates: list[Signal] = []
    for bucket in ("spot_results", "perpetual_results"):
        for result in scan_output.get(bucket, []):
            price = price_lookup.get(result["symbol"])
            if price is None:
                continue
            sig, _reason = generate_signal(
                result, price, cooldown, now,
                higher_tf_trend=higher_tf_trend_lookup.get(result["symbol"]),
                btc_regime=btc_regime,
                account_balance_usdt=account_balance_usdt,
                margin_pct_per_trade=margin_pct_per_trade,
                max_leverage_cap=max_leverage_cap,
            )
            if sig is not None:
                candidates.append(sig)

    candidates.sort(key=lambda s: -s.score)
    return candidates[:max_signals]


def check_invalidation(original_direction: Direction, fresh_result: dict) -> Optional[str]:
    """Compare a fresh Stage-1 result for a symbol against the direction
    of an already-open signal. Returns a human-readable reason if the
    setup that justified the signal has broken down, else None.
    Intended for a periodic monitoring job, not the initial /signals call."""
    metrics = fresh_result["metrics"]
    ema_state = metrics.get("ema_state")
    if original_direction == "LONG" and ema_state == "bearish":
        return "روند به بیریش برگشته"
    if original_direction == "SHORT" and ema_state == "bullish":
        return "روند به بولیش برگشته"
    if ema_state == "neutral":
        return "روند از بین رفته (خنثی شده)"
    rsi = metrics.get("rsi_value")
    if rsi is not None:
        if original_direction == "LONG" and rsi < 45:
            return "مومنتوم لانگ ضعیف شده (RSI زیر ۴۵)"
        if original_direction == "SHORT" and rsi > 55:
            return "مومنتوم شورت ضعیف شده (RSI بالای ۵۵)"
    return None


def format_signal_fa(sig: Signal) -> str:
    """Telegram-friendly Persian message for a single signal (HTML parse mode)."""
    arrow = "🟢 لانگ" if sig.direction == "LONG" else "🔴 شورت"
    lines = [
        f"{arrow} — <b>{sig.symbol}</b> ({sig.market_type})",
        f"امتیاز: {sig.score} ({round(sig.score_pct * 100)}٪ سقف)",
        f"ورود: {sig.entry_price:.6g}",
        f"حد ضرر: {sig.stop_loss:.6g}",
        f"هدف ۱: {sig.take_profit_1:.6g}  (R:R {sig.risk_reward_1})",
        f"هدف ۲: {sig.take_profit_2:.6g}  (R:R {sig.risk_reward_2})",
    ]
    if sig.position_size_units is not None:
        lines.append(f"حجم پیشنهادی: {sig.position_size_units:g} واحد (ریسک {sig.risk_amount_usdt}$)")
    if sig.suggested_leverage is not None:
        cap_note = " (محدود به سقف مجاز)" if sig.leverage_capped else ""
        lines.append(f"لوریج پیشنهادی: {sig.suggested_leverage}x{cap_note} | مارجین: {sig.margin_usdt}$")
    if sig.risk_flags:
        lines.append("فلگ‌ها: " + "، ".join(sig.risk_flags))
    if sig.reasons:
        lines.append("دلایل: " + " | ".join(sig.reasons))
    return "\n".join(lines)
