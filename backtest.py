"""
BACKTEST — Stage 1 scoring + Stage 2 signals, walk-forward, on real
historical TOOBIT candles.

-----------------------------------------------------------------
RUN THIS ON YOUR OWN MACHINE / RAILWAY (needs internet access)
-----------------------------------------------------------------
    pip install -r requirements.txt --break-system-packages
    python3 backtest.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --market SPOT --days 60

Options:
    --symbols   comma-separated symbols (default: top 10 SPOT by volume)
    --market    SPOT or PERPETUAL (default SPOT — simpler, no funding/OI needed)
    --days      how many days of 5m history to pull (default 60)
    --timeout-candles  max candles to hold a trade before force-closing (default 288 = 24h)
    --risk-pct  % of account risked per trade, for the equity curve (default 1.0)
    --out       output prefix for CSV/summary files (default "backtest")

-----------------------------------------------------------------
IMPORTANT LIMITATION — read this before trusting the numbers
-----------------------------------------------------------------
TOOBIT's public REST API does not expose historical order-book or
historical open-interest data. Because of that, this backtest can
NOT reproduce the orderbook_score, liquidity_score, or (for
PERPETUAL) open_interest_score / funding_adjustment components from
Stage 1 — those need live/recent data that simply doesn't exist
looking backward in time.

What IS reproduced exactly, using the same functions as the live
scanner (imported directly from toobit_stage1_scanner.py, not
reimplemented — so this can never silently drift out of sync):
    trend_quality, momentum_quality, volume_quality,
    volatility_quality, overextension_check, manipulation_check,
    btc_adjustment

The score threshold and max_theoretical_score used here are
adjusted down to match (see `backtest_max_score()`), so the 70%
threshold from toobit_stage2_signals.py is still comparing like
with like. Treat results as a read on trend/momentum/volatility
signal quality — not a claim about liquidity or orderbook-driven
entries, which the live bot still checks that this backtest can't.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from toobit_stage1_scanner import (
    compute_indicators, slope_direction, trend_quality, overextension_check,
    momentum_quality, volume_quality, volatility_quality, manipulation_check,
    btc_adjustment, classify_btc_regime, MIN_CANDLES,
)
from toobit_stage2_signals import (
    CooldownTracker, generate_signal, SL_ATR_MULT, TP1_ATR_MULT, TP2_ATR_MULT,
)
from toobit_real_client import RealToobitClient, BASE_URL

WARMUP_CANDLES = 210          # EMA200 + buffer needs to stabilize before we trust a score
STEP_CANDLES = 3              # evaluate a score every 3 closed 5m candles (15m), not every candle


def backtest_max_score(market_type: str) -> float:
    """max_theoretical_score, minus the components this backtest can't
    compute (orderbook, liquidity, and for perpetual: open interest)."""
    trend_max, momentum_max, volume_max, volat_max, btc_adj_max = 7, 5, 2, 3, 1
    return trend_max + momentum_max + volume_max + volat_max + btc_adj_max


# ----------------------------------------------------------------
# 1. DOWNLOAD HISTORICAL CANDLES (paginated, since TOOBIT caps
#    /quote/v1/klines at 1000 candles per request)
# ----------------------------------------------------------------

def download_candles(client: RealToobitClient, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    interval_ms = client._INTERVAL_MS.get(timeframe, 300_000)
    now_ms = int(time.time() * 1000)
    end_ms = (now_ms // interval_ms) * interval_ms - 1
    total_span_ms = days * 24 * 3600 * 1000
    start_ms = end_ms - total_span_ms

    all_rows = []
    cursor_end = end_ms
    real_symbol = client._real_symbol(symbol)
    while cursor_end > start_ms:
        cursor_start = max(start_ms, cursor_end - 1000 * interval_ms)
        raw = client._get("/quote/v1/klines", {
            "symbol": real_symbol, "interval": timeframe,
            "startTime": cursor_start, "endTime": cursor_end, "limit": 1000,
        })
        if not raw:
            break
        for k in raw:
            all_rows.append({
                "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                "close": float(k[4]), "volume": float(k[5]),
                "close_time": pd.Timestamp(int(float(k[0])) + interval_ms, unit="ms", tz="UTC"),
            })
        cursor_end = cursor_start - 1
        time.sleep(0.15)  # be polite to the API

    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "close_time"])

    df = pd.DataFrame(all_rows).drop_duplicates(subset="close_time")
    df = df.sort_values("close_time").reset_index(drop=True)
    return df


def top_symbols_by_volume(client: RealToobitClient, market_type: str, n: int) -> list[str]:
    symbols = client.fetch_active_markets(market_type)
    vols = []
    for s in symbols[:60]:  # cap probe count so this doesn't take forever
        try:
            c = client.fetch_candles(s, "5m", 288)
            vols.append((s, float(c["volume"].sum() * c["close"].iloc[-1])))
        except Exception:
            continue
    vols.sort(key=lambda x: -x[1])
    return [s for s, _ in vols[:n]]


# ----------------------------------------------------------------
# 2. WALK-FORWARD SCORING (reuses live scanner's pure functions —
#    only the data-source differs from the live bot)
# ----------------------------------------------------------------

def score_at_index(candles: pd.DataFrame, i: int, btc_regime: str) -> dict | None:
    """Same math as toobit_stage1_scanner.score_symbol(), restricted to
    the components computable from candles alone. `i` is the index of
    the last CLOSED candle to use (no look-ahead past i)."""
    window = candles.iloc[max(0, i - MIN_CANDLES + 1): i + 1]
    if len(window) < WARMUP_CANDLES:
        return None

    ind = compute_indicators(window)
    price = float(window["close"].iloc[-1])
    ema50, ema200 = ind["ema50"].iloc[-1], ind["ema200"].iloc[-1]
    adx = ind["adx14"].iloc[-1]
    atr14 = ind["atr14"].iloc[-1]
    rsi_series, hist_series = ind["rsi14"], ind["macd_hist"]
    vol_ma20 = ind["volume_ma20"].iloc[-1]
    current_vol = window["volume"].iloc[-1]

    if pd.isna(ema200) or pd.isna(atr14) or vol_ma20 == 0 or pd.isna(vol_ma20):
        return None

    ema50_slope = slope_direction(ind["ema50"], 20)
    ema200_slope = slope_direction(ind["ema200"], 20)

    risk_flags: list[str] = []

    trend_score, ema_state = trend_quality(price, ema50, ema200, adx, ema50_slope, ema200_slope)

    over_penalty, over_flags = overextension_check(price, ema50)
    risk_flags += over_flags

    momentum = momentum_quality(rsi_series, hist_series)
    if momentum is None:
        return None  # RSI out of tradeable range

    vol_score, vol_flags = volume_quality(current_vol, vol_ma20)
    risk_flags += vol_flags
    vol_score = max(vol_score, 0.0)

    volat_result = volatility_quality(atr14, price, ema50)
    if volat_result is None:
        return None  # ATR% too low
    volat_score, volat_flags = volat_result
    risk_flags += volat_flags
    volat_score = max(volat_score, 0.0)

    price_move_pct = (window["close"].iloc[-1] - window["open"].iloc[-1]) / window["open"].iloc[-1] * 100
    risk_flags += manipulation_check(current_vol, vol_ma20, price_move_pct)

    btc_adj = btc_adjustment(btc_regime, ema50, ema200, price, trend_score)

    risk_penalty_total = 0.0
    for flag, penalty in (
        ("OVEREXTENDED", 2), ("PUMP_RISK", 2), ("EXTREME_VOLATILITY", 2), ("MANIPULATION_RISK", 3),
    ):
        if flag in risk_flags:
            risk_penalty_total += penalty

    raw_score = trend_score + momentum + vol_score + volat_score
    final_score = raw_score + btc_adj - risk_penalty_total
    final_score = max(0.0, final_score)

    return {
        "symbol": None, "market_type": None, "score": final_score,
        "metrics": {
            "ema_state": ema_state, "adx_value": round(float(adx), 2),
            "atr_percent": round(float(atr14 / price * 100), 4),
        },
        "risk_flags": sorted(set(risk_flags)),
        "_price": price, "_atr14": float(atr14),
        "_close_time": window["close_time"].iloc[-1],
    }


def btc_regime_at_index(btc_candles: pd.DataFrame, close_time: pd.Timestamp) -> str:
    sub = btc_candles[btc_candles["close_time"] <= close_time]
    if len(sub) < WARMUP_CANDLES:
        return "NEUTRAL"
    ind = compute_indicators(sub.tail(MIN_CANDLES))
    price = sub["close"].iloc[-1]
    return classify_btc_regime(price, ind["ema50"].iloc[-1], ind["ema200"].iloc[-1], ind["adx14"].iloc[-1])


# ----------------------------------------------------------------
# 3. TRADE SIMULATION — given a signal at index i, walk forward
#    candle-by-candle until SL / TP1 / TP2 / timeout.
# ----------------------------------------------------------------

FEE_PCT = 0.05      # taker fee, % per side (adjust to your actual TOOBIT fee tier)
SLIPPAGE_PCT = 0.02  # % assumed slippage per fill


@dataclass
class Trade:
    symbol: str
    direction: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    outcome: str          # "TP1", "TP2", "SL", "TIMEOUT"
    pnl_pct: float         # net of fees+slippage, % of entry price
    rr_realized: float


def simulate_trade(symbol: str, candles: pd.DataFrame, entry_idx: int, direction: str,
                    entry: float, sl: float, tp1: float, tp2: float,
                    timeout_candles: int) -> Trade:
    risk = abs(entry - sl)
    end_idx = min(len(candles) - 1, entry_idx + timeout_candles)
    outcome, exit_price, exit_time = "TIMEOUT", float(candles["close"].iloc[end_idx]), candles["close_time"].iloc[end_idx]

    hit_tp1 = False
    for j in range(entry_idx + 1, end_idx + 1):
        hi, lo = float(candles["high"].iloc[j]), float(candles["low"].iloc[j])
        if direction == "LONG":
            sl_hit, tp1_hit, tp2_hit = lo <= sl, hi >= tp1, hi >= tp2
        else:
            sl_hit, tp1_hit, tp2_hit = hi >= sl, lo <= tp1, lo <= tp2

        if not hit_tp1:
            if sl_hit and tp1_hit:
                # both in the same candle — assume the worse outcome (SL) for safety
                outcome, exit_price, exit_time = "SL", sl, candles["close_time"].iloc[j]
                break
            if sl_hit:
                outcome, exit_price, exit_time = "SL", sl, candles["close_time"].iloc[j]
                break
            if tp1_hit:
                hit_tp1 = True
                # move on, now hunting for TP2 or a stop pulled to breakeven
                continue
        else:
            if tp2_hit:
                outcome, exit_price, exit_time = "TP2", tp2, candles["close_time"].iloc[j]
                break
            # simple breakeven-after-TP1 rule
            be_hit = (lo <= entry) if direction == "LONG" else (hi >= entry)
            if be_hit:
                outcome, exit_price, exit_time = "TP1", entry, candles["close_time"].iloc[j]
                break

    if hit_tp1 and outcome == "TIMEOUT":
        outcome, exit_price = "TP1", tp1

    raw_move_pct = (exit_price - entry) / entry * 100 if direction == "LONG" else (entry - exit_price) / entry * 100
    net_pct = raw_move_pct - (FEE_PCT * 2 + SLIPPAGE_PCT * 2)  # entry+exit, each side
    rr_realized = (raw_move_pct / 100 * entry) / risk if risk else 0.0

    return Trade(
        symbol=symbol, direction=direction,
        entry_time=str(candles["close_time"].iloc[entry_idx]), entry_price=entry,
        exit_time=str(exit_time), exit_price=exit_price,
        outcome=outcome, pnl_pct=round(net_pct, 4), rr_realized=round(rr_realized, 3),
    )


# ----------------------------------------------------------------
# 4. MAIN LOOP
# ----------------------------------------------------------------

def run_backtest(symbols: list[str], market_type: str, days: int, timeout_candles: int) -> pd.DataFrame:
    client = RealToobitClient()
    print(f"Downloading BTCUSDT reference candles ({days}d, 5m)...")
    btc_candles = download_candles(client, "BTCUSDT", "5m", days)

    all_trades: list[Trade] = []
    for symbol in symbols:
        print(f"Downloading {symbol} ({days}d, 5m)...")
        candles = download_candles(client, symbol, "5m", days)
        if len(candles) < WARMUP_CANDLES + STEP_CANDLES:
            print(f"  skip {symbol}: not enough history returned ({len(candles)} candles)")
            continue

        cooldown = CooldownTracker()
        i = WARMUP_CANDLES
        max_score = backtest_max_score(market_type)
        signal_count = 0

        while i < len(candles) - 1:  # leave room for at least 1 forward candle
            result = score_at_index(candles, i, btc_regime_at_index(btc_candles, candles["close_time"].iloc[i]))
            if result is not None:
                result["symbol"], result["market_type"] = symbol, market_type
                price = result["_price"]
                now_epoch = result["_close_time"].timestamp()

                sig, reason = generate_signal(
                    {k: v for k, v in result.items() if not k.startswith("_")},
                    price, cooldown, now_epoch,
                )
                # generate_signal() checks score against the FULL max_theoretical_score;
                # rescale using our reduced max instead for an apples-to-apples threshold.
                if sig is None and reason == "SCORE_BELOW_THRESHOLD":
                    if result["score"] / max_score >= 0.70:
                        # retry bypassing the internal (wrong-denominator) score check
                        from toobit_stage2_signals import (
                            BLOCKING_FLAGS, SL_ATR_MULT, TP1_ATR_MULT, TP2_ATR_MULT, MIN_RR, Signal,
                        )
                        flags = set(result["risk_flags"])
                        if not (flags & BLOCKING_FLAGS):
                            atr_val = result["_atr14"]
                            direction = "LONG" if result["metrics"]["ema_state"] == "bullish" else (
                                "SHORT" if result["metrics"]["ema_state"] == "bearish" else None)
                            if direction:
                                sl = price - SL_ATR_MULT * atr_val if direction == "LONG" else price + SL_ATR_MULT * atr_val
                                tp1 = price + TP1_ATR_MULT * atr_val if direction == "LONG" else price - TP1_ATR_MULT * atr_val
                                tp2 = price + TP2_ATR_MULT * atr_val if direction == "LONG" else price - TP2_ATR_MULT * atr_val
                                risk = abs(price - sl)
                                rr1 = abs(tp1 - price) / risk if risk else 0
                                key = f"{symbol}:{direction}"
                                if rr1 >= MIN_RR and not cooldown.is_blocked(key, now_epoch):
                                    sig = Signal(
                                        symbol=symbol, market_type=market_type, direction=direction,
                                        score=result["score"], score_pct=round(result["score"] / max_score, 3),
                                        entry_price=price, stop_loss=sl, take_profit_1=tp1, take_profit_2=tp2,
                                        risk_reward_1=round(rr1, 2), risk_reward_2=round(abs(tp2 - price) / risk, 2),
                                        atr_value=atr_val, risk_flags=sorted(flags), reasons=[],
                                    )
                                    cooldown.mark(key, now_epoch)

                if sig is not None:
                    signal_count += 1
                    trade = simulate_trade(
                        symbol, candles, i, sig.direction,
                        sig.entry_price, sig.stop_loss, sig.take_profit_1, sig.take_profit_2,
                        timeout_candles,
                    )
                    all_trades.append(trade)

            i += STEP_CANDLES

        print(f"  {symbol}: {signal_count} signals generated")

    if not all_trades:
        return pd.DataFrame()
    return pd.DataFrame([asdict(t) for t in all_trades])


def summarize(trades: pd.DataFrame, risk_pct: float) -> None:
    if trades.empty:
        print("\nNo trades were generated — try more symbols, more days, or check your data.")
        return

    n = len(trades)
    wins = trades[trades["pnl_pct"] > 0]
    losses = trades[trades["pnl_pct"] <= 0]
    win_rate = len(wins) / n * 100
    gross_win = wins["pnl_pct"].sum()
    gross_loss = abs(losses["pnl_pct"].sum())
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    avg_rr = trades["rr_realized"].mean()

    # equity curve assuming fixed risk_pct of account risked per trade
    equity = [100.0]
    for _, row in trades.iterrows():
        r_multiple = row["rr_realized"]
        equity.append(equity[-1] * (1 + (risk_pct / 100) * r_multiple))
    equity = pd.Series(equity)
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100
    max_dd = drawdown.min()

    print("\n" + "=" * 50)
    print("BACKTEST SUMMARY")
    print("=" * 50)
    print(f"Total trades:      {n}")
    print(f"Win rate:          {win_rate:.1f}%")
    print(f"Profit factor:     {profit_factor:.2f}")
    print(f"Avg R multiple:    {avg_rr:.2f}")
    print(f"Max drawdown:      {max_dd:.1f}%  (at {risk_pct}% risk/trade)")
    print(f"Final equity:      {equity.iloc[-1]:.1f}  (started at 100)")
    print("\nBy outcome:")
    print(trades["outcome"].value_counts().to_string())
    print("\nBy symbol:")
    print(trades.groupby("symbol")["pnl_pct"].agg(["count", "mean", "sum"]).to_string())


def main():
    parser = argparse.ArgumentParser(description="Backtest TOOBIT Stage 1 + Stage 2 signals")
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--market", type=str, default="SPOT", choices=["SPOT", "PERPETUAL"])
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--timeout-candles", type=int, default=288)
    parser.add_argument("--risk-pct", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="backtest")
    args = parser.parse_args()

    client = RealToobitClient()
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        print(f"No --symbols given, auto-selecting top 10 {args.market} by volume...")
        symbols = top_symbols_by_volume(client, args.market, 10)
        print("Selected:", symbols)

    trades = run_backtest(symbols, args.market, args.days, args.timeout_candles)

    if not trades.empty:
        out_csv = f"{args.out}_trades.csv"
        trades.to_csv(out_csv, index=False)
        print(f"\nTrade log saved to {out_csv}")

    summarize(trades, args.risk_pct)


if __name__ == "__main__":
    main()
