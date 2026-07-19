"""
TOOBIT STAGE 1 — TELEGRAM BOT
==============================
Wraps toobit_stage1_scanner.run_scan() behind a manual /scan command.
No scheduling, no auto-run — the scan only happens when a user sends
/scan in the chat (per your choice).

-----------------------------------------------------------------
SETUP
-----------------------------------------------------------------
1. pip install "python-telegram-bot[job-queue]"==21.6 pandas numpy --break-system-packages
   (the [job-queue] extra is REQUIRED for automatic monitoring — without
   it /signals and /check still work, but the every-15-min auto-check is
   silently disabled)
2. Get a bot token from @BotFather on Telegram.
3. Wire the 4 stub methods in toobit_stage1_scanner.ToobitClient to the
   real TOOBIT API (this bot file does NOT touch that part — it only
   calls client.run_scan(), which was added to the scanner module).
4. Set the token below via environment variable TELEGRAM_BOT_TOKEN, then:
       python3 telegram_bot.py
5. Optional, for position sizing in /signals: set ACCOUNT_BALANCE_USDT
   (your trading balance). If it isn't set, dollar amounts are skipped
   but leverage and risk % still show (they don't need a balance).
6. Optional, for leverage/risk reporting in /signals: MARGIN_PCT_PER_TRADE
   (% of balance you commit as margin per trade — you choose this,
   default 15), LEVERAGE_MIN (leverage floor for a barely-qualifying
   signal, default 20), and MAX_LEVERAGE_CAP (the ceiling leverage
   climbs toward as signal quality rises — continuously, no fixed
   mid-point, default 200; also a hard safety limit it never exceeds).
   You do NOT set a risk %; instead each signal tells you what % of
   your account is actually at risk given margin_pct, the picked
   leverage, and the ATR-based stop distance (which is wider now —
   2.5x ATR for SL — so normal noise is less likely to clip it early).
   Note: leverage is automatically clamped (via MAX_MARGIN_LOSS_AT_SL,
   default 0.85) so hitting SL never risks near-total margin loss —
   with a wide SL, that clamp will keep leverage around ~18-25x even
   if MAX_LEVERAGE_CAP is set much higher, because going higher would
   mean the exchange liquidates the position before price ever reaches
   your stop-loss.
7. Optional: CHECK_INTERVAL_SECONDS controls the auto-monitor frequency
   (default 900 = 15 min). SIGNAL_DB_PATH controls where signal history
   is stored (default ./signals.db) — on Railway this resets on every
   redeploy unless you mount a Volume at that path; a plain restart
   (not a new deploy) keeps it.

-----------------------------------------------------------------
COMMANDS
-----------------------------------------------------------------
/start          — shows usage
/scan           — scans SPOT + PERPETUAL, sends both lists
/scan spot      — scans SPOT only
/scan perp      — scans PERPETUAL only
/signals        — scans + generates Stage-2 trade signals (top 5, best score first)
/signals spot   — signals from SPOT only
/signals perp   — signals from PERPETUAL only
/check          — re-checks every signal /signals has sent you and tells
                   you whether the trend that justified it is still valid,
                   whether TP1/TP2/SL hit, and reminds you to move SL to
                   breakeven after TP1
/stats          — real win-rate stats from closed signals (this deployment)

Signals are also checked automatically every CHECK_INTERVAL_SECONDS
(default 900 = 15 min) — you only get pinged when something actually
changes (TP/SL hit, breakeven armed, trend invalidated), not on every
check. Requires the job-queue extra: see SETUP step 1 below.
"""

import logging
import os
import time
import traceback
from typing import Optional

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from toobit_stage1_scanner import (
    run_scan, compute_indicators, slope_direction, trend_quality,
    score_symbol, _build_symbol_data,
)
from toobit_real_client import RealToobitClient
from toobit_stage2_signals import (
    CooldownTracker, generate_signals_from_scan, format_signal_fa, check_invalidation,
    ACCOUNT_BALANCE_USDT, MARGIN_PCT_PER_TRADE, LEVERAGE_MIN, MAX_LEVERAGE_CAP,
)
import signal_store as store

CHECK_INTERVAL_SECONDS = int(os.environ.get("CHECK_INTERVAL_SECONDS", "900") or 900)  # auto-monitor job interval, default 15 min

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE_LEN = 3500  # stay safely under Telegram's 4096 char hard limit

# Plain-language translations for risk flags (kept short — these show
# inline per symbol, not as a lecture).
FLAG_LABELS = {
    "ORDERBOOK_IMBALANCE": "نامتعادلی دفتر سفارش",
    "OVEREXTENDED": "دور از میانگین (اورایکستند)",
    "VOLUME_SPIKE": "جهش حجم",
    "PUMP_RISK": "ریسک پامپ",
    "EXTREME_VOLATILITY": "نوسان شدید",
    "MANIPULATION_RISK": "ریسک دستکاری قیمت",
    "OI_SPIKE": "جهش open interest",
    "EXTREME_NEGATIVE_FUNDING": "فاندینگ منفی شدید",
    "EXTREME_POSITIVE_FUNDING": "فاندینگ مثبت شدید",
}

REGIME_LABELS = {
    "BULLISH": "صعودی",
    "BEARISH": "نزولی",
    "RANGE": "رنج",
    "NEUTRAL": "خنثی",
    "MIXED": "مختلط",
}

QUALITY_LABELS = {"HIGH": "بالا", "MEDIUM": "متوسط", "LOW": "پایین"}


def _format_symbol_line(rank: int, r: dict) -> str:
    m = r["metrics"]
    flags = r["risk_flags"]
    flags_txt = "، ".join(FLAG_LABELS.get(f, f) for f in flags) if flags else "بدون فلگ"

    line = (
        f"{rank}. <b>{r['symbol']}</b> — امتیاز: {r['score']}\n"
        f"   RSI: {m.get('rsi_value')} | ADX: {m.get('adx_value')} | "
        f"ATR%: {m.get('atr_percent')} | نسبت حجم: {m.get('volume_ratio')}\n"
        f"   ساختار: {m.get('ema_state')} | فلگ‌ها: {flags_txt}"
    )
    if "funding_adjustment" in m:
        line += f"\n   تعدیل فاندینگ: {m['funding_adjustment']}"
    return line


def _format_debug_reasons(scan_result: dict, market_key: str) -> str:
    reasons = scan_result.get("debug_removal_reasons", {}).get(market_key, {})
    if not reasons:
        return ""
    lines = "\n".join(f"  - {k}: {v}" for k, v in reasons.items())
    return f"\n🔍 دلایل حذف (بیشترین موارد):\n{lines}"


def _build_messages(scan_result: dict, market_key: str, market_title: str, quality_key: str) -> list[str]:
    results = scan_result[market_key]
    if not results:
        debug_key = "SPOT" if market_key == "spot_results" else "PERPETUAL"
        return [f"⚠️ هیچ نمادی در {market_title} واجد شرایط نشد.{_format_debug_reasons(scan_result, debug_key)}"]

    header = (
        f"📊 <b>{market_title}</b> — کیفیت بازار: "
        f"{QUALITY_LABELS.get(scan_result[quality_key], scan_result[quality_key])}\n"
        f"تعداد واجد شرایط: {len(results)}\n\n"
    )

    lines = [_format_symbol_line(r["rank"], r) for r in results]

    messages, current = [], header
    for line in lines:
        if len(current) + len(line) + 2 > TELEGRAM_MAX_MESSAGE_LEN:
            messages.append(current)
            current = ""
        current += line + "\n\n"
    if current.strip():
        messages.append(current)
    return messages


DIR_LABELS = {"LONG": "لانگ", "SHORT": "شورت"}


def _evaluate_signal_row(client: RealToobitClient, row) -> tuple[Optional[str], str]:
    """Checks one open/armed signal against live price + trend. Updates
    the DB as a side effect when something changes. Returns
    (event_type_or_None, human_readable_line). event_type is None when
    nothing changed (still open, still valid) — used by the periodic
    job to decide whether a message is worth sending."""
    symbol, market_type, direction = row["symbol"], row["market_type"], row["direction"]
    entry, sl, tp1, tp2 = row["entry_price"], row["stop_loss"], row["take_profit_1"], row["take_profit_2"]
    dir_label = DIR_LABELS.get(direction, direction)

    try:
        candles = client.fetch_candles(symbol, "5m", 1)
        if not len(candles):
            return None, f"⚪ <b>{symbol}</b> — قیمت در دسترس نبود"
        high, low = float(candles["high"].iloc[-1]), float(candles["low"].iloc[-1])
    except Exception:
        logger.error("price fetch failed for %s:\n%s", symbol, traceback.format_exc())
        return None, f"⚪ <b>{symbol}</b> — خطا در دریافت قیمت"

    if row["status"] == "OPEN":
        if direction == "LONG":
            sl_hit, tp1_hit, tp2_hit = low <= sl, high >= tp1, high >= tp2
        else:
            sl_hit, tp1_hit, tp2_hit = high >= sl, low <= tp1, low <= tp2

        if sl_hit and not tp1_hit:
            store.close_signal(row["id"], "SL", sl, "SL hit")
            return "SL", f"🔴 <b>{symbol}</b> ({dir_label}) — حد ضرر خورد، بسته شد"
        if tp2_hit:
            store.close_signal(row["id"], "TP2", tp2, "TP2 hit")
            return "TP2", f"🟢🟢 <b>{symbol}</b> ({dir_label}) — هدف دوم خورد! کامل بسته شد"
        if tp1_hit:
            store.mark_breakeven(row["id"])
            return "TP1_BE", (
                f"🟡 <b>{symbol}</b> ({dir_label}) — هدف اول خورد!\n"
                f"   پیشنهاد: حد ضرر رو ببر روی نقطه‌ی ورود ({entry:.6g}) تا این معامله دیگه هیچ‌وقت ضررده نشه."
            )

    elif row["status"] == "TP1_BE":
        if direction == "LONG":
            be_hit, tp2_hit = low <= entry, high >= tp2
        else:
            be_hit, tp2_hit = high >= entry, low <= tp2

        if tp2_hit:
            store.close_signal(row["id"], "TP2", tp2, "TP2 hit after BE armed")
            return "TP2", f"🟢🟢 <b>{symbol}</b> ({dir_label}) — هدف دوم خورد! کامل بسته شد"
        if be_hit:
            store.close_signal(row["id"], "BE_EXIT", entry, "returned to breakeven after TP1")
            return "BE_EXIT", f"⚪ <b>{symbol}</b> ({dir_label}) — به نقطه‌ی سربه‌سر برگشت، بسته شد (سود جزئی از هدف اول حفظ شد)"

    # No SL/TP event this round — check whether the trend that
    # justified the signal is still there.
    try:
        data, reason = _build_symbol_data(client, symbol, market_type)
        fresh_result = score_symbol(data, btc_regime="NEUTRAL")[0] if data else None
    except Exception:
        fresh_result = None

    if fresh_result is not None:
        fresh_dict = {"metrics": fresh_result.metrics}
        invalidation_reason = check_invalidation(direction, fresh_dict)
        if invalidation_reason:
            store.close_signal(row["id"], "INVALIDATED", (high + low) / 2, invalidation_reason)
            note = " (سود جزئی از هدف اول قبلاً حفظ شده)" if row["status"] == "TP1_BE" else ""
            return "INVALIDATED", f"🟠 <b>{symbol}</b> ({dir_label}) — روند دیگه معتبر نیست: {invalidation_reason}{note}"

    be_note = " [SL روی سربه‌سره]" if row["status"] == "TP1_BE" else ""
    return None, f"🟢 <b>{symbol}</b> ({dir_label}) — هنوز باز و معتبره{be_note}"



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "سلام! برای اسکن بازار TOOBIT از دستورهای زیر استفاده کن:\n\n"
        "/scan — اسکن کامل SPOT و PERPETUAL\n"
        "/scan spot — فقط SPOT\n"
        "/scan perp — فقط PERPETUAL\n"
        "/signals — اسکن + تولید سیگنال (Stage 2)\n"
        "/check — بررسی دستی سیگنال‌های باز (وضعیت، TP/SL، breakeven)\n"
        "/stats — آمار واقعی سیگنال‌های بسته‌شده (win rate)\n\n"
        f"سیگنال‌های باز هر {CHECK_INTERVAL_SECONDS // 60} دقیقه خودکار هم چک می‌شن و اگه "
        "TP/SL خورد یا روند برگشت، خودم بهت خبر می‌دم."
    )


async def signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    scope = context.args[0].lower() if context.args else "all"
    if scope not in ("all", "spot", "perp"):
        await update.message.reply_text("فقط spot، perp یا بدون آرگومان (برای هر دو) مجازه.")
        return

    await update.message.reply_text("⏳ در حال اسکن و بررسی سیگنال‌ها...")

    client: RealToobitClient = context.bot_data["toobit_client"]
    cooldown: CooldownTracker = context.bot_data["cooldown"]
    market_types = {"all": ("SPOT", "PERPETUAL"), "spot": ("SPOT",), "perp": ("PERPETUAL",)}[scope]

    try:
        result = run_scan(client, market_types=market_types)
    except Exception:
        logger.error("Scan for signals failed:\n%s", traceback.format_exc())
        await update.message.reply_text("❌ اسکن با خطا مواجه شد. جزئیات در لاگ سرور ثبت شد.")
        return

    # Stage 1 doesn't carry raw price in its output, only ATR% — fetch
    # the latest close for each qualified symbol (top 20 per market max,
    # so this stays cheap).
    price_lookup: dict[str, float] = {}
    higher_tf_trend_lookup: dict[str, str] = {}
    for bucket in ("spot_results", "perpetual_results"):
        for r in result.get(bucket, []):
            symbol = r["symbol"]
            try:
                candles_5m = client.fetch_candles(symbol, "5m", 1)
                if len(candles_5m):
                    price_lookup[symbol] = float(candles_5m["close"].iloc[-1])
            except Exception:
                continue
            try:
                candles_15m = client.fetch_candles(symbol, "15m", 210)
                if len(candles_15m) >= 210:
                    ind15 = compute_indicators(candles_15m)
                    price15 = float(candles_15m["close"].iloc[-1])
                    ema50_15, ema200_15 = ind15["ema50"].iloc[-1], ind15["ema200"].iloc[-1]
                    adx15 = ind15["adx14"].iloc[-1]
                    slope50 = slope_direction(ind15["ema50"], 20)
                    slope200 = slope_direction(ind15["ema200"], 20)
                    _, ema_state_15m = trend_quality(price15, ema50_15, ema200_15, adx15, slope50, slope200)
                    higher_tf_trend_lookup[symbol] = ema_state_15m
            except Exception:
                continue

    signals = generate_signals_from_scan(
        result, price_lookup, cooldown,
        higher_tf_trend_lookup=higher_tf_trend_lookup,
        account_balance_usdt=ACCOUNT_BALANCE_USDT or None,
        leverage_min=LEVERAGE_MIN,
        margin_pct_per_trade=MARGIN_PCT_PER_TRADE,
        max_leverage_cap=MAX_LEVERAGE_CAP,
    )

    if not signals:
        await update.message.reply_text("⚠️ در حال حاضر سیگنالی که واجد شرایط باشه پیدا نشد.")
        return

    for sig in signals:
        store.record_signal(update.effective_chat.id, sig)
        await update.message.reply_text(format_signal_fa(sig), parse_mode=ParseMode.HTML)


async def check_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rows = store.get_open_signals(update.effective_chat.id)
    if not rows:
        await update.message.reply_text(
            "هنوز سیگنالی ثبت نشده. اول یه بار /signals رو بزن، بعد می‌تونی وضعیتش رو با /check چک کنی."
        )
        return

    await update.message.reply_text(f"⏳ در حال بررسی {len(rows)} سیگنال باز...")

    client: RealToobitClient = context.bot_data["toobit_client"]
    lines = []
    for row in rows:
        try:
            _event, line = _evaluate_signal_row(client, row)
        except Exception:
            logger.error("check_command failed for %s:\n%s", row["symbol"], traceback.format_exc())
            line = f"⚪ <b>{row['symbol']}</b> — خطا در بررسی"
        lines.append(line)

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    s = store.get_stats(update.effective_chat.id)
    if not s["closed_total"]:
        await update.message.reply_text("هنوز هیچ سیگنالی بسته نشده که آماری ازش داشته باشیم.")
        return

    status_labels = {
        "TP2": "هدف دوم کامل", "SL": "حد ضرر خورد",
        "BE_EXIT": "سربه‌سر (بعد از هدف اول)", "INVALIDATED": "نامعتبر شد (روند برگشت)",
    }
    lines = [
        f"📈 آمار واقعی سیگنال‌ها (این دیپلوی)",
        f"تعداد بسته‌شده: {s['closed_total']}",
        f"نرخ برد: {s['win_rate_pct']}٪" if s["win_rate_pct"] is not None else "نرخ برد: -",
        "",
    ]
    for status, count in s["by_status"].items():
        lines.append(f"  {status_labels.get(status, status)}: {count}")
    await update.message.reply_text("\n".join(lines))


async def _auto_check_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Runs every CHECK_INTERVAL_SECONDS. Only messages a chat when
    something actually changed (TP/SL/breakeven/invalidation) — stays
    silent for signals that are simply still open and still valid."""
    client: RealToobitClient = context.bot_data["toobit_client"]
    for chat_id in store.open_chat_ids():
        rows = store.get_open_signals(chat_id)
        events = []
        for row in rows:
            try:
                event, line = _evaluate_signal_row(client, row)
            except Exception:
                logger.error("auto-check failed for %s:\n%s", row["symbol"], traceback.format_exc())
                continue
            if event is not None:
                events.append(line)
        if events:
            try:
                await context.bot.send_message(chat_id, "\n".join(events), parse_mode=ParseMode.HTML)
            except Exception:
                logger.error("failed to send auto-check message to %s:\n%s", chat_id, traceback.format_exc())


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    scope = context.args[0].lower() if context.args else "all"
    if scope not in ("all", "spot", "perp"):
        await update.message.reply_text("فقط spot، perp یا بدون آرگومان (برای هر دو) مجازه.")
        return

    await update.message.reply_text("⏳ در حال اسکن بازار TOOBIT... (ممکنه کمی طول بکشه)")

    client: RealToobitClient = context.bot_data["toobit_client"]

    market_types = {"all": ("SPOT", "PERPETUAL"), "spot": ("SPOT",), "perp": ("PERPETUAL",)}[scope]

    try:
        result = run_scan(client, market_types=market_types)
    except NotImplementedError as e:
        await update.message.reply_text(
            "⚠️ کلاینت TOOBIT هنوز به API واقعی وصل نشده.\n"
            f"جزئیات: {e}\n\n"
            "متدهای fetch_active_markets / fetch_candles / fetch_orderbook / "
            "fetch_funding_and_oi رو در ToobitClient تکمیل کن."
        )
        return
    except Exception:
        logger.error("Scan failed:\n%s", traceback.format_exc())
        await update.message.reply_text(
            "❌ اسکن با خطا مواجه شد. جزئیات فنی در لاگ سرور ثبت شد."
        )
        return

    btc_line = (
        f"🟠 رژیم بیت‌کوین: {REGIME_LABELS.get(result['btc_regime'], result['btc_regime'])} "
        f"(5m: {REGIME_LABELS.get(result['btc_regime_5m'], result['btc_regime_5m'])}, "
        f"15m: {REGIME_LABELS.get(result['btc_regime_15m'], result['btc_regime_15m'])})"
    )
    if result.get("market_quality_flags"):
        btc_line += "\n⚠️ کیفیت بازار پایین است (LOW_MARKET_QUALITY)."
    probe = result.get("debug_btcusdt_probe", {})
    if probe:
        btc_line += "\n\n🧪 تست BTCUSDT:\n" + "\n".join(f"  {k}: {v}" for k, v in probe.items())
    await update.message.reply_text(btc_line)

    if scope in ("all", "spot"):
        for msg in _build_messages(result, "spot_results", "SPOT", "spot_market_quality"):
            await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    if scope in ("all", "perp"):
        for msg in _build_messages(result, "perpetual_results", "PERPETUAL", "perpetual_market_quality"):
            await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Set the TELEGRAM_BOT_TOKEN environment variable first.")

    store.init_db()
    toobit_client = RealToobitClient()

    app = Application.builder().token(token).build()
    app.bot_data["toobit_client"] = toobit_client
    app.bot_data["cooldown"] = CooldownTracker()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CommandHandler("signals", signals_command))
    app.add_handler(CommandHandler("check", check_command))
    app.add_handler(CommandHandler("stats", stats_command))

    if app.job_queue is not None:
        app.job_queue.run_repeating(_auto_check_job, interval=CHECK_INTERVAL_SECONDS, first=CHECK_INTERVAL_SECONDS)
    else:
        logger.warning(
            "JobQueue not available — auto-monitoring is OFF. "
            "Install with: pip install \"python-telegram-bot[job-queue]\"==21.6 --break-system-packages"
        )

    logger.info("Bot starting (polling)...")
    app.run_polling()


if __name__ == "__main__":
    main()
