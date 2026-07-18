"""
TOOBIT STAGE 1 — TELEGRAM BOT
==============================
Wraps toobit_stage1_scanner.run_scan() behind a manual /scan command.
No scheduling, no auto-run — the scan only happens when a user sends
/scan in the chat (per your choice).

-----------------------------------------------------------------
SETUP
-----------------------------------------------------------------
1. pip install python-telegram-bot==21.* pandas numpy --break-system-packages
2. Get a bot token from @BotFather on Telegram.
3. Wire the 4 stub methods in toobit_stage1_scanner.ToobitClient to the
   real TOOBIT API (this bot file does NOT touch that part — it only
   calls client.run_scan(), which was added to the scanner module).
4. Set the token below via environment variable TELEGRAM_BOT_TOKEN, then:
       python3 telegram_bot.py

-----------------------------------------------------------------
COMMANDS
-----------------------------------------------------------------
/start          — shows usage
/scan           — scans SPOT + PERPETUAL, sends both lists
/scan spot      — scans SPOT only
/scan perp      — scans PERPETUAL only
"""

import logging
import os
import traceback

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from toobit_stage1_scanner import run_scan
from toobit_real_client import RealToobitClient

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


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "سلام! برای اسکن بازار TOOBIT از دستورهای زیر استفاده کن:\n\n"
        "/scan — اسکن کامل SPOT و PERPETUAL\n"
        "/scan spot — فقط SPOT\n"
        "/scan perp — فقط PERPETUAL"
    )


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

    toobit_client = RealToobitClient()

    app = Application.builder().token(token).build()
    app.bot_data["toobit_client"] = toobit_client
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("scan", scan_command))

    logger.info("Bot starting (polling)...")
    app.run_polling()


if __name__ == "__main__":
    main()
