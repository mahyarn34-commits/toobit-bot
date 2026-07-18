"""
REAL TOOBIT API CLIENT
=======================
Implements toobit_stage1_scanner.ToobitClient against the real, documented
TOOBIT REST API (https://api-docs.toobit.com/).

Endpoints used (all public, no API key needed for these):
- GET /api/v1/exchangeInfo          -> symbol/contract list + status
- GET /quote/v1/klines              -> candles
- GET /quote/v1/depth               -> order book snapshot
- GET /api/v1/futures/fundingRate   -> funding rate (perpetual only)
- GET /quote/v1/openInterest        -> current open interest (perpetual only)

-----------------------------------------------------------------
IMPORTANT CAVEAT — Open Interest history
-----------------------------------------------------------------
TOOBIT's public REST API only exposes the CURRENT open interest value,
not a historical time series. The Stage-1 spec needs a 20-period
lookback to compute OI change % and the OI-spike z-score.

Fix used here: this client keeps its own small in-memory rolling
history per symbol, appending one point every time /scan calls it.
This means:
- Right after your bot starts, OI-based scoring will be flat/neutral
  (not enough history yet).
- After ~1.5-2 hours of periodic /scan calls (or once you add
  scheduled polling), the OI trend becomes meaningful.
- Restarting the bot process resets this history (it is NOT persisted
  to disk). If you need it to survive restarts, write self._oi_history
  to a small JSON/SQLite file instead of keeping it only in memory.

-----------------------------------------------------------------
SYMBOL NORMALIZATION
-----------------------------------------------------------------
- Spot symbols from TOOBIT are already in the scanner's expected
  format (e.g. "ETHUSDT") -> used as-is.
- Perpetual symbols from TOOBIT look like "BTC-SWAP-USDT" (linear) or
  "BTC-SWAP" (inverse, EXCLUDED per Stage-1 spec: no inverse contracts).
  This client normalizes them to "<BASE>USDT_PERP" (e.g. "BTCUSDT_PERP")
  using the contract's "index" field, and keeps an internal map back
  to the real TOOBIT symbol string for actual API calls.
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import requests

from toobit_stage1_scanner import ToobitClient, OrderbookSnapshot, MarketType

BASE_URL = "https://api.toobit.com"
REQUEST_TIMEOUT_SEC = 10


class RealToobitClient(ToobitClient):
    def __init__(self):
        self.session = requests.Session()
        self._perp_symbol_map: dict[str, str] = {}   # normalized -> real TOOBIT symbol
        self._oi_history: dict[str, list[tuple[float, float]]] = {}  # real symbol -> [(ts, oi)]

    # ---------------------------------------------------------------
    # low-level helper
    # ---------------------------------------------------------------
    def _get(self, path: str, params: Optional[dict] = None) -> dict | list:
        resp = self.session.get(f"{BASE_URL}{path}", params=params, timeout=REQUEST_TIMEOUT_SEC)
        resp.raise_for_status()
        return resp.json()

    def _real_symbol(self, symbol: str) -> str:
        return self._perp_symbol_map.get(symbol, symbol)

    # ---------------------------------------------------------------
    # ToobitClient interface
    # ---------------------------------------------------------------
    def fetch_active_markets(self, market_type: MarketType) -> list[str]:
        data = self._get("/api/v1/exchangeInfo")

        if market_type == "SPOT":
            return [
                s["symbol"] for s in data.get("symbols", [])
                if s.get("status") == "TRADING"
            ]

        # PERPETUAL
        out = []
        self._perp_symbol_map = {}
        for c in data.get("contracts", []):
            if c.get("status") != "TRADING":
                continue
            if c.get("inverse"):
                continue  # exclude inverse contracts, per Stage-1 removal rules
            base_pair = c.get("index") or c["symbol"]     # e.g. "BTCUSDT"
            normalized = f"{base_pair}_PERP"
            self._perp_symbol_map[normalized] = c["symbol"]  # e.g. "BTC-SWAP-USDT"
            out.append(normalized)
        return out

    # milliseconds per candle for each supported interval
    _INTERVAL_MS = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
        "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
        "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000,
        "1d": 86_400_000, "1w": 604_800_000,
    }

    def fetch_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        real_symbol = self._real_symbol(symbol)
        limit = min(limit, 1000)  # TOOBIT max per request is 1000

        # IMPORTANT: TOOBIT's /quote/v1/klines returns ONLY the single
        # latest candle if startTime/endTime are omitted. We must supply
        # an explicit window to get a real history.
        interval_ms = self._INTERVAL_MS.get(timeframe, 300_000)
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - limit * interval_ms

        raw = self._get("/quote/v1/klines", {
            "symbol": real_symbol,
            "interval": timeframe,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        })
        rows = [{
            "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
            "close": float(k[4]), "volume": float(k[5]),
            "close_time": pd.Timestamp(int(float(k[6])), unit="ms", tz="UTC"),
        } for k in raw]
        df = pd.DataFrame(rows)
        # Do NOT assume the API returns oldest-first: sort explicitly.
        # (An unsorted/descending response made close_time.iloc[-1] look
        # like the OLDEST candle, which made every symbol look stale.)
        df = df.sort_values("close_time").reset_index(drop=True)
        # Drop any still-forming candle (close_time in the future) —
        # Stage-1 spec requires completed candles only.
        now_ts = pd.Timestamp.now(tz="UTC")
        df = df[df["close_time"] <= now_ts].reset_index(drop=True)
        return df

    def fetch_orderbook(self, symbol: str) -> OrderbookSnapshot:
        real_symbol = self._real_symbol(symbol)
        raw = self._get("/quote/v1/depth", {"symbol": real_symbol, "limit": 50})
        bids = raw.get("b", [])
        asks = raw.get("a", [])

        bid_price = float(bids[0][0]) if bids else 0.0
        ask_price = float(asks[0][0]) if asks else 0.0
        bid_depth = sum(float(q) for _, q in bids)
        ask_depth = sum(float(q) for _, q in asks)

        server_ms = raw.get("t")
        age_seconds = max(0.0, (time.time() * 1000 - server_ms) / 1000) if server_ms else 0.0

        return OrderbookSnapshot(
            bid=bid_price, ask=ask_price,
            bid_depth=bid_depth, ask_depth=ask_depth,
            levels_available=min(len(bids), len(asks)),
            age_seconds=age_seconds,
        )

    def fetch_funding_and_oi(self, symbol: str) -> tuple[float, pd.Series]:
        real_symbol = self._real_symbol(symbol)

        funding_raw = self._get("/api/v1/futures/fundingRate", {"symbol": real_symbol})
        funding_pct = 0.0
        if funding_raw:
            entry = funding_raw[0]
            rate_pct = float(entry["rate"]) * 100
            period = entry.get("period", "8H")
            # Normalize to an 8h-equivalent percentage if the settlement
            # period isn't already 8 hours.
            try:
                period_hours = float(period.rstrip("H"))
                funding_pct = rate_pct * (8.0 / period_hours) if period_hours else rate_pct
            except (ValueError, ZeroDivisionError):
                funding_pct = rate_pct

        oi_raw = self._get("/quote/v1/openInterest", {"symbol": real_symbol})
        oi_list = oi_raw.get("openInterestList", [])
        current_oi = float(oi_list[0]["size"]) if oi_list else 0.0

        now = time.time()
        history = self._oi_history.setdefault(real_symbol, [])
        history.append((now, current_oi))
        cutoff = now - 2 * 3600  # keep ~2 hours of history
        self._oi_history[real_symbol] = [(t, v) for t, v in history if t >= cutoff]

        oi_series = pd.Series([v for _, v in self._oi_history[real_symbol]])
        return funding_pct, oi_series
    def fetch_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        real_symbol = self._real_symbol(symbol)
        limit = min(limit, 1000)  # TOOBIT max per request is 1000

        # IMPORTANT: TOOBIT's /quote/v1/klines returns ONLY the single
        # latest candle if startTime/endTime are omitted. We must supply
        # an explicit window to get a real history.
        interval_ms = self._INTERVAL_MS.get(timeframe, 300_000)
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - limit * interval_ms

        raw = self._get("/quote/v1/klines", {
            "symbol": real_symbol,
            "interval": timeframe,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        })
        rows = [{
            "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
            "close": float(k[4]), "volume": float(k[5]),
            "close_time": pd.Timestamp(int(float(k[6])), unit="ms", tz="UTC"),
        } for k in raw]
        df = pd.DataFrame(rows)
        # Do NOT assume the API returns oldest-first: sort explicitly.
        # (An unsorted/descending response made close_time.iloc[-1] look
        # like the OLDEST candle, which made every symbol look stale.)
        df = df.sort_values("close_time").reset_index(drop=True)
        # Drop any still-forming candle (close_time in the future) —
        # Stage-1 spec requires completed candles only.
        now_ts = pd.Timestamp.now(tz="UTC")
        df = df[df["close_time"] <= now_ts].reset_index(drop=True)
        return df

    def fetch_orderbook(self, symbol: str) -> OrderbookSnapshot:
        real_symbol = self._real_symbol(symbol)
        raw = self._get("/quote/v1/depth", {"symbol": real_symbol, "limit": 50})
        bids = raw.get("b", [])
        asks = raw.get("a", [])

        bid_price = float(bids[0][0]) if bids else 0.0
        ask_price = float(asks[0][0]) if asks else 0.0
        bid_depth = sum(float(q) for _, q in bids)
        ask_depth = sum(float(q) for _, q in asks)

        server_ms = raw.get("t")
        age_seconds = max(0.0, (time.time() * 1000 - server_ms) / 1000) if server_ms else 0.0

        return OrderbookSnapshot(
            bid=bid_price, ask=ask_price,
            bid_depth=bid_depth, ask_depth=ask_depth,
            levels_available=min(len(bids), len(asks)),
            age_seconds=age_seconds,
        )

    def fetch_funding_and_oi(self, symbol: str) -> tuple[float, pd.Series]:
        real_symbol = self._real_symbol(symbol)

        funding_raw = self._get("/api/v1/futures/fundingRate", {"symbol": real_symbol})
        funding_pct = 0.0
        if funding_raw:
            entry = funding_raw[0]
            rate_pct = float(entry["rate"]) * 100
            period = entry.get("period", "8H")
            # Normalize to an 8h-equivalent percentage if the settlement
            # period isn't already 8 hours.
            try:
                period_hours = float(period.rstrip("H"))
                funding_pct = rate_pct * (8.0 / period_hours) if period_hours else rate_pct
            except (ValueError, ZeroDivisionError):
                funding_pct = rate_pct

        oi_raw = self._get("/quote/v1/openInterest", {"symbol": real_symbol})
        oi_list = oi_raw.get("openInterestList", [])
        current_oi = float(oi_list[0]["size"]) if oi_list else 0.0

        now = time.time()
        history = self._oi_history.setdefault(real_symbol, [])
        history.append((now, current_oi))
        cutoff = now - 2 * 3600  # keep ~2 hours of history
        self._oi_history[real_symbol] = [(t, v) for t, v in history if t >= cutoff]

        oi_series = pd.Series([v for _, v in self._oi_history[real_symbol]])
        return funding_pct, oi_series
    def fetch_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        real_symbol = self._real_symbol(symbol)
        limit = min(limit, 1000)  # TOOBIT max per request is 1000

        # IMPORTANT: TOOBIT's /quote/v1/klines returns ONLY the single
        # latest candle if startTime/endTime are omitted. We must supply
        # an explicit window to get a real history.
        interval_ms = self._INTERVAL_MS.get(timeframe, 300_000)
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - limit * interval_ms

        raw = self._get("/quote/v1/klines", {
            "symbol": real_symbol,
            "interval": timeframe,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        })
        rows = [{
            "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
            "close": float(k[4]), "volume": float(k[5]),
            "close_time": pd.Timestamp(k[6], unit="ms", tz="UTC"),
        } for k in raw]
        df = pd.DataFrame(rows)
        # Do NOT assume the API returns oldest-first: sort explicitly.
        # (An unsorted/descending response made close_time.iloc[-1] look
        # like the OLDEST candle, which made every symbol look stale.)
        df = df.sort_values("close_time").reset_index(drop=True)
        # Drop any still-forming candle (close_time in the future) —
        # Stage-1 spec requires completed candles only.
        now_ts = pd.Timestamp.now(tz="UTC")
        df = df[df["close_time"] <= now_ts].reset_index(drop=True)
        return df

    def fetch_orderbook(self, symbol: str) -> OrderbookSnapshot:
        real_symbol = self._real_symbol(symbol)
        raw = self._get("/quote/v1/depth", {"symbol": real_symbol, "limit": 50})
        bids = raw.get("bids", [])
        asks = raw.get("asks", [])

        bid_price = float(bids[0][0]) if bids else 0.0
        ask_price = float(asks[0][0]) if asks else 0.0
        bid_depth = sum(float(q) for _, q in bids)
        ask_depth = sum(float(q) for _, q in asks)

        # The REST depth snapshot has no per-response timestamp field
        # (only "lastUpdateId"), so treat it as fresh at fetch time.
        age_seconds = 0.0

        return OrderbookSnapshot(
            bid=bid_price, ask=ask_price,
            bid_depth=bid_depth, ask_depth=ask_depth,
            levels_available=min(len(bids), len(asks)),
            age_seconds=age_seconds,
        )

    def fetch_funding_and_oi(self, symbol: str) -> tuple[float, pd.Series]:
        real_symbol = self._real_symbol(symbol)

        funding_raw = self._get("/api/v1/futures/fundingRate", {"symbol": real_symbol})
        funding_pct = 0.0
        if funding_raw:
            entry = funding_raw[0]
            rate_pct = float(entry["rate"]) * 100
            period = entry.get("period", "8H")
            # Normalize to an 8h-equivalent percentage if the settlement
            # period isn't already 8 hours.
            try:
                period_hours = float(period.rstrip("H"))
                funding_pct = rate_pct * (8.0 / period_hours) if period_hours else rate_pct
            except (ValueError, ZeroDivisionError):
                funding_pct = rate_pct

        oi_raw = self._get("/quote/v1/openInterest", {"symbol": real_symbol})
        oi_list = oi_raw.get("openInterestList", [])
        current_oi = float(oi_list[0]["size"]) if oi_list else 0.0

        now = time.time()
        history = self._oi_history.setdefault(real_symbol, [])
        history.append((now, current_oi))
        cutoff = now - 2 * 3600  # keep ~2 hours of history
        self._oi_history[real_symbol] = [(t, v) for t, v in history if t >= cutoff]

        oi_series = pd.Series([v for _, v in self._oi_history[real_symbol]])
        return funding_pct, oi_series
