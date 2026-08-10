"""
Read-only client for the external `ticker` PostgreSQL data layer.

Provides CN A-share daily bars (qfq-adjusted), sparse as-of adjust factors, and
financial-fundamental snapshots. Every function is best-effort: on any error or
missing DSN it returns empty data so callers fall through to the online chain.

Connection is short-lived per call (fallback path only) — we do NOT reuse the
main application DB pool, and we never write to this data layer.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

_DSN_ENV = "TICKER_PG_READONLY_DSN"
_CONNECT_TIMEOUT = 6


def _dsn() -> str:
    return (os.getenv(_DSN_ENV, "") or os.getenv("PG_DSN", "")).strip()


def _connect():
    import psycopg2  # lazy: only needed when the data layer is configured

    return psycopg2.connect(_dsn(), connect_timeout=_CONNECT_TIMEOUT)


def _trade_ts(d: date) -> int:
    """Match Tencent daily-bar timestamps: local-timezone midnight of trade_date."""
    return int(datetime(d.year, d.month, d.day).timestamp())


def read_daily_bars(
    market: str,
    code: str,
    limit: int,
    before_time: Optional[int] = None,
    after_time: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Read unadjusted daily OHLCV from v_daily_bar (normalized K-line shape).
    before_time / after_time are Unix-seconds bounds on trade_date.
    """
    if not _dsn():
        return []
    end_date = date.fromtimestamp(before_time) if before_time else date.today()
    start_date = date.fromtimestamp(after_time) if after_time else date(2000, 1, 1)
    limit = max(1, int(limit or 300))
    try:
        conn = _connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT trade_date, open, high, low, close, volume
                FROM v_daily_bar
                WHERE market = %s AND code = %s
                  AND trade_date >= %s AND trade_date <= %s
                ORDER BY trade_date DESC
                LIMIT %s
                """,
                (market, code, start_date, end_date, limit),
            )
            rows = cur.fetchall()
            cur.close()
        finally:
            conn.close()
        out = []
        for r in rows:
            out.append(
                {
                    "time": _trade_ts(r[0]),
                    "open": round(float(r[1]), 4),
                    "high": round(float(r[2]), 4),
                    "low": round(float(r[3]), 4),
                    "close": round(float(r[4]), 4),
                    "volume": float(r[5] or 0),
                }
            )
        out.sort(key=lambda x: x["time"])
        return out
    except Exception as e:
        logger.debug("ticker PG daily bars failed %s:%s: %s", market, code, e)
        return []


def read_adjust_asof(market: str, code: str, as_of: date) -> Optional[float]:
    """Latest fore_adjust_factor with trade_date <= as_of (None if none)."""
    if not _dsn():
        return None
    try:
        conn = _connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT fore_adjust_factor FROM v_adjust_factor
                WHERE market = %s AND code = %s AND trade_date <= %s
                ORDER BY trade_date DESC
                LIMIT 1
                """,
                (market, code, as_of),
            )
            row = cur.fetchone()
            cur.close()
        finally:
            conn.close()
        return float(row[0]) if row and row[0] is not None else None
    except Exception as e:
        logger.debug("ticker PG adjust factor failed %s:%s: %s", market, code, e)
        return None


def read_daily_kline_qfq(
    market: str,
    code: str,
    limit: int,
    before_time: Optional[int] = None,
    after_time: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Daily bars from v_daily_bar, forward-adjusted (qfq) via sparse
    v_adjust_factor as-of. Bars before the first factor row use that row's
    factor (ffill), matching the data layer's as-of semantics.
    """
    rows = read_daily_bars(market, code, limit, before_time, after_time)
    if not rows:
        return []
    end_ts = rows[-1]["time"]
    end_date = date.fromtimestamp(end_ts)
    factors = []
    if _dsn():
        try:
            conn = _connect()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT trade_date, fore_adjust_factor FROM v_adjust_factor
                    WHERE market = %s AND code = %s AND trade_date <= %s
                    ORDER BY trade_date ASC
                    """,
                    (market, code, end_date),
                )
                factors = [
                    (r[0], float(r[1]))
                    for r in cur.fetchall()
                    if r[1] is not None and float(r[1]) > 0
                ]
                cur.close()
            finally:
                conn.close()
        except Exception as e:
            logger.debug("ticker PG adjust factors failed %s:%s: %s", market, code, e)
            factors = []
    if not factors:
        return rows
    f_date, f_val = factors[0]
    idx = 0
    out = []
    for b in rows:
        b_date = date.fromtimestamp(b["time"])
        while idx + 1 < len(factors) and factors[idx + 1][0] <= b_date:
            idx += 1
        f_date, f_val = factors[idx]
        f = f_val if f_date <= b_date else factors[0][1]
        out.append(
            {
                "time": b["time"],
                "open": round(b["open"] * f, 4),
                "high": round(b["high"] * f, 4),
                "low": round(b["low"] * f, 4),
                "close": round(b["close"] * f, 4),
                "volume": b["volume"],
            }
        )
    return out


def aggregate_weekly(daily_bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate daily bars into weekly bars keyed by ISO week (Monday open).
    v_daily_bar has no weekly view, so 1W consumers build it from daily rows.
    """
    weeks: Dict[tuple, Dict[str, Any]] = {}
    order: List[tuple] = []
    for b in daily_bars:
        d = date.fromtimestamp(b["time"])
        iso = d.isocalendar()
        key = (iso[0], iso[1])
        monday = datetime.fromisocalendar(iso[0], iso[1], 1)
        ts = int(monday.timestamp())
        if key not in weeks:
            weeks[key] = {
                "time": ts,
                "open": b["open"],
                "high": b["high"],
                "low": b["low"],
                "close": b["close"],
                "volume": b["volume"],
            }
            order.append(key)
        else:
            w = weeks[key]
            w["high"] = max(w["high"], b["high"])
            w["low"] = min(w["low"], b["low"])
            w["close"] = b["close"]
            w["volume"] += b["volume"]
    return [weeks[k] for k in order]


def read_financial(market: str, code: str) -> Dict[str, Any]:
    """
    Latest financial-fundamental snapshot as-of today, mapped to QuantDinger
    fundamental fields. Never raises; returns {} when unavailable.
    """
    if not _dsn():
        return {}
    out: Dict[str, Any] = {}
    try:
        conn = _connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT revenue_yoy_pct, net_profit_yoy_pct, roe_pct,
                       gross_margin_pct, net_margin_pct, pe_ttm, pb, market_cap
                FROM v_financial_fundamental
                WHERE market = %s AND code = %s
                  AND report_date <= CURRENT_DATE
                ORDER BY report_date DESC
                LIMIT 1
                """,
                (market, code),
            )
            row = cur.fetchone()
            cur.close()
        finally:
            conn.close()
        if not row:
            return {}
        revenue_yoy, net_profit_yoy, roe, gross_margin, net_margin, pe, pb, mcap = row
        if pe is not None:
            out["pe_ratio"] = float(pe)
        if pb is not None:
            out["pb_ratio"] = float(pb)
        if mcap is not None:
            out["market_cap"] = float(mcap)
        if roe is not None:
            out["roe"] = float(roe)
        if revenue_yoy is not None:
            out["revenue_growth"] = float(revenue_yoy)
        if net_profit_yoy is not None:
            out["earnings_growth"] = float(net_profit_yoy)
        # debt_to_equity is intentionally NOT mapped from debt_to_asset_pct:
        # the denominators differ (assets vs equity) and mapping would be wrong.
        margin = net_margin if net_margin is not None else gross_margin
        if margin is not None:
            out["profit_margin"] = float(margin)
        return out
    except Exception as e:
        logger.debug("ticker PG financial failed %s:%s: %s", market, code, e)
        return {}
