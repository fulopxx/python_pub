"""Optional same-contract daily data, without a new provider or subscription.

IB daily dates label the day of session close, not a UTC minute start.
Settlement publication is not timestamped by this endpoint. A conservative
availability lag is used for current analysis; this cache is not point-in-time
historical evidence and is never silently injected into historical audits.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
OHLCV = ["open", "high", "low", "close", "volume"]


def daily_bars_to_dataframe(
    raw: pd.DataFrame,
    contract_id: int,
    as_of: pd.Timestamp,
    *,
    availability_lag_hours: float = 12.0,
) -> pd.DataFrame:
    if raw.empty or any(name not in raw for name in ["date"] + OHLCV):
        raise ValueError("Üres vagy hiányos index daily válasz.")
    # IB day bars supply YYYYMMDD/trading dates; do not parse as UTC midnight.
    dates = [pd.Timestamp(str(value)).date() for value in raw["date"]]
    close_times = pd.DatetimeIndex([
        (pd.Timestamp(day) + pd.Timedelta(hours=17)).tz_localize("America/New_York")
        for day in dates
    ])
    availability = close_times + pd.Timedelta(hours=availability_lag_hours)
    result = raw[OHLCV].apply(pd.to_numeric, errors="coerce").copy()
    result.index = availability.tz_convert("UTC")
    valid = (np.isfinite(result).all(axis=1)
             & (result[["open", "high", "low", "close"]] > 0).all(axis=1)
             & (result.volume >= 0)
             & (result.high >= result[["open", "low", "close"]].max(axis=1))
             & (result.low <= result[["open", "high", "close"]].min(axis=1)))
    result = result.loc[valid].sort_index()
    result = result.loc[~result.index.duplicated(keep="last")]
    cutoff = pd.Timestamp(as_of)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    result = result.loc[result.index <= cutoff]
    result.index.name = "available_at"
    result.attrs.update(
        contract_id=int(contract_id),
        source="IBKR_DAILY_TRADES_SETTLEMENT_POSSIBLE",
        timestamp_basis="SESSION_CLOSE_PLUS_ASSUMED_AVAILABILITY_LAG",
        availability_lag_hours=availability_lag_hours,
        point_in_time_verified=False,
    )
    return result


def fetch_index_daily_context(
    ib: Any,
    contract: Any,
    as_of: pd.Timestamp,
    *,
    cache_dir: str | Path,
    duration: str = "6 M",
    cache_hours: float = 6.0,
    availability_lag_hours: float = 12.0,
) -> tuple[pd.DataFrame | None, str]:
    """Same-conId index daily OHLCV; an unavailable optional feed is nonfatal."""
    conid = int(getattr(contract, "conId", 0) or 0)
    if conid <= 0:
        return None, "NO_QUALIFIED_CONTRACT"
    path = Path(cache_dir) / f"index_daily_{conid}.csv"
    now = pd.Timestamp.now(tz="UTC")
    cached = None
    try:
        if path.exists():
            cached = pd.read_csv(path)
            age = (now.timestamp() - path.stat().st_mtime) / 3600
            if age <= cache_hours:
                df = daily_bars_to_dataframe(cached, conid, as_of,
                                             availability_lag_hours=availability_lag_hours)
                if len(df) >= 25:
                    return df, "CACHE"
    except Exception as error:
        log.warning("Index daily cache nem használható: %s", error)
    try:
        from ib_insync import Contract, util
        concrete = Contract(conId=conid, exchange=getattr(contract, "exchange", "") or "CME")
        log.info("Index daily: %s / 1 day, azonos conId=%s", duration, conid)
        bars = ib.reqHistoricalData(
            concrete, endDateTime="", durationStr=duration, barSizeSetting="1 day",
            whatToShow="TRADES", useRTH=False, formatDate=1, keepUpToDate=False,
        )
        raw = util.df(bars)
        if raw is None:
            raise ValueError("Nem érkezett daily bar.")
        df = daily_bars_to_dataframe(raw, conid, as_of,
                                     availability_lag_hours=availability_lag_hours)
        if len(df) < 25:
            raise ValueError(f"Csak {len(df)} használható napi bar; minimum 25.")
        path.parent.mkdir(parents=True, exist_ok=True)
        raw[["date"] + OHLCV].to_csv(path, index=False)
        return df, "IBKR_DAILY"
    except Exception as error:
        log.warning("Index daily N/A: %s", error)
        if cached is not None:
            try:
                df = daily_bars_to_dataframe(cached, conid, as_of,
                                             availability_lag_hours=availability_lag_hours)
                if len(df) >= 25:
                    return df, "STALE_CACHE_CHECK_FEATURE_FRESHNESS"
            except (ValueError, TypeError):
                pass
        return None, "UNAVAILABLE_OR_INSUFFICIENT"
