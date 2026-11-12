from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


# ===================================================================
# GC REGIME ENGINE VERSION 1.0.10
#
# Funkciók:
# - GC session VWAP (New York 18:00 ET reset)
# - 1m RSI(2), RSI(5), ATR
# - 5m ATR, ADX, +DI, -DI
# - Prior/current session high-low
# - NDOG/NWOG, Gap Low/CE/High
# - Stale gap felismerés
# - DXY és US 10Y real-yield macro context
# - Long/short anti-chasing
# - GC/MGC risk sizing
#
# FONTOS:
# Ez elemző/paper-trading modul.
# Nem küld IBKR ordert.
# ===================================================================


class GoldRegime(str, Enum):
    NEUTRAL = "NEUTRAL / NINCS EGYÉRTELMŰ REZHIM"
    BULL_TREND = "BULL TREND / LONG BIAS"
    BEAR_TREND = "BEAR TREND / SHORT BIAS"
    BLOWOFF_RISK = "BLOW-OFF / EXHAUSTION KOCKÁZAT"


@dataclass(frozen=True)
class GoldConfig:
    # Session
    new_york_timezone: str = "America/New_York"
    daily_close_hour_et: int = 17
    daily_open_hour_et: int = 18
    gap_close_window_start_min: int = 15
    gap_open_window_end_min: int = 20

    # Trend
    adx_trend_min: float = 25.0
    adx_strong_min: float = 30.0

    # RSI
    rsi2_blowoff: float = 95.0
    rsi5_blowoff: float = 80.0
    rsi2_extreme_oversold: float = 10.0
    rsi5_extreme_oversold: float = 20.0
    rsi2_extreme_overbought: float = 95.0
    rsi5_extreme_overbought: float = 80.0

    # Volume
    volume_expansion_ratio: float = 1.30
    volume_climax_ratio: float = 2.00
    volume_dry_ratio: float = 0.70

    # VWAP / ATR
    vwap_extension_atr: float = 2.5
    max_short_extension_below_vwap_atr: float = -3.0
    max_long_extension_above_vwap_atr: float = 3.0

    # Gap
    gap_level_proximity_atr: float = 0.50
    gap_reaction_confirmation_atr: float = 0.25
    gap_reaction_lookback_bars: int = 10
    stale_gap_distance_atr: float = 3.0

    # DXY macro
    dxy_bullish_change_pct: float = -0.05
    dxy_bearish_change_pct: float = 0.05

    # US 10Y real yield macro
    real_yield_bullish_change_bps: float = -1.0
    real_yield_bearish_change_bps: float = 1.0

    # Risk
    stop_atr_multiple: float = 1.0
    target1_atr_multiple: float = 1.0
    target2_atr_multiple: float = 2.0
    risk_per_trade_pct: float = 0.0025

    # Contract multipliers
    gc_multiplier_oz: float = 100.0
    mgc_multiplier_oz: float = 10.0


CONFIG = GoldConfig()


# ===================================================================
# INDIKÁTOROK
# ===================================================================

def calc_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def calc_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()


def calc_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=high.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=high.index,
    )

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    plus_di = (
        100
        * plus_dm.ewm(alpha=1 / period, adjust=False).mean()
        / atr.replace(0, np.nan)
    )

    minus_di = (
        100
        * minus_dm.ewm(alpha=1 / period, adjust=False).mean()
        / atr.replace(0, np.nan)
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    adx = dx.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    return pd.DataFrame(
        {
            "adx": adx.fillna(0.0),
            "plus_di": plus_di.fillna(0.0),
            "minus_di": minus_di.fillna(0.0),
        },
        index=close.index,
    )


def resample_ohlcv(
    df_1m: pd.DataFrame,
    rule: str = "5min",
) -> pd.DataFrame:
    return (
        df_1m.resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )


# ===================================================================
# ADATELLENŐRZÉS / IDŐZÓNA / SESSION
# ===================================================================

def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Hiányzó OHLCV oszlopok: {missing}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "A DataFrame indexének DatetimeIndex típusúnak kell lennie."
        )

    result = df.copy()
    result = result.sort_index()
    result = result[required].dropna()

    if len(result) < 100:
        raise ValueError(
            f"Túl kevés adat: {len(result)} db bar."
        )

    return result


def to_new_york_time(
    df: pd.DataFrame,
    config: GoldConfig = CONFIG,
) -> pd.DataFrame:
    result = df.copy()

    if result.index.tz is None:
        result.index = result.index.tz_localize("UTC")

    result.index = result.index.tz_convert(
        config.new_york_timezone
    )

    return result


def session_key(timestamp: pd.Timestamp) -> pd.Timestamp:
    """
    GC session key:

    Sunday 18:00 ET -> Sunday session.
    Monday 10:00 ET -> Sunday session.
    Monday 18:00 ET -> Monday session.
    """
    return (
        timestamp
        - pd.Timedelta(hours=18)
    ).normalize()


def add_session_key(df_ny: pd.DataFrame) -> pd.DataFrame:
    result = df_ny.copy()

    result["session_key"] = [
        session_key(timestamp)
        for timestamp in result.index
    ]

    return result


def calc_session_vwap(df_ny: pd.DataFrame) -> pd.Series:
    """
    GC session VWAP.

    Képlet:
        typical_price = (high + low + close) / 3

        VWAP =
            cumsum(typical_price * volume)
            /
            cumsum(volume)

    Reset:
        New York 18:00 ET GC session kezdete.
    """
    if "session_key" not in df_ny.columns:
        df_ny = add_session_key(df_ny)

    typical_price = (
        df_ny["high"]
        + df_ny["low"]
        + df_ny["close"]
    ) / 3.0

    cumulative_pv = (
        typical_price * df_ny["volume"]
    ).groupby(
        df_ny["session_key"]
    ).cumsum()

    cumulative_volume = (
        df_ny["volume"]
    ).groupby(
        df_ny["session_key"]
    ).cumsum()

    return cumulative_pv / cumulative_volume.replace(
        0,
        np.nan,
    )


def previous_session_levels(
    df_ny: pd.DataFrame,
) -> tuple[float, float]:
    if "session_key" not in df_ny.columns:
        df_ny = add_session_key(df_ny)

    sessions = pd.Index(
        df_ny["session_key"]
        .dropna()
        .unique()
    ).sort_values()

    if len(sessions) < 2:
        return np.nan, np.nan

    previous = sessions[-2]

    data = df_ny[
        df_ny["session_key"] == previous
    ]

    return (
        float(data["high"].max()),
        float(data["low"].min()),
    )


def current_session_levels(
    df_ny: pd.DataFrame,
) -> tuple[float, float]:
    if "session_key" not in df_ny.columns:
        df_ny = add_session_key(df_ny)

    current = df_ny["session_key"].iloc[-1]

    data = df_ny[
        df_ny["session_key"] == current
    ]

    return (
        float(data["high"].max()),
        float(data["low"].min()),
    )


# ===================================================================
# NDOG / NWOG
# ===================================================================

def close_before_break(
    df_ny: pd.DataFrame,
    target_date: pd.Timestamp,
    config: GoldConfig = CONFIG,
) -> tuple[float | None, pd.Timestamp | None]:
    date_only = target_date.normalize()

    start = date_only + pd.Timedelta(
        hours=config.daily_close_hour_et,
        minutes=-config.gap_close_window_start_min,
    )

    end = date_only + pd.Timedelta(
        hours=config.daily_close_hour_et,
        minutes=-1,
    )

    data = df_ny.loc[
        (df_ny.index >= start)
        & (df_ny.index <= end)
    ]

    if data.empty:
        return None, None

    return (
        float(data["close"].iloc[-1]),
        data.index[-1],
    )


def open_after_break(
    df_ny: pd.DataFrame,
    target_date: pd.Timestamp,
    config: GoldConfig = CONFIG,
) -> tuple[float | None, pd.Timestamp | None]:
    date_only = target_date.normalize()

    start = date_only + pd.Timedelta(
        hours=config.daily_open_hour_et,
    )

    end = date_only + pd.Timedelta(
        hours=config.daily_open_hour_et,
        minutes=config.gap_open_window_end_min,
    )

    data = df_ny.loc[
        (df_ny.index >= start)
        & (df_ny.index <= end)
    ]

    if data.empty:
        return None, None

    return (
        float(data["open"].iloc[0]),
        data.index[0],
    )


def make_gap(
    gap_type: str,
    close_price: float,
    open_price: float,
    close_timestamp: pd.Timestamp,
    open_timestamp: pd.Timestamp,
) -> dict[str, Any]:
    gap_low = min(close_price, open_price)
    gap_high = max(close_price, open_price)

    if open_price > close_price:
        direction = "UP_GAP"

    elif open_price < close_price:
        direction = "DOWN_GAP"

    else:
        direction = "FLAT"

    return {
        "type": gap_type,
        "direction": direction,
        "close_price": close_price,
        "open_price": open_price,
        "gap_low": gap_low,
        "gap_high": gap_high,
        "gap_ce": (gap_low + gap_high) / 2.0,
        "gap_size": gap_high - gap_low,
        "close_timestamp": close_timestamp,
        "open_timestamp": open_timestamp,
    }


def calculate_ndogs(
    df_ny: pd.DataFrame,
    config: GoldConfig = CONFIG,
) -> list[dict[str, Any]]:
    dates = pd.Index(
        df_ny.index.normalize().unique()
    ).sort_values()

    gaps: list[dict[str, Any]] = []

    for date_value in dates:
        # Friday close -> Sunday open = NWOG.
        if date_value.weekday() >= 4:
            continue

        close_price, close_time = close_before_break(
            df_ny,
            date_value,
            config,
        )

        open_price, open_time = open_after_break(
            df_ny,
            date_value,
            config,
        )

        if (
            close_price is None
            or open_price is None
            or close_time is None
            or open_time is None
        ):
            continue

        gaps.append(
            make_gap(
                "NDOG",
                close_price,
                open_price,
                close_time,
                open_time,
            )
        )

    return gaps


def calculate_nwogs(
    df_ny: pd.DataFrame,
    config: GoldConfig = CONFIG,
) -> list[dict[str, Any]]:
    dates = pd.Index(
        df_ny.index.normalize().unique()
    ).sort_values()

    gaps: list[dict[str, Any]] = []

    for sunday_date in dates:
        if sunday_date.weekday() != 6:
            continue

        friday_date = sunday_date - pd.Timedelta(days=2)

        close_price, close_time = close_before_break(
            df_ny,
            friday_date,
            config,
        )

        open_price, open_time = open_after_break(
            df_ny,
            sunday_date,
            config,
        )

        if (
            close_price is None
            or open_price is None
            or close_time is None
            or open_time is None
        ):
            continue

        gaps.append(
            make_gap(
                "NWOG",
                close_price,
                open_price,
                close_time,
                open_time,
            )
        )

    return gaps


def select_latest_gap(
    gaps: list[dict[str, Any]],
    current_timestamp: pd.Timestamp,
) -> dict[str, Any] | None:
    valid = [
        gap
        for gap in gaps
        if gap["open_timestamp"] <= current_timestamp
    ]

    if not valid:
        return None

    return max(
        valid,
        key=lambda gap: gap["open_timestamp"],
    )


def gap_fill_status(
    df_ny: pd.DataFrame,
    gap: dict[str, Any] | None,
) -> dict[str, Any]:
    if gap is None:
        return {
            "available": False,
        }

    data = df_ny.loc[
        df_ny.index >= gap["open_timestamp"]
    ]

    if data.empty:
        return {
            "available": False,
        }

    current_price = float(data["close"].iloc[-1])

    low = gap["gap_low"]
    high = gap["gap_high"]
    ce = gap["gap_ce"]

    touched_low = (
        (data["low"] <= low)
        & (data["high"] >= low)
    ).any()

    touched_high = (
        (data["low"] <= high)
        & (data["high"] >= high)
    ).any()

    touched_ce = (
        (data["low"] <= ce)
        & (data["high"] >= ce)
    ).any()

    if gap["direction"] == "UP_GAP":
        fully_filled = (
            data["low"] <= low
        ).any()

    elif gap["direction"] == "DOWN_GAP":
        fully_filled = (
            data["high"] >= high
        ).any()

    else:
        fully_filled = True

    if current_price > high:
        location = "A GAP FELETT"

    elif current_price < low:
        location = "A GAP ALATT"

    else:
        location = "A GAP TARTOMÁNYÁBAN"

    return {
        "available": True,
        "price_location": location,
        "fully_filled": bool(fully_filled),
        "touched_low": bool(touched_low),
        "touched_high": bool(touched_high),
        "touched_ce": bool(touched_ce),
    }


def gap_reaction(
    df_ny: pd.DataFrame,
    gap: dict[str, Any] | None,
    atr_1m: float,
    config: GoldConfig = CONFIG,
) -> dict[str, Any]:
    if gap is None or not np.isfinite(atr_1m) or atr_1m <= 0:
        return {
            "available": False,
            "reaction": "N/A",
            "distance_atr": np.nan,
        }

    recent = df_ny.tail(
        config.gap_reaction_lookback_bars
    )

    if recent.empty:
        return {
            "available": False,
            "reaction": "N/A",
            "distance_atr": np.nan,
        }

    current_close = float(
        recent["close"].iloc[-1]
    )

    levels = {
        "GAP_LOW": gap["gap_low"],
        "GAP_CE": gap["gap_ce"],
        "GAP_HIGH": gap["gap_high"],
    }

    closest_name = min(
        levels,
        key=lambda name: abs(
            current_close - levels[name]
        ),
    )

    level = levels[closest_name]

    distance_atr = (
        current_close - level
    ) / atr_1m

    touched = (
        (recent["low"] <= level)
        & (recent["high"] >= level)
    ).any()

    confirm = (
        config.gap_reaction_confirmation_atr
        * atr_1m
    )

    near = (
        config.gap_level_proximity_atr
        * atr_1m
    )

    if touched and current_close > level + confirm:
        reaction = "BULLISH_RECLAIM"

    elif touched and current_close < level - confirm:
        reaction = "BEARISH_REJECTION"

    elif abs(current_close - level) <= near:
        reaction = "NEAR_LEVEL"

    else:
        reaction = "NO_RECENT_INTERACTION"

    return {
        "available": True,
        "reaction": reaction,
        "closest_level_name": closest_name,
        "closest_level_price": level,
        "distance_atr": distance_atr,
        "touched_level": bool(touched),
    }


def build_gap_context(
    df_ny: pd.DataFrame,
    atr_1m: float,
    config: GoldConfig = CONFIG,
) -> dict[str, Any]:
    ndogs = calculate_ndogs(
        df_ny,
        config,
    )

    nwogs = calculate_nwogs(
        df_ny,
        config,
    )

    timestamp = df_ny.index[-1]

    latest_ndog = select_latest_gap(
        ndogs,
        timestamp,
    )

    latest_nwog = select_latest_gap(
        nwogs,
        timestamp,
    )

    return {
        "latest_ndog": latest_ndog,
        "latest_nwog": latest_nwog,

        "ndog_fill": gap_fill_status(
            df_ny,
            latest_ndog,
        ),

        "nwog_fill": gap_fill_status(
            df_ny,
            latest_nwog,
        ),

        "ndog_reaction": gap_reaction(
            df_ny,
            latest_ndog,
            atr_1m,
            config,
        ),

        "nwog_reaction": gap_reaction(
            df_ny,
            latest_nwog,
            atr_1m,
            config,
        ),
    }


# ===================================================================
# DXY / REAL YIELD MACRO CONTEXT
# ===================================================================

def analyze_gold_macro_context(
    macro_context: dict[str, float] | None,
    config: GoldConfig = CONFIG,
) -> dict[str, Any]:
    """
    Várt macro_context:

    {
        "dxy_price": 102.015,
        "dxy_vwap": 101.973,
        "dxy_change_15m_pct": 0.0004,

        "real_yield_10y": np.nan,
        "real_yield_change_15m_bps": np.nan,
    }

    DXY:
        - DXY gyengül + DXY VWAP alatt
          -> Gold bullish macro confirmation.

        - DXY erősödik + DXY VWAP felett
          -> Gold bearish macro confirmation.

    Real yield:
        - Csökkenő reálhozam
          -> Gold bullish macro confirmation.

        - Emelkedő reálhozam
          -> Gold bearish macro confirmation.

    A macro context csak konfirmáció vagy konfliktusfilter.
    A GC saját price action az elsődleges.
    """
    if not macro_context:
        return {
            "available": False,
            "bull_score": 0,
            "bear_score": 0,
            "state": "N/A",
            "message": (
                "Nincs DXY vagy US real yield adat. "
                "A GC rendszer price-action alapon fut."
            ),
            "bullish_checks": {},
            "bearish_checks": {},
        }

    dxy_price = macro_context.get(
        "dxy_price",
        np.nan,
    )

    dxy_vwap = macro_context.get(
        "dxy_vwap",
        np.nan,
    )

    dxy_change_15m = macro_context.get(
        "dxy_change_15m_pct",
        np.nan,
    )

    real_yield_10y = macro_context.get(
        "real_yield_10y",
        np.nan,
    )

    real_yield_change = macro_context.get(
        "real_yield_change_15m_bps",
        np.nan,
    )

    bullish_checks = {
        "DXY 15m gyengül": (
            np.isfinite(dxy_change_15m)
            and dxy_change_15m
            <= config.dxy_bullish_change_pct
        ),

        "DXY a saját VWAP alatt": (
            np.isfinite(dxy_price)
            and np.isfinite(dxy_vwap)
            and dxy_price < dxy_vwap
        ),

        "10Y real yield 15m csökken": (
            np.isfinite(real_yield_change)
            and real_yield_change
            <= config.real_yield_bullish_change_bps
        ),
    }

    bearish_checks = {
        "DXY 15m erősödik": (
            np.isfinite(dxy_change_15m)
            and dxy_change_15m
            >= config.dxy_bearish_change_pct
        ),

        "DXY a saját VWAP felett": (
            np.isfinite(dxy_price)
            and np.isfinite(dxy_vwap)
            and dxy_price > dxy_vwap
        ),

        "10Y real yield 15m emelkedik": (
            np.isfinite(real_yield_change)
            and real_yield_change
            >= config.real_yield_bearish_change_bps
        ),
    }

    bull_score = sum(
        bullish_checks.values()
    )

    bear_score = sum(
        bearish_checks.values()
    )

    available = (
        np.isfinite(dxy_price)
        or np.isfinite(dxy_change_15m)
        or np.isfinite(real_yield_10y)
        or np.isfinite(real_yield_change)
    )

    if bull_score >= 2 and bear_score == 0:
        state = "GOLD BULLISH MACRO CONFIRMATION"

        message = (
            "DXY és/vagy reálhozam kontextus támogatja a GC long oldalt."
        )

    elif bear_score >= 2 and bull_score == 0:
        state = "GOLD BEARISH MACRO CONFIRMATION"

        message = (
            "DXY és/vagy reálhozam kontextus támogatja a GC short oldalt."
        )

    elif bull_score >= 1 and bear_score >= 1:
        state = "MIXED MACRO"

        message = (
            "Vegyes makrójelzés. A GC saját price action fontosabb."
        )

    else:
        state = "NO STRONG MACRO CONFIRMATION"

        message = (
            "Nincs elég erős DXY/reálhozam makró-konfirmáció."
        )

    return {
        "available": available,

        "dxy_price": dxy_price,
        "dxy_vwap": dxy_vwap,
        "dxy_change_15m_pct": dxy_change_15m,

        "real_yield_10y": real_yield_10y,
        "real_yield_change_15m_bps": real_yield_change,

        "bull_score": bull_score,
        "bear_score": bear_score,

        "state": state,
        "message": message,

        "bullish_checks": bullish_checks,
        "bearish_checks": bearish_checks,
    }


# ===================================================================
# SNAPSHOT
# ===================================================================

def build_gold_snapshot(
    df_1m: pd.DataFrame,
    macro_context: dict[str, float] | None = None,
    config: GoldConfig = CONFIG,
) -> dict[str, Any]:
    df = validate_ohlcv(df_1m)

    df_ny = to_new_york_time(
        df,
        config,
    )

    df_ny = add_session_key(df_ny)
    df_ny["vwap"] = calc_session_vwap(df_ny)

    df_5m = resample_ohlcv(
        df_ny[
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]
    )

    if len(df_5m) < 30:
        raise ValueError(
            "Túl kevés 5 perces bar az ADX/ATR számításhoz."
        )

    atr_1m = calc_atr(
        df_ny["high"],
        df_ny["low"],
        df_ny["close"],
    )

    atr_5m = calc_atr(
        df_5m["high"],
        df_5m["low"],
        df_5m["close"],
    )

    rsi_2 = calc_rsi(
        df_ny["close"],
        2,
    )

    rsi_5 = calc_rsi(
        df_ny["close"],
        5,
    )

    adx_5m = calc_adx(
        df_5m["high"],
        df_5m["low"],
        df_5m["close"],
    )

    close = float(df_ny["close"].iloc[-1])
    vwap = float(df_ny["vwap"].iloc[-1])

    atr_1m_value = float(
        atr_1m.iloc[-1]
    )

    atr_5m_value = float(
        atr_5m.iloc[-1]
    )

    volume = float(
        df_ny["volume"].iloc[-1]
    )

    volume_avg_20 = float(
        df_ny["volume"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    volume_ratio = (
        volume / volume_avg_20
        if volume_avg_20 > 0
        else np.nan
    )

    prior_high, prior_low = previous_session_levels(
        df_ny
    )

    current_high, current_low = current_session_levels(
        df_ny
    )

    return {
        "timestamp": df_ny.index[-1],

        "close": close,
        "vwap": vwap,

        "atr_1m": atr_1m_value,
        "atr_5m": atr_5m_value,

        "atr_1m_pct": (
            100 * atr_1m_value / close
            if close > 0
            else np.nan
        ),

        "rsi_2": float(rsi_2.iloc[-1]),
        "rsi_5": float(rsi_5.iloc[-1]),

        "adx_5m": float(
            adx_5m["adx"].iloc[-1]
        ),

        "plus_di_5m": float(
            adx_5m["plus_di"].iloc[-1]
        ),

        "minus_di_5m": float(
            adx_5m["minus_di"].iloc[-1]
        ),

        "volume": volume,
        "volume_avg_20": volume_avg_20,
        "volume_ratio": volume_ratio,

        "vwap_distance_atr": (
            (close - vwap) / atr_1m_value
            if atr_1m_value > 0
            else np.nan
        ),

        "prior_session_high": prior_high,
        "prior_session_low": prior_low,

        "current_session_high": current_high,
        "current_session_low": current_low,

        "gap_context": build_gap_context(
            df_ny,
            atr_1m_value,
            config,
        ),

        "macro_analysis": analyze_gold_macro_context(
            macro_context,
            config,
        ),
    }


# ===================================================================
# REZHIM DETEKTOR
# ===================================================================

def gold_regime_active(
    snapshot: dict[str, Any],
    config: GoldConfig = CONFIG,
) -> dict[str, Any]:
    close = snapshot["close"]
    vwap = snapshot["vwap"]

    adx = snapshot["adx_5m"]
    plus_di = snapshot["plus_di_5m"]
    minus_di = snapshot["minus_di_5m"]

    rsi_2 = snapshot["rsi_2"]
    rsi_5 = snapshot["rsi_5"]

    volume_ratio = snapshot["volume_ratio"]
    vwap_distance = snapshot["vwap_distance_atr"]

    bullish_checks = {
        "Ár a session VWAP felett": close > vwap,
        f"ADX(5m) > {config.adx_trend_min:.0f}": (
            adx > config.adx_trend_min
        ),
        "+DI dominál -DI felett": plus_di > minus_di,
        f"Volume >= {config.volume_expansion_ratio:.1f}× 20-bar átlag": (
            volume_ratio >= config.volume_expansion_ratio
        ),
        "RSI(5) > 50": rsi_5 > 50,
    }

    bearish_checks = {
        "Ár a session VWAP alatt": close < vwap,
        f"ADX(5m) > {config.adx_strong_min:.0f}": (
            adx > config.adx_strong_min
        ),
        "-DI dominál +DI felett": minus_di > plus_di,
        f"Volume >= {config.volume_expansion_ratio:.1f}× 20-bar átlag": (
            volume_ratio >= config.volume_expansion_ratio
        ),
        "RSI(5) < 50": rsi_5 < 50,
    }

    blowoff_checks = {
        f"RSI(2) > {config.rsi2_blowoff:.0f}": (
            rsi_2 > config.rsi2_blowoff
        ),
        f"RSI(5) > {config.rsi5_blowoff:.0f}": (
            rsi_5 > config.rsi5_blowoff
        ),
        f"Ár > VWAP + {config.vwap_extension_atr:.1f}× ATR(1m)": (
            vwap_distance > config.vwap_extension_atr
        ),
        f"Volume >= {config.volume_climax_ratio:.1f}× 20-bar átlag": (
            volume_ratio >= config.volume_climax_ratio
        ),
        "+DI dominál -DI felett": plus_di > minus_di,
    }

    bull_score = sum(
        bullish_checks.values()
    )

    bear_score = sum(
        bearish_checks.values()
    )

    blowoff_score = sum(
        blowoff_checks.values()
    )

    if blowoff_score >= 4:
        regime = GoldRegime.BLOWOFF_RISK

    elif bear_score >= 4:
        regime = GoldRegime.BEAR_TREND

    elif bull_score >= 4:
        regime = GoldRegime.BULL_TREND

    else:
        regime = GoldRegime.NEUTRAL

    return {
        "regime": regime,

        "bullish_checks": bullish_checks,
        "bearish_checks": bearish_checks,
        "blowoff_checks": blowoff_checks,

        "bull_score": bull_score,
        "bear_score": bear_score,
        "blowoff_score": blowoff_score,
    }


# ===================================================================
# GAP KONTEXTUS
# ===================================================================

def gap_trade_context(
    snapshot: dict[str, Any],
    regime: GoldRegime,
) -> dict[str, Any]:
    context = snapshot["gap_context"]

    ndog_reaction = context["ndog_reaction"]["reaction"]
    nwog_reaction = context["nwog_reaction"]["reaction"]

    messages: list[str] = []

    if nwog_reaction == "BULLISH_RECLAIM":
        messages.append(
            "NWOG bullish reclaim: az ár érintette a szintet, majd fölé zárt."
        )

    elif nwog_reaction == "BEARISH_REJECTION":
        messages.append(
            "NWOG bearish rejection: az ár érintette a szintet, majd alá zárt."
        )

    elif nwog_reaction == "NEAR_LEVEL":
        messages.append(
            "Az ár NWOG szint közelében van; várj 1m/5m megerősítésre."
        )

    if ndog_reaction == "BULLISH_RECLAIM":
        messages.append(
            "NDOG bullish reclaim: az ár visszafoglalta a közeli NDOG szintet."
        )

    elif ndog_reaction == "BEARISH_REJECTION":
        messages.append(
            "NDOG bearish rejection: az ár nem tudott a közeli NDOG szint fölött maradni."
        )

    elif ndog_reaction == "NEAR_LEVEL":
        messages.append(
            "Az ár NDOG szint közelében van; ne lépj be megerősítés nélkül."
        )

    if regime == GoldRegime.BEAR_TREND:
        if (
            ndog_reaction == "BULLISH_RECLAIM"
            or nwog_reaction == "BULLISH_RECLAIM"
        ):
            messages.append(
                "Bear trendben bullish gap reclaim látszik: "
                "fresh short chasing kockázata nőtt."
            )

    if regime == GoldRegime.BULL_TREND:
        if (
            ndog_reaction == "BEARISH_REJECTION"
            or nwog_reaction == "BEARISH_REJECTION"
        ):
            messages.append(
                "Bull trendben bearish gap rejection látszik: "
                "új long előtt várj megerősítésre."
            )

    if not messages:
        messages.append(
            "Nincs közeli, megerősített NDOG/NWOG reakció. "
            "A gapek referencia- és célzónák."
        )

    return {
        "messages": messages,
    }


# ===================================================================
# RISK / TRADE TERV
# ===================================================================

def calculate_gold_risk_plan(
    snapshot: dict[str, Any],
    regime: GoldRegime,
    equity_usd: float,
    config: GoldConfig = CONFIG,
) -> dict[str, Any]:
    entry = snapshot["close"]
    atr_5m = snapshot["atr_5m"]

    rsi_2 = snapshot["rsi_2"]
    rsi_5 = snapshot["rsi_5"]
    vwap_distance = snapshot["vwap_distance_atr"]

    gap_context = snapshot["gap_context"]
    macro = snapshot["macro_analysis"]

    ndog_reaction = gap_context["ndog_reaction"]["reaction"]
    nwog_reaction = gap_context["nwog_reaction"]["reaction"]

    if not np.isfinite(atr_5m) or atr_5m <= 0:
        raise ValueError(
            "Érvénytelen ATR(5m); nem lehet risk tervet készíteni."
        )

    strong_macro_bullish = (
        macro["bull_score"] >= 2
        and macro["bear_score"] == 0
    )

    strong_macro_bearish = (
        macro["bear_score"] >= 2
        and macro["bull_score"] == 0
    )

    # ---------------------------------------------------------------
    # SHORT CHASING TILTÁS
    # ---------------------------------------------------------------
    extreme_oversold = (
        rsi_2 < config.rsi2_extreme_oversold
        or rsi_5 < config.rsi5_extreme_oversold
    )

    too_far_below_vwap = (
        np.isfinite(vwap_distance)
        and vwap_distance
        < config.max_short_extension_below_vwap_atr
    )

    bullish_gap_reclaim = (
        ndog_reaction == "BULLISH_RECLAIM"
        or nwog_reaction == "BULLISH_RECLAIM"
    )

    if regime == GoldRegime.BEAR_TREND and (
        extreme_oversold
        or too_far_below_vwap
        or bullish_gap_reclaim
        or strong_macro_bullish
    ):
        blocking_reasons: list[str] = []

        if rsi_2 < config.rsi2_extreme_oversold:
            blocking_reasons.append(
                f"RSI(2)={rsi_2:.1f} < "
                f"{config.rsi2_extreme_oversold:.0f}"
            )

        if rsi_5 < config.rsi5_extreme_oversold:
            blocking_reasons.append(
                f"RSI(5)={rsi_5:.1f} < "
                f"{config.rsi5_extreme_oversold:.0f}"
            )

        if too_far_below_vwap:
            blocking_reasons.append(
                f"ár {vwap_distance:.2f}× ATR-rel a VWAP alatt"
            )

        if bullish_gap_reclaim:
            blocking_reasons.append(
                "bullish NDOG/NWOG reclaim"
            )

        if strong_macro_bullish:
            blocking_reasons.append(
                "DXY/reálhozam macro context GC bullish"
            )

        return {
            "trade_allowed": False,

            "reason": (
                "Bearish trend van, de új SHORT belépés TILTVA. "
                "Legalább egy anti-chasing vagy macro-konfliktus tiltó feltétel aktív. "
                "A pontos okokat lásd a 'Konkrét tiltó jelek' sorban."
            ),

            "details": "; ".join(blocking_reasons),

            "next_setup": (
                "Várj 5 perces felpattanásra, majd failed VWAP reclaimre, "
                "NDOG/NWOG bearish rejectionre vagy lower high struktúrára."
            ),
        }

    # ---------------------------------------------------------------
    # LONG CHASING TILTÁS
    # ---------------------------------------------------------------
    extreme_overbought = (
        rsi_2 > config.rsi2_extreme_overbought
        or rsi_5 > config.rsi5_extreme_overbought
    )

    too_far_above_vwap = (
        np.isfinite(vwap_distance)
        and vwap_distance
        > config.max_long_extension_above_vwap_atr
    )

    bearish_gap_rejection = (
        ndog_reaction == "BEARISH_REJECTION"
        or nwog_reaction == "BEARISH_REJECTION"
    )

    if regime == GoldRegime.BULL_TREND and (
        extreme_overbought
        or too_far_above_vwap
        or bearish_gap_rejection
        or strong_macro_bearish
    ):
        blocking_reasons = []

        if rsi_2 > config.rsi2_extreme_overbought:
            blocking_reasons.append(
                f"RSI(2)={rsi_2:.1f} > "
                f"{config.rsi2_extreme_overbought:.0f}"
            )

        if rsi_5 > config.rsi5_extreme_overbought:
            blocking_reasons.append(
                f"RSI(5)={rsi_5:.1f} > "
                f"{config.rsi5_extreme_overbought:.0f}"
            )

        if too_far_above_vwap:
            blocking_reasons.append(
                f"ár {vwap_distance:.2f}× ATR-rel a VWAP fölött"
            )

        if bearish_gap_rejection:
            blocking_reasons.append(
                "bearish NDOG/NWOG rejection"
            )

        if strong_macro_bearish:
            blocking_reasons.append(
                "DXY/reálhozam macro context GC bearish"
            )

        return {
            "trade_allowed": False,

            "reason": (
                "Bullish trend van, de új LONG belépés TILTVA. "
                "Legalább egy anti-chasing vagy macro-konfliktus tiltó feltétel aktív. "
                "A pontos okokat lásd a 'Konkrét tiltó jelek' sorban."
            ),

            "details": "; ".join(blocking_reasons),

            "next_setup": (
                "Várj VWAP pullbackre, higher low-ra, bullish NDOG/NWOG reclaimre "
                "vagy breakout-retest setupra."
            ),
        }

    # ---------------------------------------------------------------
    # NEUTRAL / BLOW-OFF
    # ---------------------------------------------------------------
    if regime == GoldRegime.NEUTRAL:
        return {
            "trade_allowed": False,

            "reason": (
                "Nincs elég tiszta trend-rezhim. "
                "Automatikus belépés tiltott."
            ),

            "next_setup": (
                "Várj VWAP reclaimre, failed reclaimre, "
                "NDOG/NWOG reakcióra vagy tiszta 5 perces struktúrára."
            ),
        }

    if regime == GoldRegime.BLOWOFF_RISK:
        return {
            "trade_allowed": False,

            "reason": (
                "Blow-off / exhaustion kockázat van. "
                "Ne üldözd az extrém emelkedést új long belépéssel."
            ),

            "next_setup": (
                "Várj 5 perces struktúratörésre, lower high-ra, "
                "VWAP alá visszatérésre vagy bearish gap rejectionre."
            ),
        }

    # ---------------------------------------------------------------
    # ELVI TRADE TERV
    # ---------------------------------------------------------------
    risk_budget = (
        equity_usd * config.risk_per_trade_pct
    )

    stop_distance = (
        atr_5m * config.stop_atr_multiple
    )

    if regime == GoldRegime.BULL_TREND:
        side = "LONG"
        stop_price = entry - stop_distance
        target_1 = entry + atr_5m * config.target1_atr_multiple
        target_2 = entry + atr_5m * config.target2_atr_multiple

    elif regime == GoldRegime.BEAR_TREND:
        side = "SHORT"
        stop_price = entry + stop_distance
        target_1 = entry - atr_5m * config.target1_atr_multiple
        target_2 = entry - atr_5m * config.target2_atr_multiple

    else:
        return {
            "trade_allowed": False,
            "reason": "Ismeretlen rezhim.",
        }

    gc_risk = stop_distance * config.gc_multiplier_oz
    mgc_risk = stop_distance * config.mgc_multiplier_oz

    return {
        "trade_allowed": True,

        "side": side,
        "entry": round(entry, 2),
        "stop_price": round(stop_price, 2),
        "target_1": round(target_1, 2),
        "target_2": round(target_2, 2),

        "atr_5m": round(atr_5m, 2),
        "stop_distance_usd_per_oz": round(stop_distance, 2),

        "risk_budget_usd": round(risk_budget, 2),

        "gc_risk_per_contract_usd": round(gc_risk, 2),
        "mgc_risk_per_contract_usd": round(mgc_risk, 2),

        "max_gc_contracts": int(risk_budget // gc_risk),
        "max_mgc_contracts": int(risk_budget // mgc_risk),

        "warning": (
            "A kontraktusszám elméleti maximum. Margin, spread, slippage, "
            "makroadat és napi veszteséglimit nincs benne."
        ),
    }


# ===================================================================
# REPORT
# ===================================================================

def format_checks(
    title: str,
    checks: dict[str, bool],
) -> list[str]:
    lines = [
        "",
        "-" * 68,
        title,
        "-" * 68,
    ]

    for name, value in checks.items():
        status = "✅ IGEN" if value else "❌ NEM"
        lines.append(f"[{status}] {name}")

    return lines


def format_macro(
    macro: dict[str, Any],
) -> list[str]:
    lines = [
        "",
        "-" * 68,
        "DXY / US 10Y REAL YIELD MACRO CONTEXT",
        "-" * 68,
    ]

    if not macro["available"]:
        lines.append(macro["message"])
        return lines

    if np.isfinite(macro["dxy_price"]):
        lines.append(
            f"DXY ár:                 {macro['dxy_price']:.3f}"
        )

    if np.isfinite(macro["dxy_vwap"]):
        lines.append(
            f"DXY VWAP:               {macro['dxy_vwap']:.3f}"
        )

    if np.isfinite(macro["dxy_change_15m_pct"]):
        lines.append(
            f"DXY 15m változás:       "
            f"{macro['dxy_change_15m_pct']:.4f}%"
        )

    if np.isfinite(macro["real_yield_10y"]):
        lines.append(
            f"US 10Y real yield:      "
            f"{macro['real_yield_10y']:.3f}%"
        )

    else:
        lines.append(
            "US 10Y real yield:      N/A"
        )

    if np.isfinite(macro["real_yield_change_15m_bps"]):
        lines.append(
            f"Real yield 15m change:  "
            f"{macro['real_yield_change_15m_bps']:.2f} bp"
        )

    lines.extend(
        [
            f"Macro Bull Score:       {macro['bull_score']}/3",
            f"Macro Bear Score:       {macro['bear_score']}/3",
            f"Macro állapot:          {macro['state']}",
            f"Értelmezés:             {macro['message']}",
            "",
            "Bullish macro checks:",
        ]
    )

    for name, value in macro["bullish_checks"].items():
        status = "✅ IGEN" if value else "❌ NEM"
        lines.append(f"[{status}] {name}")

    lines.extend(
        [
            "",
            "Bearish macro checks:",
        ]
    )

    for name, value in macro["bearish_checks"].items():
        status = "✅ IGEN" if value else "❌ NEM"
        lines.append(f"[{status}] {name}")

    return lines


def format_gap(
    title: str,
    gap: dict[str, Any] | None,
    fill: dict[str, Any],
    reaction: dict[str, Any],
    config: GoldConfig = CONFIG,
) -> list[str]:
    lines = [
        "",
        "-" * 68,
        title,
        "-" * 68,
    ]

    if gap is None:
        lines.append(
            "Nincs számolható gap az aktuális adathalmazban."
        )
        return lines

    lines.extend(
        [
            f"Irány:                  {gap['direction']}",
            f"Close reference:        {gap['close_price']:.2f}",
            f"Open reference:         {gap['open_price']:.2f}",
            f"Gap Low:                {gap['gap_low']:.2f}",
            f"Gap CE (50%):           {gap['gap_ce']:.2f}",
            f"Gap High:               {gap['gap_high']:.2f}",
            f"Gap méret:              {gap['gap_size']:.2f} USD/oz",
            f"Gap close időpont:      {gap['close_timestamp']}",
            f"Gap open időpont:       {gap['open_timestamp']}",
        ]
    )

    if fill.get("available", False):
        lines.extend(
            [
                f"Jelenlegi helyzet:      {fill['price_location']}",
                f"CE érintve:             {'IGEN' if fill['touched_ce'] else 'NEM'}",
                f"Gap Low érintve:        {'IGEN' if fill['touched_low'] else 'NEM'}",
                f"Gap High érintve:       {'IGEN' if fill['touched_high'] else 'NEM'}",
                f"Teljes gap fill:        {'IGEN' if fill['fully_filled'] else 'NEM'}",
            ]
        )

    if reaction.get("available", False):
        lines.extend(
            [
                f"Legközelebbi szint:     {reaction['closest_level_name']}",
                f"Szint ára:              {reaction['closest_level_price']:.2f}",
                f"Távolság:               {reaction['distance_atr']:.2f} × ATR(1m)",
                f"Reakció:                {reaction['reaction']}",
            ]
        )

        if (
            fill.get("fully_filled", False)
            and abs(reaction.get("distance_atr", 0))
            > config.stale_gap_distance_atr
        ):
            lines.append(
                "Gap státusz:            KORÁBBI / MÁR KITÖLTÖTT REFERENCIA, "
                "JELENLEG NEM KÖZVETLEN TRADE SZINT"
            )

    return lines


# ===================================================================
# FŐ FUTTATÓ
# ===================================================================

def run_gold_checklist(
    df_1m: pd.DataFrame,
    equity_usd: float = 10_000.0,
    macro_context: dict[str, float] | None = None,
    config: GoldConfig = CONFIG,
) -> str:
    snapshot = build_gold_snapshot(
        df_1m=df_1m,
        macro_context=macro_context,
        config=config,
    )

    result = gold_regime_active(
        snapshot=snapshot,
        config=config,
    )

    regime = result["regime"]
    gaps = snapshot["gap_context"]

    lines = [
        "",
        "=" * 68,
        "GC / GOLD REGIME ENGINE VERSION 1.0.10",
        "=" * 68,

        f"Időbélyeg:              {snapshot['timestamp']}",
        f"GC utolsó ár:            {snapshot['close']:.2f} USD/oz",
        f"Session VWAP:            {snapshot['vwap']:.2f}",
        f"VWAP-távolság:           {snapshot['vwap_distance_atr']:.2f} × ATR(1m)",

        f"ATR(1m,14):              {snapshot['atr_1m']:.2f} USD/oz",
        f"ATR(5m,14):              {snapshot['atr_5m']:.2f} USD/oz",
        f"ATR(1m) relatív:         {snapshot['atr_1m_pct']:.4f}%",

        f"RSI(2):                  {snapshot['rsi_2']:.1f}",
        f"RSI(5):                  {snapshot['rsi_5']:.1f}",

        f"ADX(5m):                 {snapshot['adx_5m']:.1f}",
        f"+DI / -DI:               {snapshot['plus_di_5m']:.1f} / {snapshot['minus_di_5m']:.1f}",

        f"Utolsó 1m volumen:       {snapshot['volume']:,.0f}",
        f"20-bar átlagvolumen:     {snapshot['volume_avg_20']:,.0f}",
        f"Volume ratio:            {snapshot['volume_ratio']:.2f}×",

        (
            f"Előző session high:      {snapshot['prior_session_high']:.2f}"
            if np.isfinite(snapshot["prior_session_high"])
            else "Előző session high:      N/A"
        ),

        (
            f"Előző session low:       {snapshot['prior_session_low']:.2f}"
            if np.isfinite(snapshot["prior_session_low"])
            else "Előző session low:       N/A"
        ),

        (
            f"Aktuális session high:   {snapshot['current_session_high']:.2f}"
            if np.isfinite(snapshot["current_session_high"])
            else "Aktuális session high:   N/A"
        ),

        (
            f"Aktuális session low:    {snapshot['current_session_low']:.2f}"
            if np.isfinite(snapshot["current_session_low"])
            else "Aktuális session low:    N/A"
        ),

        "",
        "=" * 68,
        f"AKTUÁLIS REZHIM: {regime.value}",
        (
            f"Pontszámok — "
            f"Bull: {result['bull_score']}/5 | "
            f"Bear: {result['bear_score']}/5 | "
            f"Blow-off: {result['blowoff_score']}/5"
        ),
        "=" * 68,
    ]

    lines.extend(
        format_checks(
            "BULL TREND CHECKLIST",
            result["bullish_checks"],
        )
    )

    lines.extend(
        format_checks(
            "BEAR TREND CHECKLIST",
            result["bearish_checks"],
        )
    )

    lines.extend(
        format_checks(
            "BLOW-OFF / EXHAUSTION CHECKLIST",
            result["blowoff_checks"],
        )
    )

    lines.extend(
        format_macro(
            snapshot["macro_analysis"]
        )
    )

    lines.extend(
        format_gap(
            "LEGUTÓBBI NDOG (NEW DAY OPENING GAP)",
            gaps["latest_ndog"],
            gaps["ndog_fill"],
            gaps["ndog_reaction"],
            config,
        )
    )

    lines.extend(
        format_gap(
            "AKTÍV / LEGUTÓBBI NWOG (NEW WEEK OPENING GAP)",
            gaps["latest_nwog"],
            gaps["nwog_fill"],
            gaps["nwog_reaction"],
            config,
        )
    )

    gap_context = gap_trade_context(
        snapshot,
        regime,
    )

    lines.extend(
        [
            "",
            "=" * 68,
            "NDOG / NWOG PIACI KONTEXTUS",
            "=" * 68,
        ]
    )

    for message in gap_context["messages"]:
        lines.append(f"- {message}")

    risk = calculate_gold_risk_plan(
        snapshot,
        regime,
        equity_usd,
        config,
    )

    lines.extend(
        [
            "",
            "=" * 68,
            "RISK / TRADE TERV",
            "=" * 68,
        ]
    )

    if not risk["trade_allowed"]:
        lines.extend(
            [
                "ÁLLAPOT: NINCS AUTOMATIKUS TRADE.",
                f"Ok: {risk['reason']}",
            ]
        )

        if "details" in risk:
            lines.append(
                f"Konkrét tiltó jelek: {risk['details']}"
            )

        if "next_setup" in risk:
            lines.append(
                f"Következő keresendő setup: {risk['next_setup']}"
            )

        lines.extend(
            [
                "",
                "Gyakorlati értelmezés:",
                "- Neutral: várj VWAP reclaimre vagy failed VWAP reclaimre.",
                "- Bull trend: pullback, VWAP-retest vagy breakout-retest után keresd a longot.",
                "- Bear trend: extrém RSI-nél vagy nagy VWAP-extensionnél ne shortold a mélypontot.",
                "- Blow-off: ne üldözd az extrém emelkedést új long belépéssel.",
                "- DXY/reálhozam: konfirmáció vagy konfliktusfilter, nem önálló trigger.",
                "- NDOG/NWOG: mindig 1m/5m price actionnel együtt használd.",
            ]
        )

    else:
        lines.extend(
            [
                f"Javasolt irány:          {risk['side']}",
                f"Elméleti entry:          {risk['entry']:.2f}",
                f"ATR-alapú stop:          {risk['stop_price']:.2f}",
                f"1. célár:                {risk['target_1']:.2f}",
                f"2. célár:                {risk['target_2']:.2f}",

                f"Stop-távolság:           {risk['stop_distance_usd_per_oz']:.2f} USD/oz",
                f"Risk budget:             ${risk['risk_budget_usd']:,.2f}",

                f"GC risk / kontrakt:      ${risk['gc_risk_per_contract_usd']:,.2f}",
                f"MGC risk / kontrakt:     ${risk['mgc_risk_per_contract_usd']:,.2f}",

                f"Max GC kontraktus:       {risk['max_gc_contracts']}",
                f"Max MGC kontraktus:      {risk['max_mgc_contracts']}",

                "",
                f"Figyelmeztetés: {risk['warning']}",
            ]
        )

    lines.extend(
        [
            "",
            "=" * 68,
            "FONTOS: Elemző / paper-trading rendszer.",
            "Nem küld IBKR ordert és nem kezel valós pozíciót.",
            "A DXY, real yield, NDOG/NWOG és indikátorok nem önálló buy/sell jelek.",
            "A GC price action marad az elsődleges döntési réteg.",
            "=" * 68,
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    print(
        "\n"
        "gc_regime_1_0_10.py betöltődött.\n"
        "\n"
        "A runner importját így állítsd:\n"
        "from gc_regime_1_0_10 import run_gold_checklist\n"
    )