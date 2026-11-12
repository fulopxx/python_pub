from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


# ===================================================================
# 1. KONFIGURÁCIÓ
# ===================================================================

class GoldRegime(str, Enum):
    NEUTRAL = "NEUTRAL / NINCS EGYÉRTELMŰ REZHIM"
    BULL_TREND = "BULL TREND / LONG BIAS"
    BEAR_LIQUIDATION = "BEAR LIQUIDATION / SHORT BIAS"
    BLOWOFF_RISK = "BLOW-OFF / EXHAUSTION KOCKÁZAT"


@dataclass(frozen=True)
class GoldConfig:
    # ---------------------------------------------------------------
    # Időzóna és session
    # ---------------------------------------------------------------
    new_york_timezone: str = "America/New_York"

    # GC/COMEX futures napi Globex session:
    # 17:00 ET környékén break, 18:00 ET környékén új session.
    daily_close_hour_et: int = 17
    daily_open_hour_et: int = 18

    # Gap close keresési ablak:
    # 16:45-16:59 ET között keressük az utolsó bar záróárát.
    gap_close_window_start_min: int = 15

    # Gap open keresési ablak:
    # 18:00-18:20 ET között keressük az első bar nyitóárát.
    gap_open_window_end_min: int = 20

    # ---------------------------------------------------------------
    # Trend / rezhim küszöbök
    # ---------------------------------------------------------------
    adx_trend_min: float = 25.0
    adx_strong_min: float = 30.0

    # ---------------------------------------------------------------
    # RSI küszöbök
    # ---------------------------------------------------------------
    rsi2_blowoff: float = 95.0
    rsi5_blowoff: float = 80.0

    # Bearish esésben ezek alatt TILOS új shortot üldözni.
    rsi2_extreme_oversold: float = 10.0
    rsi5_extreme_oversold: float = 20.0

    # Bull trendben ezek felett TILOS új longot üldözni.
    rsi2_extreme_overbought: float = 95.0
    rsi5_extreme_overbought: float = 80.0

    # ---------------------------------------------------------------
    # Volumen küszöbök
    # ---------------------------------------------------------------
    volume_expansion_ratio: float = 1.30
    volume_climax_ratio: float = 2.00
    volume_dry_ratio: float = 0.70

    # ---------------------------------------------------------------
    # VWAP / ATR szabályok
    # ---------------------------------------------------------------
    vwap_extension_atr: float = 2.5

    # Ennél messzebb a VWAP alatt új short entry tiltott.
    max_short_extension_below_vwap_atr: float = -3.0

    # Ennél messzebb a VWAP fölött új long entry tiltott.
    max_long_extension_above_vwap_atr: float = 3.0

    # Gap szint közelinek számít, ha ezen belül van az ár.
    gap_level_proximity_atr: float = 0.50

    # Reclaim/rejection megerősítéshez szükséges minimum távolság.
    gap_reaction_confirmation_atr: float = 0.25

    # Ennyi utolsó 1m gyertyán nézzük a gap-reactiont.
    gap_reaction_lookback_bars: int = 10

    # Ha egy gap teljesen kitöltött és az ár ennél távolabb van,
    # már csak történeti referenciaként kezeljük.
    stale_gap_distance_atr: float = 3.0

    # ---------------------------------------------------------------
    # Stop / target
    # ---------------------------------------------------------------
    stop_atr_multiple: float = 1.0
    target1_atr_multiple: float = 1.0
    target2_atr_multiple: float = 2.0

    # ---------------------------------------------------------------
    # Risk
    # ---------------------------------------------------------------
    risk_per_trade_pct: float = 0.0025

    # Standard Gold Futures: GC = 100 troy ounce
    gc_multiplier_oz: float = 100.0

    # Micro Gold Futures: MGC = 10 troy ounce
    mgc_multiplier_oz: float = 10.0


CONFIG = GoldConfig()


# ===================================================================
# 2. INDIKÁTOROK
# ===================================================================

def calc_rsi(
    close: pd.Series,
    period: int,
) -> pd.Series:
    """
    Exponenciális / Wilder-szerű RSI.
    """
    delta = close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    rs = average_gain / average_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50.0)


def calc_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    ATR.

    GC esetén az ATR értéke USD / troy ounce egységben értendő.
    """
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
    """
    ADX, +DI és -DI számítás.
    """
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
        * plus_dm.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()
        / atr.replace(0, np.nan)
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()
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
    """
    1 perces OHLCV adatból például 5 perces OHLCV adatot készít.
    """
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
# 3. ADATELLENŐRZÉS ÉS IDŐZÓNA KEZELÉS
# ===================================================================

def validate_ohlcv(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Elvárt oszlopok:
    open, high, low, close, volume.

    Az indexnek pandas DatetimeIndex típusúnak kell lennie.
    """
    required_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Hiányzó oszlopok: {missing_columns}. "
            f"Kötelező oszlopok: {required_columns}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "A DataFrame indexének pandas DatetimeIndex típusúnak kell lennie. "
            "Példa: df = df.set_index('date')"
        )

    clean_df = df.copy()
    clean_df = clean_df.sort_index()
    clean_df = clean_df[required_columns].dropna()

    if len(clean_df) < 100:
        raise ValueError(
            f"Túl kevés adat: {len(clean_df)} bar. "
            "Legalább 100 db 1 perces GC bar szükséges."
        )

    return clean_df


def to_new_york_time(
    df: pd.DataFrame,
    config: GoldConfig = CONFIG,
) -> pd.DataFrame:
    """
    Az indexet America/New_York időzónára konvertálja.

    Timezone-naive index esetén UTC-ként értelmezi azt.
    """
    result = df.copy()

    if result.index.tz is None:
        result.index = result.index.tz_localize("UTC")

    result.index = result.index.tz_convert(
        config.new_york_timezone
    )

    return result


def session_key(
    timestamp: pd.Timestamp,
) -> pd.Timestamp:
    """
    GC session kulcs.

    A futures session 18:00 ET körül indul.
    Ezért a session_key a timestamp - 18 óra naptári dátuma.

    Példa:
    - Vasárnap 18:00 ET -> vasárnapi session.
    - Hétfő 10:00 ET -> még mindig vasárnapi session.
    - Hétfő 18:00 ET -> hétfői session.
    """
    return (
        timestamp
        - pd.Timedelta(hours=18)
    ).normalize()


def add_session_key(
    df_ny: pd.DataFrame,
) -> pd.DataFrame:
    """
    Session key oszlop hozzáadása.
    """
    result = df_ny.copy()

    result["session_key"] = [
        session_key(timestamp)
        for timestamp in result.index
    ]

    return result


def calc_session_vwap(
    df_ny: pd.DataFrame,
) -> pd.Series:
    """
    Session-anchored VWAP.

    A VWAP nem helyi naptári napváltással resetelődik,
    hanem a New York-i 18:00 ET futures session kezdete szerint.
    """
    if "session_key" not in df_ny.columns:
        df_ny = add_session_key(df_ny)

    typical_price = (
        df_ny["high"]
        + df_ny["low"]
        + df_ny["close"]
    ) / 3.0

    cumulative_price_volume = (
        typical_price * df_ny["volume"]
    ).groupby(
        df_ny["session_key"]
    ).cumsum()

    cumulative_volume = (
        df_ny["volume"]
    ).groupby(
        df_ny["session_key"]
    ).cumsum()

    return cumulative_price_volume / cumulative_volume.replace(
        0,
        np.nan,
    )


# ===================================================================
# 4. SESSION HIGH / LOW
# ===================================================================

def previous_session_levels(
    df_ny: pd.DataFrame,
) -> tuple[float, float]:
    """
    Előző futures session high és low.
    """
    if "session_key" not in df_ny.columns:
        df_ny = add_session_key(df_ny)

    sessions = pd.Index(
        df_ny["session_key"]
        .dropna()
        .unique()
    ).sort_values()

    if len(sessions) < 2:
        return np.nan, np.nan

    previous_session = sessions[-2]

    previous_df = df_ny[
        df_ny["session_key"] == previous_session
    ]

    return (
        float(previous_df["high"].max()),
        float(previous_df["low"].min()),
    )


def current_session_levels(
    df_ny: pd.DataFrame,
) -> tuple[float, float]:
    """
    Aktuális futures session high és low.
    """
    if "session_key" not in df_ny.columns:
        df_ny = add_session_key(df_ny)

    current_session = df_ny["session_key"].iloc[-1]

    current_df = df_ny[
        df_ny["session_key"] == current_session
    ]

    return (
        float(current_df["high"].max()),
        float(current_df["low"].min()),
    )


# ===================================================================
# 5. NDOG / NWOG GAP SEGÉDFÜGGVÉNYEK
# ===================================================================

def get_last_close_before_daily_break(
    df_ny: pd.DataFrame,
    target_date: pd.Timestamp,
    config: GoldConfig = CONFIG,
) -> tuple[float | None, pd.Timestamp | None]:
    """
    Megkeresi a 17:00 ET break előtti utolsó close értéket.

    Keresési ablak:
    16:45 - 16:59 ET.
    """
    date_only = target_date.normalize()

    start = (
        date_only
        + pd.Timedelta(
            hours=config.daily_close_hour_et,
            minutes=-config.gap_close_window_start_min,
        )
    )

    end = (
        date_only
        + pd.Timedelta(
            hours=config.daily_close_hour_et,
            minutes=-1,
        )
    )

    window = df_ny.loc[
        (df_ny.index >= start)
        & (df_ny.index <= end)
    ]

    if window.empty:
        return None, None

    return (
        float(window["close"].iloc[-1]),
        window.index[-1],
    )


def get_first_open_after_daily_break(
    df_ny: pd.DataFrame,
    target_date: pd.Timestamp,
    config: GoldConfig = CONFIG,
) -> tuple[float | None, pd.Timestamp | None]:
    """
    Megkeresi a 18:00 ET utáni első open értéket.

    Keresési ablak:
    18:00 - 18:20 ET.
    """
    date_only = target_date.normalize()

    start = (
        date_only
        + pd.Timedelta(
            hours=config.daily_open_hour_et,
        )
    )

    end = (
        date_only
        + pd.Timedelta(
            hours=config.daily_open_hour_et,
            minutes=config.gap_open_window_end_min,
        )
    )

    window = df_ny.loc[
        (df_ny.index >= start)
        & (df_ny.index <= end)
    ]

    if window.empty:
        return None, None

    return (
        float(window["open"].iloc[0]),
        window.index[0],
    )


def make_gap_record(
    gap_type: str,
    close_price: float,
    open_price: float,
    close_timestamp: pd.Timestamp,
    open_timestamp: pd.Timestamp,
) -> dict[str, Any]:
    """
    Egységes gap rekord NDOG és NWOG számára.
    """
    gap_low = min(close_price, open_price)
    gap_high = max(close_price, open_price)
    gap_ce = (gap_low + gap_high) / 2.0

    if open_price > close_price:
        direction = "UP_GAP"

    elif open_price < close_price:
        direction = "DOWN_GAP"

    else:
        direction = "FLAT"

    return {
        "type": gap_type,

        "close_price": close_price,
        "open_price": open_price,

        "gap_low": gap_low,
        "gap_high": gap_high,
        "gap_ce": gap_ce,
        "gap_size": gap_high - gap_low,

        "direction": direction,

        "close_timestamp": close_timestamp,
        "open_timestamp": open_timestamp,
    }


# ===================================================================
# 6. NDOG / NWOG SZÁMÍTÁS
# ===================================================================

def calculate_ndogs(
    df_ny: pd.DataFrame,
    config: GoldConfig = CONFIG,
) -> list[dict[str, Any]]:
    """
    NDOG-ok számítása.

    NDOG:
    hétfő-csütörtök 17:00 ET körüli zárás
    és ugyanazon nap 18:00 ET utáni nyitás között.

    Péntek close -> vasárnap open már NWOG, ezért a péntek kimarad.
    """
    normalized_dates = pd.Index(
        df_ny.index.normalize().unique()
    ).sort_values()

    ndogs: list[dict[str, Any]] = []

    for date_value in normalized_dates:
        weekday = date_value.weekday()

        # Monday=0 ... Friday=4.
        # Pénteket és hétvégét kihagyjuk.
        if weekday >= 4:
            continue

        close_price, close_timestamp = get_last_close_before_daily_break(
            df_ny=df_ny,
            target_date=date_value,
            config=config,
        )

        open_price, open_timestamp = get_first_open_after_daily_break(
            df_ny=df_ny,
            target_date=date_value,
            config=config,
        )

        if (
            close_price is None
            or open_price is None
            or close_timestamp is None
            or open_timestamp is None
        ):
            continue

        ndogs.append(
            make_gap_record(
                gap_type="NDOG",
                close_price=close_price,
                open_price=open_price,
                close_timestamp=close_timestamp,
                open_timestamp=open_timestamp,
            )
        )

    return ndogs


def calculate_nwogs(
    df_ny: pd.DataFrame,
    config: GoldConfig = CONFIG,
) -> list[dict[str, Any]]:
    """
    NWOG-ok számítása.

    NWOG:
    - péntek 17:00 ET előtti utolsó close;
    - vasárnap 18:00 ET utáni első open.
    """
    normalized_dates = pd.Index(
        df_ny.index.normalize().unique()
    ).sort_values()

    nwogs: list[dict[str, Any]] = []

    for sunday_date in normalized_dates:
        if sunday_date.weekday() != 6:
            continue

        friday_date = sunday_date - pd.Timedelta(days=2)

        close_price, close_timestamp = get_last_close_before_daily_break(
            df_ny=df_ny,
            target_date=friday_date,
            config=config,
        )

        open_price, open_timestamp = get_first_open_after_daily_break(
            df_ny=df_ny,
            target_date=sunday_date,
            config=config,
        )

        if (
            close_price is None
            or open_price is None
            or close_timestamp is None
            or open_timestamp is None
        ):
            continue

        nwogs.append(
            make_gap_record(
                gap_type="NWOG",
                close_price=close_price,
                open_price=open_price,
                close_timestamp=close_timestamp,
                open_timestamp=open_timestamp,
            )
        )

    return nwogs


def select_latest_gap(
    gaps: list[dict[str, Any]],
    current_timestamp: pd.Timestamp,
) -> dict[str, Any] | None:
    """
    A jelenlegi időpontnál korábban létrejött legutóbbi gapet adja vissza.

    Az NDOG-nál ez nem feltétlenül "mai aktív" gap:
    vasárnap esti / hétfő hajnali sessionben például a legutóbbi
    normál NDOG sokszor csütörtöki lehet, mert péntek-vasárnap NWOG.
    """
    valid_gaps = [
        gap
        for gap in gaps
        if gap["open_timestamp"] <= current_timestamp
    ]

    if not valid_gaps:
        return None

    return max(
        valid_gaps,
        key=lambda gap: gap["open_timestamp"],
    )


def evaluate_gap_fill_status(
    df_ny: pd.DataFrame,
    gap: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Gap fill státusz.

    Meghatározza:
    - ár a gap fölött/alatt/benne;
    - CE érintve;
    - low/high érintve;
    - gap teljesen kitöltve.
    """
    if gap is None:
        return {
            "available": False,
            "status": "N/A",
        }

    after_open = df_ny.loc[
        df_ny.index >= gap["open_timestamp"]
    ]

    if after_open.empty:
        return {
            "available": False,
            "status": "N/A",
        }

    current_price = float(
        after_open["close"].iloc[-1]
    )

    gap_low = gap["gap_low"]
    gap_high = gap["gap_high"]
    gap_ce = gap["gap_ce"]

    touched_ce = (
        (after_open["low"] <= gap_ce)
        & (after_open["high"] >= gap_ce)
    ).any()

    touched_low = (
        (after_open["low"] <= gap_low)
        & (after_open["high"] >= gap_low)
    ).any()

    touched_high = (
        (after_open["low"] <= gap_high)
        & (after_open["high"] >= gap_high)
    ).any()

    if gap["direction"] == "UP_GAP":
        fully_filled = (
            after_open["low"] <= gap_low
        ).any()

    elif gap["direction"] == "DOWN_GAP":
        fully_filled = (
            after_open["high"] >= gap_high
        ).any()

    else:
        fully_filled = True

    if current_price > gap_high:
        price_location = "A GAP FELETT"

    elif current_price < gap_low:
        price_location = "A GAP ALATT"

    else:
        price_location = "A GAP TARTOMÁNYÁBAN"

    return {
        "available": True,

        "price_location": price_location,

        "touched_ce": bool(touched_ce),
        "touched_low": bool(touched_low),
        "touched_high": bool(touched_high),

        "fully_filled": bool(fully_filled),

        "current_price": current_price,
    }


def evaluate_gap_reaction(
    df_ny: pd.DataFrame,
    gap: dict[str, Any] | None,
    atr_1m: float,
    config: GoldConfig = CONFIG,
) -> dict[str, Any]:
    """
    Gap reakció vizsgálat az utolsó N darab 1 perces gyertyán.

    Lehetséges értékek:
    - BULLISH_RECLAIM
    - BEARISH_REJECTION
    - NEAR_LEVEL
    - NO_RECENT_INTERACTION
    """
    if gap is None:
        return {
            "available": False,
            "reaction": "N/A",
            "closest_level_name": None,
            "closest_level_price": np.nan,
            "distance_atr": np.nan,
            "touched_level": False,
        }

    if not np.isfinite(atr_1m) or atr_1m <= 0:
        return {
            "available": False,
            "reaction": "N/A",
            "closest_level_name": None,
            "closest_level_price": np.nan,
            "distance_atr": np.nan,
            "touched_level": False,
        }

    recent = df_ny.tail(
        config.gap_reaction_lookback_bars
    )

    if recent.empty:
        return {
            "available": False,
            "reaction": "N/A",
            "closest_level_name": None,
            "closest_level_price": np.nan,
            "distance_atr": np.nan,
            "touched_level": False,
        }

    current_close = float(
        recent["close"].iloc[-1]
    )

    levels = {
        "GAP_LOW": gap["gap_low"],
        "GAP_CE": gap["gap_ce"],
        "GAP_HIGH": gap["gap_high"],
    }

    closest_level_name = min(
        levels,
        key=lambda name: abs(
            current_close - levels[name]
        ),
    )

    closest_level_price = levels[
        closest_level_name
    ]

    distance_atr = (
        current_close - closest_level_price
    ) / atr_1m

    touched_level = (
        (recent["low"] <= closest_level_price)
        & (recent["high"] >= closest_level_price)
    ).any()

    confirmation_distance = (
        config.gap_reaction_confirmation_atr
        * atr_1m
    )

    proximity_distance = (
        config.gap_level_proximity_atr
        * atr_1m
    )

    if touched_level and (
        current_close
        > closest_level_price + confirmation_distance
    ):
        reaction = "BULLISH_RECLAIM"

    elif touched_level and (
        current_close
        < closest_level_price - confirmation_distance
    ):
        reaction = "BEARISH_REJECTION"

    elif abs(
        current_close - closest_level_price
    ) <= proximity_distance:
        reaction = "NEAR_LEVEL"

    else:
        reaction = "NO_RECENT_INTERACTION"

    return {
        "available": True,

        "reaction": reaction,

        "closest_level_name": closest_level_name,
        "closest_level_price": closest_level_price,

        "distance_atr": distance_atr,

        "touched_level": bool(touched_level),
    }


def build_gap_context(
    df_ny: pd.DataFrame,
    atr_1m: float,
    config: GoldConfig = CONFIG,
) -> dict[str, Any]:
    """
    NDOG/NWOG adatok, fill státuszok és reakciók egyben.
    """
    ndogs = calculate_ndogs(
        df_ny=df_ny,
        config=config,
    )

    nwogs = calculate_nwogs(
        df_ny=df_ny,
        config=config,
    )

    current_timestamp = df_ny.index[-1]

    latest_ndog = select_latest_gap(
        gaps=ndogs,
        current_timestamp=current_timestamp,
    )

    latest_nwog = select_latest_gap(
        gaps=nwogs,
        current_timestamp=current_timestamp,
    )

    ndog_fill = evaluate_gap_fill_status(
        df_ny=df_ny,
        gap=latest_ndog,
    )

    nwog_fill = evaluate_gap_fill_status(
        df_ny=df_ny,
        gap=latest_nwog,
    )

    ndog_reaction = evaluate_gap_reaction(
        df_ny=df_ny,
        gap=latest_ndog,
        atr_1m=atr_1m,
        config=config,
    )

    nwog_reaction = evaluate_gap_reaction(
        df_ny=df_ny,
        gap=latest_nwog,
        atr_1m=atr_1m,
        config=config,
    )

    return {
        "latest_ndog": latest_ndog,
        "latest_nwog": latest_nwog,

        "ndog_fill": ndog_fill,
        "nwog_fill": nwog_fill,

        "ndog_reaction": ndog_reaction,
        "nwog_reaction": nwog_reaction,

        "ndog_count": len(ndogs),
        "nwog_count": len(nwogs),
    }


# ===================================================================
# 7. GOLD SNAPSHOT
# ===================================================================

def build_gold_snapshot(
    df_1m: pd.DataFrame,
    config: GoldConfig = CONFIG,
) -> dict[str, Any]:
    """
    GC/MGC 1 perces OHLCV DataFrame-ből készít market snapshotot.
    """
    df = validate_ohlcv(df_1m)

    df_ny = to_new_york_time(
        df=df,
        config=config,
    )

    df_ny = add_session_key(df_ny)

    df_5m = resample_ohlcv(
        df_1m=df_ny[
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ],
        rule="5min",
    )

    if len(df_5m) < 30:
        raise ValueError(
            "Túl kevés 5 perces bar az ADX és ATR számításhoz."
        )

    df_ny["vwap"] = calc_session_vwap(df_ny)

    atr_1m_series = calc_atr(
        high=df_ny["high"],
        low=df_ny["low"],
        close=df_ny["close"],
        period=14,
    )

    rsi_2_series = calc_rsi(
        close=df_ny["close"],
        period=2,
    )

    rsi_5_series = calc_rsi(
        close=df_ny["close"],
        period=5,
    )

    atr_5m_series = calc_atr(
        high=df_5m["high"],
        low=df_5m["low"],
        close=df_5m["close"],
        period=14,
    )

    adx_5m_df = calc_adx(
        high=df_5m["high"],
        low=df_5m["low"],
        close=df_5m["close"],
        period=14,
    )

    current_close = float(
        df_ny["close"].iloc[-1]
    )

    current_vwap = float(
        df_ny["vwap"].iloc[-1]
    )

    current_atr_1m = float(
        atr_1m_series.iloc[-1]
    )

    current_atr_5m = float(
        atr_5m_series.iloc[-1]
    )

    current_volume = float(
        df_ny["volume"].iloc[-1]
    )

    volume_average_20 = float(
        df_ny["volume"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    volume_ratio = (
        current_volume / volume_average_20
        if volume_average_20 > 0
        else np.nan
    )

    vwap_distance_atr = (
        (current_close - current_vwap)
        / current_atr_1m
        if current_atr_1m > 0
        else np.nan
    )

    prior_session_high, prior_session_low = (
        previous_session_levels(df_ny)
    )

    current_session_high, current_session_low = (
        current_session_levels(df_ny)
    )

    gap_context = build_gap_context(
        df_ny=df_ny,
        atr_1m=current_atr_1m,
        config=config,
    )

    return {
        "timestamp": df_ny.index[-1],

        "close": current_close,
        "vwap": current_vwap,

        "atr_1m": current_atr_1m,
        "atr_5m": current_atr_5m,

        "atr_1m_pct": (
            100 * current_atr_1m / current_close
            if current_close > 0
            else np.nan
        ),

        "rsi_2": float(
            rsi_2_series.iloc[-1]
        ),

        "rsi_5": float(
            rsi_5_series.iloc[-1]
        ),

        "adx_5m": float(
            adx_5m_df["adx"].iloc[-1]
        ),

        "plus_di_5m": float(
            adx_5m_df["plus_di"].iloc[-1]
        ),

        "minus_di_5m": float(
            adx_5m_df["minus_di"].iloc[-1]
        ),

        "volume": current_volume,
        "volume_avg_20": volume_average_20,
        "volume_ratio": volume_ratio,

        "vwap_distance_atr": vwap_distance_atr,

        "prior_session_high": prior_session_high,
        "prior_session_low": prior_session_low,

        "current_session_high": current_session_high,
        "current_session_low": current_session_low,

        "gap_context": gap_context,
    }


# ===================================================================
# 8. REZHIM DETEKTOR
# ===================================================================

def gold_regime_active(
    snapshot: dict[str, Any],
    config: GoldConfig = CONFIG,
) -> dict[str, Any]:
    """
    Bull / bear / blow-off / neutral rezhimdetektor.
    """
    close = snapshot["close"]
    vwap = snapshot["vwap"]

    adx = snapshot["adx_5m"]
    plus_di = snapshot["plus_di_5m"]
    minus_di = snapshot["minus_di_5m"]

    volume_ratio = snapshot["volume_ratio"]

    rsi_2 = snapshot["rsi_2"]
    rsi_5 = snapshot["rsi_5"]

    vwap_distance_atr = snapshot["vwap_distance_atr"]

    bullish_checks = {
        "Ár a session VWAP felett": close > vwap,

        f"ADX(5m) > {config.adx_trend_min:.0f}": (
            adx > config.adx_trend_min
        ),

        "+DI dominál -DI felett": (
            plus_di > minus_di
        ),

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

        "-DI dominál +DI felett": (
            minus_di > plus_di
        ),

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
            vwap_distance_atr > config.vwap_extension_atr
        ),

        f"Volume >= {config.volume_climax_ratio:.1f}× 20-bar átlag": (
            volume_ratio >= config.volume_climax_ratio
        ),

        "+DI dominál -DI felett": (
            plus_di > minus_di
        ),
    }

    bull_score = sum(bullish_checks.values())
    bear_score = sum(bearish_checks.values())
    blowoff_score = sum(blowoff_checks.values())

    if blowoff_score >= 4:
        regime = GoldRegime.BLOWOFF_RISK

    elif bear_score >= 4:
        regime = GoldRegime.BEAR_LIQUIDATION

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
# 9. GAP-ALAPÚ PIACI KONTEXTUS
# ===================================================================

def gap_trade_context(
    snapshot: dict[str, Any],
    regime: GoldRegime,
) -> dict[str, Any]:
    """
    Az NDOG/NWOG szintekhez kapcsolódó értelmező szövegek.

    Nem generál önálló entry-t.
    """
    context = snapshot["gap_context"]

    ndog_reaction = context["ndog_reaction"]
    nwog_reaction = context["nwog_reaction"]

    messages: list[str] = []

    if nwog_reaction["reaction"] == "BULLISH_RECLAIM":
        messages.append(
            "NWOG bullish reclaim: a szintet érintette, majd fölé zárt."
        )

    elif nwog_reaction["reaction"] == "BEARISH_REJECTION":
        messages.append(
            "NWOG bearish rejection: a szintet érintette, majd alá zárt."
        )

    elif nwog_reaction["reaction"] == "NEAR_LEVEL":
        messages.append(
            "Az ár aktív NWOG szint közelében van; várj 1m/5m megerősítésre."
        )

    if ndog_reaction["reaction"] == "BULLISH_RECLAIM":
        messages.append(
            "NDOG bullish reclaim: az ár visszafoglalta a közeli NDOG szintet."
        )

    elif ndog_reaction["reaction"] == "BEARISH_REJECTION":
        messages.append(
            "NDOG bearish rejection: az ár nem tudott a közeli NDOG szint fölött maradni."
        )

    elif ndog_reaction["reaction"] == "NEAR_LEVEL":
        messages.append(
            "Az ár az utolsó NDOG szint közelében van; ne lépj be megerősítés nélkül."
        )

    if regime == GoldRegime.BEAR_LIQUIDATION:
        if (
            nwog_reaction["reaction"] == "BULLISH_RECLAIM"
            or ndog_reaction["reaction"] == "BULLISH_RECLAIM"
        ):
            messages.append(
                "Bearish rezhimben bullish gap reclaim van: "
                "a short chasing kockázata nőtt."
            )

    if regime == GoldRegime.BULL_TREND:
        if (
            nwog_reaction["reaction"] == "BEARISH_REJECTION"
            or ndog_reaction["reaction"] == "BEARISH_REJECTION"
        ):
            messages.append(
                "Bullish rezhimben bearish gap rejection van: "
                "új long belépés előtt várj további megerősítésre."
            )

    if not messages:
        messages.append(
            "Nincs közeli, megerősített NDOG/NWOG reakció. "
            "A gapek továbbra is referencia- és célzónák."
        )

    return {
        "messages": messages,
    }


# ===================================================================
# 10. RISK / TRADE TERV ÉS ANTI-CHASING
# ===================================================================

def calculate_gold_risk_plan(
    snapshot: dict[str, Any],
    regime: GoldRegime,
    equity_usd: float,
    config: GoldConfig = CONFIG,
) -> dict[str, Any]:
    """
    Elméleti GC/MGC kockázati terv.

    Ez nem küld ordert.
    """
    entry = snapshot["close"]
    atr_5m = snapshot["atr_5m"]

    rsi_2 = snapshot["rsi_2"]
    rsi_5 = snapshot["rsi_5"]

    vwap_distance_atr = snapshot["vwap_distance_atr"]

    gap_context = snapshot["gap_context"]

    ndog_reaction = gap_context["ndog_reaction"]["reaction"]
    nwog_reaction = gap_context["nwog_reaction"]["reaction"]

    if not np.isfinite(atr_5m) or atr_5m <= 0:
        raise ValueError(
            "Érvénytelen ATR(5m); nem lehet kockázati tervet készíteni."
        )

    # ---------------------------------------------------------------
    # SHORT CHASING TILTÁS
    # ---------------------------------------------------------------
    extreme_oversold = (
        rsi_2 < config.rsi2_extreme_oversold
        or rsi_5 < config.rsi5_extreme_oversold
    )

    too_far_below_vwap = (
        np.isfinite(vwap_distance_atr)
        and vwap_distance_atr
        < config.max_short_extension_below_vwap_atr
    )

    bullish_gap_reclaim = (
        ndog_reaction == "BULLISH_RECLAIM"
        or nwog_reaction == "BULLISH_RECLAIM"
    )

    if regime == GoldRegime.BEAR_LIQUIDATION and (
        extreme_oversold
        or too_far_below_vwap
        or bullish_gap_reclaim
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
                f"ár {vwap_distance_atr:.2f}× ATR-rel a VWAP alatt"
            )

        if bullish_gap_reclaim:
            blocking_reasons.append(
                "bullish NDOG/NWOG reclaim"
            )

        return {
            "trade_allowed": False,

            "reason": (
                "BEARISH rezhim van, de új SHORT belépés TILTVA. "
                "A mozgás lefelé túlnyúlt vagy bullish reclaim alakult ki."
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
        np.isfinite(vwap_distance_atr)
        and vwap_distance_atr
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
                f"ár {vwap_distance_atr:.2f}× ATR-rel a VWAP fölött"
            )

        if bearish_gap_rejection:
            blocking_reasons.append(
                "bearish NDOG/NWOG rejection"
            )

        return {
            "trade_allowed": False,

            "reason": (
                "BULLISH rezhim van, de új LONG belépés TILTVA. "
                "A piac túlzottan megnyúlt felfelé vagy bearish gap rejection alakult ki."
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
    # TISZTA BULL / BEAR TREND: ELVI TRADE TERV
    # ---------------------------------------------------------------
    risk_budget_usd = (
        equity_usd * config.risk_per_trade_pct
    )

    stop_distance_usd_per_oz = (
        atr_5m * config.stop_atr_multiple
    )

    if regime == GoldRegime.BULL_TREND:
        side = "LONG"

        stop_price = entry - stop_distance_usd_per_oz

        target_1 = entry + (
            atr_5m * config.target1_atr_multiple
        )

        target_2 = entry + (
            atr_5m * config.target2_atr_multiple
        )

    elif regime == GoldRegime.BEAR_LIQUIDATION:
        side = "SHORT"

        stop_price = entry + stop_distance_usd_per_oz

        target_1 = entry - (
            atr_5m * config.target1_atr_multiple
        )

        target_2 = entry - (
            atr_5m * config.target2_atr_multiple
        )

    else:
        return {
            "trade_allowed": False,
            "reason": "Ismeretlen vagy nem támogatott rezhim.",
        }

    gc_risk_per_contract_usd = (
        stop_distance_usd_per_oz
        * config.gc_multiplier_oz
    )

    mgc_risk_per_contract_usd = (
        stop_distance_usd_per_oz
        * config.mgc_multiplier_oz
    )

    max_gc_contracts = int(
        risk_budget_usd // gc_risk_per_contract_usd
    )

    max_mgc_contracts = int(
        risk_budget_usd // mgc_risk_per_contract_usd
    )

    return {
        "trade_allowed": True,

        "side": side,

        "entry": round(entry, 2),
        "stop_price": round(stop_price, 2),

        "target_1": round(target_1, 2),
        "target_2": round(target_2, 2),

        "atr_5m": round(atr_5m, 2),

        "stop_distance_usd_per_oz": round(
            stop_distance_usd_per_oz,
            2,
        ),

        "risk_budget_usd": round(
            risk_budget_usd,
            2,
        ),

        "gc_risk_per_contract_usd": round(
            gc_risk_per_contract_usd,
            2,
        ),

        "mgc_risk_per_contract_usd": round(
            mgc_risk_per_contract_usd,
            2,
        ),

        "max_gc_contracts": max_gc_contracts,
        "max_mgc_contracts": max_mgc_contracts,

        "warning": (
            "A kontraktusszám elméleti maximum. "
            "Margin, makroadat, likviditás, spread, slippage és "
            "napi veszteséglimit nincs benne."
        ),
    }


# ===================================================================
# 11. REPORT SEGÉDFÜGGVÉNYEK
# ===================================================================

def format_checks(
    title: str,
    checks: dict[str, bool],
) -> list[str]:
    """
    Bool checklistből formázott riportblokk.
    """
    lines = [
        "",
        "-" * 68,
        title,
        "-" * 68,
    ]

    for name, passed in checks.items():
        status = "✅ IGEN" if passed else "❌ NEM"
        lines.append(f"[{status}] {name}")

    return lines


def format_gap_block(
    title: str,
    gap: dict[str, Any] | None,
    fill_status: dict[str, Any],
    reaction: dict[str, Any],
    config: GoldConfig = CONFIG,
) -> list[str]:
    """
    NDOG/NWOG részletes szöveges blokk.

    Tartalmazza a javítást:
    Ha a gap teljesen kitöltött ÉS több mint 3 ATR-re van,
    akkor történeti referenciaként jelöli, nem közvetlen trade szintként.
    """
    lines = [
        "",
        "-" * 68,
        title,
        "-" * 68,
    ]

    if gap is None:
        lines.append(
            "Nincs számolható gap az aktuális letöltött adathalmazban."
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

    if fill_status.get("available", False):
        lines.extend(
            [
                f"Jelenlegi helyzet:      {fill_status['price_location']}",

                f"CE érintve:             "
                f"{'IGEN' if fill_status['touched_ce'] else 'NEM'}",

                f"Gap Low érintve:        "
                f"{'IGEN' if fill_status['touched_low'] else 'NEM'}",

                f"Gap High érintve:       "
                f"{'IGEN' if fill_status['touched_high'] else 'NEM'}",

                f"Teljes gap fill:        "
                f"{'IGEN' if fill_status['fully_filled'] else 'NEM'}",
            ]
        )

    if reaction.get("available", False):
        lines.extend(
            [
                f"Legközelebbi szint:     "
                f"{reaction['closest_level_name']}",

                f"Szint ára:              "
                f"{reaction['closest_level_price']:.2f}",

                f"Távolság:               "
                f"{reaction['distance_atr']:.2f} × ATR(1m)",

                f"Reakció:                "
                f"{reaction['reaction']}",
            ]
        )

        # -----------------------------------------------------------
        # JAVÍTÁS: stale / régi gap felismerés
        # -----------------------------------------------------------
        if (
            fill_status.get("fully_filled", False)
            and abs(reaction.get("distance_atr", 0))
            > config.stale_gap_distance_atr
        ):
            lines.append(
                "Gap státusz:            KORÁBBI / MÁR KITÖLTÖTT REFERENCIA, "
                "JELENLEG NEM KÖZVETLEN TRADE SZINT"
            )

    return lines


# ===================================================================
# 12. TELJES CHECKLIST
# ===================================================================

def run_gold_checklist(
    df_1m: pd.DataFrame,
    equity_usd: float = 10_000.0,
    config: GoldConfig = CONFIG,
) -> str:
    """
    Fő futtató funkció.

    df_1m:
        1 perces GC/MGC OHLCV DataFrame.

    equity_usd:
        Csak az elméleti risk budget kiszámítására.
    """
    snapshot = build_gold_snapshot(
        df_1m=df_1m,
        config=config,
    )

    regime_result = gold_regime_active(
        snapshot=snapshot,
        config=config,
    )

    regime = regime_result["regime"]

    gap_context = snapshot["gap_context"]

    lines = [
        "",
        "=" * 68,
        "GC / GOLD REGIME + NDOG / NWOG CHECKLIST",
        "=" * 68,

        f"Időbélyeg:              {snapshot['timestamp']}",

        f"GC utolsó ár:            {snapshot['close']:.2f} USD/oz",
        f"Session VWAP:            {snapshot['vwap']:.2f}",

        f"VWAP-távolság:           "
        f"{snapshot['vwap_distance_atr']:.2f} × ATR(1m)",

        f"ATR(1m,14):              {snapshot['atr_1m']:.2f} USD/oz",
        f"ATR(5m,14):              {snapshot['atr_5m']:.2f} USD/oz",

        f"ATR(1m) relatív:         {snapshot['atr_1m_pct']:.4f}%",

        f"RSI(2):                  {snapshot['rsi_2']:.1f}",
        f"RSI(5):                  {snapshot['rsi_5']:.1f}",

        f"ADX(5m):                 {snapshot['adx_5m']:.1f}",

        f"+DI / -DI:               "
        f"{snapshot['plus_di_5m']:.1f} / "
        f"{snapshot['minus_di_5m']:.1f}",

        f"Utolsó 1m volumen:       {snapshot['volume']:,.0f}",
        f"20-bar átlagvolumen:     {snapshot['volume_avg_20']:,.0f}",
        f"Volume ratio:            {snapshot['volume_ratio']:.2f}×",

        (
            f"Előző session high:      "
            f"{snapshot['prior_session_high']:.2f}"
            if np.isfinite(snapshot["prior_session_high"])
            else "Előző session high:      N/A"
        ),

        (
            f"Előző session low:       "
            f"{snapshot['prior_session_low']:.2f}"
            if np.isfinite(snapshot["prior_session_low"])
            else "Előző session low:       N/A"
        ),

        (
            f"Aktuális session high:   "
            f"{snapshot['current_session_high']:.2f}"
            if np.isfinite(snapshot["current_session_high"])
            else "Aktuális session high:   N/A"
        ),

        (
            f"Aktuális session low:    "
            f"{snapshot['current_session_low']:.2f}"
            if np.isfinite(snapshot["current_session_low"])
            else "Aktuális session low:    N/A"
        ),

        "",
        "=" * 68,

        f"AKTUÁLIS REZHIM: {regime.value}",

        (
            f"Pontszámok — "
            f"Bull: {regime_result['bull_score']}/5 | "
            f"Bear: {regime_result['bear_score']}/5 | "
            f"Blow-off: {regime_result['blowoff_score']}/5"
        ),

        "=" * 68,
    ]

    # ---------------------------------------------------------------
    # BULL / BEAR / BLOW-OFF CHECKLISTEK
    # ---------------------------------------------------------------
    lines.extend(
        format_checks(
            title="BULL TREND CHECKLIST",
            checks=regime_result["bullish_checks"],
        )
    )

    lines.extend(
        format_checks(
            title="BEAR LIQUIDATION CHECKLIST",
            checks=regime_result["bearish_checks"],
        )
    )

    lines.extend(
        format_checks(
            title="BLOW-OFF / EXHAUSTION CHECKLIST",
            checks=regime_result["blowoff_checks"],
        )
    )

    # ---------------------------------------------------------------
    # LEGUTÓBBI NDOG
    # ---------------------------------------------------------------
    lines.extend(
        format_gap_block(
            title="LEGUTÓBBI NDOG (NEW DAY OPENING GAP)",
            gap=gap_context["latest_ndog"],
            fill_status=gap_context["ndog_fill"],
            reaction=gap_context["ndog_reaction"],
            config=config,
        )
    )

    # ---------------------------------------------------------------
    # LEGUTÓBBI NWOG
    # ---------------------------------------------------------------
    lines.extend(
        format_gap_block(
            title="AKTÍV / LEGUTÓBBI NWOG (NEW WEEK OPENING GAP)",
            gap=gap_context["latest_nwog"],
            fill_status=gap_context["nwog_fill"],
            reaction=gap_context["nwog_reaction"],
            config=config,
        )
    )

    # ---------------------------------------------------------------
    # NDOG / NWOG KONTEXTUS
    # ---------------------------------------------------------------
    gap_context_messages = gap_trade_context(
        snapshot=snapshot,
        regime=regime,
    )

    lines.extend(
        [
            "",
            "=" * 68,
            "NDOG / NWOG PIACI KONTEXTUS",
            "=" * 68,
        ]
    )

    for message in gap_context_messages["messages"]:
        lines.append(f"- {message}")

    # ---------------------------------------------------------------
    # RISK / TRADE TERV
    # ---------------------------------------------------------------
    risk_plan = calculate_gold_risk_plan(
        snapshot=snapshot,
        regime=regime,
        equity_usd=equity_usd,
        config=config,
    )

    lines.extend(
        [
            "",
            "=" * 68,
            "RISK / TRADE TERV",
            "=" * 68,
        ]
    )

    if not risk_plan["trade_allowed"]:
        lines.extend(
            [
                "ÁLLAPOT: NINCS AUTOMATIKUS TRADE.",
                f"Ok: {risk_plan['reason']}",
            ]
        )

        if "details" in risk_plan:
            lines.append(
                f"Konkrét tiltó jelek: {risk_plan['details']}"
            )

        if "next_setup" in risk_plan:
            lines.append(
                f"Következő keresendő setup: {risk_plan['next_setup']}"
            )

        lines.extend(
            [
                "",
                "Gyakorlati értelmezés:",
                "- Neutral: várj VWAP reclaimre vagy failed VWAP reclaimre.",
                "- Bull trend: csak pullback, VWAP-retest vagy breakout-retest után keress longot.",
                "- Bear trend: extrém RSI-nél ne shortold a mélypontot.",
                "- Blow-off: ne üldözd az extrém emelkedést új long belépéssel.",
                "- NDOG/NWOG: mindig 1m/5m reakcióval együtt használd.",
                "- Nagy makroadat előtt és után nincs automatikus entry.",
            ]
        )

    else:
        lines.extend(
            [
                f"Javasolt irány:          {risk_plan['side']}",
                f"Elméleti entry:          {risk_plan['entry']:.2f}",
                f"ATR-alapú stop:          {risk_plan['stop_price']:.2f}",
                f"1. célár:                {risk_plan['target_1']:.2f}",
                f"2. célár:                {risk_plan['target_2']:.2f}",

                f"Stop-távolság:           "
                f"{risk_plan['stop_distance_usd_per_oz']:.2f} USD/oz",

                f"Risk budget:             "
                f"${risk_plan['risk_budget_usd']:,.2f}",

                f"GC risk / kontrakt:      "
                f"${risk_plan['gc_risk_per_contract_usd']:,.2f}",

                f"MGC risk / kontrakt:     "
                f"${risk_plan['mgc_risk_per_contract_usd']:,.2f}",

                f"Max GC kontraktus:       "
                f"{risk_plan['max_gc_contracts']}",

                f"Max MGC kontraktus:      "
                f"{risk_plan['max_mgc_contracts']}",

                "",
                f"Figyelmeztetés: {risk_plan['warning']}",
            ]
        )

    lines.extend(
        [
            "",
            "=" * 68,
            "FONTOS: Ez kizárólag elemző / paper-trading modul.",
            "Nem küld IBKR ordert és nem kezel valódi pozíciót.",
            "Az NDOG/NWOG önmagában nem buy/sell signal.",
            "A gap szinteket mindig 1m/5m reakcióval, VWAP-pal, ATR-rel,",
            "ADX-szel, volumennel és makróhelyzettel együtt értelmezd.",
            "=" * 68,
        ]
    )

    return "\n".join(lines)


# ===================================================================
# 13. KÖZVETLEN FUTTATÁS
# ===================================================================

if __name__ == "__main__":
    print(
        "\n"
        "A gc_regime.py modul sikeresen betöltődött.\n"
        "\n"
        "A teljes GC checklist futtatásához a run_gc_checklist.py fájlt indítsd:\n"
        "\n"
        "python run_gc_checklist.py\n"
        "\n"
        "NDOG/NWOG használathoz a futtató scriptben ajánlott:\n"
        'durationStr="10 D"\n'
    )