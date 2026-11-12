from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from market_exhaustion import (
    ES_EXHAUSTION,
    NQ_EXHAUSTION,
    ExhaustionConfig,
    build_index_cross_context,
    evaluate_exhaustion,
    exhaustion_long_block,
    format_exhaustion,
)
from market_transition import (
    TransitionRiskConfig, evaluate_transition_risk, format_transition_risk,
    transition_trade_block,
)
from index_stress_context import evaluate_index_short_context, format_index_short_context

# VERSION 1.2.0: closed-bar consistency, HTF/whipsaw and timestamped VX context.
# Legacy score thresholds and default 1m ATR gates are not relaxed.
VERSION = "1.2.0"

# ===================================================================
# 1. ENUMS
# ===================================================================

class IndexRegime(str, Enum):
    NEUTRAL = "NEUTRAL / NINCS EGYÉRTELMŰ REZHIM"
    BULL_TREND = "BULL TREND / LONG BIAS"
    BEAR_TREND = "BEAR TREND / SHORT BIAS"
    BLOWOFF_RISK = "BLOW-OFF / LONG CHASING KOCKÁZAT"
    CRASH_RISK = "CRASH / VOLATILITY EXPANSION KOCKÁZAT"


# ===================================================================
# 2. INSTRUMENTUM KONFIGURÁCIÓK
# ===================================================================

@dataclass(frozen=True)
class IndexInstrumentConfig:
    """
    Egy index futures instrumentum specifikus konfigurációja.
    """

    symbol: str
    display_name: str

    # Standard futures kontraktus:
    standard_contract_name: str
    standard_multiplier_usd_per_point: float

    # Micro futures kontraktus:
    micro_contract_name: str
    micro_multiplier_usd_per_point: float

    # ES esetén használható a VIX/VIX3M.
    # NQ esetén ez csak széles piac-context, nem NQ-specifikus volatilitás.
    vix_context_enabled: bool

    # NQ általában nagyobb pont-alapú ATR-t produkál.
    # Ez csak megjelenítési és validációs információ.
    typical_atr_5m_min: float


ES_CONFIG = IndexInstrumentConfig(
    symbol="ES",
    display_name="E-MINI S&P 500 FUTURES",

    standard_contract_name="ES",
    standard_multiplier_usd_per_point=50.0,

    micro_contract_name="MES",
    micro_multiplier_usd_per_point=5.0,

    vix_context_enabled=True,

    typical_atr_5m_min=2.0,
)


NQ_CONFIG = IndexInstrumentConfig(
    symbol="NQ",
    display_name="E-MINI NASDAQ-100 FUTURES",

    standard_contract_name="NQ",
    standard_multiplier_usd_per_point=20.0,

    micro_contract_name="MNQ",
    micro_multiplier_usd_per_point=2.0,

    # A VIX itt továbbra is piaci stressz-context lehet,
    # de nem NQ-specifikus volatilitásmérő.
    vix_context_enabled=False,

    typical_atr_5m_min=5.0,
)


# ===================================================================
# 3. ÁLTALÁNOS REGIME CONFIG
# ===================================================================

@dataclass(frozen=True)
class IndexRegimeConfig:
    # Research profiles; do not treat the initial thresholds as crash probabilities.
    es_exhaustion: ExhaustionConfig = ES_EXHAUSTION
    nq_exhaustion: ExhaustionConfig = NQ_EXHAUSTION
    es_transition: TransitionRiskConfig = TransitionRiskConfig()
    nq_transition: TransitionRiskConfig = TransitionRiskConfig()
    vwap_distance_atr_basis: str = "1m"
    vix_max_age_minutes: float = 15.0
    vix_max_pair_skew_minutes: float = 5.0
    max_data_age_minutes: float = 45.0

    # ---------------------------------------------------------------
    # Időzóna / Globex futures session
    # ---------------------------------------------------------------
    new_york_timezone: str = "America/New_York"

    # Equity index futures normál napi session-break:
    # 17:00 ET körül szünet, 18:00 ET körül újraindulás.
    daily_close_hour_et: int = 17
    daily_open_hour_et: int = 18

    # NDOG/NWOG close/open keresési ablakok
    gap_close_window_start_min: int = 15
    gap_open_window_end_min: int = 20

    # ---------------------------------------------------------------
    # Trend-rezhim
    # ---------------------------------------------------------------
    adx_trend_min: float = 25.0
    adx_strong_min: float = 30.0

    # ---------------------------------------------------------------
    # RSI
    # ---------------------------------------------------------------
    rsi2_blowoff: float = 95.0
    rsi5_blowoff: float = 80.0

    rsi2_extreme_oversold: float = 10.0
    rsi5_extreme_oversold: float = 20.0

    rsi2_extreme_overbought: float = 95.0
    rsi5_extreme_overbought: float = 80.0

    # ---------------------------------------------------------------
    # Volumen
    # ---------------------------------------------------------------
    volume_expansion_ratio: float = 1.30
    volume_climax_ratio: float = 2.00
    volume_dry_ratio: float = 0.70

    # ---------------------------------------------------------------
    # VWAP / ATR
    # ---------------------------------------------------------------
    vwap_extension_atr: float = 2.5

    max_short_extension_below_vwap_atr: float = -3.0
    max_long_extension_above_vwap_atr: float = 3.0

    # ---------------------------------------------------------------
    # Gap szintek
    # ---------------------------------------------------------------
    gap_level_proximity_atr: float = 0.50
    gap_reaction_confirmation_atr: float = 0.25
    gap_reaction_lookback_bars: int = 10
    stale_gap_distance_atr: float = 3.0

    # ---------------------------------------------------------------
    # VIX/VIX3M
    # ---------------------------------------------------------------
    vix_contango_ratio: float = 0.90
    vix_transition_ratio: float = 1.00
    vix_stress_ratio: float = 1.10

    # ---------------------------------------------------------------
    # Risk
    # ---------------------------------------------------------------
    risk_per_trade_pct: float = 0.0025

    # Indexek közös, korrelált kockázati limite.
    # ES és NQ ugyanabba az irányba történő trade-jénél
    # ezt célszerű alkalmazni runner szinten.
    max_correlated_index_risk_pct: float = 0.0050

    # ---------------------------------------------------------------
    # Stop / target
    # ---------------------------------------------------------------
    stop_atr_multiple: float = 1.0
    target1_atr_multiple: float = 1.0
    target2_atr_multiple: float = 2.0


CONFIG = IndexRegimeConfig()


# ===================================================================
# 4. ALAP INDIKÁTOROK
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
    Average True Range indexpontban.
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
    ADX, +DI, -DI számítás.

    Returns:
        DataFrame:
        - adx
        - plus_di
        - minus_di
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
    *, as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    1 perces OHLCV adatból nagyobb időtávú OHLCV DataFrame.
    """
    result = (
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
    if as_of is not None:
        count = df_1m.close.resample(rule).count()
        duration = pd.Timedelta(rule)
        expected = duration / pd.Timedelta(minutes=1)
        result = result.loc[(result.index+duration <= as_of)
                            & (count.reindex(result.index) == expected)]
    return result


# ===================================================================
# 5. ADATELLENŐRZÉS / IDŐZÓNA / SESSION
# ===================================================================

def validate_ohlcv(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Kötelező oszlopok:
    - open
    - high
    - low
    - close
    - volume
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
            f"Kötelező: {required_columns}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "A DataFrame indexe pandas DatetimeIndex legyen. "
            "Példa: df = df.set_index('date')"
        )

    clean_df = df.copy()
    clean_df = clean_df.sort_index()
    clean_df = clean_df[required_columns].dropna()
    clean_df = clean_df.apply(pd.to_numeric, errors="raise")
    if clean_df.index.has_duplicates or clean_df.index.hasnans:
        raise ValueError("Duplikált vagy hiányzó index timestamp.")
    if not np.isfinite(clean_df.to_numpy(float)).all():
        raise ValueError("Nem véges index OHLCV.")
    if ((clean_df[["open", "high", "low", "close"]] <= 0).any().any()
            or (clean_df.volume < 0).any()
            or (clean_df.high < clean_df[["open", "close", "low"]].max(axis=1)).any()
            or (clean_df.low > clean_df[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("Érvénytelen index OHLCV.")

    if len(clean_df) < 100:
        raise ValueError(
            f"Túl kevés adat: {len(clean_df)} bar. "
            "Legalább 100 db 1 perces bar kell."
        )

    return clean_df


def to_new_york_time(
    df: pd.DataFrame,
    config: IndexRegimeConfig = CONFIG,
) -> pd.DataFrame:
    """
    A DataFrame indexét New York időre konvertálja.

    Timezone-naive index esetén UTC-ként kezeli.
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
    Equity-index futures session kulcs.

    A session 18:00 ET körül indul.
    A session key a timestamp - 18 óra naptári dátuma.
    """
    wall = timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
    key = (wall-pd.Timedelta(hours=18)).normalize()
    return key.tz_localize(timestamp.tz) if timestamp.tzinfo is not None else key


def add_session_key(
    df_ny: pd.DataFrame,
) -> pd.DataFrame:
    """
    session_key oszlopot tesz a DataFrame-be.
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
    Session VWAP.

    A VWAP New York-i 18:00 ET session-reset szerint indul újra.
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


def previous_session_levels(
    df_ny: pd.DataFrame,
) -> tuple[float, float]:
    """
    Előző futures session high / low.
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
    Aktuális futures session high / low.
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
# 6. NDOG / NWOG GAP SZÁMÍTÁS
# ===================================================================

def get_last_close_before_daily_break(
    df_ny: pd.DataFrame,
    target_date: pd.Timestamp,
    config: IndexRegimeConfig = CONFIG,
) -> tuple[float | None, pd.Timestamp | None]:
    """
    17:00 ET break előtti utolsó close.

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
    config: IndexRegimeConfig = CONFIG,
) -> tuple[float | None, pd.Timestamp | None]:
    """
    18:00 ET utáni első open.

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
    Egységes NDOG/NWOG gap rekord.
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


def calculate_ndogs(
    df_ny: pd.DataFrame,
    config: IndexRegimeConfig = CONFIG,
) -> list[dict[str, Any]]:
    """
    NDOG számítás.

    Monday-Thursday:
    17:00 ET előtti close -> 18:00 ET utáni open.

    Friday close -> Sunday open:
    NWOG, ezért a pénteket itt kihagyjuk.
    """
    normalized_dates = pd.Index(
        df_ny.index.normalize().unique()
    ).sort_values()

    ndogs: list[dict[str, Any]] = []

    for date_value in normalized_dates:
        weekday = date_value.weekday()

        # Friday=4, Saturday=5, Sunday=6.
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
    config: IndexRegimeConfig = CONFIG,
) -> list[dict[str, Any]]:
    """
    NWOG számítás.

    Péntek 17:00 ET előtti close
    -> vasárnap 18:00 ET utáni open.
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
    Az aktuális időpontig létrejött legutóbbi gap.
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
    Gap fill státusz:
    - ár gap felett/alatt/benne;
    - CE / low / high érintés;
    - teljes fill.
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
    config: IndexRegimeConfig = CONFIG,
) -> dict[str, Any]:
    """
    Gap szintre adott utolsó N perces reakció.

    Lehetséges:
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
    config: IndexRegimeConfig = CONFIG,
) -> dict[str, Any]:
    """
    NDOG/NWOG, fill és reaction adatokat egyben adja vissza.
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
# 7. VIX / VIX3M KONTEXTUS
# ===================================================================

def analyze_vix_context(
    vix_context: dict[str, float] | None,
    instrument: IndexInstrumentConfig,
    config: IndexRegimeConfig = CONFIG,
    *, as_of: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """
    Opcionális VIX/VIX3M kontextus.

    Várt input:

    {
        "vix": 14.84,
        "vix3m": 17.76,
        "vix_change_pct": -1.25
    }

    Ha nincs VIX input, a rendszer price-action alapon tovább működik.
    """
    if not vix_context:
        return {
            "available": False,
            "state": "N/A",
            "message": (
                "Nincs VIX/VIX3M adat. "
                "A rezhim kizárólag index price-action alapján fut."
            ),
        }
    # A receipt timestamp or a previous-close snapshot is not causal evidence.
    try:
        def stamp(value: Any) -> pd.Timestamp:
            ts = pd.Timestamp(value)
            if pd.isna(ts) or ts.tzinfo is None:
                raise ValueError("Timezone-aware source timestamp required.")
            return ts.tz_convert("UTC")
        source = stamp(vix_context.get("timestamp"))
        cutoff = stamp(as_of)
        first = stamp(vix_context.get("vix_timestamp", source))
        second = stamp(vix_context.get("vix3m_timestamp", source))
        ages = [(cutoff-x).total_seconds()/60 for x in (source, first, second)]
        quality = ("FUTURE_INPUT" if min(ages) < 0 else
                   "STALE_INPUT" if max(ages) > config.vix_max_age_minutes else
                   "PAIR_TIME_MISMATCH" if abs(first-second).total_seconds()/60 >
                   config.vix_max_pair_skew_minutes else "OK")
    except (ValueError, TypeError):
        quality = "UNVERIFIED_TIMESTAMP"
    if quality != "OK":
        return dict(available=False, state="N/A", quality=quality,
                    message=f"VIX/VIX3M nem használható: {quality}. Nem helyettesítjük friss receipt-idővel.")

    try:
        vix = float(vix_context.get("vix", np.nan))
        vix3m = float(vix_context.get("vix3m", np.nan))
        vix_change_pct = float(vix_context.get("vix_change_pct", np.nan))
    except (TypeError, ValueError):
        return dict(available=False, state="N/A", quality="INVALID_VALUES",
                    message="Érvénytelen VIX/VIX3M érték.")

    if (
        not np.isfinite(vix)
        or vix <= 0
        or not np.isfinite(vix3m)
        or vix3m <= 0
    ):
        return {
            "available": False,
            "state": "N/A",
            "message": (
                "Érvénytelen vagy hiányzó VIX/VIX3M érték."
            ),
        }

    ratio = vix / vix3m

    if ratio < config.vix_contango_ratio:
        state = "CONTANGO / LOW SHORT-TERM VOL"

        message = (
            "VIX/VIX3M contango: alacsony rövid távú stressz. "
            "Trend-following környezet lehetséges, de nem automatikus long jel."
        )

    elif ratio < config.vix_transition_ratio:
        state = "NORMAL / TRANSITION"

        message = (
            "VIX/VIX3M átmeneti állapot: nincs egyértelmű stressz."
        )

    elif ratio < config.vix_stress_ratio:
        state = "STRESS BUILDING"

        message = (
            "A rövid távú volatilitás közelít vagy meghaladja "
            "a 3 hónapos volatilitást. Csökkentett méret indokolt."
        )

    else:
        state = "BACKWARDATION / CRASH RISK"

        message = (
            "VIX/VIX3M backwardation: emelkedett rövid távú stressz "
            "és crash/liquidation kockázat."
        )

    if instrument.symbol == "NQ":
        message += (
            " NQ-nál a VIX csak széles piaci stressz-context; "
            "nem NQ-specifikus volatilitásmérő."
        )

    return {
        "available": True,
        "quality": quality,
        "timestamp": source,

        "vix": float(vix),
        "vix3m": float(vix3m),
        "ratio": float(ratio),

        "vix_change_pct": (
            float(vix_change_pct)
            if np.isfinite(vix_change_pct)
            else np.nan
        ),

        "state": state,
        "message": message,
    }


# ===================================================================
# 8. INDEX SNAPSHOT
# ===================================================================

def build_index_snapshot(
    df_1m: pd.DataFrame,
    instrument: IndexInstrumentConfig,
    vix_context: dict[str, float] | None = None,
    config: IndexRegimeConfig = CONFIG,
    *,
    daily_history: pd.DataFrame | None = None,
    cross_market_context: dict[str, Any] | None = None,
    exhaustion_memory: dict[str, Any] | None = None,
    data_context: dict[str, Any] | None = None,
    vx_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    ES vagy NQ 1 perces OHLCV adatból teljes market snapshot.
    """
    df = validate_ohlcv(df_1m)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    metadata = dict(data_context or {})
    input_timestamp = df.index[-1]
    analysis_timestamp = input_timestamp+(
        pd.Timedelta(minutes=1) if metadata.get("last_bar_is_complete", False)
        else pd.Timedelta(0))
    df = df.loc[df.index < analysis_timestamp]
    closed_context = {**metadata, "last_bar_is_complete": True}
    if config.vwap_distance_atr_basis not in ("1m", "5m"):
        raise ValueError("VWAP ATR basis must be 1m or 5m.")

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
        as_of=analysis_timestamp,
    )

    if len(df_5m) < 30:
        raise ValueError(
            "Túl kevés 5 perces bar az ADX / ATR számításhoz."
        )

    df_ny["vwap"] = calc_session_vwap(df_ny)
    # Never label a truncated-session VWAP as a complete session benchmark.
    for day, part in df_ny.groupby("session_key"):
        expected_open = pd.Timestamp(day).tz_localize(None)+pd.Timedelta(
            hours=config.daily_open_hour_et)
        expected_open = expected_open.tz_localize(config.new_york_timezone)
        if part.index[0] != expected_open:
            df_ny.loc[part.index, "vwap"] = np.nan

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

    average_volume_20 = float(
        df_ny["volume"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    volume_ratio = (
        current_volume / average_volume_20
        if average_volume_20 > 0
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

    vix_analysis = analyze_vix_context(
        vix_context=vix_context,
        instrument=instrument,
        config=config,
        as_of=analysis_timestamp,
    )

    snapshot = {
        "instrument": instrument,

        "timestamp": analysis_timestamp.tz_convert(config.new_york_timezone),
        "input_bar_start": input_timestamp,
        "last_closed_1m_start": df_ny.index[-1],
        "last_closed_1m_at": df_ny.index[-1]+pd.Timedelta(minutes=1),
        "last_closed_5m_at": df_5m.index[-1]+pd.Timedelta(minutes=5),
        "data_mode": metadata.get("mode", "UNVERIFIED"),
        "data_quality": "OK" if np.isfinite(current_vwap) else "PARTIAL_SESSION_VWAP",

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
        "volume_avg_20": average_volume_20,
        "volume_ratio": volume_ratio,

        "vwap_distance_atr_1m": vwap_distance_atr,
        "vwap_distance_atr_5m": ((current_close-current_vwap)/current_atr_5m
                               if current_atr_5m > 0 else np.nan),
        "vwap_distance_atr": (vwap_distance_atr if config.vwap_distance_atr_basis == "1m"
                             else (current_close-current_vwap)/current_atr_5m),

        "prior_session_high": prior_session_high,
        "prior_session_low": prior_session_low,

        "current_session_high": current_session_high,
        "current_session_low": current_session_low,

        "gap_context": gap_context,
        "vix_analysis": vix_analysis,
    }
    if metadata.get("as_of") is not None:
        wall = pd.Timestamp(metadata["as_of"])
        wall = wall.tz_localize("UTC") if wall.tzinfo is None else wall.tz_convert("UTC")
        age = (wall-analysis_timestamp.tz_convert("UTC")).total_seconds()/60
        snapshot["data_age_minutes"] = age
        if age < 0 or age > config.max_data_age_minutes:
            snapshot["data_quality"] = "FUTURE_INPUT" if age < 0 else "STALE_INPUT"
    else:
        snapshot["data_age_minutes"] = None

    profile = (
        config.es_exhaustion if instrument.symbol == "ES" else config.nq_exhaustion
    )
    overlay_context = dict(cross_market_context or {})
    if not vix_analysis.get("available", False):
        overlay_context.pop("vix_stress", None)
    if any(gap_context[k].get("available", False)
           for k in ("ndog_reaction", "nwog_reaction")):
        overlay_context.setdefault("bearish_gap_rejection", {
            "value": any(
                gap_context[k].get("reaction") == "BEARISH_REJECTION"
                for k in ("ndog_reaction", "nwog_reaction")
            ),
            "timestamp": df_ny.index[-1],
        })
    if vix_context and vix_context.get("timestamp") is not None:
        if vix_analysis.get("available", False):
            overlay_context.setdefault("vix_stress", {
                "value": vix_analysis["ratio"] >= config.vix_stress_ratio,
                "timestamp": vix_context["timestamp"],
            })
    snapshot["exhaustion"] = evaluate_exhaustion(
        df_ny,
        instrument.symbol,
        config=profile,
        daily_history=daily_history,
        cross_market_context=overlay_context,
        previous_state=exhaustion_memory,
        data_context=closed_context,
    )
    transition_profile = config.es_transition if instrument.symbol == "ES" else config.nq_transition
    snapshot["transition_risk"] = evaluate_transition_risk(
        df_ny, config=transition_profile, data_context=closed_context)
    snapshot["short_stress_context"] = evaluate_index_short_context(
        snapshot, vx_context, analysis_timestamp=analysis_timestamp,
        wall_clock=metadata.get("as_of"), config=config)
    return snapshot


# ===================================================================
# 9. REZHIM DETEKTOR
# ===================================================================

def index_regime_active(
    snapshot: dict[str, Any],
    config: IndexRegimeConfig = CONFIG,
) -> dict[str, Any]:
    """
    ES/NQ rezhimdetektor.

    Állapotok:
    - Bull trend
    - Bear trend
    - Blow-off risk
    - Crash / volatility expansion risk
    - Neutral
    """
    close = snapshot["close"]
    vwap = snapshot["vwap"]

    adx = snapshot["adx_5m"]
    plus_di = snapshot["plus_di_5m"]
    minus_di = snapshot["minus_di_5m"]

    rsi_2 = snapshot["rsi_2"]
    rsi_5 = snapshot["rsi_5"]

    volume_ratio = snapshot["volume_ratio"]

    vwap_distance_atr = snapshot["vwap_distance_atr"]

    prior_session_low = snapshot["prior_session_low"]
    current_session_low = snapshot["current_session_low"]

    vix_analysis = snapshot["vix_analysis"]

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

        f"Ár > VWAP + {config.vwap_extension_atr:.1f}× ATR({config.vwap_distance_atr_basis})": (
            vwap_distance_atr > config.vwap_extension_atr
        ),

        f"Volume >= {config.volume_climax_ratio:.1f}× 20-bar átlag": (
            volume_ratio >= config.volume_climax_ratio
        ),

        "+DI dominál -DI felett": (
            plus_di > minus_di
        ),
    }

    crash_checks = {
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

        "RSI(5) < 40": rsi_5 < 40,

        "Ár közel az aktuális session low-hoz": (
            close <= current_session_low + snapshot["atr_1m"]
        ),

        "Ár az előző session low alatt": (
            np.isfinite(prior_session_low)
            and close < prior_session_low
        ),
    }

    if vix_analysis.get("available", False):
        crash_checks["VIX/VIX3M >= 1.10"] = (
            vix_analysis["ratio"] >= config.vix_stress_ratio
        )

    bull_score = sum(
        bullish_checks.values()
    )

    bear_score = sum(
        bearish_checks.values()
    )

    blowoff_score = sum(
        blowoff_checks.values()
    )

    crash_score = sum(
        crash_checks.values()
    )

    vix_crash_condition = (
        vix_analysis.get("available", False)
        and vix_analysis.get("ratio", 0.0)
        >= config.vix_stress_ratio
    )

    # Crash rezhimhez szigorúbb feltétel kell:
    # legalább 5 crash jel, vagy 4 jel + VIX backwardation.
    if snapshot.get("data_quality", "OK") != "OK":
        regime = IndexRegime.NEUTRAL
    elif (
        crash_score >= 5
        or (
            crash_score >= 4
            and vix_crash_condition
        )
    ):
        regime = IndexRegime.CRASH_RISK

    elif blowoff_score >= 4:
        regime = IndexRegime.BLOWOFF_RISK

    elif bear_score >= 4:
        regime = IndexRegime.BEAR_TREND

    elif bull_score >= 4:
        regime = IndexRegime.BULL_TREND

    else:
        regime = IndexRegime.NEUTRAL

    return {
        "regime": regime,

        "bullish_checks": bullish_checks,
        "bearish_checks": bearish_checks,
        "blowoff_checks": blowoff_checks,
        "crash_checks": crash_checks,

        "bull_score": bull_score,
        "bear_score": bear_score,
        "blowoff_score": blowoff_score,
        "crash_score": crash_score,
        # Parallel risk state: it does not fabricate a bear/crash regime.
        "exhaustion_state": snapshot.get("exhaustion", {}).get("state", "N/A"),
        "block_new_longs": (
            snapshot.get("exhaustion", {}).get("block_new_longs", False)
            or snapshot.get("transition_risk", {}).get("block_new_longs", False)
            or snapshot.get("data_quality", "OK") != "OK"),
        "block_new_shorts": (snapshot.get("transition_risk", {}).get("block_new_shorts", False)
                             or snapshot.get("data_quality", "OK") != "OK"),
        "whipsaw_state": snapshot.get("transition_risk", {}).get("whipsaw", {}).get("state", "N/A"),
        "short_stress_state": snapshot.get("short_stress_context", {}).get("state", "N/A"),
    }


# ===================================================================
# 10. GAP KONTEXTUS
# ===================================================================

def gap_trade_context(
    snapshot: dict[str, Any],
    regime: IndexRegime,
) -> dict[str, Any]:
    """
    NDOG/NWOG értelmező kontextus.
    """
    context = snapshot["gap_context"]

    ndog_reaction = context["ndog_reaction"]
    nwog_reaction = context["nwog_reaction"]

    messages: list[str] = []

    if nwog_reaction["reaction"] == "BULLISH_RECLAIM":
        messages.append(
            "NWOG bullish reclaim: az ár érintette a szintet, "
            "majd megerősítve fölé zárt."
        )

    elif nwog_reaction["reaction"] == "BEARISH_REJECTION":
        messages.append(
            "NWOG bearish rejection: az ár érintette a szintet, "
            "majd megerősítve alá zárt."
        )

    elif nwog_reaction["reaction"] == "NEAR_LEVEL":
        messages.append(
            "Az ár NWOG szint közelében van; "
            "várj 1m/5m struktúra-megerősítésre."
        )

    if ndog_reaction["reaction"] == "BULLISH_RECLAIM":
        messages.append(
            "NDOG bullish reclaim: az ár visszafoglalta "
            "a közeli napi gap szintet."
        )

    elif ndog_reaction["reaction"] == "BEARISH_REJECTION":
        messages.append(
            "NDOG bearish rejection: az ár nem tudott "
            "a napi gap szint fölött maradni."
        )

    elif ndog_reaction["reaction"] == "NEAR_LEVEL":
        messages.append(
            "Az ár NDOG szint közelében van; "
            "ne lépj be kizárólag a gap alapján."
        )

    if regime in (
        IndexRegime.BEAR_TREND,
        IndexRegime.CRASH_RISK,
    ):
        if (
            ndog_reaction["reaction"] == "BULLISH_RECLAIM"
            or nwog_reaction["reaction"] == "BULLISH_RECLAIM"
        ):
            messages.append(
                "Bearish környezetben bullish gap reclaim alakult ki: "
                "a friss short belépés kockázatosabb."
            )

    if regime == IndexRegime.BULL_TREND:
        if (
            ndog_reaction["reaction"] == "BEARISH_REJECTION"
            or nwog_reaction["reaction"] == "BEARISH_REJECTION"
        ):
            messages.append(
                "Bullish környezetben bearish gap rejection látszik: "
                "új long előtt várj további megerősítésre."
            )

    if not messages:
        messages.append(
            "Nincs közeli, megerősített NDOG/NWOG reakció. "
            "A gapek referencia- és célzónák maradnak."
        )

    return {
        "messages": messages,
    }


# ===================================================================
# 11. KOCKÁZATI TERV / ANTI-CHASING
# ===================================================================

def calculate_index_risk_plan(
    snapshot: dict[str, Any],
    regime: IndexRegime,
    equity_usd: float,
    config: IndexRegimeConfig = CONFIG,
) -> dict[str, Any]:
    """
    ES/MES vagy NQ/MNQ elméleti risk plan.

    Nem küld ordert.
    """
    instrument: IndexInstrumentConfig = snapshot[
        "instrument"
    ]

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
            "Érvénytelen ATR(5m); risk plan nem készíthető."
        )
    if snapshot.get("data_quality", "OK") != "OK":
        return dict(trade_allowed=False, reason="Index core adatkapu zárt.",
                    details=snapshot["data_quality"],
                    next_setup="Friss, teljes session-VWAP és lezárt gyertyák szükségesek.")

    # Hard gate BEFORE the existing long anti-chasing filters.
    exhaustion = snapshot.get("exhaustion", {})
    if regime == IndexRegime.BULL_TREND and exhaustion.get("block_new_longs", False):
        return exhaustion_long_block(exhaustion)
    transition = snapshot.get("transition_risk", {})
    if regime == IndexRegime.BULL_TREND and transition.get("block_new_longs", False):
        return transition_trade_block(transition, 1)
    if regime in (IndexRegime.BEAR_TREND, IndexRegime.CRASH_RISK) and transition.get("block_new_shorts", False):
        return transition_trade_block(transition, -1)

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

    if regime in (
        IndexRegime.BEAR_TREND,
        IndexRegime.CRASH_RISK,
    ) and (
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
                "Bearish rezhim van, de új SHORT belépés TILTVA. "
                "Az ár túlzottan kiterjedt lefelé vagy bullish reclaim alakult ki."
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

    if regime == IndexRegime.BULL_TREND and (
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
                "Bullish rezhim van, de új LONG belépés TILTVA. "
                "Az ár túlzottan megnyúlt felfelé vagy bearish gap rejection alakult ki."
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
    if regime == IndexRegime.NEUTRAL:
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

    if regime == IndexRegime.BLOWOFF_RISK:
        return {
            "trade_allowed": False,

            "reason": (
                "Blow-off / exhaustion kockázat van. "
                "Ne üldözd az extrém emelkedést új long belépéssel."
            ),

            "next_setup": (
                "Várj pullbackre, VWAP retestre, 5m struktúratörésre, "
                "lower high-ra vagy bearish gap rejectionre."
            ),
        }

    # ---------------------------------------------------------------
    # TISZTA BULL / BEAR TREND
    # ---------------------------------------------------------------
    risk_budget_usd = (
        equity_usd * config.risk_per_trade_pct
    )

    stop_distance_points = (
        atr_5m * config.stop_atr_multiple
    )

    if regime == IndexRegime.BULL_TREND:
        side = "LONG"

        stop_price = entry - stop_distance_points

        target_1 = entry + (
            atr_5m * config.target1_atr_multiple
        )

        target_2 = entry + (
            atr_5m * config.target2_atr_multiple
        )

    elif regime in (
        IndexRegime.BEAR_TREND,
        IndexRegime.CRASH_RISK,
    ):
        side = "SHORT"

        stop_price = entry + stop_distance_points

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

    standard_risk_per_contract_usd = (
        stop_distance_points
        * instrument.standard_multiplier_usd_per_point
    )

    micro_risk_per_contract_usd = (
        stop_distance_points
        * instrument.micro_multiplier_usd_per_point
    )

    max_standard_contracts = int(
        risk_budget_usd
        // standard_risk_per_contract_usd
    )

    max_micro_contracts = int(
        risk_budget_usd
        // micro_risk_per_contract_usd
    )

    return {
        "trade_allowed": True,

        "side": side,

        "entry": round(entry, 2),
        "stop_price": round(stop_price, 2),

        "target_1": round(target_1, 2),
        "target_2": round(target_2, 2),

        "atr_5m": round(atr_5m, 2),

        "stop_distance_points": round(
            stop_distance_points,
            2,
        ),

        "risk_budget_usd": round(
            risk_budget_usd,
            2,
        ),

        "standard_contract_name": instrument.standard_contract_name,
        "micro_contract_name": instrument.micro_contract_name,

        "standard_risk_per_contract_usd": round(
            standard_risk_per_contract_usd,
            2,
        ),

        "micro_risk_per_contract_usd": round(
            micro_risk_per_contract_usd,
            2,
        ),

        "max_standard_contracts": max_standard_contracts,
        "max_micro_contracts": max_micro_contracts,

        "warning": (
            "A kontraktusszám elméleti maximum. "
            "Margin, spread, slippage, likviditás, hírek és napi veszteséglimit "
            "nincs benne."
        ),
    }


# ===================================================================
# 12. REPORT FORMÁZÁS
# ===================================================================

def format_checks(
    title: str,
    checks: dict[str, bool],
) -> list[str]:
    """
    Checklist formázása.
    """
    lines = [
        "",
        "-" * 72,
        title,
        "-" * 72,
    ]

    for name, passed in checks.items():
        status = "N/A" if passed is None else "IGEN" if passed else "NEM"
        lines.append(f"[{status}] {name}")

    return lines


def format_gap_block(
    title: str,
    gap: dict[str, Any] | None,
    fill_status: dict[str, Any],
    reaction: dict[str, Any],
    config: IndexRegimeConfig = CONFIG,
) -> list[str]:
    """
    NDOG/NWOG riportblokk.
    """
    lines = [
        "",
        "-" * 72,
        title,
        "-" * 72,
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

            f"Gap méret:              {gap['gap_size']:.2f} pont",

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


def format_vix_block(
    vix_analysis: dict[str, Any],
) -> list[str]:
    """
    VIX/VIX3M riportblokk.
    """
    lines = [
        "",
        "-" * 72,
        "VIX / VIX3M VOLATILITY KONTEXTUS",
        "-" * 72,
    ]

    if not vix_analysis.get("available", False):
        lines.append(
            vix_analysis.get(
                "message",
                "Nincs VIX/VIX3M adat.",
            )
        )
        return lines

    lines.extend(
        [
            f"VIX:                    {vix_analysis['vix']:.2f}",
            f"VIX3M:                  {vix_analysis['vix3m']:.2f}",
            f"VIX / VIX3M:            {vix_analysis['ratio']:.3f}",
            f"Állapot:                {vix_analysis['state']}",
            f"Értelmezés:             {vix_analysis['message']}",
        ]
    )

    if np.isfinite(vix_analysis["vix_change_pct"]):
        lines.append(
            f"VIX változás:           "
            f"{vix_analysis['vix_change_pct']:.2f}%"
        )

    return lines


# ===================================================================
# 13. FŐ FUTTATÓ FÜGGVÉNY
# ===================================================================

def run_index_checklist(
    df_1m: pd.DataFrame,
    instrument: IndexInstrumentConfig,
    equity_usd: float = 10_000.0,
    vix_context: dict[str, float] | None = None,
    config: IndexRegimeConfig = CONFIG,
    *,
    daily_history: pd.DataFrame | None = None,
    cross_market_context: dict[str, Any] | None = None,
    exhaustion_memory: dict[str, Any] | None = None,
    data_context: dict[str, Any] | None = None,
    vx_context: dict[str, Any] | None = None,
) -> str:
    """
    ES vagy NQ teljes rezhim checklist.

    Példa ES-re:
        report = run_index_checklist(
            df_1m=df_es,
            instrument=ES_CONFIG,
            equity_usd=10_000,
            vix_context={
                "vix": 14.84,
                "vix3m": 17.76,
                "vix_change_pct": -1.25,
            },
        )

    Példa NQ-ra:
        report = run_index_checklist(
            df_1m=df_nq,
            instrument=NQ_CONFIG,
            equity_usd=10_000,
            vix_context=None,
        )
    """
    snapshot = build_index_snapshot(
        df_1m=df_1m,
        instrument=instrument,
        vix_context=vix_context,
        config=config,
        daily_history=daily_history,
        cross_market_context=cross_market_context,
        exhaustion_memory=exhaustion_memory,
        data_context=data_context,
        vx_context=vx_context,
    )

    regime_result = index_regime_active(
        snapshot=snapshot,
        config=config,
    )

    regime = regime_result["regime"]

    gap_context = snapshot["gap_context"]
    def shown(value: Any, precision: int = 2) -> str:
        return f"{value:.{precision}f}" if value is not None and np.isfinite(value) else "N/A"

    lines = [
        "",
        "=" * 72,
        f"{instrument.symbol} / {instrument.display_name} REGIME CHECKLIST",
        "=" * 72,

        f"Időbélyeg:              {snapshot['timestamp']}",
        f"Engine verzió:          {VERSION}",
        f"Core adatminőség:       {snapshot['data_quality']} | adatmód: {snapshot['data_mode']}",
        f"Falióra szerinti adatkor: {shown(snapshot['data_age_minutes'], 1)} perc",
        f"Utolsó lezárt 1m/5m:    {snapshot['last_closed_1m_at']} / {snapshot['last_closed_5m_at']}",

        f"{instrument.symbol} utolsó ár:       {snapshot['close']:.2f}",
        f"Session VWAP:            {shown(snapshot['vwap'])}",

        f"VWAP-távolság:           "
        f"{shown(snapshot['vwap_distance_atr'])} × ATR({config.vwap_distance_atr_basis})",
        f"VWAP-táv 1m / 5m:       {shown(snapshot['vwap_distance_atr_1m'])} / "
        f"{shown(snapshot['vwap_distance_atr_5m'])} ATR (külön alapok)",

        f"ATR(1m,14):              {snapshot['atr_1m']:.2f} pont",
        f"ATR(5m,14):              {snapshot['atr_5m']:.2f} pont",

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
        "=" * 72,

        f"AKTUÁLIS REZHIM: {regime.value}",

        (
            f"Pontszámok — "
            f"Bull: {regime_result['bull_score']}/5 | "
            f"Bear: {regime_result['bear_score']}/5 | "
            f"Blow-off: {regime_result['blowoff_score']}/5 | "
            f"Crash: {regime_result['crash_score']}/"
            f"{len(regime_result['crash_checks'])}"
        ),

        "=" * 72,
    ]

    lines.extend(
        format_checks(
            title="BULL TREND CHECKLIST",
            checks=regime_result["bullish_checks"],
        )
    )

    lines.extend(
        format_checks(
            title="BEAR TREND CHECKLIST",
            checks=regime_result["bearish_checks"],
        )
    )

    lines.extend(
        format_checks(
            title="BLOW-OFF / EXHAUSTION CHECKLIST",
            checks=regime_result["blowoff_checks"],
        )
    )

    lines.extend(
        format_checks(
            title="CRASH / VOLATILITY EXPANSION CHECKLIST",
            checks=regime_result["crash_checks"],
        )
    )

    lines.extend(
        format_vix_block(
            vix_analysis=snapshot["vix_analysis"],
        )
    )

    lines.extend(format_exhaustion(snapshot["exhaustion"]))
    lines.extend(format_transition_risk(snapshot["transition_risk"], instrument=instrument.symbol))
    lines.extend(format_index_short_context(snapshot["short_stress_context"], instrument.symbol))

    lines.extend(
        format_gap_block(
            title="LEGUTÓBBI NDOG (NEW DAY OPENING GAP)",
            gap=gap_context["latest_ndog"],
            fill_status=gap_context["ndog_fill"],
            reaction=gap_context["ndog_reaction"],
            config=config,
        )
    )

    lines.extend(
        format_gap_block(
            title="AKTÍV / LEGUTÓBBI NWOG (NEW WEEK OPENING GAP)",
            gap=gap_context["latest_nwog"],
            fill_status=gap_context["nwog_fill"],
            reaction=gap_context["nwog_reaction"],
            config=config,
        )
    )

    gap_messages = gap_trade_context(
        snapshot=snapshot,
        regime=regime,
    )

    lines.extend(
        [
            "",
            "=" * 72,
            "NDOG / NWOG PIACI KONTEXTUS",
            "=" * 72,
        ]
    )

    for message in gap_messages["messages"]:
        lines.append(f"- {message}")

    risk_plan = calculate_index_risk_plan(
        snapshot=snapshot,
        regime=regime,
        equity_usd=equity_usd,
        config=config,
    )

    lines.extend(
        [
            "",
            "=" * 72,
            "RISK / TRADE TERV",
            "=" * 72,
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
                "- Bull trend: csak pullback, VWAP-retest vagy breakout-retest után long.",
                "- Bear trend: csak bounce / failed reclaim / lower high után short.",
                "- Crash: mélyponton ne shortolj, várj visszatesztre vagy struktúrára.",
                "- Blow-off: ne üldözd az extrém emelkedést új long belépéssel.",
                "- NDOG/NWOG: csak 1m/5m reakcióval együtt használd.",
                "- ES és NQ együtt magas korrelációjú; a teljes index-risket kezeld közösen.",
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
                f"{risk_plan['stop_distance_points']:.2f} pont",

                f"Risk budget:             "
                f"${risk_plan['risk_budget_usd']:,.2f}",

                f"{risk_plan['standard_contract_name']} risk / kontrakt: "
                f"${risk_plan['standard_risk_per_contract_usd']:,.2f}",

                f"{risk_plan['micro_contract_name']} risk / kontrakt: "
                f"${risk_plan['micro_risk_per_contract_usd']:,.2f}",

                f"Max {risk_plan['standard_contract_name']} kontraktus: "
                f"{risk_plan['max_standard_contracts']}",

                f"Max {risk_plan['micro_contract_name']} kontraktus: "
                f"{risk_plan['max_micro_contracts']}",

                "",
                f"Figyelmeztetés: {risk_plan['warning']}",
            ]
        )

    lines.extend(
        [
            "",
            "=" * 72,
            "FONTOS: Ez elemző / paper-trading modul.",
            "Nem küld IBKR ordert és nem kezel valós pozíciót.",
            "A VIX/VIX3M kontextus nem önálló buy/sell jel.",
            "Az NDOG/NWOG szint nem önálló buy/sell jel.",
            "ES és NQ ugyanabba az irányba felvett trade-jei korreláltak.",
            "Közös index-risk limitet alkalmazz, ne külön kockázati limitet.",
            "=" * 72,
        ]
    )

    return "\n".join(lines)


# ===================================================================
# 14. KÖZÖS ES/NQ RISK ÖSSZEGZŐ
# ===================================================================

def build_cross_market_risk_summary(
    es_regime: IndexRegime,
    nq_regime: IndexRegime,
    config: IndexRegimeConfig = CONFIG,
    *,
    es_exhaustion: dict[str, Any] | None = None,
    nq_exhaustion: dict[str, Any] | None = None,
    es_transition: dict[str, Any] | None = None,
    nq_transition: dict[str, Any] | None = None,
) -> str:
    """
    Egyszerű ES/NQ közös kockázati értelmezés.

    Ezt majd a run_es_nq_checklist.py fogja meghívni.
    """
    overlays = {"ES": es_exhaustion or {}, "NQ": nq_exhaustion or {}}
    blocked = [
        f"{symbol}: {value.get('state', 'N/A')}"
        for symbol, value in overlays.items() if value.get("block_new_longs", False)
    ]
    if blocked:
        return (
            "KÖZÖS INDEX LONG VÉDELEM: " + "; ".join(blocked) + ". "
            "A felsorolt indexeken az új long kapu zárt, a bullish trendpontszámtól "
            "függetlenül. Ez önmagában nem short vagy QQQ put belépési jel. "
            "Az ES/NQ kitettséget továbbra is közösen kezeld."
        )
    gates = [
        f"{symbol}: LONG={'TILTVA' if value.get('block_new_longs') else 'nincs új tiltás'}, "
        f"SHORT={'TILTVA' if value.get('block_new_shorts') else 'nincs új tiltás'}"
        for symbol, value in (("ES", es_transition or {}), ("NQ", nq_transition or {}))
        if value.get("block_new_longs") or value.get("block_new_shorts")
    ]
    if gates:
        return ("KÖZÖS HTF/WHIPSAW ADATKAPU: "+"; ".join(gates)+
                ". Az iránykonszenzus nem írja felül a belépési tiltást.")

    bullish_regimes = {
        IndexRegime.BULL_TREND,
    }

    bearish_regimes = {
        IndexRegime.BEAR_TREND,
        IndexRegime.CRASH_RISK,
    }

    if (
        es_regime in bullish_regimes
        and nq_regime in bullish_regimes
    ):
        return (
            "ES és NQ is bullish. "
            "A két long pozíció erősen korrelált; "
            f"a teljes közös index-risk cap legyen legfeljebb "
            f"{config.max_correlated_index_risk_pct * 100:.2f}%."
        )

    if (
        es_regime in bearish_regimes
        and nq_regime in bearish_regimes
    ):
        return (
            "ES és NQ is bearish. "
            "A két short pozíció erősen korrelált; "
            f"a teljes közös index-risk cap legyen legfeljebb "
            f"{config.max_correlated_index_risk_pct * 100:.2f}%."
        )

    if (
        es_regime in bullish_regimes
        and nq_regime in bearish_regimes
    ):
        return (
            "ES bullish, NQ bearish: leadership divergence vagy "
            "tech-specific gyengeség. Ne kezeld ezt egyetlen széles piaci trade-ként."
        )

    if (
        es_regime in bearish_regimes
        and nq_regime in bullish_regimes
    ):
        return (
            "ES bearish, NQ bullish: divergens indexkörnyezet. "
            "Ne nyiss automatikusan mindkét irányba pozíciót."
        )

    return (
        "Nincs egységes ES/NQ trendkonszenzus. "
        "Csökkentett méret vagy no-trade környezet."
    )

# ===================================================================
# 16. QQQ LONG PUT ADVISORY
# ===================================================================

def _build_qqq_put_advisory_legacy(
    es_snapshot: dict[str, Any],
    es_regime_result: dict[str, Any],
    nq_snapshot: dict[str, Any],
    nq_regime_result: dict[str, Any],
    config: IndexRegimeConfig = CONFIG,
) -> dict[str, Any]:
    """
    QQQ long put / put debit spread döntéstámogató modul.

    FONTOS:
    - Nem küld ordert.
    - Nem választ konkrét strike-ot.
    - Nem számol konkrét prémiumot.
    - Nem helyettesíti a QQQ opciós lánc vizsgálatát.
    - Csak azt mondja meg, hogy a jelenlegi ES/NQ/VIX környezet
      alkalmas-e QQQ bearish opciós setup keresésére.

    A tényleges strike és lejárat kiválasztásához később kell:
    - QQQ aktuális ár;
    - QQQ option chain;
    - delta;
    - implied volatility;
    - bid/ask spread;
    - open interest;
    - választott lejárat.
    """

    es_regime = es_regime_result["regime"]
    nq_regime = nq_regime_result["regime"]

    es_crash_score = es_regime_result["crash_score"]
    nq_crash_score = nq_regime_result["crash_score"]

    nq_rsi_2 = nq_snapshot["rsi_2"]
    nq_rsi_5 = nq_snapshot["rsi_5"]

    nq_vwap_distance_atr = nq_snapshot["vwap_distance_atr"]

    nq_gap_context = nq_snapshot["gap_context"]

    ndog_reaction = nq_gap_context["ndog_reaction"]["reaction"]
    nwog_reaction = nq_gap_context["nwog_reaction"]["reaction"]

    vix_analysis = es_snapshot["vix_analysis"]

    # ---------------------------------------------------------------
    # 1. ES/NQ strukturális bearish / crash konfirmáció
    # ---------------------------------------------------------------
    both_indexes_crash = (
        es_regime == IndexRegime.CRASH_RISK
        and nq_regime == IndexRegime.CRASH_RISK
    )

    nq_crash_es_bear = (
        nq_crash_score >= 5
        and es_crash_score >= 4
        and es_regime in (
            IndexRegime.BEAR_TREND,
            IndexRegime.CRASH_RISK,
        )
    )

    broad_bearish_confirmation = (
        both_indexes_crash
        or nq_crash_es_bear
    )

    # ---------------------------------------------------------------
    # 2. VIX stress-konfirmáció
    # ---------------------------------------------------------------
    vix_available = vix_analysis.get(
        "available",
        False,
    )

    vix_ratio = vix_analysis.get(
        "ratio",
        np.nan,
    )

    if vix_available:
        vix_stress_building = (
            vix_ratio >= config.vix_transition_ratio
        )

        vix_backwardation = (
            vix_ratio >= config.vix_stress_ratio
        )

    else:
        vix_stress_building = False
        vix_backwardation = False

    # ---------------------------------------------------------------
    # 3. Anti-chasing szabály
    # ---------------------------------------------------------------
    # Ha NQ túlzottan túladott, a fresh put entry tiltott,
    # mert short covering / reflex bounce jöhet.
    nq_extreme_oversold = (
        nq_rsi_2 < config.rsi2_extreme_oversold
        or nq_rsi_5 < config.rsi5_extreme_oversold
    )

    nq_too_far_below_vwap = (
        np.isfinite(nq_vwap_distance_atr)
        and nq_vwap_distance_atr
        < config.max_short_extension_below_vwap_atr
    )

    put_chasing_risk = (
        nq_extreme_oversold
        or nq_too_far_below_vwap
    )

    # ---------------------------------------------------------------
    # 4. Gap-alapú bearish rejection
    # ---------------------------------------------------------------
    bearish_gap_rejection = (
        ndog_reaction == "BEARISH_REJECTION"
        or nwog_reaction == "BEARISH_REJECTION"
    )

    bullish_gap_reclaim = (
        ndog_reaction == "BULLISH_RECLAIM"
        or nwog_reaction == "BULLISH_RECLAIM"
    )

    # ---------------------------------------------------------------
    # 5. Általános NQ price-action állapot
    # ---------------------------------------------------------------
    nq_below_vwap = (
        nq_snapshot["close"] < nq_snapshot["vwap"]
    )

    nq_bearish_di = (
        nq_snapshot["minus_di_5m"]
        > nq_snapshot["plus_di_5m"]
    )

    nq_adx_strong = (
        nq_snapshot["adx_5m"]
        > config.adx_strong_min
    )

    nq_volume_expansion = (
        nq_snapshot["volume_ratio"]
        >= config.volume_expansion_ratio
    )

    # ---------------------------------------------------------------
    # 6. Advisory-state logika
    # ---------------------------------------------------------------
    reasons: list[str] = []
    checklist: dict[str, bool] = {
        "ES crash score >= 4": es_crash_score >= 4,
        "NQ crash score >= 5": nq_crash_score >= 5,
        "NQ session VWAP alatt": nq_below_vwap,
        "NQ -DI > +DI": nq_bearish_di,
        "NQ ADX(5m) > 30": nq_adx_strong,
        "NQ volume expansion >= 1.3x": nq_volume_expansion,
        "VIX stress building >= 1.00": vix_stress_building,
        "VIX backwardation >= 1.10": vix_backwardation,
        "NQ nem extrém túladott": not nq_extreme_oversold,
        "NQ nincs >3 ATR-rel VWAP alatt": not nq_too_far_below_vwap,
        "Bearish NDOG/NWOG rejection": bearish_gap_rejection,
        "Nincs bullish NDOG/NWOG reclaim": not bullish_gap_reclaim,
    }

    # ---------------------------------------------------------------
    # Nincs crash-környezet
    # ---------------------------------------------------------------
    if not broad_bearish_confirmation:
        reasons.append(
            "Az ES/NQ crash-környezet még nincs kellően megerősítve."
        )

        return {
            "state": "NO_QQQ_PUT_SETUP",
            "trade_candidate": False,
            "entry_allowed": False,
            "reasons": reasons,
            "checklist": checklist,
            "recommended_structure": None,
            "instructions": [
                "Ne vegyél QQQ putot pusztán egy rövid intraday esés miatt.",
                "Várj legalább NQ crash score >= 5 és ES crash score >= 4 környezetre.",
                "Figyeld, hogy a NQ a session VWAP alatt marad-e.",
                "Figyeld, hogy kialakul-e volume-expansion és lower-high struktúra.",
            ],
        }

    # ---------------------------------------------------------------
    # Crash-környezet van, de túl késő putot venni
    # ---------------------------------------------------------------
    if put_chasing_risk:
        reasons.append(
            "Crash-környezet megerősített, de a NQ túlzottan túladott "
            "vagy túl messze van a VWAP alatt."
        )

        if nq_rsi_2 < config.rsi2_extreme_oversold:
            reasons.append(
                f"NQ RSI(2)={nq_rsi_2:.1f} < "
                f"{config.rsi2_extreme_oversold:.0f}"
            )

        if nq_rsi_5 < config.rsi5_extreme_oversold:
            reasons.append(
                f"NQ RSI(5)={nq_rsi_5:.1f} < "
                f"{config.rsi5_extreme_oversold:.0f}"
            )

        if nq_too_far_below_vwap:
            reasons.append(
                f"NQ ár {nq_vwap_distance_atr:.2f}× ATR-rel "
                "a session VWAP alatt van."
            )

        return {
            "state": "CRASH_CONFIRMED_WAIT_FOR_BOUNCE",
            "trade_candidate": True,
            "entry_allowed": False,
            "reasons": reasons,
            "checklist": checklist,
            "recommended_structure": "WAIT",
            "instructions": [
                "Ne vegyél friss QQQ putot a pánikgyertya vagy a mélypont közelében.",
                "Várj NQ / QQQ felpattanásra.",
                "Várj failed VWAP reclaimre vagy lower-high struktúrára.",
                "Várj arra, hogy a lefelé volumen ismét emelkedjen.",
                "Csak a bounce elutasítása után vizsgáld újra a put entryt.",
            ],
        }

    # ---------------------------------------------------------------
    # Crash-környezet van, de még nincs entry rejection
    # ---------------------------------------------------------------
    if not bearish_gap_rejection:
        reasons.append(
            "A crash-környezet megerősített, de nincs friss bearish "
            "NDOG/NWOG rejection."
        )

        if bullish_gap_reclaim:
            reasons.append(
                "Bullish gap reclaim látszik, ezért a friss put belépés tiltott."
            )

        return {
            "state": "WAIT_FOR_FAILED_RECLAIM",
            "trade_candidate": True,
            "entry_allowed": False,
            "reasons": reasons,
            "checklist": checklist,
            "recommended_structure": "WAIT",
            "instructions": [
                "Várj NQ/QQQ bounce-ra VWAP, NDOG, NWOG vagy prior-low zónába.",
                "A short/put trigger egy bearish rejection vagy failed reclaim legyen.",
                "Várj 1m lower high-ra és 5m bearish zárásra.",
                "Ideális esetben a volume ratio ismét 1.3x fölé emelkedik.",
                "Ne vegyél putot, ha a QQQ már egy nagy piros gyertyával megnyúlt.",
            ],
        }

    # ---------------------------------------------------------------
    # Crash + bearish rejection + nincs chasing
    # ---------------------------------------------------------------
    reasons.append(
        "ES/NQ crash-környezet, bearish gap rejection és nem-extrem "
        "NQ extension együtt teljesül."
    )

    if vix_backwardation:
        structure = "QQQ_LONG_PUT_OR_DEBIT_SPREAD"

        instructions = [
            "Ellenőrizd a QQQ aktuális árát és az opciós láncot TWS-ben.",
            "Nézz 7-14 DTE lejáratot, ha a további esést 1-3 napon belül várod.",
            "Long put jelölt: kb. 0.40-0.60 delta.",
            "Ha gyors, erős folytatást vársz és vállalod a teljes prémiumvesztést: sima long put.",
            "Ha magas az IV vagy plafonált downside célod van: bear put debit spread.",
            "Bear put spreadnél a long put lehet kb. 0.40-0.55 delta.",
            "A short put lehet kb. 0.15-0.30 delta, alacsonyabb strike-on.",
            "Ne használj naked short putot vagy naked short callt ehhez a crash setuphoz.",
            "A pozíció maximális kockázata a long put prémiuma vagy a debit spread nettó debitje legyen.",
        ]

    else:
        structure = "QQQ_PUT_DEBIT_SPREAD_PREFERRED"

        instructions = [
            "A crash price-action igazolt, de a VIX nem backwardated.",
            "Ez lehet technikai selloff, nem feltétlenül teljes rendszerszintű pánik.",
            "Első választásként 7-14 DTE bear put debit spreadet vizsgálj.",
            "Long put láb: kb. 0.40-0.55 delta.",
            "Short put láb: kb. 0.15-0.30 delta, lejjebb lévő strike.",
            "A spread csökkentheti a belépési prémiumot és az IV-visszaesés kockázatát.",
            "Sima long put csak akkor előnyösebb, ha gyors, erős, többnapos esést vársz.",
            "Ellenőrizd a bid/ask spreadet, open interestet és a QQQ opciós lánc likviditását.",
        ]

    return {
        "state": "QQQ_PUT_CANDIDATE",
        "trade_candidate": True,
        "entry_allowed": True,
        "reasons": reasons,
        "checklist": checklist,
        "recommended_structure": structure,
        "instructions": instructions,
    }


def build_qqq_put_advisory(
    es_snapshot: dict[str, Any], es_regime_result: dict[str, Any],
    nq_snapshot: dict[str, Any], nq_regime_result: dict[str, Any],
    config: IndexRegimeConfig = CONFIG,
) -> dict[str, Any]:
    """Preserve legacy setup rules, add non-bypassable data/transition vetoes."""
    result = _build_qqq_put_advisory_legacy(
        es_snapshot, es_regime_result, nq_snapshot, nq_regime_result, config)
    result["execution_allowed"] = False
    result["entry_scope"] = "PAPER_ONLY"
    nq_gate = nq_snapshot.get("transition_risk", {})
    core_ok = all(s.get("data_quality", "OK") == "OK" for s in (es_snapshot, nq_snapshot))
    transition_ok = not nq_gate.get("block_new_shorts", False)
    result["checklist"]["ES/NQ core adatminőség rendben"] = core_ok
    result["checklist"]["NQ transition/whipsaw short kapu nyitott"] = transition_ok
    # Keep the N/A distinction for optional volatility context.
    if not es_snapshot.get("vix_analysis", {}).get("available"):
        for key in ("VIX stress building >= 1.00", "VIX backwardation >= 1.10"):
            result["checklist"][key] = None
    result["checklist"][f"NQ nincs >{abs(config.max_short_extension_below_vwap_atr):g} ATR-rel "
                        f"a VWAP alatt (alap: {config.vwap_distance_atr_basis})"] = (
        result["checklist"].pop("NQ nincs >3 ATR-rel VWAP alatt"))
    result["vx_short_context_state"] = nq_snapshot.get("short_stress_context", {}).get("state", "N/A")
    if not core_ok or not transition_ok:
        result["entry_allowed"] = False
        result["state"] = "QQQ_PUT_BLOCKED_BY_DATA_OR_TRANSITION"
        result["recommended_structure"] = "WAIT"
        result["reasons"].append("A core/HTF/whipsaw kapu tiltása a legacy QQQ feltételeket is felülírja.")
        result["reasons"] += nq_gate.get("short_block_reasons", [])
    result["instructions"].append(
        "A VX-stressz nem bizonyít VIX-short zárást, és nem önálló QQQ put trigger.")
    return result


def format_qqq_put_advisory(
    advisory: dict[str, Any],
) -> str:
    """
    A QQQ put advisory konzolos, magyar nyelvű riportja.
    """
    lines = [
        "",
        "=" * 72,
        "QQQ LONG PUT / PUT DEBIT SPREAD ADVISORY",
        "=" * 72,
        f"Állapot:                {advisory['state']}",
        f"Put környezet jelölt:   {'IGEN' if advisory['trade_candidate'] else 'NEM'}",
        f"Paper belépési feltételek: {'IGEN' if advisory['entry_allowed'] else 'NEM'}",
        "Valós végrehajtási engedély: NEM",
        f"VX/NQ short-környezet: {advisory.get('vx_short_context_state', 'N/A')}",
    ]

    if advisory.get("recommended_structure"):
        lines.append(
            f"Javasolt struktúra:     {advisory['recommended_structure']}"
        )

    lines.extend(
        [
            "",
            "-" * 72,
            "CHECKLIST",
            "-" * 72,
        ]
    )

    for name, passed in advisory["checklist"].items():
        status = "N/A" if passed is None else "IGEN" if passed else "NEM"
        lines.append(f"[{status}] {name}")

    lines.extend(
        [
            "",
            "-" * 72,
            "ÉRTELMEZÉS",
            "-" * 72,
        ]
    )

    for reason in advisory["reasons"]:
        lines.append(f"- {reason}")

    lines.extend(
        [
            "",
            "-" * 72,
            "INSTRUKCIÓK",
            "-" * 72,
        ]
    )

    for index, instruction in enumerate(
        advisory["instructions"],
        start=1,
    ):
        lines.append(f"{index}. {instruction}")

    lines.extend(
        [
            "",
            "-" * 72,
            "FONTOS",
            "-" * 72,
            "Ez döntéstámogató / paper-trading jelzés.",
            "Nem küld ordert és nem választ automatikusan strike-ot.",
            "A konkrét QQQ option chain, delta, IV, bid/ask és lejárat",
            "ellenőrzése minden trade előtt szükséges.",
            "=" * 72,
        ]
    )

    return "\n".join(lines)


# ===================================================================
# 15. KÖZVETLEN FUTTATÁS
# ===================================================================

if __name__ == "__main__":
    print(
        "\n"
        "Az index_regime.py 1.2.0 modul sikeresen betöltődött.\n"
        "\n"
        "Adatlekéréshez futtasd a mellékelt run_es_nq_checklist.py fájlt.\n"
        "Ez a modul önmagában nem kapcsolódik a TWS-hez.\n"
        "\n"
        "Elérhető konfigurációk:\n"
        "- ES_CONFIG\n"
        "- NQ_CONFIG\n"
    )
