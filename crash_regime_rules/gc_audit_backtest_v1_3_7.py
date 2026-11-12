from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gc_regime_1_0_14 import (
    GoldRegime,
    build_gold_snapshot,
    calculate_gold_risk_plan,
    gold_regime_active,
)


# ===================================================================
# GC REGIME AUDIT BACKTEST VERSION 1.5.0
#
# Funkciók:
# - gc_1m_history.csv betöltése
# - GC rezhimlogika historikus auditálása
# - Bull Trend / Bear Liquidation / Neutral / Blow-off vizsgálata
# - Trade candidate és blocked trade összehasonlítása
# - Forward return, MFE, MAE számítás ATR-ben
# - RSI, VWAP-extension, ADX, volume, NDOG/NWOG filter audit
# - 1.0 ATR stop audit
# - Részletes státuszjelzés és hátralévő idő becslés
#
# FONTOS:
# - Nem küld ordert.
# - Nem nyit pozíciót.
# - Nem PnL backtest commission/slippage modelllel.
# - Ez egy rezhim- és filter-minőség audit.
# ===================================================================


# ===================================================================
# 1. FÁJLNEVEK
# ===================================================================

INPUT_CSV_PATH = "gc_1m_history.csv"

OUTPUT_OBSERVATIONS_CSV = "gc_audit_observations.csv"
OUTPUT_EVENTS_CSV = "gc_audit_events.csv"
OUTPUT_SUMMARY_CSV = "gc_audit_summary.csv"


# ===================================================================
# 2. AUDIT KONFIGURÁCIÓ
# ===================================================================

# Első futáshoz gyorsabb beállítás:
# 15 percenként készül egy audit snapshot.
EVALUATION_STEP_BARS = 15

# Minden snapshot előtt ennyi korábbi 1 perces GC bar kerül
# a regime engine-be.
# 1.3.2: 12 000 bar (~8-9 kereskedési nap). A 7000 bar mellett az
# observationök 73%-ából még hiányzott az NWOG (az előző hétvégi
# Péntek-záró / Vasárnap-nyitó pár nem esett az ablakba).
# Régi értékek: 2_000 (1.3.0), 7_000 (1.3.1).
SNAPSHOT_LOOKBACK_BARS = 12_000

# Mekkora időtávokon mérjük a signal utáni mozgást.
FORWARD_HORIZONS_MINUTES = [5, 15, 30, 60]

# Egy tartósan fennálló azonos state-ből ne legyen sok, egymástól
# alig különböző event.
EVENT_DEDUPLICATION_BARS = 30

# Milyen gyakran írjon státuszjelzőt a konzolra.
STATUS_UPDATE_EVERY = 50

# Ennyi részletes snapshot hiba jelenhet meg.
MAX_DETAILED_SNAPSHOT_ERRORS = 3

# Minimum ajánlott observation darabszám egy érdemi következtetéshez.
MINIMUM_SAMPLE_SIZE = 20

# Az engine elméleti risk-plan számításához.
EQUITY_USD = 10_000.0

# VWAP anti-chasing audit küszöb (a gc_regime_1_0_11 CONFIG értékével
# egyező, 5m ATR alapon). Régi érték (1m ATR): 3.0
VWAP_AUDIT_THRESHOLD_ATR = 5.0


# ===================================================================
# 3. LOGOLÁS
# ===================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

log = logging.getLogger(__name__)


# ===================================================================
# 4. CSV BETÖLTÉS
# ===================================================================

def load_gc_history(
    csv_path: str,
) -> pd.DataFrame:
    """
    Betölti a run_gc_checklist.py által elmentett GC 1 perces adatot.

    Várt formátum:

    date,open,high,low,close,volume
    """
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Nem található input CSV: {path.resolve()}\n"
            f"Előbb futtasd a run_gc_checklist.py scriptet, "
            f"hogy létrejöjjön a {INPUT_CSV_PATH}."
        )

    df = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    required_columns = [
        "date",
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
            f"Hiányzó oszlopok: {missing_columns}\n"
            f"Elérhető oszlopok: {list(df.columns)}"
        )

    df = df.set_index("date")

    df.columns = [
        str(column).lower()
        for column in df.columns
    ]

    df = df[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].copy()

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna()
    df = df.sort_index()

    if not isinstance(
        df.index,
        pd.DatetimeIndex,
    ):
        raise ValueError(
            "Nem sikerült DatetimeIndexet létrehozni a date oszlopból."
        )

    # Ha az index timezone-naive, UTC-ként kezeljük.
    if df.index.tz is None:
        df.index = df.index.tz_localize(
            "UTC"
        )

    minimum_required_bars = (
        SNAPSHOT_LOOKBACK_BARS
        + max(FORWARD_HORIZONS_MINUTES)
    )

    if len(df) < minimum_required_bars:
        raise ValueError(
            f"Túl kevés adat: {len(df):,} bar.\n"
            f"Minimum szükséges: {minimum_required_bars:,} bar."
        )

    return df


# ===================================================================
# 5. SEGÉDFÜGGVÉNYEK
# ===================================================================

def safe_float(
    value: Any,
    default: float = np.nan,
) -> float:
    """
    Biztonságos float konverzió.
    """
    try:
        result = float(value)

        if np.isfinite(result):
            return result

        return default

    except (
        TypeError,
        ValueError,
    ):
        return default


def safe_text(
    value: Any,
    default: str = "N/A",
) -> str:
    """
    Biztonságos szöveges konverzió.
    """
    if value is None:
        return default

    text = str(value).strip()

    if not text:
        return default

    return text


def format_duration(
    seconds: float,
) -> str:
    """
    Másodpercek olvasható formázása.

    Példák:
    - 8 mp
    - 2 perc 14 mp
    - 1 óra 04 perc 06 mp
    """
    if (
        not np.isfinite(seconds)
        or seconds < 0
    ):
        return "N/A"

    total_seconds = int(
        round(seconds)
    )

    hours, remainder = divmod(
        total_seconds,
        3600,
    )

    minutes, secs = divmod(
        remainder,
        60,
    )

    if hours > 0:
        return (
            f"{hours} óra "
            f"{minutes:02d} perc "
            f"{secs:02d} mp"
        )

    if minutes > 0:
        return (
            f"{minutes} perc "
            f"{secs:02d} mp"
        )

    return f"{secs} mp"


def regime_name(
    regime: Any,
) -> str:
    """
    GoldRegime enum értékéből stabil auditnév.

    Várt értékek:
    - BULLTREND
    - BEARLIQUIDATION
    - NEUTRAL
    - BLOWOFFRISK
    """
    if hasattr(regime, "name"):
        raw = str(regime.name)
    else:
        raw = str(regime)

    normalized = (
        raw.upper()
        .replace("_", "")
        .replace(" ", "")
        .replace("-", "")
    )

    # A gc_regime_1_0_10 modulban a bear rezsim neve BEAR_TREND,
    # régebbi verzióban BEAR_LIQUIDATION. Mindkettő egységesen
    # BEARLIQUIDATION néven kezelendő.
    if normalized in ("BEARTREND", "BEARLIQUIDATION"):
        return "BEARLIQUIDATION"

    return normalized


def direction_from_regime(
    regime: Any,
) -> int:
    """
    +1 = Long bias
    -1 = Short bias
     0 = Nincs directionális bias
    """
    name = regime_name(
        regime
    )

    if name == "BULLTREND":
        return 1

    if name == "BEARLIQUIDATION":
        return -1

    return 0


def direction_name(
    direction: int,
) -> str:
    """
    Irány számból olvasható név.
    """
    if direction == 1:
        return "LONG"

    if direction == -1:
        return "SHORT"

    return "NONE"


def snapshot_value(
    snapshot: dict[str, Any],
    compact_name: str,
    snake_name: str | None = None,
    default: Any = np.nan,
) -> Any:
    """
    Kezeli, ha a regime modul kulcsai compact vagy snake_case alakúak.

    Példák:
    - atr5m vagy atr_5m
    - rsi2 vagy rsi_2
    """
    if compact_name in snapshot:
        return snapshot[compact_name]

    if (
        snake_name is not None
        and snake_name in snapshot
    ):
        return snapshot[snake_name]

    return default


def get_gap_context(
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    """
    Gap context objektum lekérése.
    """
    gap_context = snapshot.get(
        "gapcontext",
        snapshot.get(
            "gap_context",
            {},
        ),
    )

    if isinstance(
        gap_context,
        dict,
    ):
        return gap_context

    return {}


def get_gap_reaction(
    snapshot: dict[str, Any],
    gap_name: str,
) -> str:
    """
    NDOG vagy NWOG reaction lekérése.

    Lehetséges értékek:
    - BULLISHRECLAIM
    - BEARISHREJECTION
    - NEARLEVEL
    - NORECENTINTERACTION
    - N/A
    """
    gap_context = get_gap_context(
        snapshot
    )

    reaction_data = gap_context.get(
        f"{gap_name}reaction",
        gap_context.get(
            f"{gap_name}_reaction",
            {},
        ),
    )

    if isinstance(
        reaction_data,
        dict,
    ):
        text = safe_text(
            reaction_data.get(
                "reaction",
                "N/A",
            )
        )

        # A modul BULLISH_RECLAIM / BEARISH_REJECTION /
        # NO_RECENT_INTERACTION alakot ad, a riport pedig az
        # alávonás nélküli neveket hasonlítja. Egységesítés:
        if text == "N/A":
            return text

        return text.upper().replace("_", "")

    return "N/A"


def get_gap_status(
    snapshot: dict[str, Any],
    gap_name: str,
) -> str:
    """
    NDOG vagy NWOG fill státusz / price location lekérése.
    """
    gap_context = get_gap_context(
        snapshot
    )

    fill_data = gap_context.get(
        f"{gap_name}fill",
        gap_context.get(
            f"{gap_name}_fill",
            {},
        ),
    )

    if isinstance(
        fill_data,
        dict,
    ):
        return safe_text(
            fill_data.get(
                "price_location",
                fill_data.get(
                    "pricelocation",
                    fill_data.get(
                        "status",
                        "N/A",
                    ),
                ),
            )
        )

    return "N/A"


# ===================================================================
# 6. FORWARD RETURN, MFE ÉS MAE
# ===================================================================

def record_forward_metrics(
    record: dict[str, Any],
    df: pd.DataFrame,
    row_index: int,
    atr_5m: float,
) -> None:
    """
    Elmenti minden horizontra:

    - Long return ATR-ben
    - Short return ATR-ben
    - Directionális return ATR-ben
    - Long/Short MFE ATR-ben
    - Long/Short MAE ATR-ben
    - Directionális MFE/MAE ATR-ben
    """
    entry_price = safe_float(
        df["close"].iloc[row_index]
    )

    direction = int(
        record.get(
            "direction",
            0,
        )
    )

    for horizon in FORWARD_HORIZONS_MINUTES:
        empty_columns = [
            f"forward_long_{horizon}m_atr",
            f"forward_short_{horizon}m_atr",
            f"forward_directional_{horizon}m_atr",
            f"long_mfe_{horizon}m_atr",
            f"long_mae_{horizon}m_atr",
            f"short_mfe_{horizon}m_atr",
            f"short_mae_{horizon}m_atr",
            f"mfe_{horizon}m_atr",
            f"mae_{horizon}m_atr",
        ]

        future_end_index = (
            row_index
            + horizon
        )

        valid_input = (
            np.isfinite(entry_price)
            and np.isfinite(atr_5m)
            and atr_5m > 0
            and future_end_index < len(df)
        )

        if not valid_input:
            for column in empty_columns:
                record[column] = np.nan

            continue

        future_df = df.iloc[
            row_index + 1:future_end_index + 1
        ]

        if future_df.empty:
            for column in empty_columns:
                record[column] = np.nan

            continue

        exit_price = safe_float(
            future_df["close"].iloc[-1]
        )

        future_high = safe_float(
            future_df["high"].max()
        )

        future_low = safe_float(
            future_df["low"].min()
        )

        # Long outcome.
        long_return = (
            exit_price
            - entry_price
        ) / atr_5m

        long_mfe = (
            future_high
            - entry_price
        ) / atr_5m

        long_mae = (
            future_low
            - entry_price
        ) / atr_5m

        # Short outcome.
        short_return = (
            entry_price
            - exit_price
        ) / atr_5m

        short_mfe = (
            entry_price
            - future_low
        ) / atr_5m

        short_mae = (
            entry_price
            - future_high
        ) / atr_5m

        record[
            f"forward_long_{horizon}m_atr"
        ] = long_return

        record[
            f"forward_short_{horizon}m_atr"
        ] = short_return

        record[
            f"long_mfe_{horizon}m_atr"
        ] = long_mfe

        record[
            f"long_mae_{horizon}m_atr"
        ] = long_mae

        record[
            f"short_mfe_{horizon}m_atr"
        ] = short_mfe

        record[
            f"short_mae_{horizon}m_atr"
        ] = short_mae

        if direction == 1:
            directional_return = long_return
            directional_mfe = long_mfe
            directional_mae = long_mae

        elif direction == -1:
            directional_return = short_return
            directional_mfe = short_mfe
            directional_mae = short_mae

        else:
            directional_return = np.nan
            directional_mfe = np.nan
            directional_mae = np.nan

        record[
            f"forward_directional_{horizon}m_atr"
        ] = directional_return

        record[
            f"mfe_{horizon}m_atr"
        ] = directional_mfe

        record[
            f"mae_{horizon}m_atr"
        ] = directional_mae


# ===================================================================
# 7. SNAPSHOT → AUDIT RECORD
# ===================================================================

def build_observation_record(
    snapshot: dict[str, Any],
    regime_result: dict[str, Any],
    risk_plan: dict[str, Any],
    row_index: int,
) -> dict[str, Any]:
    """
    Snapshotból és risk-planből egységes audit rekord.
    """
    regime = regime_result.get(
        "regime",
        GoldRegime.NEUTRAL,
    )

    regime_text = regime_name(
        regime
    )

    direction = direction_from_regime(
        regime
    )

    trade_allowed = bool(
        risk_plan.get(
            "tradeallowed",
            risk_plan.get(
                "trade_allowed",
                False,
            ),
        )
    )

    gap_context = get_gap_context(
        snapshot
    )

    return {
        "row_index": row_index,

        "timestamp": snapshot.get(
            "timestamp",
            pd.NaT,
        ),

        "regime": regime_text,

        "direction": direction,

        "direction_name": direction_name(
            direction
        ),

        "is_bull_regime": regime_text == "BULLTREND",

        "is_bear_regime": regime_text == "BEARLIQUIDATION",

        "is_neutral_regime": regime_text == "NEUTRAL",

        "is_blowoff_regime": regime_text == "BLOWOFFRISK",

        "trade_allowed": trade_allowed,

        "exhaustion_state": snapshot.get("exhaustion", {}).get("state", "N/A"),
        "exhaustion_data_quality": snapshot.get("exhaustion", {}).get(
            "data_quality", "N/A"
        ),
        "exhaustion_block_new_longs": snapshot.get("exhaustion", {}).get(
            "block_new_longs", False
        ),
        "exhaustion_short_candidate": snapshot.get("exhaustion", {}).get(
            "short_setup_candidate", False
        ),
        "exhaustion_hourly_extension_atr": snapshot.get("exhaustion", {}).get(
            "metrics", {}
        ).get("hourly_extension_atr", np.nan),
        "exhaustion_daily_extension_atr": snapshot.get("exhaustion", {}).get(
            "metrics", {}
        ).get("daily_extension_atr", np.nan),
        "exhaustion_vwap_extension_atr_5m": snapshot.get("exhaustion", {}).get(
            "metrics", {}
        ).get("vwap_extension_atr", np.nan),
        **transition_audit_fields(snapshot),

        "risk_reason": safe_text(
            risk_plan.get(
                "reason",
                "N/A",
            )
        ),

        "risk_details": safe_text(
            risk_plan.get(
                "details",
                "N/A",
            )
        ),

        "close": safe_float(
            snapshot_value(
                snapshot,
                "close",
            )
        ),

        "vwap": safe_float(
            snapshot_value(
                snapshot,
                "vwap",
            )
        ),

        "atr_1m": safe_float(
            snapshot_value(
                snapshot,
                "atr1m",
                "atr_1m",
            )
        ),

        "atr_5m": safe_float(
            snapshot_value(
                snapshot,
                "atr5m",
                "atr_5m",
            )
        ),

        "rsi_2": safe_float(
            snapshot_value(
                snapshot,
                "rsi2",
                "rsi_2",
            )
        ),

        "rsi_5": safe_float(
            snapshot_value(
                snapshot,
                "rsi5",
                "rsi_5",
            )
        ),

        "adx_5m": safe_float(
            snapshot_value(
                snapshot,
                "adx5m",
                "adx_5m",
            )
        ),

        "plus_di_5m": safe_float(
            snapshot_value(
                snapshot,
                "plusdi5m",
                "plus_di_5m",
            )
        ),

        "minus_di_5m": safe_float(
            snapshot_value(
                snapshot,
                "minusdi5m",
                "minus_di_5m",
            )
        ),

        "volume": safe_float(
            snapshot_value(
                snapshot,
                "volume",
            )
        ),

        "volume_avg_20": safe_float(
            snapshot_value(
                snapshot,
                "volumeavg20",
                "volume_avg_20",
            )
        ),

        "volume_ratio": safe_float(
            snapshot_value(
                snapshot,
                "volumeratio",
                "volume_ratio",
            )
        ),

        "vwap_distance_atr": safe_float(
            snapshot_value(
                snapshot,
                "vwapdistanceatr",
                "vwap_distance_atr",
            )
        ),

        "bull_score": safe_float(
            regime_result.get(
                "bullscore",
                regime_result.get(
                    "bull_score",
                    np.nan,
                ),
            )
        ),

        "bear_score": safe_float(
            regime_result.get(
                "bearscore",
                regime_result.get(
                    "bear_score",
                    np.nan,
                ),
            )
        ),

        "blowoff_score": safe_float(
            regime_result.get(
                "blowoffscore",
                regime_result.get(
                    "blowoff_score",
                    np.nan,
                ),
            )
        ),

        "ndog_reaction": get_gap_reaction(
            snapshot,
            "ndog",
        ),

        "nwog_reaction": get_gap_reaction(
            snapshot,
            "nwog",
        ),

        "ndog_status": get_gap_status(
            snapshot,
            "ndog",
        ),

        "nwog_status": get_gap_status(
            snapshot,
            "nwog",
        ),

        "ndog_count": safe_float(
            gap_context.get(
                "ndogcount",
                gap_context.get(
                    "ndog_count",
                    np.nan,
                ),
            )
        ),

        "nwog_count": safe_float(
            gap_context.get(
                "nwogcount",
                gap_context.get(
                    "nwog_count",
                    np.nan,
                ),
            )
        ),
    }


# ===================================================================
# 8. EVENT KÉPZÉS
# ===================================================================

def transition_audit_fields(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Flatten the causal HTF/whipsaw overlay into observation/event CSVs."""
    overlay = snapshot.get("transition_risk", {})
    whip = overlay.get("whipsaw", {})
    fields = {
        "transition_asof": overlay.get("asof"),
        "transition_block_new_longs": overlay.get("block_new_longs", False),
        "transition_block_new_shorts": overlay.get("block_new_shorts", False),
        "transition_confirmation_conflict": overlay.get("confirmation_conflict", False),
        "transition_long_block_reasons": "; ".join(overlay.get("long_block_reasons", [])),
        "transition_short_block_reasons": "; ".join(overlay.get("short_block_reasons", [])),
        "whipsaw_state": whip.get("state", "N/A"),
    }
    for key, value in whip.get("metrics", {}).items():
        fields[f"whipsaw_{key}"] = value
    for name, frame in overlay.get("timeframes", {}).items():
        for key in (
            "state", "current_context", "confirmed_direction", "pending_direction",
            "provisional_direction", "forming_state", "last_closed_at",
            "cross_at", "retest_at", "failed_at", "close_vs_vwap_atr_5m",
        ):
            fields[f"htf_{name}_{key}"] = frame.get(key)
    return fields


def event_type_for_observation(
    observation: dict[str, Any],
) -> str | None:
    """
    Bull vagy Bear contextből eventet képez.

    Lehetséges:
    - TRADE_CANDIDATE
    - BLOCKED_TRADE
    """
    direction = int(
        observation.get(
            "direction",
            0,
        )
    )

    if direction == 0:
        return None

    if bool(
        observation.get(
            "trade_allowed",
            False,
        )
    ):
        return "TRADE_CANDIDATE"

    return "BLOCKED_TRADE"


def build_event_record(
    observation: dict[str, Any],
    event_type: str,
) -> dict[str, Any]:
    """
    Observation rekordból event rekord.
    """
    event = observation.copy()

    event["event_type"] = event_type

    return event


# ===================================================================
# 9. FŐ AUDIT FUTTATÁS
# ===================================================================

def run_audit(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Végigfuttatja az auditot a GC 1 perces adatokon.

    FONTOS:
    A gc_regime_1_0_10.py alapértelmezett konfigurációját használjuk.

    Nem adunk át CONFIG objektumot sem a build_gold_snapshot(),
    sem a gold_regime_active(), sem a calculate_gold_risk_plan()
    hívásnak, mert a jelenlegi engine verzióban az extra második
    argumentum macro_contextként értelmeződhet.

    Ez megakadályozza ezt a hibát:

    AttributeError: 'GoldConfig' object has no attribute 'get'
    """
    observations: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    first_row_index = (
        SNAPSHOT_LOOKBACK_BARS
        - 1
    )

    max_forward_horizon = max(
        FORWARD_HORIZONS_MINUTES
    )

    if (
        len(df)
        <= first_row_index
        + max_forward_horizon
    ):
        raise ValueError(
            "Nincs elég adat a snapshot lookback és forward horizon mellett."
        )

    # 1.3.3: a mintavételi rács a CSV globális sorindexéhez van
    # igazítva (row_index % EVALUATION_STEP_BARS == 0). Így a különböző
    # lookback-kal futtatott auditok UGYANAZOKAT a bárokat értékelik,
    # tehát közvetlenül összehasonlíthatók. Korábban a rács fázisa a
    # lookbackkal együtt tolódott el.
    aligned_start = first_row_index + (
        (-first_row_index) % EVALUATION_STEP_BARS
    )

    evaluation_indices = list(
        range(
            aligned_start,
            len(df) - max_forward_horizon,
            EVALUATION_STEP_BARS,
        )
    )

    total_observations = len(
        evaluation_indices
    )

    if total_observations == 0:
        raise ValueError(
            "Nincs feldolgozható audit observation."
        )

    last_event_index_by_type: dict[
        str,
        int,
    ] = {}

    successful_snapshots = 0
    skipped_snapshots = 0

    audit_start_time = time.perf_counter()

    print("\n" + "=" * 100)
    print("AUDIT FELDOLGOZÁS INDUL")
    print("=" * 100)

    print(
        f"GC 1 perces barok:            {len(df):,}"
    )

    print(
        f"Snapshot lookback:            "
        f"{SNAPSHOT_LOOKBACK_BARS:,} bar"
    )

    print(
        f"Evaluation step:              "
        f"{EVALUATION_STEP_BARS} bar"
    )

    print(
        f"Várható observationök:        "
        f"{total_observations:,}"
    )

    print(
        f"Forward horizontok:           "
        f"{FORWARD_HORIZONS_MINUTES}"
    )

    print(
        f"Státusz frissítés:            "
        f"minden {STATUS_UPDATE_EVERY} observation"
    )

    print("=" * 100)

    for sequence_number, row_index in enumerate(
        evaluation_indices,
        start=1,
    ):
        start_index = max(
            0,
            row_index
            - SNAPSHOT_LOOKBACK_BARS
            + 1,
        )

        df_window = df.iloc[
            start_index:row_index + 1
        ].copy()

        current_timestamp = df.index[
            row_index
        ]

        try:
            # -------------------------------------------------------
            # FONTOS:
            #
            # Csak a szükséges adatot adjuk át.
            #
            # Így az engine a saját alapértelmezett GoldConfig
            # konfigurációját használja, és nincs rosszul átadott
            # macro_context / CONFIG paraméter.
            # -------------------------------------------------------
            snapshot = build_gold_snapshot(
                df_window
            )

            regime_result = gold_regime_active(
                snapshot
            )

            regime = regime_result.get(
                "regime",
                GoldRegime.NEUTRAL,
            )

            risk_plan = calculate_gold_risk_plan(
                snapshot,
                regime,
                EQUITY_USD,
            )

            observation = build_observation_record(
                snapshot=snapshot,
                regime_result=regime_result,
                risk_plan=risk_plan,
                row_index=row_index,
            )

            record_forward_metrics(
                record=observation,
                df=df,
                row_index=row_index,
                atr_5m=observation["atr_5m"],
            )

            observations.append(
                observation
            )

            successful_snapshots += 1

            event_type = event_type_for_observation(
                observation
            )

            if event_type is not None:
                previous_event_index = (
                    last_event_index_by_type.get(
                        event_type,
                        -10_000_000,
                    )
                )

                enough_distance = (
                    row_index
                    - previous_event_index
                    >= EVENT_DEDUPLICATION_BARS
                )

                if enough_distance:
                    event = build_event_record(
                        observation=observation,
                        event_type=event_type,
                    )

                    events.append(
                        event
                    )

                    last_event_index_by_type[
                        event_type
                    ] = row_index

        except Exception as error:
            skipped_snapshots += 1

            if (
                skipped_snapshots
                <= MAX_DETAILED_SNAPSHOT_ERRORS
            ):
                log.warning(
                    f"Audit snapshot kihagyva | "
                    f"index={row_index} | "
                    f"time={current_timestamp} | "
                    f"hiba={type(error).__name__}: {error}"
                )

            # Fail-fast:
            # Ha az első három snapshot is hibás, akkor nem futtatjuk
            # végig feleslegesen a maradék több száz iterationt.
            if (
                skipped_snapshots
                >= MAX_DETAILED_SNAPSHOT_ERRORS
                and successful_snapshots == 0
            ):
                raise RuntimeError(
                    "Az első snapshotok mind hibára futottak. "
                    "Az audit leáll, mert a gc_regime_1_0_10.py "
                    "függvényinterfésze nem kompatibilis az audit "
                    "aktuális hívásával."
                )

            elif skipped_snapshots == (
                MAX_DETAILED_SNAPSHOT_ERRORS
                + 1
            ):
                print(
                    "\n⚠️ További egyedi snapshot hibák elnyomva. "
                    "A futás végén az összesített kihagyott "
                    "snapshot szám megjelenik.\n"
                )

        # -----------------------------------------------------------
        # STÁTUSZJELZŐ
        # -----------------------------------------------------------
        is_status_update = (
            sequence_number
            % STATUS_UPDATE_EVERY
            == 0
        )

        is_last_observation = (
            sequence_number
            == total_observations
        )

        if (
            is_status_update
            or is_last_observation
        ):
            elapsed_seconds = (
                time.perf_counter()
                - audit_start_time
            )

            progress_pct = (
                100
                * sequence_number
                / total_observations
            )

            seconds_per_observation = (
                elapsed_seconds
                / sequence_number
            )

            remaining_observations = (
                total_observations
                - sequence_number
            )

            estimated_remaining_seconds = (
                remaining_observations
                * seconds_per_observation
            )

            observations_per_second = (
                sequence_number
                / elapsed_seconds
                if elapsed_seconds > 0
                else 0.0
            )

            print(
                "\n"
                f"[AUDIT STÁTUSZ] "
                f"{sequence_number:,}/{total_observations:,} "
                f"({progress_pct:6.2f}%)"
            )

            print(
                f"  Aktuális timestamp:        "
                f"{current_timestamp}"
            )

            print(
                f"  Sikeres snapshotok:        "
                f"{successful_snapshots:,}"
            )

            print(
                f"  Kihagyott snapshotok:      "
                f"{skipped_snapshots:,}"
            )

            print(
                f"  Eddigi eventek:            "
                f"{len(events):,}"
            )

            print(
                f"  Feldolgozási sebesség:     "
                f"{observations_per_second:.3f} observation/mp"
            )

            print(
                f"  Eltelt idő:                "
                f"{format_duration(elapsed_seconds)}"
            )

            print(
                f"  Becsült hátralévő idő:     "
                f"{format_duration(estimated_remaining_seconds)}"
            )

    total_elapsed_seconds = (
        time.perf_counter()
        - audit_start_time
    )

    print("\n" + "=" * 100)
    print("AUDIT FELDOLGOZÁS BEFEJEZVE")
    print("=" * 100)

    print(
        f"Tervezett observationök:      "
        f"{total_observations:,}"
    )

    print(
        f"Sikeres snapshotok:           "
        f"{successful_snapshots:,}"
    )

    print(
        f"Kihagyott snapshotok:         "
        f"{skipped_snapshots:,}"
    )

    print(
        f"Létrehozott observationök:    "
        f"{len(observations):,}"
    )

    print(
        f"Létrehozott eventek:          "
        f"{len(events):,}"
    )

    print(
        f"Teljes futási idő:            "
        f"{format_duration(total_elapsed_seconds)}"
    )

    print("=" * 100)

    observations_df = pd.DataFrame(
        observations
    )

    events_df = pd.DataFrame(
        events
    )

    if events_df.empty:
        events_df = pd.DataFrame(
            columns=["event_type"]
        )

    if observations_df.empty:
        raise RuntimeError(
            "Az audit nem hozott létre observation rekordokat. "
            "Ellenőrizd a fenti első részletes hibaüzenetet."
        )

    return (
        observations_df,
        events_df,
    )


# ===================================================================
# 10. STATISZTIKAI SEGÉDFÜGGVÉNYEK
# ===================================================================

def safe_metric_stats(
    df: pd.DataFrame,
    column: str,
) -> dict[str, float]:
    """
    Egy outcome oszlop alapstatisztikája.
    """
    if (
        df.empty
        or column not in df.columns
    ):
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "win_rate": np.nan,
            "std": np.nan,
        }

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "win_rate": np.nan,
            "std": np.nan,
        }

    return {
        "n": len(values),
        "mean": float(
            values.mean()
        ),
        "median": float(
            values.median()
        ),
        "win_rate": float(
            100
            * (
                values > 0
            ).mean()
        ),
        "std": float(
            values.std()
        ),
    }


def sample_quality(
    n: int,
) -> str:
    """
    Mintanagyság minősítése.
    """
    if n == 0:
        return "NINCS ADAT"

    if n < MINIMUM_SAMPLE_SIZE:
        return (
            f"KIS MINTA: N={n}, "
            f"minimum {MINIMUM_SAMPLE_SIZE} ajánlott"
        )

    return "MINTA RENDBEN"


def format_stats_line(
    label: str,
    stats: dict[str, float],
) -> str:
    """
    Konzolos statisztikai sor.
    """
    return (
        f"{label:<38} | "
        f"N={int(stats['n']):>5} | "
        f"Mean={stats['mean']:>7.3f} ATR | "
        f"Median={stats['median']:>7.3f} ATR | "
        f"WinRate={stats['win_rate']:>6.2f}% | "
        f"{sample_quality(int(stats['n']))}"
    )


BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 42


def day_block_bootstrap_diff(
    left_df: pd.DataFrame,
    left_column: str,
    right_df: pd.DataFrame,
    right_column: str,
) -> dict[str, float]:
    """
    Napi blokk-bootstrap a (left - right) átlagkülönbségre.

    Miért napi blokk?
    Az observationök 15 percenként készülnek, a forward horizont
    15-60 perc, ezért az egymás utáni minták átfedők és korreláltak.
    Az egyszerű t-érték ezt nem veszi figyelembe, és túlbecsüli a
    biztonságot. A napokat egészben húzzuk vissza, így a
    napon belüli korreláció megmarad.
    """
    nan_result = {
        "diff": np.nan,
        "lo": np.nan,
        "hi": np.nan,
        "days": 0,
    }

    for frame, column in (
        (left_df, left_column),
        (right_df, right_column),
    ):
        if (
            frame.empty
            or column not in frame.columns
            or "timestamp" not in frame.columns
        ):
            return nan_result

    def per_day(
        frame: pd.DataFrame,
        column: str,
    ) -> pd.DataFrame:
        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        stamps = pd.to_datetime(
            frame["timestamp"],
            utc=True,
            errors="coerce",
        ).dt.tz_convert("America/New_York")

        days = stamps.dt.date

        data = pd.DataFrame(
            {
                "day": days,
                "value": values,
            }
        ).dropna()

        return data.groupby("day")["value"].agg(
            ["sum", "count"]
        )

    left_days = per_day(
        left_df,
        left_column,
    )

    right_days = per_day(
        right_df,
        right_column,
    )

    all_days = left_days.index.union(
        right_days.index
    )

    if len(all_days) < 5:
        return nan_result

    left_days = left_days.reindex(
        all_days,
        fill_value=0,
    )

    right_days = right_days.reindex(
        all_days,
        fill_value=0,
    )

    left_sum = left_days["sum"].to_numpy(float)
    left_count = left_days["count"].to_numpy(float)
    right_sum = right_days["sum"].to_numpy(float)
    right_count = right_days["count"].to_numpy(float)

    if left_count.sum() == 0 or right_count.sum() == 0:
        return nan_result

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    day_count = len(all_days)

    picks = rng.integers(
        0,
        day_count,
        size=(
            BOOTSTRAP_RESAMPLES,
            day_count,
        ),
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        left_mean = (
            left_sum[picks].sum(axis=1)
            / left_count[picks].sum(axis=1)
        )

        right_mean = (
            right_sum[picks].sum(axis=1)
            / right_count[picks].sum(axis=1)
        )

    diffs = left_mean - right_mean
    diffs = diffs[np.isfinite(diffs)]

    if len(diffs) < 100:
        return nan_result

    return {
        "diff": float(
            left_sum.sum() / left_count.sum()
            - right_sum.sum() / right_count.sum()
        ),
        "lo": float(np.percentile(diffs, 2.5)),
        "hi": float(np.percentile(diffs, 97.5)),
        "days": day_count,
    }


def print_comparison(
    title: str,
    left_label: str,
    left_df: pd.DataFrame,
    left_column: str,
    right_label: str,
    right_df: pd.DataFrame,
    right_column: str,
) -> None:
    """
    Két auditcsoport összehasonlítása.
    """
    left_stats = safe_metric_stats(
        left_df,
        left_column,
    )

    right_stats = safe_metric_stats(
        right_df,
        right_column,
    )

    print("\n" + "-" * 110)
    print(title)
    print("-" * 110)

    print(
        format_stats_line(
            left_label,
            left_stats,
        )
    )

    print(
        format_stats_line(
            right_label,
            right_stats,
        )
    )

    boot = day_block_bootstrap_diff(
        left_df,
        left_column,
        right_df,
        right_column,
    )

    if np.isfinite(boot["diff"]):
        significant = (
            boot["lo"] > 0
            or boot["hi"] < 0
        )

        verdict = (
            "KÜLÖNBSÉG STATISZTIKAILAG KIMUTATHATÓ"
            if significant
            else "NEM SZIGNIFIKÁNS (a CI tartalmazza a 0-t)"
        )

        print(
            f"  Különbség (bal - jobb): {boot['diff']:+.3f} ATR | "
            f"95% napi-blokk bootstrap CI: "
            f"[{boot['lo']:+.3f}, {boot['hi']:+.3f}] | "
            f"{boot['days']} nap | {verdict}"
        )


def summary_row(
    label: str,
    df: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    """
    Summary CSV-hez egy audit rekord.
    """
    stats = safe_metric_stats(
        df,
        column,
    )

    return {
        "label": label,
        "metric": column,
        "n": stats["n"],
        "mean_atr": stats["mean"],
        "median_atr": stats["median"],
        "win_rate_pct": stats["win_rate"],
        "std_atr": stats["std"],
        "sample_quality": sample_quality(
            int(stats["n"])
        ),
    }


# ===================================================================
# 11. ALAPRIORT
# ===================================================================

def print_basic_audit_report(
    observations_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> None:
    """
    Rezhim- és event-eloszlás.
    """
    print("\n" + "=" * 100)
    print("GC REGIME AUDIT - ALAP ÖSSZEFOGLALÓ")
    print("=" * 100)

    print(
        f"\nÖsszes observation: "
        f"{len(observations_df):,}"
    )

    print(
        f"Összes deduplikált event: "
        f"{len(events_df):,}"
    )

    print("\nREZHIM ELOSZLÁS:")

    regime_counts = (
        observations_df["regime"]
        .value_counts(
            dropna=False,
        )
    )

    for regime, count in regime_counts.items():
        percentage = (
            100
            * count
            / len(observations_df)
        )

        print(
            f"  {str(regime):<24} "
            f"{count:>5,} db | "
            f"{percentage:>6.2f}%"
        )

    print("\nEVENT ELOSZLÁS:")

    if events_df.empty:
        print("  Nincs directionális event.")

    else:
        event_counts = (
            events_df["event_type"]
            .value_counts(
                dropna=False,
            )
        )

        for event_type, count in event_counts.items():
            percentage = (
                100
                * count
                / len(events_df)
            )

            print(
                f"  {str(event_type):<24} "
                f"{count:>5,} db | "
                f"{percentage:>6.2f}%"
            )

    print("=" * 100)


# ===================================================================
# 12. RÉSZLETES PARAMÉTER- ÉS FILTER AUDIT
# ===================================================================

def print_detailed_audit_questions(
    observations_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> None:
    """
    Strukturált válasz a fejlesztési kérdésekre.

    1. Túl sok trade-et tilt-e a rendszer?
    2. Túl kevés lehetőséget enged-e?
    3. Túl laza-e a Bull Trend filter?
    4. Túl laza-e a Bear Trend filter?
    5. Jó-e a -3 ATR short tiltás?
    6. Jó-e a +3 ATR long tiltás?
    7. Jó-e az RSI(2) < 10 tiltás?
    8. Hasznos-e az ADX > 30?
    9. Hasznos-e a Volume >= 1.3x?
    10. Hasznos-e az NDOG/NWOG?
    11. Jó-e az 1.0 ATR stop?
    """
    print("\n" + "=" * 100)
    print("RÉSZLETES PARAMÉTER- ÉS FILTER AUDIT")
    print("=" * 100)

    if observations_df.empty:
        print("Nincs observation adat.")
        return

    if events_df.empty:
        print(
            "Nincs directionális event. "
            "A részletes event-audit nem futtatható."
        )
        return

    all_observations = observations_df.copy()

    directional = all_observations[
        all_observations["direction"] != 0
    ].copy()

    bull = all_observations[
        all_observations["is_bull_regime"]
    ].copy()

    bear = all_observations[
        all_observations["is_bear_regime"]
    ].copy()

    candidates = events_df[
        events_df["event_type"]
        == "TRADE_CANDIDATE"
    ].copy()

    blocked = events_df[
        events_df["event_type"]
        == "BLOCKED_TRADE"
    ].copy()

    # ===============================================================
    # 1. TÚL SOK TRADE-ET TILT-E?
    # ===============================================================
    print("\n" + "=" * 100)
    print("1. TÚL SOK TRADE-ET TILT-E A RENDSZER?")
    print("=" * 100)

    print_comparison(
        title=(
            "TRADE_CANDIDATE VS BLOCKED_TRADE "
            "- 15 PERCES DIRECTIONÁLIS RETURN"
        ),
        left_label="TRADE_CANDIDATE",
        left_df=candidates,
        left_column="forward_directional_15m_atr",
        right_label="BLOCKED_TRADE",
        right_df=blocked,
        right_column="forward_directional_15m_atr",
    )

    print_comparison(
        title=(
            "TRADE_CANDIDATE VS BLOCKED_TRADE "
            "- 30 PERCES DIRECTIONÁLIS RETURN"
        ),
        left_label="TRADE_CANDIDATE",
        left_df=candidates,
        left_column="forward_directional_30m_atr",
        right_label="BLOCKED_TRADE",
        right_df=blocked,
        right_column="forward_directional_30m_atr",
    )

    print(
        "\nÉrtelmezés:"
        "\n- A filter valószínűleg hasznos, ha a TRADE_CANDIDATE jobb."
        "\n- A filter túl szigorú lehet, ha a BLOCKED_TRADE jobb."
    )

    # ===============================================================
    # 2. TÚL KEVÉS LEHETŐSÉGET ENGED?
    # ===============================================================
    print("\n" + "=" * 100)
    print("2. TÚL KEVÉS LEHETŐSÉGET ENGED-E A RENDSZER?")
    print("=" * 100)

    allowed_directional = directional[
        directional["trade_allowed"]
    ].copy()

    blocked_directional = directional[
        ~directional["trade_allowed"]
    ].copy()

    print(
        f"Összes observation:                   "
        f"{len(all_observations):,}"
    )

    print(
        f"Directionális trend context:           "
        f"{len(directional):,} "
        f"({100 * len(directional) / len(all_observations):.2f}%)"
    )

    print(
        f"Engedélyezett trend context:           "
        f"{len(allowed_directional):,}"
    )

    print(
        f"Tiltott trend context:                 "
        f"{len(blocked_directional):,}"
    )

    if len(directional) > 0:
        allowed_percentage = (
            100
            * len(allowed_directional)
            / len(directional)
        )

        print(
            f"Engedélyezési arány trend-contextben:  "
            f"{allowed_percentage:.2f}%"
        )

    # ===============================================================
    # 3. BULL TREND FILTER
    # ===============================================================
    print("\n" + "=" * 100)
    print("3. TÚL LAZA-E A BULL TREND FILTER?")
    print("=" * 100)

    non_bull = all_observations[
        ~all_observations["is_bull_regime"]
    ].copy()

    print_comparison(
        title=(
            "BULL TREND VS NEM-BULL "
            "- 15 PERCES LONG RETURN"
        ),
        left_label="BULL_TREND",
        left_df=bull,
        left_column="forward_long_15m_atr",
        right_label="NEM_BULL",
        right_df=non_bull,
        right_column="forward_long_15m_atr",
    )

    print_comparison(
        title=(
            "BULL TREND VS NEM-BULL "
            "- 30 PERCES LONG RETURN"
        ),
        left_label="BULL_TREND",
        left_df=bull,
        left_column="forward_long_30m_atr",
        right_label="NEM_BULL",
        right_df=non_bull,
        right_column="forward_long_30m_atr",
    )

    # ===============================================================
    # 4. BEAR TREND FILTER
    # ===============================================================
    print("\n" + "=" * 100)
    print("4. TÚL LAZA-E A BEAR TREND FILTER?")
    print("=" * 100)

    non_bear = all_observations[
        ~all_observations["is_bear_regime"]
    ].copy()

    print_comparison(
        title=(
            "BEAR LIQUIDATION VS NEM-BEAR "
            "- 15 PERCES SHORT RETURN"
        ),
        left_label="BEAR_LIQUIDATION",
        left_df=bear,
        left_column="forward_short_15m_atr",
        right_label="NEM_BEAR",
        right_df=non_bear,
        right_column="forward_short_15m_atr",
    )

    print_comparison(
        title=(
            "BEAR LIQUIDATION VS NEM-BEAR "
            "- 30 PERCES SHORT RETURN"
        ),
        left_label="BEAR_LIQUIDATION",
        left_df=bear,
        left_column="forward_short_30m_atr",
        right_label="NEM_BEAR",
        right_df=non_bear,
        right_column="forward_short_30m_atr",
    )

    # ===============================================================
    # 5. -3 ATR SHORT ANTI-CHASING
    # ===============================================================
    print("\n" + "=" * 100)
    print(f"5. JÓ-E A -{VWAP_AUDIT_THRESHOLD_ATR:g} ATR SHORT ANTI-CHASING TILTÁS?")
    print("=" * 100)

    short_extension = bear[
        bear["vwap_distance_atr"] < -VWAP_AUDIT_THRESHOLD_ATR
    ].copy()

    short_normal = bear[
        bear["vwap_distance_atr"] >= -VWAP_AUDIT_THRESHOLD_ATR
    ].copy()

    print_comparison(
        title=(
            f"BEAR: VWAP < -{VWAP_AUDIT_THRESHOLD_ATR:g} ATR VS VWAP >= -{VWAP_AUDIT_THRESHOLD_ATR:g} ATR "
            "- 15 PERCES SHORT RETURN"
        ),
        left_label=f"VWAP_EXTENSION_LT_-{VWAP_AUDIT_THRESHOLD_ATR:g}_ATR",
        left_df=short_extension,
        left_column="forward_short_15m_atr",
        right_label=f"VWAP_EXTENSION_GE_-{VWAP_AUDIT_THRESHOLD_ATR:g}_ATR",
        right_df=short_normal,
        right_column="forward_short_15m_atr",
    )

    # ===============================================================
    # 6. +3 ATR LONG ANTI-CHASING
    # ===============================================================
    print("\n" + "=" * 100)
    print(f"6. JÓ-E A +{VWAP_AUDIT_THRESHOLD_ATR:g} ATR LONG ANTI-CHASING TILTÁS?")
    print("=" * 100)

    long_extension = bull[
        bull["vwap_distance_atr"] > VWAP_AUDIT_THRESHOLD_ATR
    ].copy()

    long_normal = bull[
        bull["vwap_distance_atr"] <= VWAP_AUDIT_THRESHOLD_ATR
    ].copy()

    print_comparison(
        title=(
            f"BULL: VWAP > +{VWAP_AUDIT_THRESHOLD_ATR:g} ATR VS VWAP <= +{VWAP_AUDIT_THRESHOLD_ATR:g} ATR "
            "- 15 PERCES LONG RETURN"
        ),
        left_label=f"VWAP_EXTENSION_GT_+{VWAP_AUDIT_THRESHOLD_ATR:g}_ATR",
        left_df=long_extension,
        left_column="forward_long_15m_atr",
        right_label=f"VWAP_EXTENSION_LE_+{VWAP_AUDIT_THRESHOLD_ATR:g}_ATR",
        right_df=long_normal,
        right_column="forward_long_15m_atr",
    )

    # ===============================================================
    # 7. RSI(2) < 10 SHORT TILTÁS
    # ===============================================================
    print("\n" + "=" * 100)
    print("7. JÓ-E AZ RSI(2) < 10 SHORT TILTÁS?")
    print("=" * 100)

    rsi2_oversold = bear[
        bear["rsi_2"] < 10
    ].copy()

    rsi2_normal = bear[
        bear["rsi_2"] >= 10
    ].copy()

    print_comparison(
        title=(
            "BEAR: RSI(2) < 10 VS RSI(2) >= 10 "
            "- 15 PERCES SHORT RETURN"
        ),
        left_label="RSI2_LT_10",
        left_df=rsi2_oversold,
        left_column="forward_short_15m_atr",
        right_label="RSI2_GE_10",
        right_df=rsi2_normal,
        right_column="forward_short_15m_atr",
    )

    # ===============================================================
    # 8. ADX > 30
    # ===============================================================
    print("\n" + "=" * 100)
    print("8. JAVÍTJA-E AZ ADX(5M) > 30 A JEL MINŐSÉGÉT?")
    print("=" * 100)

    adx_strong = directional[
        directional["adx_5m"] > 30
    ].copy()

    adx_normal = directional[
        directional["adx_5m"] <= 30
    ].copy()

    print_comparison(
        title=(
            "DIRECTIONÁLIS CONTEXT: ADX > 30 VS ADX <= 30 "
            "- 15 PERCES RETURN"
        ),
        left_label="ADX_GT_30",
        left_df=adx_strong,
        left_column="forward_directional_15m_atr",
        right_label="ADX_LE_30",
        right_df=adx_normal,
        right_column="forward_directional_15m_atr",
    )

    # ===============================================================
    # 9. VOLUME >= 1.3X
    # ===============================================================
    print("\n" + "=" * 100)
    print("9. HASZNOS-E A VOLUME >= 1.3X SZABÁLY?")
    print("=" * 100)

    volume_expansion = directional[
        directional["volume_ratio"] >= 1.30
    ].copy()

    volume_normal = directional[
        directional["volume_ratio"] < 1.30
    ].copy()

    print_comparison(
        title=(
            "DIRECTIONÁLIS CONTEXT: VOLUME >= 1.3X VS VOLUME < 1.3X "
            "- 15 PERCES RETURN"
        ),
        left_label="VOLUME_GE_1_3X",
        left_df=volume_expansion,
        left_column="forward_directional_15m_atr",
        right_label="VOLUME_LT_1_3X",
        right_df=volume_normal,
        right_column="forward_directional_15m_atr",
    )

    # ===============================================================
    # 10. NDOG / NWOG
    # ===============================================================
    print("\n" + "=" * 100)
    print("10. HASZNOS-E AZ NDOG / NWOG REACTION FILTER?")
    print("=" * 100)

    bullish_gap_reclaim = bull[
        (
            bull["ndog_reaction"]
            == "BULLISHRECLAIM"
        )
        | (
            bull["nwog_reaction"]
            == "BULLISHRECLAIM"
        )
    ].copy()

    bull_no_gap_reaction = bull[
        (
            bull["ndog_reaction"]
            == "NORECENTINTERACTION"
        )
        & (
            bull["nwog_reaction"]
            == "NORECENTINTERACTION"
        )
    ].copy()

    bearish_gap_rejection = bear[
        (
            bear["ndog_reaction"]
            == "BEARISHREJECTION"
        )
        | (
            bear["nwog_reaction"]
            == "BEARISHREJECTION"
        )
    ].copy()

    bear_no_gap_reaction = bear[
        (
            bear["ndog_reaction"]
            == "NORECENTINTERACTION"
        )
        & (
            bear["nwog_reaction"]
            == "NORECENTINTERACTION"
        )
    ].copy()

    print_comparison(
        title=(
            "BULL: BULLISH GAP RECLAIM VS NINCS GAP REAKCIÓ "
            "- 15 PERCES LONG RETURN"
        ),
        left_label="BULLISH_GAP_RECLAIM",
        left_df=bullish_gap_reclaim,
        left_column="forward_long_15m_atr",
        right_label="NO_RECENT_GAP_INTERACTION",
        right_df=bull_no_gap_reaction,
        right_column="forward_long_15m_atr",
    )

    print_comparison(
        title=(
            "BEAR: BEARISH GAP REJECTION VS NINCS GAP REAKCIÓ "
            "- 15 PERCES SHORT RETURN"
        ),
        left_label="BEARISH_GAP_REJECTION",
        left_df=bearish_gap_rejection,
        left_column="forward_short_15m_atr",
        right_label="NO_RECENT_GAP_INTERACTION",
        right_df=bear_no_gap_reaction,
        right_column="forward_short_15m_atr",
    )

    # ===============================================================
    # SHARED MULTI-TIMEFRAME EXHAUSTION OVERLAY
    # ===============================================================
    print("\n" + "=" * 100)
    print("BLOW-OFF / PARABOLIKUS KIFÁRADÁS RÉTEG - KUTATÁSI AUDIT")
    print("=" * 100)

    if "exhaustion_state" in all_observations.columns:
        print(all_observations["exhaustion_state"].value_counts().to_string())
        print(
            "Az állapotok nem független minták és nem crash-valószínűségek. "
            "Napi magasabb idősík hiányában 60m-alapú elemzés."
        )
        allowed_overlay = bull[
            ~bull["exhaustion_block_new_longs"].astype(bool)
        ].copy()
        blocked_overlay = bull[
            bull["exhaustion_block_new_longs"].astype(bool)
        ].copy()
        for horizon in ("15m", "30m", "60m"):
            print_comparison(
                title=f"BULL CONTEXT: ÚJ RÉTEG ÁLTAL TILTOTT VS NEM TILTOTT - {horizon}",
                left_label="OVERLAY_BLOCKED_BULL",
                left_df=blocked_overlay,
                left_column=f"forward_long_{horizon}_atr",
                right_label="OVERLAY_UNBLOCKED_BULL",
                right_df=allowed_overlay,
                right_column=f"forward_long_{horizon}_atr",
            )

    # ===============================================================
    # 11. ATR STOP
    # ===============================================================
    print("\n" + "=" * 100)
    print("11. JÓ-E AZ 1.0 ATR(5M) STOP?")
    print("=" * 100)

    if candidates.empty:
        print(
            "Nincs TRADE_CANDIDATE event, ezért nincs ATR stop audit."
        )

    else:
        for horizon in [
            15,
            30,
            60,
        ]:
            mae_column = (
                f"mae_{horizon}m_atr"
            )

            mfe_column = (
                f"mfe_{horizon}m_atr"
            )

            mae_values = pd.to_numeric(
                candidates[mae_column],
                errors="coerce",
            ).dropna()

            mfe_values = pd.to_numeric(
                candidates[mfe_column],
                errors="coerce",
            ).dropna()

            if mae_values.empty:
                continue

            stop_hit_rate = (
                100
                * (
                    mae_values <= -1.0
                ).mean()
            )

            print(
                f"\n{horizon} perces horizon:"
            )

            print(
                f"  Observation darabszám:     "
                f"{len(mae_values):,}"
            )

            print(
                f"  Átlag MAE:                 "
                f"{mae_values.mean():.3f} ATR"
            )

            print(
                f"  Median MAE:                "
                f"{mae_values.median():.3f} ATR"
            )

            print(
                f"  1.0 ATR stop-hit arány:    "
                f"{stop_hit_rate:.2f}%"
            )

            if not mfe_values.empty:
                print(
                    f"  Átlag MFE:                 "
                    f"{mfe_values.mean():.3f} ATR"
                )

                print(
                    f"  Median MFE:                "
                    f"{mfe_values.median():.3f} ATR"
                )

    # ===============================================================
    # VÉGSŐ ÉRTELMEZÉS
    # ===============================================================
    print("\n" + "=" * 100)
    print("VÉGSŐ AUDIT ÉRTELMEZÉSI SZABÁLY")
    print("=" * 100)

    print(
        "1. Egy paramétert nem azért módosítunk, mert kevés vagy sok trade-et ad."
    )

    print(
        "2. Akkor módosítjuk, ha a tiltott vagy engedélyezett csoport "
        "forward outcome-ja történetileg gyengébb, mint az alternatíva."
    )

    print(
        "3. 20 observation alatt a következtetés csak jelzésértékű."
    )

    print(
        "4. 50+ observation már használható kezdeti összehasonlításhoz."
    )

    print(
        "5. A rövid történet főleg pipeline- és logikaellenőrzésre alkalmas."
    )

    print(
        "6. Paraméterváltoztatás előtt célszerű legalább 3-6 hónap, "
        "inkább 6-12 hónap GC 1 perces adatot használni."
    )

    print("=" * 100)


# ===================================================================
# 13. SUMMARY CSV
# ===================================================================

def build_summary_dataframe(
    observations_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rövid, exportálható summary tábla.
    """
    rows: list[dict[str, Any]] = []

    directional = observations_df[
        observations_df["direction"] != 0
    ].copy()

    bull = observations_df[
        observations_df["is_bull_regime"]
    ].copy()

    bear = observations_df[
        observations_df["is_bear_regime"]
    ].copy()

    if (
        events_df.empty
        or "event_type" not in events_df.columns
    ):
        events_df = pd.DataFrame(
            columns=["event_type"]
        )

    candidates = events_df[
        events_df["event_type"]
        == "TRADE_CANDIDATE"
    ].copy()

    blocked = events_df[
        events_df["event_type"]
        == "BLOCKED_TRADE"
    ].copy()

    rows.append(
        summary_row(
            "TRADE_CANDIDATE_15M_DIRECTIONAL",
            candidates,
            "forward_directional_15m_atr",
        )
    )

    rows.append(
        summary_row(
            "BLOCKED_TRADE_15M_DIRECTIONAL",
            blocked,
            "forward_directional_15m_atr",
        )
    )

    rows.append(
        summary_row(
            "TRADE_CANDIDATE_30M_DIRECTIONAL",
            candidates,
            "forward_directional_30m_atr",
        )
    )

    rows.append(
        summary_row(
            "BLOCKED_TRADE_30M_DIRECTIONAL",
            blocked,
            "forward_directional_30m_atr",
        )
    )

    rows.append(
        summary_row(
            "BULL_TREND_15M_LONG",
            bull,
            "forward_long_15m_atr",
        )
    )

    rows.append(
        summary_row(
            "BEAR_LIQUIDATION_15M_SHORT",
            bear,
            "forward_short_15m_atr",
        )
    )

    rows.append(
        summary_row(
            "ADX_GT_30_15M_DIRECTIONAL",
            directional[
                directional["adx_5m"] > 30
            ],
            "forward_directional_15m_atr",
        )
    )

    rows.append(
        summary_row(
            "ADX_LE_30_15M_DIRECTIONAL",
            directional[
                directional["adx_5m"] <= 30
            ],
            "forward_directional_15m_atr",
        )
    )

    rows.append(
        summary_row(
            "VOLUME_GE_1_3X_15M_DIRECTIONAL",
            directional[
                directional["volume_ratio"] >= 1.30
            ],
            "forward_directional_15m_atr",
        )
    )

    rows.append(
        summary_row(
            "VOLUME_LT_1_3X_15M_DIRECTIONAL",
            directional[
                directional["volume_ratio"] < 1.30
            ],
            "forward_directional_15m_atr",
        )
    )

    rows.append(
        summary_row(
            "RSI2_LT_10_BEAR_15M_SHORT",
            bear[
                bear["rsi_2"] < 10
            ],
            "forward_short_15m_atr",
        )
    )

    rows.append(
        summary_row(
            f"VWAP_LT_MINUS_{VWAP_AUDIT_THRESHOLD_ATR:g}_ATR_BEAR_15M_SHORT",
            bear[
                bear["vwap_distance_atr"] < -VWAP_AUDIT_THRESHOLD_ATR
            ],
            "forward_short_15m_atr",
        )
    )

    rows.append(
        summary_row(
            f"VWAP_GT_PLUS_{VWAP_AUDIT_THRESHOLD_ATR:g}_ATR_BULL_15M_LONG",
            bull[
                bull["vwap_distance_atr"] > VWAP_AUDIT_THRESHOLD_ATR
            ],
            "forward_long_15m_atr",
        )
    )

    if "exhaustion_state" in observations_df.columns:
        for state in (
            "NORMAL", "PARABOLIC_EXTENSION", "EXHAUSTION_RISK",
            "BREAKDOWN_CONFIRMED", "UNKNOWN",
        ):
            subset = observations_df[
                observations_df["exhaustion_state"] == state
            ]
            rows.append(summary_row(
                f"EXHAUSTION_{state}_30M_LONG", subset, "forward_long_30m_atr"
            ))
        for blocked_flag in (True, False):
            subset = bull[
                bull["exhaustion_block_new_longs"].astype(bool) == blocked_flag
            ]
            rows.append(summary_row(
                f"OVERLAY_{'BLOCKED' if blocked_flag else 'UNBLOCKED'}_BULL_30M_LONG",
                subset, "forward_long_30m_atr",
            ))

    if "whipsaw_state" in directional.columns:
        for state in ("LOW", "ELEVATED", "HIGH", "UNKNOWN"):
            rows.append(summary_row(
                f"WHIPSAW_{state}_30M_DIRECTIONAL",
                directional[directional["whipsaw_state"] == state],
                "forward_directional_30m_atr",
            ))
        for flag in (True, False):
            for label, subset, column, metric in (
                ("BULL", bull, "transition_block_new_longs", "forward_long_30m_atr"),
                ("BEAR", bear, "transition_block_new_shorts", "forward_short_30m_atr"),
            ):
                rows.append(summary_row(
                    f"TRANSITION_{'BLOCKED' if flag else 'UNBLOCKED'}_{label}_30M",
                    subset[subset[column].astype(bool) == flag], metric,
                ))

    return pd.DataFrame(rows)


# ===================================================================
# 14. CSV EXPORT
# ===================================================================

def save_audit_outputs(
    observations_df: pd.DataFrame,
    events_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    """
    Három audit CSV fájl mentése.
    """
    observations_df.to_csv(
        OUTPUT_OBSERVATIONS_CSV,
        index=False,
    )

    events_df.to_csv(
        OUTPUT_EVENTS_CSV,
        index=False,
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY_CSV,
        index=False,
    )

    print("\n" + "=" * 100)
    print("AUDIT CSV EXPORTOK")
    print("=" * 100)

    print(
        f"Observation export: "
        f"{OUTPUT_OBSERVATIONS_CSV} | "
        f"{len(observations_df):,} sor"
    )

    print(
        f"Event export:       "
        f"{OUTPUT_EVENTS_CSV} | "
        f"{len(events_df):,} sor"
    )

    print(
        f"Summary export:     "
        f"{OUTPUT_SUMMARY_CSV} | "
        f"{len(summary_df):,} sor"
    )

    print("=" * 100)


# ===================================================================
# 15. MAIN
# ===================================================================

def main() -> None:
    """
    Teljes audit futtatása.
    """
    try:
        print("=" * 100)
        print("GC REGIME AUDIT BACKTEST VERSION 1.5.0")
        print("=" * 100)

        print(
            f"Input CSV:                  "
            f"{INPUT_CSV_PATH}"
        )

        print(
            f"Evaluation step:            "
            f"{EVALUATION_STEP_BARS} bar"
        )

        print(
            f"Snapshot lookback:          "
            f"{SNAPSHOT_LOOKBACK_BARS:,} bar"
        )

        print(
            f"Forward horizontok:         "
            f"{FORWARD_HORIZONS_MINUTES}"
        )

        print(
            f"Event deduplikáció:         "
            f"{EVENT_DEDUPLICATION_BARS} bar"
        )

        print(
            f"Státusz frissítés:          "
            f"minden {STATUS_UPDATE_EVERY} observation"
        )

        print("=" * 100)

        print(
            "\n[1/5] GC 1 perces CSV betöltése..."
        )

        df = load_gc_history(
            INPUT_CSV_PATH
        )

        print(
            f"GC CSV sikeresen betöltve: "
            f"{len(df):,} db 1 perces bar"
        )

        print(
            f"Első timestamp: {df.index[0]}"
        )

        print(
            f"Utolsó timestamp: {df.index[-1]}"
        )

        print(
            "\n[2/5] Regime snapshotok és forward outcome-ok számítása..."
        )

        observations_df, events_df = run_audit(
            df
        )

        print(
            "\n[3/5] Summary dataframe építése..."
        )

        summary_df = build_summary_dataframe(
            observations_df,
            events_df,
        )

        print(
            "\n[4/5] Konzolos audit riport készítése..."
        )

        print_basic_audit_report(
            observations_df,
            events_df,
        )

        print_detailed_audit_questions(
            observations_df,
            events_df,
        )

        print(
            "\n[5/5] CSV exportok mentése..."
        )

        save_audit_outputs(
            observations_df,
            events_df,
            summary_df,
        )

        print(
            "\nAUDIT SIKERESEN LEFUTOTT."
        )

    except Exception as error:
        print("\n" + "=" * 100)
        print("KRITIKUS AUDIT HIBA")
        print("=" * 100)

        print(
            f"Hibatípus: {type(error).__name__}"
        )

        print(
            f"Üzenet: {error}"
        )

        print("=" * 100)

        log.exception(
            "GC audit kritikus hiba:"
        )


if __name__ == "__main__":
    main()
