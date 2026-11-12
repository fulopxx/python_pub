from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ib_insync import IB, ContFuture, util

# FONTOS:
# A modul neve a jelenlegi verziószámozott fájlnak megfelelően:
from gc_regime_1_0_12 import run_gold_checklist


# ===================================================================
# RUN GC CHECKLIST VERSION 1.2.0 (inkrementális GC letöltés) (1 hónapos történet, CSV merge)
#
# Feladat:
# - Kapcsolódás TWS-hez
# - GC 1 perces historical adat letöltése
# - DXY / DX 1 perces historical adat letöltése
# - DXY session VWAP és 15 perces változás számítás
# - Macro context átadása a gc_regime_1_0_10.py modulnak
# - GC 1 perces adat export audit/backtest célra
# - Konzolos GC regime report
#
# Nem küld ordert.
# Nem nyit pozíciót.
# ===================================================================


# ===================================================================
# 1. TWS / IBKR KONFIGURÁCIÓ
# ===================================================================

IB_HOST = "127.0.0.1"

# TWS Paper Trading:
IB_PORT = 7497

# TWS Live Trading esetén:
# IB_PORT = 7496

# Egyedi API kliensazonosító.
# Ne legyen ugyanaz, mint más futó scripté.
CLIENT_ID = 20

# Csak elméleti GC/MGC risk sizinghez.
EQUITY_USD = 10_000.0

# -------------------------------------------------------------------
# GC adat
# -------------------------------------------------------------------
# 1.1.0: 1 hónap (régi: 10 nap). Szükséges:
# - NDOG/NWOG számításhoz
# - prior session high/low-hoz
# - session VWAP-hoz
# - audit CSV exporthoz
# 1.1: 10 D helyett 1 hónap (első körben).
# Régi érték: GC_HISTORICAL_DURATION = "10 D"
GC_HISTORICAL_DURATION = "1 M"

# Ha az 1 hónapos egyetlen kérés hibára fut vagy túl kevés bar jön,
# a script ennyi napos darabokban, visszafelé lapozva tölt le.
GC_HISTORY_TOTAL_DAYS = 30
GC_CHUNK_DAYS = 5
CHUNK_PAUSE_SECONDS = 3.0

# True = a meglévő gc_1m_history.csv-hez hozzáfűzi az újat
# (duplikátumok nélkül), így a történet idővel nő.
MERGE_WITH_EXISTING_HISTORY = True

# 1.1.1: az IBKR az egyetlen "1 M" kérésre 60 mp után timeoutolt,
# a chunkolt letöltés viszont működik. False = rögtön chunkolva tölt.
GC_TRY_SINGLE_REQUEST_FIRST = False

# 1.2.0: inkrementális letöltés. Ha a gc_1m_history.csv létezik, csak a
# hiányzó időszakot kérjük le (+ átfedés), nem az egész hónapot.
INCREMENTAL_GC_DOWNLOAD = True

# Átfedés (nap) a legutolsó mentett bar előtt, a duplikátumokat a
# merge kiszűri. Segít, ha az utolsó bar csonka volt.
INCREMENTAL_OVERLAP_DAYS = 1

# Ha a CSV-ben ennél régebbi az utolsó bar, a script a teljes
# GC_HISTORY_TOTAL_DAYS időszakot tölti újra (lyukmentes történet).
INCREMENTAL_MAX_GAP_DAYS = 25

# A checklist ennyi utolsó 1 perces bart kap a teljes történetből
# (a gyors futás miatt; kb. 10 nap).
GC_CHECKLIST_LOOKBACK_BARS = 14_000

# -------------------------------------------------------------------
# DXY adat
# -------------------------------------------------------------------
# DXY session VWAP és DXY 15m változás számításához.
DXY_HISTORICAL_DURATION = "2 D"

BAR_SIZE = "1 min"

# False = teljes Globex / ICE session.
USE_RTH = False

# 4 = Delayed market data.
# Ha később van real-time előfizetésed, használhatsz 1-et.
MARKET_DATA_TYPE = 4


# ===================================================================
# 2. REAL YIELD VERSION 1
# ===================================================================

# -------------------------------------------------------------------
# REAL YIELD V1 KORLÁT
# -------------------------------------------------------------------
#
# A gc_regime_1_0_10.py támogatja az alábbi adatokat:
#
# real_yield_10y
# real_yield_change_15m_bps
#
# De jelenleg nem töltünk le automatikus real-yield intraday adatot.
#
# Miért?
# - Az FRED DFII10 napi reálhozam-adat, nem intraday.
# - Nem korrekt napi adatból 15 perces hozamváltozást képezni.
# - Valódi intraday TIPS/reálhozam feed kellene.
#
# Version 1-ben:
# - DXY: automatikus IBKR futures adat.
# - 10Y real yield: N/A.
#
# Ha később van saját megbízható intraday real-yield forrásod,
# akkor állítsd True-ra, és töltsd ki a manuális értékeket.
# -------------------------------------------------------------------

USE_MANUAL_REAL_YIELD_CONTEXT = False

MANUAL_REAL_YIELD_10Y = np.nan
MANUAL_REAL_YIELD_CHANGE_15M_BPS = np.nan


# ===================================================================
# 3. CSV EXPORT
# ===================================================================

# A GC audit/backtest script ezt a fájlt fogja használni.
GC_HISTORY_EXPORT_PATH = "gc_1m_history.csv"

# Ha True:
# minden run_gc_checklist.py futáskor frissíti / felülírja a fájlt.
EXPORT_GC_HISTORY_FOR_AUDIT = True


# ===================================================================
# 4. LOGOLÁS
# ===================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

log = logging.getLogger(__name__)


# ===================================================================
# 5. IBKR DATA SEGÉDFÜGGVÉNYEK
# ===================================================================

def bars_to_dataframe(
    bars: list[Any],
    label: str,
) -> pd.DataFrame:
    """
    IBKR HistoricalData listából tiszta OHLCV DataFrame.

    Várt oszlopok:
    - open
    - high
    - low
    - close
    - volume

    Index:
    - date
    """
    if not bars:
        raise RuntimeError(
            f"{label}: az IBKR nem adott vissza történeti adatot."
        )

    df = util.df(bars)

    if df.empty:
        raise RuntimeError(
            f"{label}: a DataFrame üres."
        )

    if "date" not in df.columns:
        raise RuntimeError(
            f"{label}: nincs 'date' oszlop. "
            f"Elérhető oszlopok: {list(df.columns)}"
        )

    df = df.set_index("date")

    df.columns = [
        str(column).lower()
        for column in df.columns
    ]

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
        raise RuntimeError(
            f"{label}: hiányzó oszlopok: {missing_columns}. "
            f"Elérhető oszlopok: {list(df.columns)}"
        )

    df = df[
        required_columns
    ].copy()

    df = df.dropna()
    df = df.sort_index()

    if len(df) < 100:
        raise RuntimeError(
            f"{label}: túl kevés adat érkezett: {len(df)} db bar."
        )

    return df


def fetch_historical_1m_data(
    ib: IB,
    contract: Any,
    duration: str,
    label: str,
) -> pd.DataFrame:
    """
    Historical 1 perces TRADES adat letöltése.
    """
    log.info(
        f"📊 {label}: "
        f"{duration} / {BAR_SIZE} "
        f"historical data letöltése..."
    )

    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting=BAR_SIZE,
        whatToShow="TRADES",
        useRTH=USE_RTH,
        formatDate=1,
    )

    df = bars_to_dataframe(
        bars=bars,
        label=label,
    )

    log.info(
        f"   ✅ {label}: "
        f"{len(df):,} db 1 perces bar letöltve."
    )

    return df


def load_existing_gc_history() -> pd.DataFrame | None:
    """
    A meglévő gc_1m_history.csv betöltése (UTC index), vagy None.
    """
    path = Path(GC_HISTORY_EXPORT_PATH)

    if not MERGE_WITH_EXISTING_HISTORY or not path.exists():
        return None

    try:
        old = pd.read_csv(
            path,
            parse_dates=["date"],
            index_col="date",
        )

        if old.empty:
            return None

        if old.index.tz is None:
            old.index = old.index.tz_localize("UTC")
        else:
            old.index = old.index.tz_convert("UTC")

        return old.sort_index()

    except Exception as error:
        log.warning(f"A meglévő CSV nem olvasható ({error}).")
        return None


def days_to_download(
    existing: pd.DataFrame | None,
) -> int:
    """
    Hány (kereskedési) napnyi adatot kell letölteni.

    - Nincs CSV vagy túl régi: a teljes GC_HISTORY_TOTAL_DAYS.
    - Van friss CSV: csak a hiányzó rész + átfedés.
    """
    if (
        not INCREMENTAL_GC_DOWNLOAD
        or existing is None
    ):
        return GC_HISTORY_TOTAL_DAYS

    now_utc = pd.Timestamp.now(tz="UTC")
    last_bar = existing.index.max()

    gap_days = (now_utc - last_bar).total_seconds() / 86400.0

    if gap_days > INCREMENTAL_MAX_GAP_DAYS:
        log.info(
            f"A CSV utolsó bar {gap_days:.1f} napos, "
            f"teljes újratöltés."
        )
        return GC_HISTORY_TOTAL_DAYS

    days = int(np.ceil(gap_days)) + INCREMENTAL_OVERLAP_DAYS

    return max(1, min(days, GC_HISTORY_TOTAL_DAYS))


def fetch_gc_history_with_fallback(
    ib: IB,
    contract: Any,
    duration: str,
    label: str,
    total_days: int = GC_HISTORY_TOTAL_DAYS,
) -> pd.DataFrame:
    """
    Először egyetlen kéréssel próbálja (ContFuture, üres endDateTime).
    Ha ez hibázik vagy túl kevés bart ad, a kvalifikált konkrét
    kontraktuson, visszafelé lapozva, darabokban tölt.

    FIGYELEM: a ContFuture csak üres endDateTime-mal kérdezhető, ezért
    a lapozás konkrét (front month) kontraktuson megy. Ha a hónap
    átível egy kontraktusváltáson (roll), a chunkok között ár-ugrás
    lehet. A GC aktív hónapjai: Feb, Apr, Jun, Aug, Oct, Dec.
    """
    expected_min_bars = int(total_days * 1380 * 0.5)

    try:
        if not GC_TRY_SINGLE_REQUEST_FIRST:
            raise RuntimeError("egyetlen kérés kihagyva (beállítás)")

        df = fetch_historical_1m_data(
            ib=ib,
            contract=contract,
            duration=duration,
            label=label,
        )

        if len(df) >= expected_min_bars:
            return df

        log.warning(
            f"{label}: csak {len(df):,} bar jött "
            f"(várt min. {expected_min_bars:,}). Chunkolt letöltés."
        )

    except Exception as error:
        log.warning(
            f"{label}: az egyetlen {duration} kérés hibázott: "
            f"{error}. Chunkolt letöltés."
        )

    from ib_insync import Contract

    chunk_contract = Contract(
        conId=contract.conId,
        exchange=contract.exchange or "COMEX",
    )

    ib.qualifyContracts(chunk_contract)

    frames: list[pd.DataFrame] = []
    end_time = ""
    chunks = max(1, -(-total_days // GC_CHUNK_DAYS))

    for index in range(chunks):
        log.info(
            f"📊 {label}: chunk {index + 1}/{chunks} "
            f"({GC_CHUNK_DAYS} D, end={end_time or 'now'})"
        )

        bars = ib.reqHistoricalData(
            chunk_contract,
            endDateTime=end_time,
            durationStr=(
                f"{min(GC_CHUNK_DAYS, total_days - index * GC_CHUNK_DAYS)} D"
            ),
            barSizeSetting=BAR_SIZE,
            whatToShow="TRADES",
            useRTH=USE_RTH,
            formatDate=2,
        )

        if not bars:
            log.warning(f"{label}: üres chunk, leállok.")
            break

        part = bars_to_dataframe(
            bars=bars,
            label=label,
        )

        frames.append(part)

        first_ts = pd.Timestamp(part.index.min())

        if first_ts.tzinfo is None:
            first_ts = first_ts.tz_localize("UTC")
        else:
            first_ts = first_ts.tz_convert("UTC")

        # Az "YYYYMMDD-HH:MM:SS" forma IBKR-nél UTC időt jelent.

        end_time = first_ts.strftime(
            "%Y%m%d-%H:%M:%S"
        )

        time.sleep(CHUNK_PAUSE_SECONDS)

    if not frames:
        raise RuntimeError(f"{label}: a chunkolt letöltés sem adott adatot.")

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    log.info(f"   ✅ {label}: {len(df):,} bar összesen (chunkolt).")

    return df


# ===================================================================
# 6. IDŐZÓNA
# ===================================================================

def to_new_york_time(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Indexet America/New_York időzónára konvertál.
    """
    result = df.copy()

    if result.index.tz is None:
        result.index = result.index.tz_localize("UTC")

    result.index = result.index.tz_convert(
        "America/New_York"
    )

    return result


# ===================================================================
# 7. DXY VERSION 1 SZÁMÍTÁS
# ===================================================================

def dxy_session_key(
    timestamp: pd.Timestamp,
) -> pd.Timestamp:
    """
    DXY futures session key.

    Version 1 session-feltételezés:

    Vasárnap:
        18:00 ET → heti első DXY session indul.

    Hétfő-csütörtök:
        20:00 ET → új DXY session indul.

    Következő nap:
        17:00 ET → napi DXY session zár.

    Ez nem azonos a GC 18:00 ET session-resetjével.
    """
    weekday = timestamp.weekday()
    hour = timestamp.hour

    # Vasárnap 18:00 után a session vasárnapi kulcsot kap.
    if weekday == 6 and hour >= 18:
        return timestamp.normalize()

    # Minden más esetben 20 órával visszatolva kapjuk meg
    # a DXY session kezdőnapját.
    return (
        timestamp
        - pd.Timedelta(hours=20)
    ).normalize()


def add_dxy_session_key(
    df_dxy_ny: pd.DataFrame,
) -> pd.DataFrame:
    """
    DXY session key hozzáadása.
    """
    result = df_dxy_ny.copy()

    result["dxy_session_key"] = [
        dxy_session_key(timestamp)
        for timestamp in result.index
    ]

    return result


def calculate_dxy_session_vwap(
    df_dxy_ny: pd.DataFrame,
) -> pd.Series:
    """
    DXY VWAP Version 1.

    Képlet:

    typical_price =
        (high + low + close) / 3

    session_vwap =
        cumsum(typical_price * volume)
        /
        cumsum(volume)

    A cumsum minden DXY session-key-nél újraindul.
    """
    if "dxy_session_key" not in df_dxy_ny.columns:
        df_dxy_ny = add_dxy_session_key(
            df_dxy_ny
        )

    typical_price = (
        df_dxy_ny["high"]
        + df_dxy_ny["low"]
        + df_dxy_ny["close"]
    ) / 3.0

    cumulative_price_volume = (
        typical_price * df_dxy_ny["volume"]
    ).groupby(
        df_dxy_ny["dxy_session_key"]
    ).cumsum()

    cumulative_volume = (
        df_dxy_ny["volume"]
    ).groupby(
        df_dxy_ny["dxy_session_key"]
    ).cumsum()

    return cumulative_price_volume / cumulative_volume.replace(
        0,
        np.nan,
    )


def calculate_dxy_macro_context(
    df_dxy_1m: pd.DataFrame,
) -> dict[str, float]:
    """
    DXY Macro Context Version 1.

    Számítások:

    1. DXY ár:
       az utolsó 1 perces close.

    2. DXY session VWAP:
       a calculate_dxy_session_vwap() képlete szerint.

    3. DXY 15 perces változás:

       dxy_change_15m_pct =
           100 * (
               current_close / close_15m_ago - 1
           )

       1 perces adatnál:
       - current close: iloc[-1]
       - 15 perccel korábbi close: iloc[-16]
    """
    df_dxy_ny = to_new_york_time(
        df_dxy_1m
    )

    df_dxy_ny = add_dxy_session_key(
        df_dxy_ny
    )

    df_dxy_ny["dxy_vwap"] = calculate_dxy_session_vwap(
        df_dxy_ny
    )

    dxy_price = float(
        df_dxy_ny["close"].iloc[-1]
    )

    dxy_vwap = float(
        df_dxy_ny["dxy_vwap"].iloc[-1]
    )

    if len(df_dxy_ny) >= 16:
        current_close = float(
            df_dxy_ny["close"].iloc[-1]
        )

        close_15m_ago = float(
            df_dxy_ny["close"].iloc[-16]
        )

        if close_15m_ago > 0:
            dxy_change_15m_pct = (
                100
                * (
                    current_close
                    / close_15m_ago
                    - 1
                )
            )

        else:
            dxy_change_15m_pct = np.nan

    else:
        dxy_change_15m_pct = np.nan

    return {
        "dxy_price": dxy_price,
        "dxy_vwap": dxy_vwap,
        "dxy_change_15m_pct": dxy_change_15m_pct,
    }


# ===================================================================
# 8. US REAL YIELD VERSION 1
# ===================================================================

def get_real_yield_context_v1() -> dict[str, float]:
    """
    Reálhozam Version 1.

    Alapeset:
        N/A

    Nem számolunk hamis 15 perces real-yield változást napi DFII10
    adatsorból.

    Később ide kerülhet:
    - valós idejű TIPS yield feed;
    - real yield vendor;
    - saját market-data adapter;
    - manuális, megbízható intraday input.
    """
    if USE_MANUAL_REAL_YIELD_CONTEXT:
        return {
            "real_yield_10y": MANUAL_REAL_YIELD_10Y,
            "real_yield_change_15m_bps": (
                MANUAL_REAL_YIELD_CHANGE_15M_BPS
            ),
        }

    return {
        "real_yield_10y": np.nan,
        "real_yield_change_15m_bps": np.nan,
    }


# ===================================================================
# 9. MACRO CONTEXT
# ===================================================================

def build_gc_macro_context(
    df_dxy_1m: pd.DataFrame,
) -> dict[str, float]:
    """
    A gc_regime_1_0_10.py számára macro_context objektum.

    Version 1:
    - DXY automatikus TWS/IBKR futures adatokból.
    - Real yield N/A vagy manuális input.
    """
    dxy_context = calculate_dxy_macro_context(
        df_dxy_1m
    )

    real_yield_context = get_real_yield_context_v1()

    return {
        "dxy_price": dxy_context["dxy_price"],
        "dxy_vwap": dxy_context["dxy_vwap"],
        "dxy_change_15m_pct": (
            dxy_context["dxy_change_15m_pct"]
        ),

        "real_yield_10y": (
            real_yield_context["real_yield_10y"]
        ),

        "real_yield_change_15m_bps": (
            real_yield_context[
                "real_yield_change_15m_bps"
            ]
        ),
    }


# ===================================================================
# 10. GC CSV EXPORT
# ===================================================================

def update_gc_history(
    df_new: pd.DataFrame,
    existing: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Új adat összefűzése a meglévő történettel, mentés CSV-be.

    A gc_audit_backtest.py a gc_1m_history.csv fájlt olvassa.
    Formátum: date,open,high,low,close,volume

    Visszaadja a teljes (összevont) történetet.
    """
    new_df = df_new.copy()

    if new_df.index.tz is None:
        new_df.index = new_df.index.tz_localize("UTC")
    else:
        new_df.index = new_df.index.tz_convert("UTC")

    if existing is not None:
        combined = pd.concat([existing, new_df])
        combined = combined[
            ~combined.index.duplicated(keep="last")
        ].sort_index()

        print(
            f"Meglévő történet: {len(existing):,} bar | "
            f"letöltött: {len(new_df):,} | "
            f"összevont: {len(combined):,}"
        )
    else:
        combined = new_df.sort_index()

    combined.index.name = "date"

    if EXPORT_GC_HISTORY_FOR_AUDIT:
        combined.to_csv(
            GC_HISTORY_EXPORT_PATH,
            index=True,
        )

        print(
            f"GC audit CSV elmentve: "
            f"{GC_HISTORY_EXPORT_PATH} | "
            f"{len(combined):,} bar"
        )

    return combined


# ===================================================================
# 11. FŐ PROGRAM
# ===================================================================

def main() -> None:
    ib = IB()

    try:
        # -----------------------------------------------------------
        # 1. TWS kapcsolat
        # -----------------------------------------------------------
        print("Kapcsolódás a TWS-hez...")

        ib.connect(
            IB_HOST,
            IB_PORT,
            clientId=CLIENT_ID,
            timeout=20,
        )

        print("Kapcsolódás sikeres.")

        ib.reqMarketDataType(
            MARKET_DATA_TYPE
        )

        if MARKET_DATA_TYPE == 4:
            print("Market data mode: DELAYED")

        else:
            print("Market data mode: LIVE / DEFAULT")

        # -----------------------------------------------------------
        # 2. Kontraktusok
        # -----------------------------------------------------------
        # Standard Gold Futures:
        gc_contract = ContFuture(
            "GC",
            "COMEX",
            "USD",
        )

        # ICE U.S. Dollar Index Futures:
        # Nálad validált konstrukció:
        # DXZ6 DX NYBOT ...
        dxy_contract = ContFuture(
            "DX",
            "NYBOT",
            "USD",
        )

        print("GC és DXY kontraktusok validálása...")

        qualified = ib.qualifyContracts(
            gc_contract,
            dxy_contract,
        )

        if not qualified:
            raise RuntimeError(
                "Nem sikerült GC vagy DXY kontraktust validálni."
            )

        print("Kontraktusok:")

        for contract in qualified:
            print(
                f"  ✅ {contract.localSymbol} | "
                f"Symbol={contract.symbol} | "
                f"Exchange={contract.exchange} | "
                f"conId={contract.conId}"
            )

        # -----------------------------------------------------------
        # 3. GC historical adat
        # -----------------------------------------------------------
        existing_gc = load_existing_gc_history()

        download_days = days_to_download(
            existing_gc
        )

        if existing_gc is not None:
            print(
                f"Meglévő GC történet: {len(existing_gc):,} bar, "
                f"utolsó: {existing_gc.index.max()} | "
                f"letöltendő: {download_days} nap"
            )

        df_gc_new = fetch_gc_history_with_fallback(
            ib=ib,
            contract=gc_contract,
            duration=(
                GC_HISTORICAL_DURATION
                if download_days >= GC_HISTORY_TOTAL_DAYS
                else f"{download_days} D"
            ),
            label="GC",
            total_days=download_days,
        )

        # -----------------------------------------------------------
        # 4. Összefűzés a meglévő történettel + CSV mentés
        # -----------------------------------------------------------
        df_gc_full = update_gc_history(
            df_gc_new,
            existing_gc,
        )

        # A checklist csak az utolsó ~10 napot kapja (gyorsabb).
        df_gc = df_gc_full.tail(
            GC_CHECKLIST_LOOKBACK_BARS
        )

        # -----------------------------------------------------------
        # 5. DXY historical adat
        # -----------------------------------------------------------
        df_dxy = fetch_historical_1m_data(
            ib=ib,
            contract=dxy_contract,
            duration=DXY_HISTORICAL_DURATION,
            label="DXY / DX",
        )

        # -----------------------------------------------------------
        # 6. Macro context
        # -----------------------------------------------------------
        print("DXY macro context számítás...")

        macro_context = build_gc_macro_context(
            df_dxy
        )

        print(
            f"DXY ár: "
            f"{macro_context['dxy_price']:.3f} | "
            f"DXY VWAP: "
            f"{macro_context['dxy_vwap']:.3f} | "
            f"DXY 15m: "
            f"{macro_context['dxy_change_15m_pct']:.4f}%"
        )

        if USE_MANUAL_REAL_YIELD_CONTEXT:
            print(
                f"US 10Y real yield: "
                f"{macro_context['real_yield_10y']:.3f}% | "
                f"15m change: "
                f"{macro_context['real_yield_change_15m_bps']:.2f} bp"
            )

        else:
            print(
                "US 10Y real yield: N/A "
                "(Version 1-ben nincs megbízható intraday feed)"
            )

        # -----------------------------------------------------------
        # 7. GC regime checklist
        # -----------------------------------------------------------
        print("Gold regime checklist futtatása...")

        report = run_gold_checklist(
            df_1m=df_gc,
            equity_usd=EQUITY_USD,
            macro_context=macro_context,
        )

        print(report)

    except Exception as error:
        print("\n" + "=" * 72)
        print("KRITIKUS HIBA")
        print("=" * 72)
        print(f"Típus: {type(error).__name__}")
        print(f"Üzenet: {error}")
        print("=" * 72)

        log.exception("💥 GC checklist runner hiba:")

    finally:
        if ib.isConnected():
            ib.disconnect()
            print("\nTWS kapcsolat bontva.")


if __name__ == "__main__":
    main()
