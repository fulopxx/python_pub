from __future__ import annotations

import logging
import math
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ib_insync import IB, ContFuture, Index, util

from index_regime import (
    ES_CONFIG,
    NQ_CONFIG,
    build_cross_market_risk_summary,
    build_index_cross_context,
    build_index_snapshot,
    build_qqq_put_advisory,
    format_qqq_put_advisory,
    index_regime_active,
    run_index_checklist,
)
from index_daily_context import fetch_index_daily_context

# VERSION 1.2.1: export a timestamped paper advisory for qqq_options_advisor.py.
# Requires index_regime.py 1.2.0, updated market_transition.py,
# index_stress_context.py, index_daily_context.py and existing market_exhaustion.py.
# No orders or position management.

IB_HOST = "127.0.0.1"
IB_PORT = 7497
CLIENT_ID = 30
EQUITY_USD = 10_000.0
HISTORICAL_DURATION = "10 D"
BAR_SIZE = "1 min"
# False = full session; True = regular trading hours only.
USE_RTH = False
MARKET_DATA_TYPE = 4
HISTORY_TIMEOUT_SECONDS = 60
BASE_DIR = Path(__file__).resolve().parent
VX_MONITOR_JSON = BASE_DIR / "vix_monitor_output" / "vix_monitor_latest.json"
FETCH_DAILY_HISTORY = True
DAILY_CACHE_DIR = BASE_DIR / "index_daily_cache"
ADVISORY_OUTPUT_JSON = BASE_DIR / "index_regime_output" / "qqq_regime_advisory.json"

# Snapshots can contain a previous close, and ticker.time may be receipt time.
# Only historical closed-minute timestamps can supply the new VIX context.
USE_TIMESTAMPED_VIX_HISTORY = True
# Off by default: an untimestamped display snapshot adds no causal evidence.
DISPLAY_UNTIMESTAMPED_VIX_SNAPSHOT = False
VIX_HISTORY_DURATION = "1 D"
VIX_HISTORY_TIMEOUT_SECONDS = 15
VIX_MAX_ANALYSIS_AGE_MINUTES = 15
VIX_MAX_PAIR_SKEW_MINUTES = 5
MAX_INDEX_TIME_SKEW_MINUTES = 5

# Optional daily CSV inputs. None = fetch daily if enabled, then intraday fallback.
# CSV format: date,open,high,low,close,volume
# IMPORTANT: date is the actual COMPLETION time, not a midnight trading-day label.
# Must use the same contract/continuous-series convention as the intraday input.
DAILY_HISTORY_CSV: dict[str, str | None] = {"ES": None, "NQ": None}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)
logging.getLogger("ib_insync.wrapper").setLevel(logging.WARNING)


def is_valid_price(value: Any) -> bool:
    if value is None:
        return False
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric_value) and numeric_value > 0


def get_ticker_price(ticker: Any) -> float:
    """Legacy snapshot fallback order; these prices are DISPLAY ONLY."""
    for value in (getattr(ticker, "last", None), getattr(ticker, "close", None)):
        if is_valid_price(value):
            return float(value)
    try:
        value = ticker.marketPrice()
        if is_valid_price(value):
            return float(value)
    except Exception:
        pass
    for value in (getattr(ticker, "bid", None), getattr(ticker, "ask", None)):
        if is_valid_price(value):
            return float(value)
    return np.nan


def bars_to_dataframe(bars: list[Any]) -> pd.DataFrame:
    if not bars:
        raise RuntimeError("Az IBKR nem adott vissza történeti adatot.")
    df = util.df(bars)
    if df is None or df.empty:
        raise RuntimeError("Az IBKR történeti DataFrame üres.")
    df.columns = [str(column).lower() for column in df.columns]
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Hiányzó oszlopok: {missing}. Elérhető: {list(df.columns)}")
    # formatDate=2 requests explicit UTC timestamps for intraday bars.
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise")
    if df["date"].isna().any():
        raise RuntimeError("Hiányzó történeti időbélyeg.")
    df = df.set_index("date")[required[1:]]
    df = df.apply(pd.to_numeric, errors="coerce").dropna().sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]
    if len(df) < 100:
        raise RuntimeError(f"Túl kevés adat érkezett: {len(df)} bar.")
    return df


def fetch_historical_1m_data(ib: IB, contract: Any, label: str) -> pd.DataFrame:
    """Existing ten-day request; this version does not add index-history caching."""
    log.info("%s: %s / %s historical letöltés...", label, HISTORICAL_DURATION, BAR_SIZE)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=HISTORICAL_DURATION,
        barSizeSetting=BAR_SIZE,
        whatToShow="TRADES",
        useRTH=USE_RTH,
        formatDate=2,
        timeout=HISTORY_TIMEOUT_SECONDS,
    )
    df = bars_to_dataframe(bars)
    df.attrs["contract_id"] = contract.conId
    log.info("%s: %s bar; %s -> %s", label, f"{len(df):,}", df.index.min(), df.index.max())
    return df


def load_daily_history(symbol: str) -> pd.DataFrame | None:
    path = DAILY_HISTORY_CSV.get(symbol)
    if not path:
        return None
    # A configured but invalid file is a real error, not silently absent data.
    data = pd.read_csv(Path(path))
    data.columns = [str(c).lower() for c in data.columns]
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in data]
    if missing:
        raise ValueError(f"{symbol} napi CSV hiányzó oszlopai: {missing}")
    data["date"] = pd.to_datetime(data["date"], utc=True, errors="raise")
    frame = data.set_index("date")[required[1:]].sort_index()
    if frame.empty or frame.index.has_duplicates:
        raise ValueError(f"{symbol} napi CSV üres vagy duplikált időbélyegű.")
    log.info("%s: %s külső napi bar, lezárási időbélyeggel.", symbol, len(frame))
    return frame


def load_vx_monitor_context(path: str | Path = VX_MONITOR_JSON) -> dict[str, Any] | None:
    """Local optional input only. Timestamp validation occurs in index_regime."""
    try:
        packet = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(packet, dict):
            raise ValueError("VX JSON nem objektum.")
        log.info("VX-monitor input: %s | forrásidő: %s | állapot: %s",
                 path, packet.get("asof"), packet.get("candidate_state"))
        return packet
    except (OSError, ValueError) as error:
        log.warning("VX-monitor input N/A (%s): %s. Az index price-action ettől még fut.", path, error)
        return None


def _closed_vix_observation(
    bars: list[Any], as_of: pd.Timestamp,
) -> tuple[float, pd.Timestamp] | None:
    """Select only a completed minute, never a future value versus futures as_of."""
    if not bars:
        return None
    df = util.df(bars)
    if df is None or df.empty or not {"date", "close"}.issubset(df.columns):
        return None
    timestamps = pd.to_datetime(df["date"], utc=True, errors="coerce")
    values = pd.to_numeric(df["close"], errors="coerce")
    completed_at = timestamps + pd.Timedelta(minutes=1)
    eligible = (timestamps.notna() & np.isfinite(values) & (values > 0)
                & (completed_at <= as_of))
    candidates = pd.DataFrame({"completed_at": completed_at[eligible],
                               "close": values[eligible]}).sort_values("completed_at")
    if candidates.empty:
        return None
    latest = candidates.iloc[-1]
    stamp = latest["completed_at"]
    if as_of - stamp > pd.Timedelta(minutes=VIX_MAX_ANALYSIS_AGE_MINUTES):
        return None
    return float(latest["close"]), stamp


def vix_history_diagnostics(
    bars: list[Any], as_of: pd.Timestamp,
) -> dict[str, Any]:
    """Explain N/A without guessing permissions, trading hours or provider gaps."""
    details: dict[str, Any] = dict(
        status="EMPTY", bars=len(bars) if bars is not None else 0,
        first_start=None, last_start=None, valid_price_bars=0,
        closed_before_asof=0, last_eligible_close=None,
        age_minutes=None, starts_after_asof=0,
    )
    if not bars:
        return details
    df = util.df(bars)
    if df is None or df.empty or not {"date", "close"}.issubset(df.columns):
        details["status"] = "INVALID_COLUMNS"
        return details
    times = pd.to_datetime(df["date"], utc=True, errors="coerce")
    values = pd.to_numeric(df["close"], errors="coerce")
    valid = times.notna() & np.isfinite(values) & (values > 0)
    usable_times = times[valid]
    details.update(
        first_start=times.min(), last_start=times.max(),
        valid_price_bars=int(valid.sum()),
        starts_after_asof=int((times > as_of).sum()),
    )
    if usable_times.empty:
        details["status"] = "NO_VALID_PRICES_OR_TIMESTAMPS"
        return details
    closes = usable_times + pd.Timedelta(minutes=1)
    eligible = closes.loc[closes <= as_of]
    details["closed_before_asof"] = len(eligible)
    if eligible.empty:
        details["status"] = "NO_CLOSED_BAR_BEFORE_ASOF"
        return details
    latest = eligible.max()
    age = (as_of - latest).total_seconds() / 60
    details.update(
        last_eligible_close=latest, age_minutes=age,
        status="STALE" if age > VIX_MAX_ANALYSIS_AGE_MINUTES else "OK",
    )
    return details


def fetch_vix_context(
    ib: IB,
    vix_contract: Any,
    vix3m_contract: Any,
    *,
    as_of: pd.Timestamp | None = None,
) -> dict[str, Any] | None:
    """Snapshot display + fail-soft, timestamp-verified historical context.

    ticker.time is NOT treated as the time of the chosen last/close price.
    If historical context is unavailable/stale, pass None to BOTH the legacy
    and the new engine. This prevents an unverified close from influencing
    crash scoring or the QQQ advisory.
    """
    stamp = pd.to_datetime(
        as_of if as_of is not None else pd.Timestamp.now(tz="UTC"),
        utc=True,
    )
    if not vix_contract.conId or not vix3m_contract.conId:
        log.warning("VIX/VIX3M kontraktus nem validált; kontextus N/A.")
        return None
    if DISPLAY_UNTIMESTAMPED_VIX_SNAPSHOT:
        try:
            tickers = ib.reqTickers(vix_contract, vix3m_contract)
            ib.sleep(2)
            if len(tickers) >= 2:
                vix, vix3m = (get_ticker_price(t) for t in tickers[:2])
                if is_valid_price(vix) and is_valid_price(vix3m):
                    log.info(
                        "VIX snapshot (csak kijelzés, ár-időbélyeg nem igazolt): "
                        "VIX=%.2f, VIX3M=%.2f, ratio=%.3f", vix, vix3m, vix / vix3m
                    )
        except Exception as error:
            log.warning("VIX snapshot nem elérhető: %s", error)
    if not USE_TIMESTAMPED_VIX_HISTORY:
        log.warning("Timestampolt VIX history kikapcsolva: engine kontextus N/A.")
        return None
    observations: list[tuple[float, pd.Timestamp]] = []
    for label, contract in (("VIX", vix_contract), ("VIX3M", vix3m_contract)):
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=VIX_HISTORY_DURATION,
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=2,
                timeout=VIX_HISTORY_TIMEOUT_SECONDS,
            )
            observation = _closed_vix_observation(bars, stamp)
            diagnostics = vix_history_diagnostics(bars, stamp)
            log.info(
                "%s history diagnosztika: status=%s; bars=%s; valid=%s; "
                "first_start=%s; last_start=%s; lezárt_asof_előtt=%s; "
                "utolsó_alkalmas_close=%s; kor=%s perc; jövőbeli_start=%s",
                label, diagnostics["status"], diagnostics["bars"],
                diagnostics["valid_price_bars"], diagnostics["first_start"],
                diagnostics["last_start"], diagnostics["closed_before_asof"],
                diagnostics["last_eligible_close"], diagnostics["age_minutes"],
                diagnostics["starts_after_asof"],
            )
        except Exception as error:
            log.warning("%s timestampolt history nem elérhető: %s", label, error)
            return None
        if observation is None:
            log.warning(
                "%s: nincs friss, az indexek %s elemzési idejéig lezárt bar. "
                "VIX kontextus N/A.", label, stamp,
            )
            return None
        observations.append(observation)
    (vix, vix_time), (vix3m, vix3m_time) = observations
    if abs(vix_time - vix3m_time) > pd.Timedelta(minutes=VIX_MAX_PAIR_SKEW_MINUTES):
        log.warning("VIX és VIX3M időbélyeg-eltérés túl nagy; kontextus N/A.")
        return None
    context = dict(
        vix=vix, vix3m=vix3m, vix_change_pct=np.nan,
        timestamp=min(vix_time, vix3m_time),
        vix_timestamp=vix_time, vix3m_timestamp=vix3m_time,
        received_at=pd.Timestamp.now(tz="UTC"),
        source="HISTORICAL_1M_CLOSED",
    )
    log.info("VIX engine kontextus: %.2f / %.2f = %.3f; forrás-idő: %s",
             vix, vix3m, vix / vix3m, context["timestamp"])
    return context


def requested_data_mode() -> str:
    return {1: "REALTIME", 2: "FROZEN", 3: "DELAYED", 4: "DELAYED_FROZEN"}.get(
        MARKET_DATA_TYPE, "ANALYSIS_ONLY"
    )


def build_analysis_bundle(
    df_es: pd.DataFrame,
    df_nq: pd.DataFrame,
    vix_context: dict[str, Any] | None,
    *,
    daily_es: pd.DataFrame | None = None,
    daily_nq: pd.DataFrame | None = None,
    clock: pd.Timestamp | None = None,
    vx_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One consistent set of contexts for snapshots, reports, summary and QQQ."""
    now = pd.to_datetime(
        clock if clock is not None else pd.Timestamp.now(tz="UTC"), utc=True
    )
    skew = abs(df_es.index.max() - df_nq.index.max())
    analysis_as_of = min(df_es.index.max(), df_nq.index.max())
    df_es, df_nq = (frame.loc[frame.index <= analysis_as_of].copy() for frame in (df_es, df_nq))
    cross = build_index_cross_context(df_es, df_nq, vix_context=vix_context)
    warnings = []
    if skew > pd.Timedelta(minutes=MAX_INDEX_TIME_SKEW_MINUTES):
        warnings.append(
            f"ES/NQ utolsó bar eltérés: {skew}. A peer/relatív erő sorok N/A."
        )
        # Keep optional timestamped VIX but do not claim synchronized peer context.
        for symbol in cross:
            cross[symbol].pop("peer_non_confirmation", None)
            cross[symbol].pop("relative_strength_falling", None)
    output: dict[str, Any] = {"warnings": warnings, "cross_context": cross, "asof": analysis_as_of}
    for symbol, frame, instrument, daily in (
        ("ES", df_es, ES_CONFIG, daily_es),
        ("NQ", df_nq, NQ_CONFIG, daily_nq),
    ):
        metadata = dict(
            mode=requested_data_mode(),
            as_of=now,
            last_bar_is_complete=False,
            requested_market_data_type=MARKET_DATA_TYPE,
            contract_id=frame.attrs.get("contract_id"),
        )
        kwargs = dict(
            df_1m=frame,
            instrument=instrument,
            vix_context=vix_context,
            daily_history=daily,
            cross_market_context=cross[symbol],
            data_context=metadata,
            vx_context=vx_context,
        )
        snapshot = build_index_snapshot(**kwargs)
        result = index_regime_active(snapshot)
        report = run_index_checklist(**kwargs, equity_usd=EQUITY_USD)
        output[symbol] = dict(snapshot=snapshot, regime_result=result, report=report)
    es, nq = output["ES"], output["NQ"]
    output["summary"] = build_cross_market_risk_summary(
        es["regime_result"]["regime"], nq["regime_result"]["regime"],
        es_exhaustion=es["snapshot"]["exhaustion"],
        nq_exhaustion=nq["snapshot"]["exhaustion"],
        es_transition=es["snapshot"]["transition_risk"],
        nq_transition=nq["snapshot"]["transition_risk"],
    )
    # Same snapshots and verified VIX context; no blow-off-only put authorization.
    output["qqq_advisory"] = build_qqq_put_advisory(
        es["snapshot"], es["regime_result"], nq["snapshot"], nq["regime_result"]
    )
    return output


def print_cross_market_summary(
    es_regime: Any,
    nq_regime: Any,
    *,
    es_exhaustion: dict[str, Any] | None = None,
    nq_exhaustion: dict[str, Any] | None = None,
    es_transition: dict[str, Any] | None = None,
    nq_transition: dict[str, Any] | None = None,
) -> None:
    summary = build_cross_market_risk_summary(
        es_regime=es_regime, nq_regime=nq_regime,
        es_exhaustion=es_exhaustion, nq_exhaustion=nq_exhaustion,
        es_transition=es_transition, nq_transition=nq_transition,
    )
    print("\n" + "=" * 72)
    print("ES / NQ CROSS-MARKET RISK ÖSSZEGZÉS")
    print("=" * 72)
    print(f"ES rezsim: {es_regime.value}")
    print(f"NQ rezsim: {nq_regime.value}")
    if es_exhaustion is not None:
        print(f"ES exhaustion: {es_exhaustion['state']}")
    if nq_exhaustion is not None:
        print(f"NQ exhaustion: {nq_exhaustion['state']}")
    print(summary)
    print("=" * 72)


def print_overall_status(bundle: dict[str, Any]) -> None:
    print("\n"+"="*100)
    print("ES / NQ ÖSSZESÍTETT ÁLLAPOT")
    print(f"Közös elemzési időpont: {bundle['asof']}")
    for symbol in ("ES", "NQ"):
        item = bundle[symbol]
        snapshot, result = item["snapshot"], item["regime_result"]
        print(f"{symbol}: {result['regime'].value}")
        print(f"  Core: {snapshot['data_quality']} | Whipsaw: {result['whipsaw_state']} | "
              f"Exhaustion: {result['exhaustion_state']}")
        print(f"  LONG: {'TILTVA' if result['block_new_longs'] else 'nincs overlay tiltás'} | "
              f"SHORT: {'TILTVA' if result['block_new_shorts'] else 'nincs overlay tiltás'}")
        print(f"  Ár + VX kontextus: {result['short_stress_state']}")
    print("VIX-short zárás: NEM IGAZOLT. A VX input ár/görbe-kontextus, nem szereplői pozícióadat.")
    print("A nyitott overlay kapu nem entry-engedély. Elemző/paper rendszer, orderküldés nélkül.")
    print("="*100)


def write_advisory_packet(packet: dict[str, Any], path: str | Path) -> None:
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=path.parent, delete=False) as file:
            name = file.name
            json.dump(packet, file, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def export_qqq_advisory(bundle: dict[str, Any], path: str | Path = ADVISORY_OUTPUT_JSON,
                        *, clock: pd.Timestamp | None = None) -> dict[str, Any]:
    """Only analysis fields: no account, balance or position information."""
    from datetime import datetime
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    indices = {}
    for symbol in ("ES", "NQ"):
        item = bundle[symbol]
        s, r = item["snapshot"], item["regime_result"]
        indices[symbol] = dict(
            asof=s["timestamp"], data_quality=s["data_quality"],
            regime=r["regime"].value, short_overlay_blocked=r["block_new_shorts"],
            whipsaw=r["whipsaw_state"], short_stress_state=r["short_stress_state"])
    packet = clean(dict(
        schema_version=1, source="ES_NQ_REGIME_RUNNER",
        observed_at=pd.Timestamp.now(tz="UTC") if clock is None else clock,
        asof=bundle["asof"], indices=indices, advisory=bundle["qqq_advisory"],
        data_mode=requested_data_mode(), execution_allowed=False))
    write_advisory_packet(packet, path)
    return packet


def main() -> None:
    ib = IB()
    try:
        print("ES/NQ RUNNER 1.2.1 - elemző / paper rendszer.")
        print("Kapcsolódás a TWS-hez...")
        ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID, timeout=20)
        ib.reqMarketDataType(MARKET_DATA_TYPE)
        print(f"Kapcsolódás sikeres. Kért market data mode: {requested_data_mode()}")
        es_contract = ContFuture("ES", "CME", "USD")
        nq_contract = ContFuture("NQ", "CME", "USD")
        vix_contract = Index("VIX", "CBOE", "USD")
        vix3m_contract = Index("VIX3M", "CBOE", "USD")
        qualified = ib.qualifyContracts(es_contract, nq_contract)
        if len(qualified) < 2 or not es_contract.conId or not nq_contract.conId:
            raise RuntimeError("ES vagy NQ kontraktus nem validálható.")
        for contract in qualified:
            print(f"  {contract.localSymbol} | Symbol={contract.symbol} | "
                  f"Exchange={contract.exchange} | conId={contract.conId}")
        try:
            ib.qualifyContracts(vix_contract, vix3m_contract)
        except Exception as error:
            log.warning("Opcionális VIX kvalifikáció nem sikerült: %s", error)
        df_es = fetch_historical_1m_data(ib, es_contract, "ES")
        df_nq = fetch_historical_1m_data(ib, nq_contract, "NQ")
        # Both futures histories now exist, so VIX can be aligned causally.
        analysis_as_of = min(df_es.index.max(), df_nq.index.max())
        vix_context = fetch_vix_context(
            ib, vix_contract, vix3m_contract, as_of=analysis_as_of
        )
        daily = {}
        for symbol, contract in (("ES", es_contract), ("NQ", nq_contract)):
            daily[symbol] = load_daily_history(symbol)
            status = "CSV" if daily[symbol] is not None else "INTRADAY_SESSION_FALLBACK"
            if daily[symbol] is None and FETCH_DAILY_HISTORY:
                daily[symbol], status = fetch_index_daily_context(
                    ib, contract, analysis_as_of, cache_dir=DAILY_CACHE_DIR)
            print(f"{symbol} napi kontextus: {status} | "
                  f"{len(daily[symbol]) if daily[symbol] is not None else 0} külön napi bar")
        bundle = build_analysis_bundle(
            df_es, df_nq, vix_context,
            daily_es=daily["ES"],
            daily_nq=daily["NQ"],
            vx_context=load_vx_monitor_context(),
        )
        print(f"\nKözös futures kontextus elemzési időpontja: {analysis_as_of}")
        if vix_context is None:
            print("Timestampolt VIX/VIX3M engine kontextus: N/A.")
        else:
            print(f"VIX/VIX3M forrás-idő: {vix_context['timestamp']} | "
                  f"fogadás: {vix_context['received_at']}")
        for warning in bundle["warnings"]:
            print(f"FIGYELMEZTETÉS: {warning}")
        for symbol in ("ES", "NQ"):
            print(bundle[symbol]["report"])
        print_cross_market_summary(
            bundle["ES"]["regime_result"]["regime"],
            bundle["NQ"]["regime_result"]["regime"],
            es_exhaustion=bundle["ES"]["snapshot"]["exhaustion"],
            nq_exhaustion=bundle["NQ"]["snapshot"]["exhaustion"],
            es_transition=bundle["ES"]["snapshot"]["transition_risk"],
            nq_transition=bundle["NQ"]["snapshot"]["transition_risk"],
        )
        print(
            "\nA következő QQQ advisory 'entry_allowed' mezője kizárólag "
            "a legacy paper feltételeket jelenti, nem valós idejű végrehajtási engedély."
        )
        print(format_qqq_put_advisory(bundle["qqq_advisory"]))
        print_overall_status(bundle)
        export_qqq_advisory(bundle, ADVISORY_OUTPUT_JSON)
        print(f"QQQ options-advisor input elmentve: {ADVISORY_OUTPUT_JSON}")
    except Exception as error:
        print("\n" + "=" * 72)
        print("KRITIKUS HIBA")
        print(f"Típus: {type(error).__name__}")
        print(f"Üzenet: {error}")
        print("=" * 72)
        log.exception("ES/NQ runner hiba:")
        try:
            # Replace an older successful packet after a failed run.
            stamp = pd.Timestamp.now(tz="UTC").isoformat()
            write_advisory_packet(dict(
                schema_version=1, source="ES_NQ_REGIME_RUNNER", observed_at=stamp,
                asof=stamp, execution_allowed=False, indices={},
                advisory=dict(state="DATA_UNAVAILABLE", entry_scope="PAPER_ONLY",
                              execution_allowed=False, entry_allowed=False,
                              trade_candidate=False, recommended_structure="WAIT",
                              reasons=[f"Runner hiba: {type(error).__name__}"])),
                ADVISORY_OUTPUT_JSON)
        except Exception as export_error:
            log.warning("Hibás advisory exportja nem sikerült: %s", export_error)
        raise  # Real failures must not masquerade as exit code 0.
    finally:
        if ib.isConnected():
            ib.disconnect()
            print("\nTWS kapcsolat bontva.")


if __name__ == "__main__":
    main()
