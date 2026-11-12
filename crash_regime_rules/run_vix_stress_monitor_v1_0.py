"""VIX futures stress heuristic, TWS read-only. No orders or position management.

One file, Python + pandas + ib_insync (ib_insync only needed for TWS mode).
Run once: python run_vix_stress_monitor_v1_0.py
Local polling: python run_vix_stress_monitor_v1_0.py --loop --interval 300 --beep
Offline synthetic example: python run_vix_stress_monitor_v1_0.py --demo

Technical references:
https://www.interactivebrokers.com/docs/tws-api/doc/market-data-delayed/introduction
https://interactivebrokers.github.io/tws-api/basic_contracts.html
https://www.volatilityshares.com/resource/faq

Price/volume cannot identify asset-manager short covering. UVIX/SVIX are
ETFs, not VX contracts. Heuristic thresholds are unvalidated research defaults.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

VERSION = "1.1.0"
log = logging.getLogger("vix_monitor")


@dataclass(frozen=True)
class Config:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 40
    market_data_type: int = 3  # 3 delayed; 4 delayed-frozen; not a guarantee.
    initial_duration: str = "10 D"
    update_duration: str = "1 D"
    max_data_age_minutes: float = 45.0
    reference_tolerance_minutes: float = 2.0
    optional_alignment_minutes: float = 5.0
    f1_rise_15m_pct: float = 2.0
    f1_rise_30m_pct: float = 3.0
    curve_compression_30m_pp: float = -1.0
    front_outperformance_30m_pp: float = 0.75
    flat_curve_pct: float = 2.0
    backwardation_pct: float = -0.25
    backwardation_price_rise_15m_pct: float = 0.5
    volume_ratio_threshold: float = 1.5
    volume_prior_days: int = 5
    uvix_rise_15m_pct: float = 3.0
    svix_fall_15m_pct: float = -1.0
    # Unvalidated starting thresholds for ROLLING elapsed-time windows.
    mtf_f1_rise_1h_pct: float = 4.0
    mtf_f1_rise_4h_pct: float = 8.0
    mtf_curve_compression_15m_pp: float = -0.5
    mtf_curve_compression_1h_pp: float = -1.5
    mtf_curve_compression_4h_pp: float = -2.5
    mtf_front_outperformance_15m_pp: float = 0.4
    mtf_front_outperformance_1h_pp: float = 1.0
    mtf_front_outperformance_4h_pp: float = 2.0
    mtf_backwardation_rise_1h_pct: float = 1.0
    mtf_backwardation_rise_4h_pct: float = 2.0
    mtf_min_coverage: float = 0.90
    mtf_max_gap_minutes: float = 5.0
    confirmations: int = 2
    minimum_confirmation_gap_minutes: float = 5.0
    maximum_confirmation_gap_minutes: float = 20.0
    cooldown_minutes: float = 30.0
    beep: bool = False

    def __post_init__(self) -> None:
        values = (self.mtf_f1_rise_1h_pct, self.mtf_f1_rise_4h_pct,
                  self.mtf_curve_compression_15m_pp, self.mtf_curve_compression_1h_pp,
                  self.mtf_curve_compression_4h_pp, self.mtf_front_outperformance_15m_pp,
                  self.mtf_front_outperformance_1h_pp, self.mtf_front_outperformance_4h_pp,
                  self.mtf_backwardation_rise_1h_pct, self.mtf_backwardation_rise_4h_pct,
                  self.mtf_min_coverage, self.mtf_max_gap_minutes)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Finite MTF configuration values required.")
        if not 0 < self.mtf_min_coverage <= 1 or self.mtf_max_gap_minutes < 1:
            raise ValueError("MTF coverage must be (0, 1], max gap >=1 minute.")


def utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def closed_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    """Minute-start input. Drop the latest potentially incomplete bar."""
    if frame.empty:
        return frame.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("OHLCV index must be a DatetimeIndex.")
    required = ["open", "high", "low", "close", "volume"]
    if any(c not in frame for c in required):
        raise ValueError("Missing OHLCV column.")
    data = frame[required].copy().sort_index()
    data.index = pd.DatetimeIndex([utc(x) for x in data.index])
    data = data.loc[~data.index.duplicated(keep="last")]
    # Remove latest timestamp BEFORE validation; a bad trailing bar must not
    # cause a previously completed minute to be unnecessarily excluded.
    data = data.iloc[:-1]
    data = data.apply(pd.to_numeric, errors="coerce")
    valid = data.apply(lambda col: col.map(finite)).all(axis=1)
    valid &= (data[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid &= data.high >= data[["open", "low", "close"]].max(axis=1)
    valid &= data.low <= data[["open", "high", "close"]].min(axis=1)
    # Some indices report volume=-1. Keep their prices, volume becomes unknown.
    data.loc[data.volume < 0, "volume"] = float("nan")
    data = data.loc[valid]
    data.index += pd.Timedelta(minutes=1)  # now completion timestamps
    return data


def reference(data: pd.DataFrame, target: pd.Timestamp, config: Config) -> pd.Series | None:
    if data.empty:
        return None
    position = data.index.searchsorted(target, side="right") - 1
    if position < 0:
        return None
    stamp = data.index[position]
    if (target - stamp).total_seconds()/60 > config.reference_tolerance_minutes:
        return None
    return data.iloc[position]


def change(data: pd.DataFrame, end: pd.Timestamp, minutes: int,
           config: Config, *, same_rth_day: bool = False) -> float | None:
    now = reference(data, end, config)
    before = reference(data, end-pd.Timedelta(minutes=minutes), config)
    if now is None or before is None:
        return None
    if same_rth_day:
        a = now.name.tz_convert("America/New_York")
        b = before.name.tz_convert("America/New_York")
        if a.date() != b.date() or b.hour*60+b.minute < 9*60+30:
            return None
    return 100*(float(now.close)/float(before.close)-1)


def time_of_day_volume_ratio(data: pd.DataFrame, end: pd.Timestamp,
                             config: Config, minutes: int = 15) -> float | None:
    """Compare a complete elapsed volume window with same-clock prior days.

    No previous-bar volume baseline masquerading as time-of-day RVOL.
    Missing scheduled minutes or negative/unknown volume => unavailable.
    """
    def volume_at(stamp: pd.Timestamp) -> float | None:
        part = data.loc[(data.index > stamp-pd.Timedelta(minutes=minutes)) & (data.index <= stamp)]
        if len(part) != minutes or part.volume.isna().any():
            return None
        return float(part.volume.sum())
    current = volume_at(end)
    if current is None:
        return None
    wall = end.tz_convert("America/New_York").tz_localize(None)
    past = []
    for days in range(1, 15):
        stamp = (wall-pd.Timedelta(days=days)).tz_localize(
            "America/New_York", ambiguous="NaT", nonexistent="NaT")
        if pd.isna(stamp):
            continue
        volume = volume_at(stamp.tz_convert("UTC"))
        if volume is not None and volume > 0:
            past.append(volume)
        if len(past) == config.volume_prior_days:
            break
    if len(past) < config.volume_prior_days:
        return None
    return current/float(pd.Series(past).median())


def mtf_profiles(config: Config) -> dict[str, dict[str, float]]:
    return {
        "15m": dict(minutes=15, price=config.f1_rise_15m_pct,
                    compression=config.mtf_curve_compression_15m_pp,
                    outperform=config.mtf_front_outperformance_15m_pp,
                    backward_rise=config.backwardation_price_rise_15m_pct),
        "1h": dict(minutes=60, price=config.mtf_f1_rise_1h_pct,
                   compression=config.mtf_curve_compression_1h_pp,
                   outperform=config.mtf_front_outperformance_1h_pp,
                   backward_rise=config.mtf_backwardation_rise_1h_pct),
        "4h": dict(minutes=240, price=config.mtf_f1_rise_4h_pct,
                   compression=config.mtf_curve_compression_4h_pp,
                   outperform=config.mtf_front_outperformance_4h_pp,
                   backward_rise=config.mtf_backwardation_rise_4h_pct),
    }


def window_quality(index: pd.DatetimeIndex, start: pd.Timestamp,
                   end: pd.Timestamp, config: Config) -> dict[str, Any]:
    part = index[(index >= start) & (index <= end)]
    expected = int((end-start)/pd.Timedelta(minutes=1))+1
    coverage = len(part)/expected if expected > 0 else 0.0
    gaps = part.to_series().diff().dt.total_seconds().div(60)
    max_gap = float(gaps.max()) if len(part) > 1 else None
    complete = (coverage >= config.mtf_min_coverage and max_gap is not None
                and max_gap <= config.mtf_max_gap_minutes)
    return dict(coverage=coverage, bars=len(part), expected_bars=expected,
                max_gap_minutes=max_gap, sufficient=bool(complete))


def optional_horizon(name: str, frame: pd.DataFrame, feed: dict[str, Any],
                     minutes: int, config: Config) -> dict[str, Any]:
    """Optional ETF/index own-timestamp return, within one normal RTH day."""
    result = dict(status=feed["status"], asof=feed.get("used_asof"),
                  reference_at=None, return_pct=None)
    if feed["status"] not in ("OK", "OK_LAGGED", "MISSING_REFERENCE") or frame.empty:
        return result
    end = utc(feed["used_asof"])
    local = end.tz_convert("America/New_York")
    if not (local.weekday() < 5 and 570 < local.hour*60+local.minute <= 960):
        result["status"] = "OUTSIDE_RTH"
        return result
    before = reference(frame, end-pd.Timedelta(minutes=minutes), config)
    if before is None:
        result["status"] = "MISSING_REFERENCE"
        return result
    result["reference_at"] = before.name.isoformat()
    previous = before.name.tz_convert("America/New_York")
    if previous.date() != local.date() or previous.hour*60+previous.minute < 571:
        result["status"] = "INSUFFICIENT_SAME_RTH_SESSION"
        return result
    quality = window_quality(frame.index, before.name, end, config)
    result.update(quality)
    if not quality["sufficient"]:
        result["status"] = "PARTIAL_WINDOW"
        return result
    value = change(frame, end, minutes, config, same_rth_day=True)
    result.update(status="OK_LAGGED" if feed["lag_to_vx_minutes"] > 0 else "OK",
                  return_pct=value)
    return result


def evaluate_timeframes(data: dict[str, pd.DataFrame], output: dict[str, Any],
                        end: pd.Timestamp, config: Config) -> None:
    """Independent rolling 15/60/240m context, not clock-aligned HTF candles."""
    f1, f2 = data["F1"], data["F2"]
    common = f1.index.intersection(f2.index)
    common = common[common <= end]
    p1, p2 = float(f1.loc[end, "close"]), float(f2.loc[end, "close"])
    curve = 100*(p2/p1-1)
    for name, profile in mtf_profiles(config).items():
        tf = output["timeframes"][name]
        minutes = int(profile["minutes"])
        target = end-pd.Timedelta(minutes=minutes)
        tf.update(asof=end.isoformat(), target_reference_at=target.isoformat())
        position = common.searchsorted(target, side="right")-1
        baseline = common[position] if position >= 0 else None
        if baseline is None or (target-baseline).total_seconds()/60 > config.reference_tolerance_minutes:
            tf.update(quality="MISSING_REFERENCE",
                      reason="Nincs közös F1/F2 időalapú referencia a megengedett toleranciával.")
        else:
            tf["reference_at"] = baseline.isoformat()
            tf["actual_elapsed_minutes"] = (end-baseline).total_seconds()/60
            quality = window_quality(common, baseline, end, config)
            tf.update(quality)
            if not quality["sufficient"]:
                tf.update(quality="PARTIAL_WINDOW",
                          reason="Hiányos közös percablak vagy túl nagy belső adatrés.")
            else:
                r1 = 100*(p1/float(f1.loc[baseline, "close"])-1)
                r2 = 100*(p2/float(f2.loc[baseline, "close"])-1)
                old_curve = 100*(float(f2.loc[baseline, "close"])/float(f1.loc[baseline, "close"])-1)
                delta, outperform = curve-old_curve, r1-r2
                price = r1 >= profile["price"]
                compression = delta <= profile["compression"] and outperform >= profile["outperform"]
                backward = curve <= config.backwardation_pct
                rising = backward and r1 >= profile["backward_rise"]
                state = ("BACKWARDATION_STRESS" if rising else "STRESS_BUILDING"
                         if price and compression else "WATCH"
                         if price or compression or curve <= config.flat_curve_pct else "NORMAL")
                rvol = time_of_day_volume_ratio(f1, end, config, minutes)
                tf.update(state=state, quality="OK",
                          reason="Lezárt 1m adatokból gördülő, időalapú stresszkontextus.",
                          metrics=dict(f1_change_pct=r1, f2_change_pct=r2,
                                       curve_pct=curve, curve_change_pp=delta,
                                       front_outperformance_pp=outperform, f1_volume_rvol=rvol),
                          checks=dict(price_acceleration=price, curve_compression=compression,
                                      backwardation=backward, backwardation_with_rising_front=rising,
                                      elevated_time_of_day_volume=(
                                          rvol >= config.volume_ratio_threshold if rvol is not None else None)))
        for feed_name in ("UVIX", "SVIX", "VIX", "VIX3M"):
            tf["optional_feeds"][feed_name] = optional_horizon(
                feed_name, data.get(feed_name, pd.DataFrame()),
                output["feed_quality"][feed_name], minutes, config)
            output["metrics"].setdefault(
                f"{feed_name.lower()}_change_{name}_pct",
                tf["optional_feeds"][feed_name]["return_pct"])
        uv, sv = tf["optional_feeds"]["UVIX"], tf["optional_feeds"]["SVIX"]
        pair_ok = (uv["return_pct"] is not None and sv["return_pct"] is not None
                   and abs((utc(uv["asof"])-utc(sv["asof"])).total_seconds()/60)
                   <= config.reference_tolerance_minutes
                   and abs((utc(uv["reference_at"])-utc(sv["reference_at"])).total_seconds()/60)
                   <= config.reference_tolerance_minutes)
        # Directional context only; not an independent evidence vote.
        tf["checks"]["uvix_up_svix_down"] = (
            uv["return_pct"] > 0 and sv["return_pct"] < 0 if pair_ok else None)
        for metric, suffix in (("f1_change_pct", "f1_change"),
                               ("f2_change_pct", "f2_change"),
                               ("curve_change_pp", "curve_change"),
                               ("front_outperformance_pp", "front_outperformance")):
            unit = "pp" if metric.endswith("_pp") else "pct"
            output["metrics"].setdefault(f"{suffix}_{name}_{unit}", tf["metrics"].get(metric))
    states = {name: tf["state"] for name, tf in output["timeframes"].items()}
    severe = {name for name, state in states.items()
              if state in ("STRESS_BUILDING", "BACKWARDATION_STRESS")}
    available = sum(tf["quality"] == "OK" for tf in output["timeframes"].values())
    if len(severe) >= 2:
        combined = "MTF_STRESS_CONSENSUS"
    elif severe.intersection(("1h", "4h")):
        combined = "HTF_STRESS_SHORT_TERM_COOLING" if states["15m"] == "NORMAL" else "HTF_STRESS"
    elif "15m" in severe:
        combined = "SHORT_TERM_STRESS_ONLY"
    elif any(s == "WATCH" for s in states.values()):
        combined = "MTF_WATCH"
    else:
        combined = "MTF_NO_TRIGGER" if available == 3 else "MTF_PARTIAL_DATA" if available else "MTF_DATA_UNAVAILABLE"
    output.update(mtf_context_state=combined, mtf_available_timeframes=available,
                  mtf_stress_timeframes=sorted(severe))


def feed_diagnostics(data: dict[str, pd.DataFrame], wall: pd.Timestamp,
                     config: Config, core_end: pd.Timestamp | None = None) -> dict[str, Any]:
    """No data after the wall clock or common VX analysis time can be used."""
    reports = {}
    for name in ("F1", "F2", "UVIX", "SVIX", "VIX", "VIX3M"):
        frame = data.get(name, pd.DataFrame())
        latest = frame.index[-1] if not frame.empty else None
        selected = frame if core_end is None or frame.empty else frame.loc[frame.index <= core_end]
        stamp = selected.index[-1] if not selected.empty else None
        age = (wall-stamp).total_seconds()/60 if stamp is not None else None
        lag = (core_end-stamp).total_seconds()/60 if core_end is not None and stamp is not None else None
        status, reason = "OK", "Használható lezárt ár-adat; realtime nem igazolt."
        if stamp is None:
            status, reason = "MISSING", "Nincs használható lezárt bar az elemzési időpontig."
        elif age > config.max_data_age_minutes:
            status, reason = "STALE", "Az adat kora meghaladja a megengedett maximumot."
        elif core_end is None:
            status, reason = "NO_CORE_ALIGNMENT", "Még nincs közös VX elemzési időpont."
        elif lag > config.optional_alignment_minutes:
            status, reason = "OUT_OF_ALIGNMENT", "Túl nagy az eltérés a közös VX elemzési időponttól."
        reports[name] = dict(
            status=status, reason=reason,
            latest_closed_at=latest.isoformat() if latest is not None else None,
            used_asof=stamp.isoformat() if stamp is not None else None,
            data_age_minutes=age, lag_to_vx_minutes=lag,
            reference_15m_at=None, return_15m_pct=None,
            volume_available=(finite(selected.volume.iloc[-1]) if stamp is not None else False),
        )
    return reports


def optional_context(data: dict[str, pd.DataFrame], output: dict[str, Any],
                     end: pd.Timestamp, config: Config) -> None:
    """Own-timestamp ETF returns; lag limits remain in force, not relaxed."""
    quality = output["feed_quality"]
    etf_returns = {}
    for name in ("UVIX", "SVIX"):
        report = quality[name]
        value = None
        if report["status"] == "OK":
            stamp = utc(report["used_asof"])
            local = stamp.tz_convert("America/New_York")
            if not (local.weekday() < 5 and 570 < local.hour*60+local.minute <= 960):
                report.update(status="OUTSIDE_RTH",
                              reason="Az ETF saját adat-időpontja nincs a normál kereskedési sávban.")
            else:
                frame = data[name]
                before = reference(frame, stamp-pd.Timedelta(minutes=15), config)
                report["reference_15m_at"] = before.name.isoformat() if before is not None else None
                value = change(frame, stamp, 15, config, same_rth_day=True)
                if value is None:
                    report.update(status="MISSING_REFERENCE",
                                  reason="Nincs megfelelő, azonos RTH napi 15 perces referencia.")
                else:
                    report["return_15m_pct"] = value
                    if report["lag_to_vx_minutes"] > 0:
                        report.update(status="OK_LAGGED",
                                      reason="Saját lezárt időpontú hozam; elfogadott késés a VX-hez képest.")
        etf_returns[name] = value
        output["metrics"][f"{name.lower()}_change_15m_pct"] = value
    uvix, svix = etf_returns["UVIX"], etf_returns["SVIX"]
    pair_ok = False
    if uvix is not None and svix is not None:
        skew = abs((utc(quality["UVIX"]["used_asof"])-utc(quality["SVIX"]["used_asof"])).total_seconds()/60)
        output["metrics"]["etf_pair_time_skew_minutes"] = skew
        pair_ok = skew <= config.reference_tolerance_minutes
    output["checks"]["uvix_up_svix_down"] = (
        uvix >= config.uvix_rise_15m_pct and svix <= config.svix_fall_15m_pct
        if pair_ok else None)
    output["etf_pair_status"] = (
        "ALIGNED_CONTEXT" if pair_ok else
        "PAIR_TIME_MISMATCH" if uvix is not None and svix is not None else "UNAVAILABLE")
    index_values = {}
    for name in ("VIX", "VIX3M"):
        report = quality[name]
        value = None
        if report["status"] == "OK":
            value = float(data[name].loc[utc(report["used_asof"]), "close"])
            if report["lag_to_vx_minutes"] > 0:
                report.update(status="OK_LAGGED", reason="Elfogadott késésű indexár; nem egyidejű a VX-szel.")
        index_values[name] = value
        output["metrics"][name.lower()] = value
    vix, vix3m = index_values["VIX"], index_values["VIX3M"]
    pair_ok = False
    if vix is not None and vix3m is not None:
        skew = abs((utc(quality["VIX"]["used_asof"])-utc(quality["VIX3M"]["used_asof"])).total_seconds()/60)
        output["metrics"]["index_pair_time_skew_minutes"] = skew
        pair_ok = skew <= config.reference_tolerance_minutes
    output["metrics"]["vix_vix3m_ratio"] = vix/vix3m if pair_ok else None
    output["index_pair_status"] = (
        "ALIGNED_CONTEXT" if pair_ok else
        "PAIR_TIME_MISMATCH" if vix is not None and vix3m is not None else "UNAVAILABLE")


def evaluate(frames: dict[str, pd.DataFrame], config: Config = Config(), *,
             now: pd.Timestamp | None = None) -> dict[str, Any]:
    wall_clock = utc(now if now is not None else pd.Timestamp.now(tz="UTC"))
    data = {name: closed_minutes(frame) for name, frame in frames.items()}
    data = {name: frame if frame.empty else frame.loc[frame.index <= wall_clock]
            for name, frame in data.items()}
    output: dict[str, Any] = {
        "version": VERSION, "observed_at": wall_clock.isoformat(), "asof": None,
        "candidate_state": "DATA_UNAVAILABLE", "metrics": {}, "checks": {},
        "notes": [], "realtime_verified": False,
        "short_covering_confirmed": False,
        "cause": "Price/volume cannot identify who buys or whether shorts close.",
        "oi_context": "N/A: daily OI confirmation not implemented",
        "cot_context": "N/A: participant-category confirmation not implemented",
        "feed_quality": feed_diagnostics(data, wall_clock, config),
        "etf_pair_status": "UNAVAILABLE", "index_pair_status": "UNAVAILABLE",
        "timeframes": {
            name: dict(state="DATA_UNAVAILABLE", quality="CORE_UNAVAILABLE", asof=None,
                       reference_at=None, target_reference_at=None, metrics={}, checks={},
                       optional_feeds={}, thresholds=profile,
                       reason="A friss közös VX mag még nem áll rendelkezésre.")
            for name, profile in mtf_profiles(config).items()},
        "mtf_context_state": "MTF_DATA_UNAVAILABLE", "mtf_available_timeframes": 0,
        "mtf_stress_timeframes": [],
    }
    f1, f2 = data.get("F1", pd.DataFrame()), data.get("F2", pd.DataFrame())
    if f1.empty or f2.empty:
        output["notes"].append("Core F1/F2 data missing. No safe NORMAL classification.")
        return output
    common = f1.index.intersection(f2.index)
    common = common[common <= wall_clock]
    if common.empty:
        output["notes"].append("No common closed F1/F2 minute.")
        return output
    end = common[-1]
    output["asof"] = end.isoformat()
    output["feed_quality"] = feed_diagnostics(data, wall_clock, config, end)
    age = (wall_clock-end).total_seconds()/60
    output["metrics"]["data_age_minutes"] = age
    if age > config.max_data_age_minutes:
        output["notes"].append("Core data stale; this is a feed warning, not market stress.")
        return output
    # Never let a future bar enter the volume or optional context.
    data = {name: frame if frame.empty else frame.loc[frame.index <= end]
            for name, frame in data.items()}
    optional_context(data, output, end, config)
    f1, f2 = data["F1"], data["F2"]
    p1, p2 = float(f1.loc[end, "close"]), float(f2.loc[end, "close"])
    spread = 100*(p2/p1-1)
    previous = end-pd.Timedelta(minutes=30)
    baseline_position = common.searchsorted(previous, side="right") - 1
    baseline = common[baseline_position] if baseline_position >= 0 else None
    if baseline is not None and (previous-baseline).total_seconds()/60 > config.reference_tolerance_minutes:
        baseline = None
    a, b = (f1.loc[baseline], f2.loc[baseline]) if baseline is not None else (None, None)
    spread_before = 100*(float(b.close)/float(a.close)-1) if a is not None and b is not None else None
    r15 = change(f1, end, 15, config)
    r30 = 100*(p1/float(a.close)-1) if a is not None else None
    f2_r30 = 100*(p2/float(b.close)-1) if b is not None else None
    delta = spread-spread_before if spread_before is not None else None
    outperform = r30-f2_r30 if r30 is not None and f2_r30 is not None else None
    rvol = time_of_day_volume_ratio(f1, end, config)
    output["metrics"].update(
        f1_price=p1, f2_price=p2, curve_pct=spread,
        f1_change_15m_pct=r15, f1_change_30m_pct=r30,
        f1_change_1h_pct=change(f1, end, 60, config),
        f2_change_30m_pct=f2_r30, curve_change_30m_pp=delta,
        front_outperformance_30m_pp=outperform, f1_volume_rvol_15m=rvol,
    )
    # Longer horizons remain assessable even when the legacy 30m reference is missing.
    evaluate_timeframes(data, output, end, config)
    if any(x is None for x in (r15, r30, delta, outperform)):
        output["notes"].append("Missing elapsed-time 15m/30m reference. No bar-count substitution.")
        for name in ("F1", "F2"):
            output["feed_quality"][name].update(
                status="CORE_REFERENCE_INCOMPLETE",
                reason="A közös VX 15m/30m referenciaellenőrzés nem teljesült; nincs core minősítés.")
        return output
    price = r15 >= config.f1_rise_15m_pct or r30 >= config.f1_rise_30m_pct
    compression = delta <= config.curve_compression_30m_pp and outperform >= config.front_outperformance_30m_pp
    backward = spread <= config.backwardation_pct
    backward_rising = backward and r15 >= config.backwardation_price_rise_15m_pct
    output["checks"].update(
        price_acceleration=price, curve_compression=compression,
        backwardation=backward, backwardation_with_rising_front=backward_rising,
        elevated_time_of_day_volume=(rvol >= config.volume_ratio_threshold if rvol is not None else None),
    )
    if backward_rising:
        state = "BACKWARDATION_STRESS"
    elif price and compression:
        state = "STRESS_BUILDING"
    elif price or compression or spread <= config.flat_curve_pct:
        state = "WATCH"
    else:
        state = "NORMAL"
    output["candidate_state"] = state
    output["notes"] += [
        "Thresholds are unvalidated; correlated checklist items are not independent evidence.",
        "UVIX/SVIX confirmation is mechanically related exposure, not an independent causal test.",
        "VIX/VIX3M index ratio does not replace the F1/F2 futures curve.",
        "NORMAL means no current heuristic trigger, not absence of crash risk.",
        "ETF returns use their own closed timestamps at/before VX asof; see feed_quality for lag.",
        "MTF: gördülő 15m/1h/4h ablak, nem órás gyertyazárás. Átfedő, nem független bizonyíték.",
        "A hosszabb kontextus nem írja felül a legacy 15m/30m index-input állapotot.",
    ]
    return output


def _alert_update_single(result: dict[str, Any], memory: dict[str, Any], fingerprint: str,
                 config: Config = Config()) -> tuple[bool, dict[str, Any], str]:
    """Confirmation uses advancing source timestamps, never repeated polls."""
    state, stamp = result["candidate_state"], result.get("asof")
    severe = state in ("STRESS_BUILDING", "BACKWARDATION_STRESS")
    old = dict(memory) if memory.get("fingerprint") == fingerprint else {}
    count = int(old.get("confirmation_count", 0))
    if not severe or state != old.get("candidate_state"):
        count = 0
    previous = old.get("last_counted_asof")
    if (previous is not None and stamp is not None
            and (utc(stamp)-utc(previous)).total_seconds()/60 > config.maximum_confirmation_gap_minutes):
        # A much later restart cannot confirm yesterday's candidate.
        count, previous = 0, None
    advanced = stamp is not None and (
        previous is None or
        (utc(stamp)-utc(previous)).total_seconds()/60 >= config.minimum_confirmation_gap_minutes
    )
    counted = stamp if severe and advanced else previous
    if severe and advanced:
        count += 1
    wall = utc(result["observed_at"])
    last_alert = old.get("last_alert_at")
    cooldown_ok = last_alert is None or (
        wall-utc(last_alert)).total_seconds()/60 >= config.cooldown_minutes
    feed_alert = state == "DATA_UNAVAILABLE" and old.get("candidate_state") != state
    market_alert = severe and count >= config.confirmations and cooldown_ok and advanced
    emit = bool(feed_alert or market_alert)
    updated = {
        "fingerprint": fingerprint, "candidate_state": state,
        "confirmation_count": count, "last_counted_asof": counted if severe else None,
        "last_alert_at": wall.isoformat() if market_alert else last_alert,
    }
    label = ("ADATHIÁNY / ELAVULT FEED, NEM PIACI CRASH-JEL" if feed_alert else
             state if market_alert else f"Nincs új alert; megerősítés {count}/{config.confirmations}")
    return emit, updated, label


def alert_update(result: dict[str, Any], memory: dict[str, Any], fingerprint: str,
                 config: Config = Config()) -> tuple[bool, dict[str, Any], str]:
    """Legacy core + separately confirmed MTF context alerts. No signal promotion."""
    emit, updated, message = _alert_update_single(result, memory, fingerprint, config)
    old = memory.get("timeframe_alerts", {}) if memory.get("fingerprint") == fingerprint else {}
    counts, alerted = {}, []
    for name, tf in result.get("timeframes", {}).items():
        mini = dict(candidate_state=tf["state"], asof=tf.get("asof"),
                    observed_at=result["observed_at"])
        tf_emit, tf_memory, _ = _alert_update_single(
            mini, old.get(name, {}),
            f"{fingerprint}:MTF:{name}:{json.dumps(tf.get('thresholds', {}), sort_keys=True)}",
            config)
        counts[name] = tf_memory
        # Missing HTF history is a displayed diagnostic, not a repeated feed beep.
        if tf_emit and tf["state"] in ("STRESS_BUILDING", "BACKWARDATION_STRESS"):
            alerted.append(name)
    updated.update(timeframe_alerts=counts, mtf_alerted_timeframes=alerted,
                   legacy_alert_emitted=emit)
    if alerted:
        suffix = "MTF KONTEXTUS ALERT: "+", ".join(
            f"{name} {result['timeframes'][name]['state']}" for name in alerted)
        message = message+" | "+suffix if emit else suffix
    return bool(emit or alerted), updated, message


def expiry_of(details: Any) -> pd.Timestamp | None:
    text = str(getattr(details, "realExpirationDate", "") or
               getattr(details.contract, "lastTradeDateOrContractMonth", ""))[:8]
    try:
        return pd.Timestamp(pd.to_datetime(text, format="%Y%m%d")) if len(text) == 8 else None
    except (ValueError, TypeError):
        return None


def select_monthlies(details: list[Any], now: pd.Timestamp, front: str | None = None,
                     second: str | None = None) -> tuple[Any, Any]:
    today = utc(now).tz_convert("America/New_York").date()
    candidates = {}
    for item in details:
        c, expiry = item.contract, expiry_of(item)
        if (getattr(c, "tradingClass", "") == "VX"
                and getattr(c, "secType", "") == "FUT"
                and expiry is not None and expiry.date() > today
                and getattr(c, "conId", 0) > 0):
            candidates[c.conId] = (expiry, c)
    ordered = sorted(candidates.values(), key=lambda x: x[0])
    if front is not None or second is not None:
        if not front or not second:
            raise ValueError("Both --front-expiry and --second-expiry required.")
        selected = []
        for text in (front, second):
            matches = [pair for pair in ordered if pair[0].strftime("%Y%m%d").startswith(text)]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous or missing monthly VX expiry: {text}")
            selected.append(matches[0])
    else:
        selected = ordered[:2]
    if len(selected) != 2 or selected[0][0] >= selected[1][0]:
        raise ValueError("Two distinct qualified monthly VX contracts required.")
    return selected[0][1], selected[1][1]


def fetch_history(ib: Any, contract: Any, name: str, folder: Path, config: Config,
                  *, regular_hours: bool = False) -> pd.DataFrame:
    from ib_insync import util
    path = folder/f"{name}_{contract.conId}_1m.csv"
    cached = pd.DataFrame()
    if path.exists():
        try:
            cached = pd.read_csv(path, index_col="date", parse_dates=["date"])
            cached.index = pd.to_datetime(cached.index, utc=True)
        except (ValueError, OSError):
            log.warning("%s cache invalid; restarting history.", name)
    bars = ib.reqHistoricalData(
        contract, endDateTime="",
        durationStr=config.update_duration if not cached.empty else config.initial_duration,
        barSizeSetting="1 min", whatToShow="TRADES", useRTH=regular_hours,
        formatDate=2, keepUpToDate=False, timeout=20,
    )
    new = util.df(bars)
    if new is None or new.empty:
        # Never return old cache as if a new historical request succeeded.
        raise RuntimeError(f"{name}: no current historical data / permission.")
    new = new.set_index("date")[["open", "high", "low", "close", "volume"]]
    new.index = pd.to_datetime(new.index, utc=True)
    data = pd.concat([cached, new]).sort_index()
    data = data.loc[~data.index.duplicated(keep="last")]
    data = data.loc[data.index >= data.index[-1]-pd.Timedelta(days=16)]
    data.index.name = "date"
    data.to_csv(path)
    log.info("%s: %s | %d cached bars | last start %s",
             name, getattr(contract, "localSymbol", ""), len(data), data.index[-1])
    return data


def tws_cycle(config: Config, folder: Path, front: str | None,
              second: str | None) -> tuple[dict[str, Any], str]:
    from ib_insync import IB, Future, Index, Stock
    ib, frames, sources, errors = IB(), {}, {}, []
    now = pd.Timestamp.now(tz="UTC")
    fingerprint = "NO_FUTURES_PAIR"
    try:
        ib.connect(config.host, config.port, clientId=config.client_id,
                   readonly=True, timeout=20)
        ib.reqMarketDataType(config.market_data_type)
        details = ib.reqContractDetails(Future(
            symbol="VIX", exchange="CFE", currency="USD", tradingClass="VX"))
        f1, f2 = select_monthlies(details, now, front, second)
        fingerprint = f"{VERSION}:{f1.conId}:{f2.conId}"
        contracts = {"F1": f1, "F2": f2}
        for name, contract in (
            ("UVIX", Stock("UVIX", "SMART", "USD")),
            ("SVIX", Stock("SVIX", "SMART", "USD")),
            ("VIX", Index("VIX", "CBOE", "USD")),
            ("VIX3M", Index("VIX3M", "CBOE", "USD")),
        ):
            try:
                qualified = ib.qualifyContracts(contract)
                if qualified:
                    contracts[name] = qualified[0]
                else:
                    errors.append(f"{name}: qualification unavailable")
            except Exception as error:
                errors.append(f"{name}: {error}")
        for name, contract in contracts.items():
            sources[name] = {
                "conId": contract.conId, "localSymbol": contract.localSymbol,
                "expiry": getattr(contract, "lastTradeDateOrContractMonth", ""),
                "tradingClass": getattr(contract, "tradingClass", ""),
            }
            if name in ("F1", "F2"):
                text = str(getattr(contract, "lastTradeDateOrContractMonth", ""))[:8]
                try:
                    expiry = pd.Timestamp(pd.to_datetime(text, format="%Y%m%d")).date()
                    if (expiry-now.tz_convert("America/New_York").date()).days <= 2:
                        errors.append(f"{name}: expiry within 2 calendar days; convergence/roll context requires care.")
                except ValueError:
                    errors.append(f"{name}: expiry-day proximity not determined.")
            try:
                frames[name] = fetch_history(ib, contract, name, folder, config,
                                             regular_hours=name not in ("F1", "F2"))
            except Exception as error:
                errors.append(f"{name}: {error}")
        result = evaluate(frames, config, now=pd.Timestamp.now(tz="UTC"))
    except Exception as error:
        result = evaluate({}, config, now=now)
        errors.append(f"TWS/core: {error}")
    finally:
        if ib.isConnected():
            ib.disconnect()
    result["sources"] = sources
    result["data_mode_requested"] = {
        1: "LIVE_REQUESTED_NOT_VERIFIED", 2: "FROZEN", 3: "DELAYED", 4: "DELAYED_FROZEN",
    }[config.market_data_type]
    result["notes"] += errors
    result["notes"].append("Daily OI and weekly participant data are not fetched by this version.")
    return result, fingerprint


def overall_status(result: dict[str, Any], memory: dict[str, Any],
                   config: Config = Config()) -> tuple[str, str]:
    """Display-only summary: never changes the heuristic or alert decisions."""
    state = result["candidate_state"]
    count = int(memory.get("confirmation_count", 0))
    higher = [name for name in ("1h", "4h")
              if result.get("timeframes", {}).get(name, {}).get("state")
              in ("STRESS_BUILDING", "BACKWARDATION_STRESS")]
    if higher and state in ("NORMAL", "WATCH", "DATA_UNAVAILABLE"):
        confirmed = any(memory.get("timeframe_alerts", {}).get(name, {}).get(
            "confirmation_count", 0) >= config.confirmations for name in higher)
        label = "MEGERŐSÍTVE" if confirmed else "MEGERŐSÍTÉSRE VÁR"
        short = {"NORMAL": "NINCS RÖVID TÁVÚ TRIGGER", "WATCH": "RÖVID TÁVÚ WATCH",
                 "DATA_UNAVAILABLE": "RÖVID TÁV NEM ÉRTÉKELHETŐ"}[state]
        return (f"HTF STRESSZ / {label} / {short}",
                f"{', '.join(higher)} gördülő ablakon stresszjelölt maradt; "
                "a rövid táv állapota ezt nem törli. Nem belépési engedély.")
    if state == "DATA_UNAVAILABLE":
        return "NEM ÉRTÉKELHETŐ", "Hiányzó vagy elavult core adat; nem NORMAL és nem crash-jel."
    if state == "NORMAL":
        return "NINCS AKTUÁLIS STRESSZTRIGGER", "A heurisztika nem jelez; ez nem jelent crash-kockázatmentességet."
    if state == "WATCH":
        return "FIGYELŐ ÁLLAPOT", "Részleges stresszfeltétel; még nincs megerősített stresszkörnyezet."
    if state in ("STRESS_BUILDING", "BACKWARDATION_STRESS"):
        label = "BACKWARDATION STRESSZ" if state == "BACKWARDATION_STRESS" else "ÉPÜLŐ STRESSZ"
        if count >= config.confirmations:
            return f"{label} / MEGERŐSÍTVE", "A kutatási megerősítési feltétel teljesült; nem belépési engedély."
        return f"{label} / MEGERŐSÍTÉSRE VÁR", "A jelölt már látszik; újabb, előrehaladó forrásadat szükséges."
    return "ISMERETLEN ÁLLAPOT", "A visszaadott állapotot ellenőrizni kell."


def show(result: dict[str, Any], emit: bool, message: str, memory: dict[str, Any],
         config: Config = Config()) -> None:
    print("\n"+"="*88)
    print(f"VIX / UVIX / SVIX STRESS MONITOR {VERSION} - elemző/paper rendszer")
    print(f"Futás: {result['observed_at']} | Közös lezárt VX adat: {result.get('asof') or 'N/A'}")
    print(f"Kért adatmód: {result.get('data_mode_requested', 'SYNTHETIC')}")
    print(f"Környezet-jelölt: {result['candidate_state']} | Alert: {message}")
    for name, source in result.get("sources", {}).items():
        print(f"{name}: {source}")
    print("-"*88)
    print("FEEDENKÉNTI ADATMINŐSÉG (időpontok UTC, lezárási idő)")
    for name, report in result.get("feed_quality", {}).items():
        age = report["data_age_minutes"]
        lag = report["lag_to_vx_minutes"]
        age_text = f"{age:.1f} perc" if finite(age) else "N/A"
        lag_text = f"{lag:.1f} perc" if finite(lag) else "N/A"
        print(f"{name:<6} | {report['status']} | adatkor={age_text} | VX-eltérés={lag_text}")
        print(f"       Használt adat: {report['used_asof'] or 'N/A'} | "
              f"Legutolsó lezárt: {report['latest_closed_at'] or 'N/A'}")
        if name in ("UVIX", "SVIX"):
            print(f"       15m referencia: {report['reference_15m_at'] or 'N/A'}")
        print(f"       {report['reason']}")
    print(f"ETF-pár időillesztés:   {result.get('etf_pair_status', 'UNAVAILABLE')}")
    print(f"Indexpár időillesztés:  {result.get('index_pair_status', 'UNAVAILABLE')}")
    print("-"*88)
    print("15m / 1h / 4h STRESSZKONTEXTUS: GÖRDÜLŐ ABLAKOK, NEM HTF GYERTYÁK")
    print("IDŐTÁV | ÁLLAPOT | ADAT | F1 % | F2 % | GÖRBE VÁLT. pp | FRONT TÖBBLET pp | MEGERŐSÍTÉS")
    def metric_text(value):
        return f"{value:+.3f}" if finite(value) else "N/A"
    for name, tf in result.get("timeframes", {}).items():
        metrics = tf["metrics"]
        count = memory.get("timeframe_alerts", {}).get(name, {}).get("confirmation_count", 0)
        print(f"{name:<6} | {tf['state']} | {tf['quality']} | "
              f"{metric_text(metrics.get('f1_change_pct'))} | "
              f"{metric_text(metrics.get('f2_change_pct'))} | "
              f"{metric_text(metrics.get('curve_change_pp'))} | "
              f"{metric_text(metrics.get('front_outperformance_pp'))} | {count}/{config.confirmations}")
        print(f"       Végpont: {tf.get('asof') or 'N/A'} | referencia: {tf.get('reference_at') or 'N/A'}")
        thresholds = tf["thresholds"]
        print(f"       Küszöbök: F1 >= {thresholds['price']:g}%; görbe <= "
              f"{thresholds['compression']:g} pp ÉS front többlet >= {thresholds['outperform']:g} pp.")
        print(f"       Lefedettség: {tf.get('coverage', 0):.1%} | "
              f"max. belső gap: {metric_text(tf.get('max_gap_minutes'))} perc | {tf['reason']}")
        for name_feed, report in tf["optional_feeds"].items():
            print(f"       {name_feed}: {metric_text(report['return_pct'])}% | {report['status']} | "
                  f"saját végpont: {report['asof'] or 'N/A'} | ref: {report['reference_at'] or 'N/A'}")
    print(f"MTF összkép: {result.get('mtf_context_state', 'N/A')} | "
          f"értékelhető ablakok: {result.get('mtf_available_timeframes', 0)}/3")
    print("A hosszabb ablakok átfednek: nem három független stresszbizonyíték.")
    print("-"*88)
    for name, value in result["metrics"].items():
        text = f"{value:.4f}" if finite(value) else "N/A"
        print(f"{name:<36} {text}")
    print("-"*88)
    for name, value in result["checks"].items():
        print(f"[{'N/A' if value is None else 'IGEN' if value else 'NEM'}] {name}")
    print(f"Short covering / asset-manager zárás: NEM IGAZOLT")
    print(result["oi_context"])
    print(result["cot_context"])
    print("Nem buy/sell jel, nem küld ordert, nem kezel pozíciót.")
    for note in result["notes"]:
        print(f"- {note}")
    if emit:
        print("\n"+"!"*88+"\nALERT: "+message+"\n"+"!"*88)
    status, meaning = overall_status(result, memory, config)
    age = result["metrics"].get("data_age_minutes")
    age_text = f"{age:.1f} perc" if finite(age) else "N/A"
    print("\n"+"="*88)
    print("ÖSSZESÍTETT VIX STRESSZ STÁTUSZ")
    print("="*88)
    print(f"Összesített státusz:    {status}")
    print(f"MTF kontextus:          {result.get('mtf_context_state', 'N/A')}")
    print(f"MTF stresszablakok:     {', '.join(result.get('mtf_stress_timeframes', [])) or 'Nincs'}")
    print(f"Új MTF alert ablakai:   {', '.join(memory.get('mtf_alerted_timeframes', [])) or 'Nincs'}")
    print(f"Heurisztika állapota:   {result['candidate_state']}")
    print(f"Megerősítések:          {memory.get('confirmation_count', 0)}/{config.confirmations}")
    print(f"Új alert ezen futásban: {'IGEN' if emit else 'NEM'}")
    print(f"Alert részlete:         {message}")
    print(f"Közös lezárt VX adat:   {result.get('asof') or 'N/A'}")
    print(f"Core adat kora:         {age_text}")
    print(f"Valós idejű adat:       {'IGAZOLT' if result.get('realtime_verified') else 'NEM IGAZOLT'}")
    for name in ("UVIX", "SVIX", "VIX", "VIX3M"):
        report = result.get("feed_quality", {}).get(name, {})
        print(f"{name} adatstátusz:".ljust(23)+report.get("status", "N/A"))
    print(f"Értelmezés:            {meaning}")
    if result["candidate_state"] == "DATA_UNAVAILABLE":
        for note in result["notes"]:
            print(f"Adatdiagnosztika:       {note}")
    print("Az új alert hiánya nem azonos a stressz hiányával; cooldown is elnyomhatja.")
    print("Elemző/paper státusz, nem buy/sell jel. Nem küld ordert.")
    print("="*88, flush=True)


def save_outputs(result: dict[str, Any], memory: dict[str, Any], folder: Path,
                 emit: bool, message: str) -> None:
    packet = {**result, "alert_emitted": emit, "alert_message": message,
              "confirmation_count": memory["confirmation_count"],
              "mtf_alerted_timeframes": memory.get("mtf_alerted_timeframes", []),
              "mtf_confirmation_counts": {
                  name: item["confirmation_count"]
                  for name, item in memory.get("timeframe_alerts", {}).items()}}
    core_emit = memory.get("legacy_alert_emitted", emit and not packet["mtf_alerted_timeframes"])
    packet["legacy_alert_emitted"] = bool(core_emit)
    packet["alert_scope"] = ("CORE_AND_MTF" if core_emit and packet["mtf_alerted_timeframes"]
                             else "CORE" if core_emit else "MTF_CONTEXT"
                             if packet["mtf_alerted_timeframes"] else "NONE")
    # Atomic local state writes. Concurrent processes using this folder are
    # unsupported; use distinct data dirs/client IDs if intentionally duplicated.
    for name, value in (("vix_monitor_state.json", memory), ("vix_monitor_latest.json", packet)):
        path = folder/name
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                        allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    columns = ["observed_at", "asof", "candidate_state", "alert_emitted",
               "confirmation_count", "f1_price", "f2_price", "curve_pct",
               "f1_change_15m_pct", "f1_change_30m_pct", "curve_change_30m_pp",
               "front_outperformance_30m_pp", "f1_volume_rvol_15m",
               "uvix_change_15m_pct", "svix_change_15m_pct"]
    row = {**packet, **result["metrics"]}
    path = folder/"vix_monitor_observations.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    # Separate schema preserves compatibility with the existing observations CSV.
    quality_path = folder/"vix_monitor_feed_quality.csv"
    quality_exists = quality_path.exists()
    quality_columns = ["observed_at", "vx_asof", "feed", "status", "reason",
                       "latest_closed_at", "used_asof", "data_age_minutes",
                       "lag_to_vx_minutes", "reference_15m_at", "return_15m_pct",
                       "volume_available"]
    with quality_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=quality_columns, extrasaction="ignore")
        if not quality_exists:
            writer.writeheader()
        for name, report in result.get("feed_quality", {}).items():
            writer.writerow({"observed_at": result["observed_at"],
                             "vx_asof": result.get("asof"), "feed": name, **report})
    # New schema: never append extra columns to the older observations file.
    mtf_path = folder/"vix_monitor_mtf_observations.csv"
    mtf_exists = mtf_path.exists()
    columns = ["observed_at", "asof", "timeframe", "state", "quality", "reference_at",
               "coverage", "max_gap_minutes", "f1_change_pct", "f2_change_pct",
               "curve_pct", "curve_change_pp", "front_outperformance_pp", "f1_volume_rvol",
               "confirmation_count", "alert_emitted", "mtf_context_state"]
    with mtf_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        if not mtf_exists:
            writer.writeheader()
        for name, tf in result.get("timeframes", {}).items():
            writer.writerow({
                **tf, **tf["metrics"], "observed_at": result["observed_at"],
                "timeframe": name, "mtf_context_state": result.get("mtf_context_state"),
                "confirmation_count": packet["mtf_confirmation_counts"].get(name, 0),
                "alert_emitted": name in packet["mtf_alerted_timeframes"]})


def synthetic_frames() -> dict[str, pd.DataFrame]:
    end = pd.Timestamp.now(tz="UTC").floor("min")-pd.Timedelta(minutes=15)
    idx = pd.date_range(end=end, periods=601, freq="min")
    frames = {}
    for name, base, rise in (("F1", 21., 2.), ("F2", 24., .2)):
        prices = [base+rise*max(0, i-569)/30 for i in range(601)]
        frames[name] = pd.DataFrame(
            dict(open=prices, high=[p+.01 for p in prices],
                 low=[p-.01 for p in prices], close=prices, volume=100.), index=idx)
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only, heuristic VX term-structure alerts.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--client-id", type=int, default=None)
    parser.add_argument("--market-data-type", type=int, choices=[1, 2, 3, 4], default=None)
    parser.add_argument("--loop", action="store_true", help="Local loop; not an OS scheduled task.")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between completed runs.")
    parser.add_argument("--confirmations", type=int, default=None)
    parser.add_argument("--max-confirmation-gap", type=float, default=None,
                        help="Maximum minutes between counted source observations.")
    parser.add_argument("--beep", action="store_true", default=None, help="Local Windows sound on emitted alerts.")
    parser.add_argument("--front-expiry", help="Explicit monthly VX expiry YYYYMM or YYYYMMDD.")
    parser.add_argument("--second-expiry")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent/"vix_monitor_data")
    parser.add_argument("--demo", action="store_true", help="Synthetic offline example, no saved real alert state.")
    args = parser.parse_args(argv)
    overrides = {field: value for field, value in (
        ("host", args.host), ("port", args.port), ("client_id", args.client_id),
        ("market_data_type", args.market_data_type), ("confirmations", args.confirmations),
        ("maximum_confirmation_gap_minutes", args.max_confirmation_gap),
        ("beep", args.beep)) if value is not None}
    try:
        config = replace(Config(), **overrides)
    except ValueError as error:
        parser.error(str(error))
    if args.interval < 60 or config.confirmations < 1:
        parser.error("interval must be >=60 seconds and confirmations >=1")
    if config.maximum_confirmation_gap_minutes < config.minimum_confirmation_gap_minutes:
        parser.error("max-confirmation-gap must be >=5 minutes")
    if bool(args.front_expiry) != bool(args.second_expiry):
        parser.error("both expiry arguments must be specified together")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logging.getLogger("ib_insync.wrapper").setLevel(logging.WARNING)
    mode = "SZINTETIKUS DEMO" if args.demo else "TWS ELEMZÉS"
    print(f"VIX STRESS MONITOR {VERSION} | {mode} | "
          f"{'CIKLIKUS' if args.loop and not args.demo else 'EGYSZERI'} FUTÁS", flush=True)
    if not args.demo:
        print(f"TWS: {config.host}:{config.port} | clientId={config.client_id} | "
              f"kért adatmód={config.market_data_type}", flush=True)
    if not args.demo:
        args.data_dir.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            if args.demo:
                result = evaluate(synthetic_frames(), config)
                result["data_mode_requested"] = "SYNTHETIC_DEMO_NOT_MARKET_DATA"
                fingerprint, memory = "DEMO", {}
            else:
                try:
                    memory = json.loads((args.data_dir/"vix_monitor_state.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    memory = {}
                try:
                    result, fingerprint = tws_cycle(config, args.data_dir, args.front_expiry, args.second_expiry)
                except ImportError:
                    result = evaluate({}, config)
                    result["notes"].append("Hiányzó ib_insync csomag: pip install ib_insync")
                    result["data_mode_requested"] = "UNAVAILABLE"
                    fingerprint = "MISSING_DEPENDENCY"
            emit, updated, message = alert_update(result, memory, fingerprint, config)
            show(result, emit, message, updated, config)
            if not args.demo:
                save_outputs(result, updated, args.data_dir, emit, message)
            if emit and args.beep and not args.demo:
                try:
                    import winsound
                    winsound.Beep(1100, 600)
                except (ImportError, RuntimeError):
                    print("Hangjelzés nem érhető el; a konzolos alert megmaradt.")
            if not args.loop or args.demo:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor leállítva.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
