"""Causal ICT-inspired OHLC zones. Research/paper, NOT institutional-flow proof.

Three-candle FVG, displaced swing break + last opposite candle OB, qualified
failed OB breaker, inversion FVG, confirmed-pivot sweeps, and closed 5m reactions.
No orders, downloads, persistent state or permission changes. Minute bar STARTS.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

VERSION = "0.1.2"


@dataclass(frozen=True)
class StructureConfig:
    enabled: bool = True
    timeframes_minutes: tuple[int, ...] = (60, 120, 240)
    timezone: str = "America/New_York"
    session_open_hour: int = 18
    session_close_hour: int = 17
    pivot_left: int = 2
    pivot_right: int = 2
    atr_period: int = 14
    min_htf_bars: int = 18
    history_bars: int = 400
    ob_search_bars: int = 6
    displacement_body_atr: float = .50
    displacement_body_share: float = .60
    structure_buffer_atr: float = .10
    min_fvg_atr: float = .05
    max_ob_range_atr: float = 2.0
    zone_bounds: str = "WICK"
    require_ob_fvg: bool = True
    require_ob_origin_sweep: bool = True
    require_breaker_sweep: bool = True
    require_breaker_mss: bool = True
    sweep_memory_bars: int = 12
    max_zone_age_hours: float = 240
    max_zones_per_timeframe: int = 50
    display_zones_per_timeframe: int = 6
    reaction_buffer_atr_5m: float = .10
    reaction_max_range_atr_5m: float = 3.0
    reaction_fresh_minutes: float = 30
    source_max_age_minutes: float = 45
    max_ltf_gap_minutes: float = 10
    enable_breaker_candidates: bool = True
    candidate_reference_bars: int = 3
    candidate_lifetime_bars: int = 12

    def __post_init__(self) -> None:
        integers = (self.pivot_left, self.pivot_right, self.atr_period, self.min_htf_bars,
                    self.history_bars, self.ob_search_bars, self.sweep_memory_bars,
                    self.max_zones_per_timeframe, self.display_zones_per_timeframe,
                    self.candidate_reference_bars, self.candidate_lifetime_bars)
        floats = (self.displacement_body_atr, self.displacement_body_share,
                  self.structure_buffer_atr, self.min_fvg_atr, self.max_ob_range_atr,
                  self.max_zone_age_hours, self.reaction_buffer_atr_5m,
                  self.reaction_max_range_atr_5m, self.reaction_fresh_minutes,
                  self.source_max_age_minutes, self.max_ltf_gap_minutes)
        if any(not isinstance(x, int) or x < 1 for x in integers):
            raise ValueError("Positive integer structure settings required.")
        if any(not np.isfinite(x) or x <= 0 for x in floats):
            raise ValueError("Positive finite structure settings required.")
        if (self.zone_bounds not in ("WICK", "BODY") or self.displacement_body_share > 1
                or not self.timeframes_minutes
                or any(x not in (60, 120, 240) for x in self.timeframes_minutes)
                or len(set(self.timeframes_minutes)) != len(self.timeframes_minutes)
                or self.min_htf_bars < self.atr_period+1
                or self.history_bars < self.min_htf_bars
                or not 0 <= self.session_close_hour < self.session_open_hour < 24):
            raise ValueError("Invalid structure settings.")


DEFAULT_STRUCTURE = StructureConfig()


def _iso(value: Any) -> str | None:
    return None if value is None else pd.Timestamp(value).isoformat()


def _atr(frame: pd.DataFrame, period: int) -> pd.Series:
    previous = frame.close.shift()
    tr = pd.concat([frame.high-frame.low, (frame.high-previous).abs(),
                    (frame.low-previous).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def _expected_ends(start: pd.Timestamp, end: pd.Timestamp, minutes: int,
                   cfg: StructureConfig) -> pd.DatetimeIndex:
    """Normal Sun-Thu futures sessions, no exchange holiday calendar.

    Full buckets only. Missing holiday bars cause conservative unknown gaps.
    """
    from market_transition import _session_bounds
    days = pd.date_range(start.tz_convert(cfg.timezone).date()-timedelta(days=1),
                         end.tz_convert(cfg.timezone).date(), freq="D")
    values = []
    for day in days:
        if day.weekday() not in (6, 0, 1, 2, 3):
            continue
        opened, closed = _session_bounds(day.date(), cfg)
        stamp = opened+pd.Timedelta(minutes=minutes)
        while stamp <= closed:
            if start <= stamp <= end:
                values.append(stamp)
            stamp += pd.Timedelta(minutes=minutes)
    return pd.DatetimeIndex(values, tz=cfg.timezone)


def _span_complete(index: pd.DatetimeIndex, expected: pd.DatetimeIndex,
                   start: pd.Timestamp, end: pd.Timestamp) -> bool:
    needed = expected[(expected >= start) & (expected <= end)]
    return len(needed) > 0 and needed.isin(index).all()


def _span_checker(index: pd.DatetimeIndex, expected: pd.DatetimeIndex):
    """Precomputed O(log n) equivalent of repeated _span_complete calls.

    Both structure schedules are chronological, so searchsorted range
    counting gives the same completeness verdict as full boolean masking.
    """
    present = np.asarray(expected.isin(index))
    cumulative = np.concatenate(([0], np.cumsum(present)))

    def complete(start: pd.Timestamp, end: pd.Timestamp) -> bool:
        low = int(np.searchsorted(expected, start, side="left"))
        high = int(np.searchsorted(expected, end, side="right"))
        return high > low and cumulative[high] - cumulative[low] == high - low

    return complete


def _displaced_at(open_: float, high: float, low: float, close: float,
                  prior_atr: float, direction: int,
                  cfg: StructureConfig) -> bool:
    """Array-accessor twin of _displaced; identical thresholds."""
    body, span = direction*(close-open_), high-low
    return bool(np.isfinite(prior_atr) and prior_atr > 0 and span > 0
                and body >= cfg.displacement_body_atr*prior_atr
                and body/span >= cfg.displacement_body_share)


def _zone(kind: str, direction: int, low: float, high: float, at: pd.Timestamp,
          origin: pd.Timestamp, timeframe: str, cfg: StructureConfig,
          parent_id: str | None = None, evidence: dict | None = None) -> dict:
    identifier = f"{timeframe}:{kind}:{direction}:{origin.isoformat()}:{at.isoformat()}"
    return dict(id=identifier, kind=kind, direction=direction, low=float(low),
                high=float(high), midpoint=float((low+high)/2), known_at=at,
                origin_at=origin, timeframe=timeframe, parent_id=parent_id,
                evidence=evidence or {}, lifecycle="ACTIVE", invalidated_at=None,
                observation_gap_at=None, expires_at=at+pd.Timedelta(hours=cfg.max_zone_age_hours),
                gate_effect="OBSERVATION_ONLY", execution_allowed=False)


def _displaced(bar: pd.Series, prior_atr: float, direction: int,
               cfg: StructureConfig) -> bool:
    body, span = direction*(bar.close-bar.open), bar.high-bar.low
    return bool(np.isfinite(prior_atr) and prior_atr > 0 and span > 0
                and body >= cfg.displacement_body_atr*prior_atr
                and body/span >= cfg.displacement_body_share)


def detect_zones(bars: pd.DataFrame, minutes: int, cfg: StructureConfig) -> dict:
    """Forward replay. Pivot usable only after its right-hand bars close."""
    name = f"{minutes//60}h"
    result = dict(timeframe=name, quality="INSUFFICIENT_HISTORY", zones=[], events=[],
                  last_closed_at=_iso(bars.index[-1]) if len(bars) else None,
                  bars=len(bars), bias="UNKNOWN", missing_scheduled_bars=0, active_zones=0)
    if len(bars) < cfg.min_htf_bars:
        return result
    b = bars.iloc[-cfg.history_bars:].copy()
    expected = _expected_ends(b.index[0], b.index[-1], minutes, cfg)
    result["missing_scheduled_bars"] = int((~expected.isin(b.index)).sum())
    atr = _atr(b, cfg.atr_period).shift()
    complete = _span_checker(b.index, expected)
    stamps = list(b.index)
    opens = b.open.to_numpy(float)
    highs = b.high.to_numpy(float)
    lows = b.low.to_numpy(float)
    closes = b.close.to_numpy(float)
    atr_values = atr.to_numpy(float)
    zones, events, seeds = [], [], []
    levels = {1: None, -1: None}
    used_levels, used_origins = set(), set()
    sweeps = {1: None, -1: None}
    last_direction = 0
    for i in range(len(b)):
        stamp = stamps[i]
        if i and not complete(stamps[i-1], stamp):
            levels = {1: None, -1: None}
            seeds.clear()
            sweeps = {1: None, -1: None}
            last_direction = 0
            for z in zones:
                if z["lifecycle"] == "ACTIVE":
                    z.update(lifecycle="UNVERIFIED_AFTER_HTF_GAP", observation_gap_at=stamp)
        candidate = i-cfg.pivot_right
        if candidate >= cfg.pivot_left:
            first = candidate-cfg.pivot_left
            if complete(stamps[first], stamp):
                for direction, values, value in ((1, highs[first:i+1], highs[candidate]),
                                                 (-1, lows[first:i+1], lows[candidate])):
                    extreme = values.max() if direction == 1 else values.min()
                    if value == extreme and int(np.sum(values == value)) == 1:
                        levels[direction] = dict(value=float(value), pivot_at=stamps[candidate],
                                                 known_at=stamp)
        at = atr_values[i]
        if not np.isfinite(at) or at <= 0:
            continue
        buffer = cfg.structure_buffer_atr*at
        # Low sweep => bullish context; high sweep => bearish context.
        for direction in (1, -1):
            level = levels[-direction]
            if level is None:
                continue
            swept = (lows[i] < level["value"]-buffer and closes[i] >= level["value"]
                     if direction == 1 else
                     highs[i] > level["value"]+buffer and closes[i] <= level["value"])
            if swept:
                sweeps[direction] = dict(index=i, at=stamp, level=level["value"],
                                         reference_known_at=level["known_at"])
                events.append(dict(kind="SELLSIDE_SWEEP" if direction == 1 else "BUYSIDE_SWEEP",
                                   direction=direction, at=stamp, level=level["value"]))
        breaks = {}
        for direction in (1, -1):
            level = levels[direction]
            if level is None:
                continue
            key = (direction, level["pivot_at"])
            crossed = (closes[i] > level["value"]+buffer if direction == 1
                       else closes[i] < level["value"]-buffer)
            if crossed and key not in used_levels and _displaced_at(
                    opens[i], highs[i], lows[i], closes[i], at, direction, cfg):
                used_levels.add(key)
                event_kind = ("MSS_CANDIDATE" if last_direction and direction != last_direction
                              else "SWING_BREAK")
                breaks[direction] = dict(kind=event_kind, direction=direction, at=stamp,
                                         level=level["value"], reference_known_at=level["known_at"])
                events.append(breaks[direction])
                last_direction = direction
        created_fvg = {}
        if i >= 2 and complete(stamps[i-2], stamp):
            for direction, low, high in ((1, highs[i-2], lows[i]), (-1, highs[i], lows[i-2])):
                if (high-low >= cfg.min_fvg_atr*at
                        and _displaced_at(opens[i-1], highs[i-1], lows[i-1], closes[i-1],
                                          atr_values[i-1], direction, cfg)):
                    z = _zone("FVG", direction, low, high, stamp, stamps[i-1], name, cfg,
                              evidence={"three_closed_bars": True, "middle_displacement": True})
                    zones.append(z)
                    created_fvg[direction] = z
        # Lifecycle checked BEFORE any newly flipped zone can see this bar.
        for z in list(zones):
            if z["lifecycle"] != "ACTIVE" or stamp <= z["known_at"]:
                continue
            if stamp >= z["expires_at"]:
                z["lifecycle"] = "EXPIRED"
                continue
            opposite = -z["direction"]
            failed = (closes[i] < z["low"]-buffer if z["direction"] == 1
                      else closes[i] > z["high"]+buffer)
            if not failed:
                continue
            z.update(lifecycle="INVALIDATED", invalidated_at=stamp)
            if z["kind"] == "FVG":
                zones.append(_zone("INVERSION_FVG", opposite, z["low"], z["high"], stamp,
                                   z["origin_at"], name, cfg, parent_id=z["id"],
                                   evidence={"closed_htf_far_edge_break": True}))
            elif z["kind"] == "ORDER_BLOCK":
                sweep = sweeps[opposite]
                sweep_ok = (sweep is not None and i-sweep["index"] <= cfg.sweep_memory_bars
                            and z["known_at"] <= sweep["at"] < stamp)
                mss_ok = opposite in breaks
                displaced = _displaced_at(opens[i], highs[i], lows[i], closes[i], at,
                                          opposite, cfg)
                z["failure_evidence"] = dict(sweep_confirmed=bool(sweep_ok),
                                             displaced_structure_break=bool(mss_ok),
                                             displacement=bool(displaced))
                if (displaced and (sweep_ok or not cfg.require_breaker_sweep)
                        and (mss_ok or not cfg.require_breaker_mss)):
                    zones.append(_zone("BREAKER", opposite, z["low"], z["high"], stamp,
                                       z["origin_at"], name, cfg, parent_id=z["id"],
                                       evidence={**z["failure_evidence"], "failed_ob": True,
                                                 "sweep_at": _iso(sweep["at"]) if sweep_ok else None}))
        # Stage OB at the displaced structure break. Wait for FVG confirmation.
        for direction, event in breaks.items():
            for j in range(i-1, max(-1, i-cfg.ob_search_bars-1), -1):
                if direction*(closes[j]-opens[j]) >= 0:
                    continue
                if not complete(stamps[j], stamp):
                    break
                low, high = ((float(lows[j]), float(highs[j])) if cfg.zone_bounds == "WICK" else (
                    float(min(opens[j], closes[j])), float(max(opens[j], closes[j]))))
                departed = closes[i] > high+buffer if direction == 1 else closes[i] < low-buffer
                key = (direction, stamps[j])
                origin_sweep = bool(j > 0 and (
                    lows[j] < lows[j-1] if direction == 1
                    else highs[j] > highs[j-1]))
                if (high > low and departed and high-low <= cfg.max_ob_range_atr*at
                        and key not in used_origins
                        and (origin_sweep or not cfg.require_ob_origin_sweep)):
                    used_origins.add(key)
                    seeds.append(dict(direction=direction, low=low, high=high, index=i,
                                      origin_at=stamps[j], event=event, origin_sweep=origin_sweep))
                break
        pending = []
        for seed in seeds:
            # An FVG's middle candle must be the displacement/BOS candle.
            fvg = created_fvg.get(seed["direction"])
            evidence_ok = fvg is not None and fvg["origin_at"] == stamps[seed["index"]]
            if not cfg.require_ob_fvg or evidence_ok:
                zones.append(_zone("ORDER_BLOCK", seed["direction"], seed["low"], seed["high"],
                                   stamp, seed["origin_at"], name, cfg,
                                   evidence={"displaced_structure_break": True,
                                             "structure_break_at": _iso(seed["event"]["at"]),
                                             "fvg_confirmed": bool(evidence_ok),
                                             "origin_candle_sweep": seed["origin_sweep"],
                                             "boundary_mode": cfg.zone_bounds}))
            elif i-seed["index"] < 1:
                pending.append(seed)
        seeds = pending
        # Bounded replay output; chronological pruning never uses future distance.
        if len(zones) > cfg.max_zones_per_timeframe:
            zones = zones[-cfg.max_zones_per_timeframe:]
    last = stamps[-1]
    for z in zones:
        if z["lifecycle"] == "ACTIVE" and last >= z["expires_at"]:
            z["lifecycle"] = "EXPIRED"
    result.update(quality="OK_WITH_HISTORICAL_GAPS" if result["missing_scheduled_bars"] else "OK",
                  zones=zones, events=events[-40:],
                  bias="BULLISH" if last_direction == 1 else "BEARISH" if last_direction == -1
                  else "NO_CONFIRMED_SWING_BREAK")
    return result


def _candidate_ends(start: pd.Timestamp, end: pd.Timestamp, minutes: int,
                    cfg: StructureConfig) -> pd.DatetimeIndex:
    """Include scheduled short session-close buckets, never missing-data fragments."""
    from market_transition import _session_bounds
    values = list(_expected_ends(start, end, minutes, cfg))
    days = pd.date_range(start.date()-timedelta(days=1), end.date(), freq="D")
    for day in days:
        if day.weekday() not in (6, 0, 1, 2, 3):
            continue
        opened, closed = _session_bounds(day.date(), cfg)
        if ((closed-opened)/pd.Timedelta(minutes=1)) % minutes and start <= closed <= end:
            values.append(closed)
    return pd.DatetimeIndex(sorted(set(values)), tz=cfg.timezone)


def detect_breaker_candidates(bars: pd.DataFrame, minutes: int,
                              cfg: StructureConfig) -> list[dict]:
    """Separate weaker hypothesis: local-extreme takeout, reclaim, then failure.

    Bearish: last up candle -> later low takeout -> displaced close above
    its high -> still later displaced close below its low. Bullish is mirrored.
    No claim of confirmed swing sweep, FVG-qualified OB or strict breaker.
    Short scheduled origin candles are explicitly labelled; confirmation must
    be a full timeframe candle. Never uses a drawn/user-provided price boundary.
    """
    if not cfg.enable_breaker_candidates or len(bars) < cfg.min_htf_bars:
        return []
    b = bars.iloc[-cfg.history_bars:].copy()
    expected = _candidate_ends(b.index[0], b.index[-1], minutes, cfg)
    b = b.loc[b.index.isin(expected)]
    if len(b) < cfg.min_htf_bars:
        return []
    atr = _atr(b, cfg.atr_period).shift()
    complete = _span_checker(b.index, expected)
    stamps = list(b.index)
    opens = b.open.to_numpy(float)
    highs = b.high.to_numpy(float)
    lows = b.low.to_numpy(float)
    closes = b.close.to_numpy(float)
    scheduled = b.scheduled_minutes.to_numpy(float)
    starts = b["start"].tolist()
    atr_values = atr.to_numpy(float)
    seeds, zones = [], []
    for i in range(len(b)):
        stamp = stamps[i]
        if i and not complete(stamps[i-1], stamp):
            seeds.clear()
            for z in zones:
                if z["lifecycle"] == "ACTIVE":
                    z.update(lifecycle="UNVERIFIED_AFTER_HTF_GAP", observation_gap_at=stamp)
        at = atr_values[i]
        if not np.isfinite(at) or at <= 0:
            continue
        buffer = cfg.structure_buffer_atr*at
        for z in zones:
            if z["lifecycle"] != "ACTIVE" or stamp <= z["known_at"]:
                continue
            if stamp >= z["expires_at"]:
                z["lifecycle"] = "EXPIRED"
            elif int(scheduled[i]) == minutes and (
                    closes[i] > z["high"]+buffer if z["direction"] == -1
                    else closes[i] < z["low"]-buffer):
                z.update(lifecycle="INVALIDATED", invalidated_at=stamp)
        pending = []
        for seed in seeds:
            if i-seed["index"] > cfg.candidate_lifetime_bars:
                continue
            direction = seed["direction"]
            full = int(scheduled[i]) == minutes
            if seed["reclaimed_at"] is None:
                recovered = (closes[i] > seed["high"]+buffer if direction == -1
                             else closes[i] < seed["low"]-buffer)
                if stamp > seed["sweep_at"] and full and recovered and _displaced_at(
                        opens[i], highs[i], lows[i], closes[i], at, -direction, cfg):
                    seed["reclaimed_at"] = stamp
                pending.append(seed)
            else:
                failed = (closes[i] < seed["low"]-buffer if direction == -1
                          else closes[i] > seed["high"]+buffer)
                if stamp > seed["reclaimed_at"] and full and failed and _displaced_at(
                        opens[i], highs[i], lows[i], closes[i], at, direction, cfg):
                    zones.append(_zone(
                        "BREAKER_CANDIDATE", direction, seed["low"], seed["high"],
                        stamp, seed["origin_at"], f"{minutes//60}h", cfg,
                        evidence=dict(
                            basis="LOCAL_TAKEOUT_RECLAIM_FAILURE",
                            strict_breaker=False, qualified_ob=False,
                            confirmed_swing_sweep=False,
                            reference_level=seed["reference_level"],
                            reference_bars=cfg.candidate_reference_bars,
                            sweep_at=_iso(seed["sweep_at"]),
                            reclaimed_at=_iso(seed["reclaimed_at"]),
                            source_start=_iso(seed["source_start"]),
                            source_scheduled_minutes=seed["source_minutes"],
                            source_is_full_timeframe=seed["source_minutes"] == minutes,
                            boundary_mode=cfg.zone_bounds,
                            confirmation_full_timeframe=True)))
                else:
                    pending.append(seed)
        seeds = pending
        n = cfg.candidate_reference_bars
        if i >= n and complete(stamps[i-n], stamp):
            # Opposing source candle must immediately precede the takeout.
            for direction in (-1, 1):
                if direction*(closes[i-1]-opens[i-1]) >= 0:
                    continue
                reference = (float(lows[i-n:i].min()) if direction == -1
                             else float(highs[i-n:i].max()))
                taken = lows[i] < reference-buffer if direction == -1 else highs[i] > reference+buffer
                if not taken:
                    continue
                low, high = ((float(lows[i-1]), float(highs[i-1])) if cfg.zone_bounds == "WICK"
                             else (float(min(opens[i-1], closes[i-1])), float(max(opens[i-1], closes[i-1]))))
                if 0 < high-low <= cfg.max_ob_range_atr*at:
                    seeds.append(dict(direction=direction, low=low, high=high, index=i,
                                      origin_at=stamps[i-1], source_start=starts[i-1],
                                      source_minutes=int(scheduled[i-1]),
                                      sweep_at=stamp, reference_level=reference, reclaimed_at=None))
        zones = zones[-cfg.max_zones_per_timeframe:]
    return zones


def react_to_zone(zone: dict, five: pd.DataFrame, cutoff: pd.Timestamp,
                  cfg: StructureConfig, expected_five: pd.DatetimeIndex | None = None,
                  shifted_close: pd.Series | None = None) -> dict:
    """5m close evidence, never presumed high/low intrabar ordering."""
    z = dict(zone)
    if z["lifecycle"] == "ACTIVE" and cutoff >= z["expires_at"]:
        z["lifecycle"] = "EXPIRED"
    end = min(cutoff, z["expires_at"])
    if z["invalidated_at"] is not None:
        end = min(end, z["invalidated_at"])
    if z["observation_gap_at"] is not None:
        end = min(end, z["observation_gap_at"])
    # Sorted-index searchsorted slices are exact equivalents of the previous
    # boolean masks, without scanning the full 5m frame for every zone.
    idx = five.index
    if idx.is_monotonic_increasing:
        low = idx.searchsorted(z["known_at"], side="right")
        part = five.iloc[low:idx.searchsorted(end, side="left")]
    else:
        part = five.loc[(five.index > z["known_at"]) & (five.index < end)]
    # An HTF invalidation/expiry timestamp cannot simultaneously confirm retest.
    if end == cutoff and z["invalidated_at"] != cutoff and z["expires_at"] != cutoff:
        if idx.is_monotonic_increasing:
            part = five.iloc[low:idx.searchsorted(end, side="right")]
        else:
            part = five.loc[(five.index > z["known_at"]) & (five.index <= end)]
    z.update(first_touch_at=None, last_touch_at=None, ce_touched=False,
             wick_fill_fraction=0.0, confirmed_retest_at=None, failed_ltf_at=None,
             reclaimed_at=None,
             last_reaction_at=None, last_reaction="NO_RETEST",
             current_position="UNKNOWN", current_interaction=False,
             current_position_at=None, reaction_quality="NO_LATER_5M",
             reaction_fresh=False, reaction_evaluated_until=None,
             touch_count=0, visit_count=0, repeated_retest_warning=False,
             mitigation_status="UNTOUCHED")
    if expected_five is None:
        expected = _expected_ends(z["known_at"]+pd.Timedelta(minutes=5), end, 5, cfg)
    elif expected_five.is_monotonic_increasing:
        low = expected_five.searchsorted(z["known_at"], side="right")
        high = expected_five.searchsorted(end, side="right")
        expected = expected_five[low:high]
    else:
        expected = expected_five[
            (expected_five > z["known_at"]) & (expected_five <= end)]
    if z["invalidated_at"] == end or z["expires_at"] == end or z["observation_gap_at"] == end:
        expected = expected[expected < end]
    part = part.loc[part.index.isin(expected)]
    observed_index = part.index
    missing = expected[~expected.isin(observed_index)]
    if len(missing):
        # Stop at first missing scheduled observation. No later false confirmation.
        part = part.loc[part.index < missing[0]]
        z["reaction_quality"] = "OBSERVATION_GAP"
    elif len(expected):
        z["reaction_quality"] = "OK"
    if part.empty:
        if z["reaction_quality"] == "OBSERVATION_GAP":
            z["last_reaction"] = "UNVERIFIED_OBSERVATIONS"
        return z
    z["reaction_evaluated_until"] = part.index[-1]
    width, direction = z["high"]-z["low"], z["direction"]
    touches = (part.low <= z["high"]) & (part.high >= z["low"])
    touched = part.loc[touches]
    z["touch_count"] = int(touches.sum())
    z["visit_count"] = int((touches & ~touches.shift(1, fill_value=False)).sum())
    z["repeated_retest_warning"] = z["visit_count"] > 1
    if len(touched):
        z["first_touch_at"], z["last_touch_at"] = touched.index[0], touched.index[-1]
        z["ce_touched"] = bool(((touched.low <= z["midpoint"]) & (touched.high >= z["midpoint"])).any())
        fill = ((z["high"]-touched.low)/width if direction == 1
                else (touched.high-z["low"])/width)
        z["wick_fill_fraction"] = float(fill.clip(0, 1).max())
        z["mitigation_status"] = ("FULL_WICK_FILL" if z["wick_fill_fraction"] >= 1
                                  else "CE_TOUCHED" if z["ce_touched"] else "PARTIAL_FILL")
    at = part["prior_atr"]
    buffer = cfg.reaction_buffer_atr_5m*at
    usable = at.notna() & (at > 0)
    if not usable.all() and z["reaction_quality"] == "OK":
        z["reaction_quality"] = "PARTIAL_ATR_HISTORY"
    range_ok = (part.high-part.low)/at <= cfg.reaction_max_range_atr_5m
    held = (part.close > z["high"]+buffer if direction == 1
            else part.close < z["low"]-buffer)
    failed = (part.close < z["low"]-buffer if direction == 1
              else part.close > z["high"]+buffer)
    signed_body = direction*(part.close-part.open) > 0
    confirmed = usable & touches & held & range_ok & signed_body
    # A close through the zone is useful failure evidence even without range overlap.
    failed = usable & failed
    if z["first_touch_at"] is not None:
        failed &= part.index >= z["first_touch_at"]
    else:
        failed[:] = False
    good_times, bad_times = part.index[confirmed], part.index[failed]
    previous_close = (shifted_close if shifted_close is not None
                      else five.close.shift()).reindex(part.index)
    wrong_side = (previous_close < z["low"]-buffer if direction == 1
                  else previous_close > z["high"]+buffer)
    held_times = part.index[confirmed & ~wrong_side]
    reclaim_times = part.index[confirmed & wrong_side]
    if len(held_times):
        z["confirmed_retest_at"] = held_times[-1]
    if len(reclaim_times):
        z["reclaimed_at"] = reclaim_times[-1]
    if len(bad_times):
        z["failed_ltf_at"] = bad_times[-1]
    event_times = [(t, "LTF_RETEST_HELD") for t in held_times]
    event_times += [(t, "LTF_RECLAIM_FROM_WRONG_SIDE") for t in reclaim_times]
    event_times += [(t, "LTF_FAILED_RETEST") for t in bad_times]
    if event_times:
        z["last_reaction_at"], z["last_reaction"] = max(event_times, key=lambda x: x[0])
        z["reaction_fresh"] = cutoff-z["last_reaction_at"] <= pd.Timedelta(minutes=cfg.reaction_fresh_minutes)
    elif z["last_touch_at"] is not None:
        z["last_reaction_at"], z["last_reaction"] = z["last_touch_at"], "TOUCH_UNCONFIRMED"
        z["reaction_fresh"] = cutoff-z["last_reaction_at"] <= pd.Timedelta(minutes=cfg.reaction_fresh_minutes)
    latest = part.iloc[-1]
    recent = cutoff-part.index[-1] <= pd.Timedelta(minutes=cfg.max_ltf_gap_minutes)
    if recent and z["reaction_quality"] == "OK" and z["lifecycle"] == "ACTIVE" and cutoff < z["expires_at"]:
        z["current_position"] = ("ABOVE_ZONE" if latest.close > z["high"]
                                 else "BELOW_ZONE" if latest.close < z["low"] else "INSIDE_ZONE")
        z["current_interaction"] = bool(touches.iloc[-1])
        z["current_position_at"] = part.index[-1]
    if z["reaction_quality"] != "OK":
        z["reaction_fresh"] = False
    return z


def _serialize_zone(zone: dict) -> dict:
    result = dict(zone)
    for key in ("known_at", "origin_at", "expires_at", "invalidated_at", "observation_gap_at",
                "first_touch_at", "last_touch_at", "confirmed_retest_at", "failed_ltf_at",
                "last_reaction_at", "reaction_evaluated_until", "reclaimed_at"):
        result[key] = _iso(result.get(key))
    result["current_position_at"] = _iso(result.get("current_position_at"))
    return result


def evaluate_market_structure(
    df_1m: pd.DataFrame, *, config: StructureConfig = DEFAULT_STRUCTURE,
    data_context: dict[str, Any] | None = None,
    prepared_bars: dict[int, pd.DataFrame] | None = None,
    prepared_five: pd.DataFrame | None = None,
) -> dict:
    """Pure snapshot replay; optional preaggregated inputs from transition."""
    from market_transition import _prepare, _active_minutes, _bars, TransitionRiskConfig
    cfg, metadata = config, data_context or {}
    output = dict(version=VERSION, state="DISABLED", asof=None, data_mode=metadata.get("mode", "UNVERIFIED"),
                  source_quality="NOT_CHECKED", timeframes={}, execution_allowed=False,
                  gate_effect="OBSERVATION_ONLY", notes=[
                      "ICT-inspired, konfigurált OHLC-szabályok; nem teljes ICT-diszkrecionális modell.",
                      "Zónák csak a megerősítésük után használhatók. Swing pivothoz jobb oldali zárások kellenek.",
                      "Az 5m reakció nem 4h gyertyazárás és nem automatikus trade-engedély.",
                      "Nincs intézményi order-flow vagy garantált reakció bizonyíték.",
                  ])
    if not cfg.enabled:
        return output
    if not isinstance(df_1m.index, pd.DatetimeIndex) or df_1m.index.tz is None:
        raise ValueError("Timezone-aware minute-start DatetimeIndex required.")
    agg_cfg = TransitionRiskConfig(timezone=cfg.timezone, session_open_hour=cfg.session_open_hour,
                                   session_close_hour=cfg.session_close_hour)
    df = _prepare(df_1m, agg_cfg)
    if df.empty:
        raise ValueError("Empty structure input.")
    cutoff = df.index[-1]+pd.Timedelta(minutes=1) if metadata.get("last_bar_is_complete") else df.index[-1]
    df = _active_minutes(df.loc[df.index < cutoff], agg_cfg)
    if df.empty:
        raise ValueError("No completed active minute.")
    output.update(asof=_iso(cutoff), state="NO_DETECTED_STRUCTURE")
    output.update(latest_price=float(df.close.iloc[-1]), latest_price_at=_iso(df.index[-1]+pd.Timedelta(minutes=1)))
    observed = metadata.get("observed_at", metadata.get("as_of"))
    if observed is not None:
        stamp = pd.Timestamp(observed)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError("Timezone-aware observation timestamp required.")
        age = (stamp-(df.index[-1]+pd.Timedelta(minutes=1))).total_seconds()/60
        output["source_age_minutes"] = age
        output["source_quality"] = ("FUTURE_SOURCE" if age < 0 else "STALE_SOURCE"
                                    if age > cfg.source_max_age_minutes else "OK_DELAY_UNVERIFIED")
    five = (prepared_five.copy() if prepared_five is not None else _bars(df, 5, cutoff, agg_cfg))
    five = five.loc[five.index <= cutoff].copy()
    five["prior_atr"] = _atr(five, cfg.atr_period).shift()
    expected_five = _expected_ends(df.index[0], cutoff, 5, cfg)
    # One full-frame shift instead of one per zone; values are identical.
    shifted_close = five.close.shift()
    for minutes in cfg.timeframes_minutes:
        raw = (prepared_bars[minutes].copy() if prepared_bars is not None and minutes in prepared_bars
               else _bars(df, minutes, cutoff, agg_cfg))
        # Never call a truncated final session bucket a full HTF candle.
        all_scheduled = raw.loc[raw.index <= cutoff].copy()
        raw = raw.loc[(raw.index <= cutoff) & (raw.scheduled_minutes == minutes)]
        if len(raw):
            scheduled = _expected_ends(raw.index[0], raw.index[-1], minutes, cfg)
            raw = raw.loc[raw.index.isin(scheduled)]
        tf = detect_zones(raw, minutes, cfg)
        tf["events"] = [{**e, "at": _iso(e["at"]),
                         **({"reference_known_at": _iso(e["reference_known_at"])}
                            if "reference_known_at" in e else {})} for e in tf["events"]]
        tf["zones"] = [_serialize_zone(react_to_zone(z, five, cutoff, cfg, expected_five,
                                                     shifted_close))
                       for z in tf["zones"]]
        tf["breaker_candidates"] = [
            _serialize_zone(react_to_zone(z, five, cutoff, cfg, expected_five,
                                         shifted_close))
            for z in detect_breaker_candidates(all_scheduled, minutes, cfg)]
        price = float(df.close.iloc[-1])
        for z in tf["zones"]+tf["breaker_candidates"]:
            z["distance_usd"] = (z["low"]-price if price < z["low"]
                                 else price-z["high"] if price > z["high"] else 0.0)
        tf["zones"].sort(key=lambda z: (z["lifecycle"] != "ACTIVE", z["distance_usd"],
                                       -pd.Timestamp(z["known_at"]).timestamp()))
        tf["active_zones"] = sum(z["lifecycle"] == "ACTIVE" and cutoff < pd.Timestamp(z["expires_at"])
                                 for z in tf["zones"])
        tf["current_direction_conflict"] = len({
            z["direction"] for z in tf["zones"]
            if z["lifecycle"] == "ACTIVE" and z["current_position"] == "INSIDE_ZONE"}) > 1
        output["timeframes"][tf["timeframe"]] = tf
    if all(t["quality"] == "INSUFFICIENT_HISTORY" for t in output["timeframes"].values()):
        output["state"] = "INSUFFICIENT_HISTORY"
    elif any(t["active_zones"] for t in output["timeframes"].values()):
        output["state"] = "ACTIVE_ZONES"
    if output["source_quality"] in ("FUTURE_SOURCE", "STALE_SOURCE"):
        output["state"] = "DATA_UNAVAILABLE"
        for tf in output["timeframes"].values():
            for z in tf["zones"]+tf.get("breaker_candidates", []):
                z["current_interaction"], z["reaction_fresh"] = False, False
                z["current_position"] = "UNVERIFIED_SOURCE_AGE"
                z["current_position_at"] = None
    output["config"] = {k: v for k, v in vars(cfg).items()}
    return output


def format_market_structure(result: dict, *, instrument: str = "GC") -> list[str]:
    lines = ["", "="*100, f"{instrument} ICT ÁRSTRUKTÚRA: ORDER BLOCK / BREAKER / FVG",
             "="*100, f"Elemzési végpont: {result.get('asof')} | {result['state']} | "
             f"forrás: {result['source_quality']}",
             "Automatikus, szabályalapú zónák. VWAP-tól független árterületek; OBSERVATION_ONLY."]
    labels = {"FVG": "FVG", "ORDER_BLOCK": "ORDER BLOCK", "BREAKER": "BREAKER-JELÖLT",
              "INVERSION_FVG": "INVERSION FVG"}
    reactions = {"NO_RETEST": "MÉG NINCS KÉSŐBBI VISSZATESZT",
                 "LTF_RETEST_HELD": "5m REAKCIÓ A ZÓNA IRÁNYÁBA MEGERŐSÍTVE",
                 "LTF_FAILED_RETEST": "5m SIKERTELEN VISSZATESZT / ELLENIRÁNYÚ ZÁRÁS",
                 "LTF_RECLAIM_FROM_WRONG_SIDE": "5m VISSZAVÉTEL A KORÁBBI ROSSZ OLDALRÓL; NEM ELSŐ RETEST",
                 "TOUCH_UNCONFIRMED": "ÉRINTÉS, DE NINCS MEGERŐSÍTETT REAKCIÓ",
                 "UNVERIFIED_OBSERVATIONS": "HIÁNYOS MEGFIGYELÉS, REAKCIÓ NEM IGAZOLT"}
    for name, tf in result.get("timeframes", {}).items():
        lines += ["", f"{name.upper()}: {tf['quality']} | lezárt teljes HTF barok: {tf['bars']} | "
                  f"struktúrairány: {tf['bias']} | aktív zónák: {tf['active_zones']}",
                  f"Utolsó teljes HTF zárás: {tf['last_closed_at'] or 'N/A'}"]
        if tf.get("current_direction_conflict"):
            lines.append("FIGYELEM: a jelenlegi árnál egymásnak ellentmondó aktív zónák fednek át.")
        candidates = tf.get("breaker_candidates", [])
        lines.append(f"Szigorú aktív breaker: {sum(z['kind'] == 'BREAKER' and z['lifecycle'] == 'ACTIVE' for z in tf['zones'])} | "
                     f"külön gyengébb breaker-jelölt: {sum(z['lifecycle'] == 'ACTIVE' for z in candidates)}")
        for z in sorted(candidates, key=lambda z: (z["lifecycle"] != "ACTIVE", z["distance_usd"]))[:result["config"]["display_zones_per_timeframe"]]:
            evidence = z["evidence"]
            source = evidence["source_scheduled_minutes"]
            lines += [
                f"- GYENGÉBB {'BEARISH' if z['direction'] == -1 else 'BULLISH'} BREAKER-JELÖLT: "
                f"{z['low']:.2f} - {z['high']:.2f} | {z['lifecycle']}",
                f"  Eredet: {evidence['source_start']} | {source} perces session-bucket; "
                f"{'TELJES HTF' if evidence['source_is_full_timeframe'] else 'RÖVIDÍTETT, NEM TELJES HTF GYERTYA'}",
                f"  Lokális szélsőérték kivétele: {evidence['sweep_at']} | visszavétel: {evidence['reclaimed_at']}",
                f"  Későbbi teljes HTF áttörés; ismert ettől: {z['known_at']}",
                f"  5m: {z['current_position']} | {z['current_position_at'] or 'N/A'} | "
                f"reakció: {z['last_reaction']} | {z['last_reaction_at'] or 'Nincs'}",
                "  NEM kvalifikált failed-OB breaker; nincs igazolt swing-sweep/FVG-OB. OBSERVATION_ONLY."]
        if not tf["zones"]:
            lines.append("Nincs e szabályokkal felismert zóna; ez nem bizonyítja, hogy a charton nincs ICT-zóna.")
        limit = result["config"]["display_zones_per_timeframe"]
        # Ensure a nearby breaker is visible even if FVGs fill the normal display slots.
        shown = tf["zones"][:limit]
        breaker = next((z for z in tf["zones"] if z["kind"] == "BREAKER" and z["lifecycle"] == "ACTIVE"), None)
        if breaker is not None and breaker not in shown:
            shown = shown[:-1]+[breaker]
        for z in shown:
            direction = "BULLISH / TÁMASZ-JELÖLT" if z["direction"] == 1 else "BEARISH / ELLENÁLLÁS-JELÖLT"
            lines += [
                f"- {labels[z['kind']]} | {direction} | {z['lifecycle']}",
                f"  Határok: {z['low']:.2f} - {z['high']:.2f} | CE/50%: {z['midpoint']:.2f} | "
                f"távolság: {z['distance_usd']:.2f} USD",
                f"  Eredeti gyertya: {z['origin_at']} | Először ismert: {z['known_at']}",
                f"  Utolsó értékelt 5m helyzet: {z['current_position']} | {z['current_position_at'] or 'N/A'} | érinti: "
                f"{'IGEN' if z['current_interaction'] else 'NEM / NEM IGAZOLT'}",
                f"  Reakció: {reactions[z['last_reaction']]} | időpont: {z['last_reaction_at'] or 'Nincs'}",
                f"  Utolsó esemény friss: {'IGEN' if z['reaction_fresh'] else 'NEM'} | "
                f"reakcióadat: {z['reaction_quality']} | értékelve eddig: {z['reaction_evaluated_until'] or 'N/A'}",
                f"  CE érintve: {'IGEN' if z['ce_touched'] else 'NEM'} | wick-lefedés: "
                f"{z['wick_fill_fraction']:.0%} (nem HTF close-invalidation)",
                f"  Érintő 5m barok: {z['touch_count']} | külön visszatérések: {z['visit_count']} | "
                f"állapot: {z['mitigation_status']}",
                f"  HTF invalidáció: {z['invalidated_at'] or 'Nincs igazolt'} | "
                f"megfigyelési rés: {z['observation_gap_at'] or 'Nincs'} | lejárat: {z['expires_at']}",
            ]
            if z["repeated_retest_warning"]:
                lines.append("  FIGYELEM: ismételt visszateszt; nem friss, első visszatérés.")
    lines += [f"- {n}" for n in result["notes"]]
    lines.append("Új LONG/SHORT engedély ebből: NINCS. Az eredeti whipsaw/anti-chasing/VWAP vétók megmaradnak.")
    return lines
