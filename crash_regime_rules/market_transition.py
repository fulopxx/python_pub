"""Closed-bar 1h/2h/4h session-VWAP transitions and 1m whipsaw risk.

Version 0.2.0: ICT-inspired OHLC structure plus legacy VWAP/whipsaw.
Does not predict the next candle. Legacy gating rules are unchanged.
Minute timestamps denote bar STARTS. No downloads, orders or disk state.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from market_structure import (
    DEFAULT_STRUCTURE, StructureConfig, evaluate_market_structure, format_market_structure,
)

OHLCV = ["open", "high", "low", "close", "volume"]
AGG = dict(open="first", high="max", low="min", close="last", volume="sum")
VERSION = "0.2.0"


@dataclass(frozen=True)
class TransitionRiskConfig:
    enabled: bool = True
    timezone: str = "America/New_York"
    session_open_hour: int = 18
    session_close_hour: int = 17
    # Buckets begin at session open: e.g. 4h 18/22/02/06/10/14 ET.
    # DST uses elapsed hours from the known session start.
    min_bucket_coverage: float = 1.0
    reclaim_buffer_atr_5m: float = 0.25
    retest_tolerance_atr_5m: float = 0.25
    max_retest_range_atr_5m: float = 3.0
    retest_wait_timeframe_bars: int = 2
    whipsaw_window_minutes: int = 30
    min_1m_coverage: float = 0.90
    max_1m_gap_minutes: int = 2
    wick_side_fraction: float = 0.25
    two_sided_wick_share: float = 0.40
    small_body_fraction: float = 0.25
    small_body_share: float = 0.60
    direction_flip_rate: float = 0.60
    max_efficiency_ratio: float = 0.25
    large_range_atr_1m: float = 1.80
    large_range_share: float = 0.25
    extreme_range_atr_1m: float = 3.0
    vwap_cross_buffer_atr_1m: float = 0.10
    vwap_cross_count: int = 4
    block_unknown_whipsaw: bool = True
    block_elevated_whipsaw: bool = False
    block_opposite_confirmed: bool = True
    block_against_pending_turn: bool = True
    block_pending_turn_entry: bool = True
    structure: StructureConfig = DEFAULT_STRUCTURE

    def __post_init__(self) -> None:
        if not 0 < self.min_bucket_coverage <= 1 or not 0 < self.min_1m_coverage <= 1:
            raise ValueError("Coverage must be in (0, 1].")
        if self.whipsaw_window_minutes < 15 or self.retest_wait_timeframe_bars < 1:
            raise ValueError("Insufficient whipsaw window or retest lifetime.")
        if not (0 <= self.session_close_hour < self.session_open_hour < 24):
            raise ValueError("This implementation requires close before open, e.g. 17/18 ET.")


GC_TRANSITION = TransitionRiskConfig()


def _prepare(df: pd.DataFrame, cfg: TransitionRiskConfig) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex) or not set(OHLCV).issubset(df):
        raise ValueError("DatetimeIndex and open/high/low/close/volume are required.")
    out = df[OHLCV].apply(pd.to_numeric, errors="raise").copy()
    out.index = pd.to_datetime(out.index, utc=True).tz_convert(cfg.timezone)
    out = out.sort_index()
    utc = out.index.tz_convert("UTC")
    if out.index.has_duplicates or utc.hasnans or (utc != utc.floor("min")).any():
        raise ValueError("Unique, valid minute-start timestamps are required.")
    if not np.isfinite(out.to_numpy(float)).all():
        raise ValueError("Non-finite OHLCV.")
    if ((out[["open", "high", "low", "close"]] <= 0).any().any()
            or (out.volume < 0).any()
            or (out.high < out[["open", "close", "low"]].max(axis=1)).any()
            or (out.low > out[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("Invalid OHLCV.")
    return out


def _session_keys(index: pd.DatetimeIndex, cfg: TransitionRiskConfig) -> pd.Index:
    local = index.tz_localize(None)
    return pd.Index((local - pd.Timedelta(hours=cfg.session_open_hour)).date)


def _session_bounds(day: Any, cfg: TransitionRiskConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = (pd.Timestamp(day) + pd.Timedelta(hours=cfg.session_open_hour)).tz_localize(cfg.timezone)
    end = (pd.Timestamp(day + timedelta(days=1))
           + pd.Timedelta(hours=cfg.session_close_hour)).tz_localize(cfg.timezone)
    return start, end


def _active_minutes(df: pd.DataFrame, cfg: TransitionRiskConfig) -> pd.DataFrame:
    # Exclude the declared maintenance interval. Not an exchange-holiday calendar.
    hour = df.index.hour
    return df.loc[(hour >= cfg.session_open_hour) | (hour < cfg.session_close_hour)].copy()


def _atr(df: pd.DataFrame) -> pd.Series:
    prev = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - prev).abs(),
                    (df.low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()


def _bars(df: pd.DataFrame, minutes: int, cutoff: pd.Timestamp,
          cfg: TransitionRiskConfig) -> pd.DataFrame:
    """Full scheduled buckets only; the last session bucket may be shorter."""
    frames = []
    for day, part in df.groupby(_session_keys(df.index, cfg), sort=True):
        start, session_end = _session_bounds(day, cfg)
        src = part.tz_convert("UTC")
        rule = f"{minutes}min"
        origin = start.tz_convert("UTC")
        grouped = src.resample(rule, origin=origin, label="left", closed="left")
        agg = grouped.agg(AGG)
        count = grouped.close.count()
        first = pd.Series(src.index, index=src.index).resample(rule, origin=origin).first()
        last = pd.Series(src.index, index=src.index).resample(rule, origin=origin).last()
        ends = agg.index + pd.Timedelta(minutes=minutes)
        ends = pd.DatetimeIndex([
            min(t, session_end.tz_convert("UTC")) for t in ends
        ])
        # pandas may store datetime arrays in us instead of ns.
        expected = np.asarray((ends - agg.index) / pd.Timedelta(minutes=1))
        good = (
            (expected > 0)
            & (count.to_numpy() >= np.ceil(expected * cfg.min_bucket_coverage))
            & (first.to_numpy() == agg.index.to_numpy())
            & (last.to_numpy() == (ends - pd.Timedelta(minutes=1)).to_numpy())
            & (ends <= cutoff.tz_convert("UTC"))
        )
        agg = agg.loc[good].copy()
        if agg.empty:
            continue
        agg["start"] = agg.index.tz_convert(cfg.timezone)
        agg["session"] = day
        agg["scheduled_minutes"] = expected[good]
        agg.index = ends[good].tz_convert(cfg.timezone)
        frames.append(agg)
    if not frames:
        return pd.DataFrame(columns=OHLCV + ["start", "session", "scheduled_minutes"],
                            index=pd.DatetimeIndex([], tz=cfg.timezone))
    return pd.concat(frames).sort_index()


def _row(label: str, value: Any, threshold: str, passed: Any,
         available: bool = True) -> dict[str, Any]:
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        available = False
    return dict(label=label, value=value, threshold=threshold,
                passed=bool(passed) if available else None)


def _sign(distance: float, buffer: float) -> int:
    if not np.isfinite(distance):
        return 0
    return 1 if distance >= buffer else -1 if distance <= -buffer else 0


def _timeframe(
    bars: pd.DataFrame, five: pd.DataFrame, vwap: pd.Series, atr5: pd.Series,
    current: pd.DataFrame, cutoff: pd.Timestamp, minutes: int,
    cfg: TransitionRiskConfig,
) -> dict[str, Any]:
    name = f"{minutes // 60}h"
    result = dict(timeframe=name, state="UNKNOWN", confirmed_direction=0,
                  pending_direction=0, provisional_direction=0,
                  last_closed_at=None, current_context="UNKNOWN",
                  cross_at=None, retest_at=None, failed_at=None,
                  close_vs_vwap_atr_5m=np.nan, forming_state="NONE",
                  checks=[], notes=[])
    if bars.empty:
        result["notes"].append("Nincs teljes, lezárt magasabb idősíkú gyertya.")
        return result
    b = bars.copy()
    minute_times = b.index - pd.Timedelta(minutes=1)
    b["vwap"] = vwap.reindex(minute_times).to_numpy()
    b["atr5"] = atr5.reindex(b.index, method="ffill").to_numpy() if len(atr5) else np.nan
    b["distance"] = (b.close - b.vwap) / b.atr5.replace(0, np.nan)
    latest = b.iloc[-1]
    result.update(last_closed_at=b.index[-1],
                  close_vs_vwap_atr_5m=float(latest.distance))
    side = _sign(latest.distance, cfg.reclaim_buffer_atr_5m)
    result["current_context"] = (
        "ABOVE_SESSION_VWAP" if side == 1
        else "BELOW_SESSION_VWAP" if side == -1 else "NEAR_OR_UNKNOWN_VWAP"
    )
    # Only the current session can arm a reclaim, never a VWAP reset across sessions.
    today = _session_keys(current.index[-1:], cfg)[0]
    session_bars = b.loc[b.session == today]
    last_non_neutral = 0
    previous_known_at = None
    cross_at, direction = None, 0
    for ts, bar in session_bars.iterrows():
        sig = _sign(bar.distance, cfg.reclaim_buffer_atr_5m)
        if sig and last_non_neutral and sig != last_non_neutral:
            # Do not bridge large holes in higher-timeframe input.
            if ts - previous_known_at <= pd.Timedelta(minutes=2 * minutes):
                cross_at, direction = ts, sig
        if sig:
            last_non_neutral, previous_known_at = sig, ts
    last5 = five.index[-1] if len(five) else None
    forming = current.index[-1] + pd.Timedelta(minutes=1) > b.index[-1]
    current_atr = float(atr5.iloc[-1]) if len(atr5) else np.nan
    current_vwap = float(vwap.iloc[-1])
    forming_distance = ((current.close.iloc[-1] - current_vwap) / current_atr
                        if np.isfinite(current_atr) and current_atr > 0 else np.nan)
    forming_side = _sign(forming_distance, cfg.reclaim_buffer_atr_5m)
    if forming and latest.session == today and side and forming_side == -side:
        result["provisional_direction"] = forming_side
        result["forming_state"] = (
            "BULLISH_RECLAIM_IN_PROGRESS" if forming_side == 1
            else "BEARISH_REJECTION_IN_PROGRESS"
        )
    ready = (
        len(session_bars) >= 2 and np.isfinite(latest.distance)
        and latest.session == today
        and cutoff - b.index[-1] <= pd.Timedelta(minutes=minutes + 5)
    )
    state, retest_at, failed_at = "NO_TURN", None, None
    expires, evaluated_until, window_complete = None, None, False
    if not ready:
        state = "UNKNOWN"
    elif cross_at is not None:
        bull = direction == 1
        state = "BULLISH_CROSS_PENDING" if bull else "BEARISH_CROSS_PENDING"
        expires = cross_at + pd.Timedelta(
            minutes=minutes * cfg.retest_wait_timeframe_bars
        )
        following = five.loc[
            (five.index > cross_at) & (five.index <= min(cutoff, expires))
        ]
        evaluated_until = min(cutoff, expires)
        expected_count = int((evaluated_until - cross_at) / pd.Timedelta(minutes=5))
        window_complete = len(following) == expected_count and expected_count > 0
        for ts, bar in following.iterrows():
            minute = ts - pd.Timedelta(minutes=1)
            vw = float(vwap.get(minute, np.nan))
            at = float(atr5.get(ts, np.nan))
            if not np.isfinite(vw) or not np.isfinite(at) or at <= 0:
                window_complete = False
                continue
            if _session_keys(pd.DatetimeIndex([minute]), cfg)[0] != today:
                continue
            dist = (bar.close - vw) / at
            invalidated = (dist <= -cfg.reclaim_buffer_atr_5m if bull
                           else dist >= cfg.reclaim_buffer_atr_5m)
            if invalidated:
                failed_at = ts
                state = "FAILED_BULLISH_RECLAIM" if bull else "FAILED_BEARISH_REJECTION"
                break
            touch = (bar.low <= vw + cfg.retest_tolerance_atr_5m * at
                     and bar.high >= vw - cfg.retest_tolerance_atr_5m * at)
            held = (dist >= cfg.reclaim_buffer_atr_5m if bull
                    else dist <= -cfg.reclaim_buffer_atr_5m)
            range_ok = (bar.high - bar.low) / at <= cfg.max_retest_range_atr_5m
            if touch and held and range_ok and retest_at is None:
                retest_at = ts
                state = "BULLISH_RETEST_CONFIRMED" if bull else "BEARISH_RETEST_CONFIRMED"
        if cutoff >= expires and failed_at is None:
            state = "TURN_EXPIRED"
        if state.endswith("RETEST_CONFIRMED"):
            result["confirmed_direction"] = direction
        elif state.endswith("CROSS_PENDING"):
            result["pending_direction"] = direction
    result.update(state=state, cross_at=cross_at, retest_at=retest_at, failed_at=failed_at,
                  expires_at=expires, evaluated_until=evaluated_until,
                  observation_window_complete=window_complete,
                  active_turn=state.endswith(("CROSS_PENDING", "RETEST_CONFIRMED")))
    result["checks"] = [
        _row("Elég lezárt gyertya az aktuális sessionben", len(session_bars),
             ">= 2; reseten át nem hasonlítunk VWAP-ot", len(session_bars) >= 2),
        _row("Lezárt close távolsága a session VWAP-tól", latest.distance,
             f"irányváltási puffer +/- {cfg.reclaim_buffer_atr_5m:g} ATR(5m)",
             abs(latest.distance) >= cfg.reclaim_buffer_atr_5m),
        _row("Történeti bearish -> bullish VWAP oldalváltás", direction == 1,
             "Korábbi lezárt BELOW, későbbi lezárt ABOVE, azonos session",
             direction == 1, available=ready),
        _row("Történeti bullish -> bearish VWAP oldalváltás", direction == -1,
             "Korábbi lezárt ABOVE, későbbi lezárt BELOW, azonos session",
             direction == -1, available=ready),
        _row("Külön későbbi lezárt 5m visszateszt", retest_at is not None,
             "VWAP-zóna érintése + zárás az új oldalon, nem túl nagy range",
             retest_at is not None, available=cross_at is not None
             and (window_complete or retest_at is not None)),
        _row("Értékelt aktív ablakban nincs ellenoldali 5m zárás",
             failed_at is None, "Csak a feltüntetett értékelési végpontig, nem mostanáig",
             failed_at is None, available=cross_at is not None
             and (window_complete or failed_at is not None)),
        _row("Forduló most is aktív", result["active_turn"],
             "Nem lejárt, nem érvénytelenített; önmagában nem entry",
             result["active_turn"], available=cross_at is not None),
    ]
    if result["provisional_direction"]:
        result["notes"].append("Folyamatban levő HTF oldalváltás: NEM lezárt megerősítés.")
    if not ready:
        result["notes"].append("A hiányzó/friss session-adat nem negatív irányjelzés.")
    if state == "TURN_EXPIRED":
        result["notes"].append("A történeti cross lejárt; az utána következő időszak nem ennek az epizódnak a vizsgálata.")
    return result


def _whipsaw(df: pd.DataFrame, vwap: pd.Series, atr1: pd.Series,
             cutoff: pd.Timestamp, cfg: TransitionRiskConfig) -> dict[str, Any]:
    result = dict(state="UNKNOWN", checks=[], metrics={}, asof=None,
                  high_risk=False, notes=[])
    if df.empty:
        return result
    latest_session = _session_keys(df.index[-1:], cfg)[0]
    lower = cutoff - pd.Timedelta(minutes=cfg.whipsaw_window_minutes)
    recent = df.loc[
        (df.index >= lower) & (_session_keys(df.index, cfg) == latest_session)
    ]
    if recent.empty:
        return result
    gap = recent.index.to_series().diff().dt.total_seconds().div(60).max()
    ready = (
        len(recent) >= np.ceil(cfg.whipsaw_window_minutes * cfg.min_1m_coverage)
        and (pd.isna(gap) or gap <= cfg.max_1m_gap_minutes)
        and cutoff - recent.index[-1] <= pd.Timedelta(minutes=2)
    )
    result["asof"] = recent.index[-1] + pd.Timedelta(minutes=1)
    ranges = recent.high - recent.low
    positive_range = ranges > 0
    if positive_range.sum() < np.ceil(len(recent) * cfg.min_1m_coverage):
        ready = False
    denominator = ranges.replace(0, np.nan)
    upper = (recent.high - recent[["open", "close"]].max(axis=1)) / denominator
    lower_wick = (recent[["open", "close"]].min(axis=1) - recent.low) / denominator
    body = (recent.close - recent.open).abs() / denominator
    two_share = float(((upper >= cfg.wick_side_fraction)
                       & (lower_wick >= cfg.wick_side_fraction)).mean())
    body_share = float((body <= cfg.small_body_fraction).mean())
    moves = recent.close.diff().dropna()
    signs = np.sign(moves.to_numpy())
    signs = signs[signs != 0]
    flip = float((signs[1:] != signs[:-1]).mean()) if len(signs) >= 2 else 0.0
    path = float(moves.abs().sum())
    efficiency = float(abs(recent.close.iloc[-1] - recent.close.iloc[0]) / path) if path > 0 else 0.0
    prior_atr = atr1.shift(1).reindex(recent.index)
    ratios = ranges / prior_atr.replace(0, np.nan)
    if ratios.notna().sum() < np.ceil(len(recent) * cfg.min_1m_coverage):
        ready = False
    large_share = float((ratios >= cfg.large_range_atr_1m).mean())
    max_ratio = float(ratios.max())
    vw = vwap.reindex(recent.index)
    cross_count, previous_side = 0, 0
    for ts in recent.index:
        at = float(prior_atr.get(ts, np.nan))
        if not np.isfinite(at) or at <= 0 or not np.isfinite(vw.loc[ts]):
            continue
        sig = _sign((recent.close.loc[ts] - vw.loc[ts]) / at,
                    cfg.vwap_cross_buffer_atr_1m)
        if sig and previous_side and sig != previous_side:
            cross_count += 1
        if sig:
            previous_side = sig
    if vw.notna().sum() < np.ceil(len(recent) * cfg.min_1m_coverage):
        ready = False
    wick_family = two_share >= cfg.two_sided_wick_share or body_share >= cfg.small_body_share
    chop_family = (efficiency <= cfg.max_efficiency_ratio
                   and (flip >= cfg.direction_flip_rate
                        or two_share >= cfg.small_body_share))
    range_family = large_share >= cfg.large_range_share or max_ratio >= cfg.extreme_range_atr_1m
    cross_family = cross_count >= cfg.vwap_cross_count
    high = wick_family and range_family and (chop_family or cross_family)
    elevated = sum((wick_family, chop_family, range_family, cross_family)) >= 2
    state = "UNKNOWN" if not ready else "HIGH" if high else "ELEVATED" if elevated else "LOW"
    result.update(
        state=state, high_risk=state == "HIGH",
        metrics=dict(bars=len(recent), two_sided_wick_share=two_share,
                     small_body_share=body_share, direction_flip_rate=flip,
                     efficiency_ratio=efficiency, large_range_share=large_share,
                     max_range_atr_1m=max_ratio, vwap_cross_count=cross_count),
    )
    result["checks"] = [
        _row("Perces adatlefedettség", len(recent),
             f">= {cfg.min_1m_coverage:.0%} az utolsó {cfg.whipsaw_window_minutes} percből",
             ready),
        _row("Kétoldali kanócos gyertyák aránya", two_share,
             f">= {cfg.two_sided_wick_share:.0%}; mindkét kanóc >= {cfg.wick_side_fraction:.0%}",
             two_share >= cfg.two_sided_wick_share, available=ready),
        _row("Kis testű gyertyák aránya", body_share,
             f">= {cfg.small_body_share:.0%}; test <= {cfg.small_body_fraction:.0%}",
             body_share >= cfg.small_body_share, available=ready),
        _row("Close-irányváltások aránya", flip,
             f">= {cfg.direction_flip_rate:.0%}", flip >= cfg.direction_flip_rate,
             available=ready),
        _row("Nettó mozgás / teljes close-út", efficiency,
             f"<= {cfg.max_efficiency_ratio:g}", efficiency <= cfg.max_efficiency_ratio,
             available=ready),
        _row("Nagy range gyertyák aránya", large_share,
             f">= {cfg.large_range_share:.0%}; range >= {cfg.large_range_atr_1m:g} előző ATR1m",
             large_share >= cfg.large_range_share, available=ready),
        _row("Extrém egyszeri range / előző ATR1m", max_ratio,
             f">= {cfg.extreme_range_atr_1m:g}", max_ratio >= cfg.extreme_range_atr_1m,
             available=ready),
        _row("Pufferelt VWAP oldalváltások száma", cross_count,
             f">= {cfg.vwap_cross_count}; puffer {cfg.vwap_cross_buffer_atr_1m:g} ATR1m",
             cross_family, available=ready),
    ]
    if not ready:
        result["notes"].append("Kevés/hiányos/friss session-adat: nem minősítjük biztonságosnak.")
    result["notes"].append(
        "Mért környezeti kockázat, nem a következő 1m kanóc előrejelzése. "
        "OHLC-ből a gyertyán belüli high/low sorrend nem ismert."
    )
    return result


def evaluate_transition_risk(
    df_1m: pd.DataFrame, *, config: TransitionRiskConfig = GC_TRANSITION,
    data_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg, metadata = config, data_context or {}
    result = dict(version=VERSION, timeframes={}, whipsaw={"state": "DISABLED"},
                  block_new_longs=False, block_new_shorts=False,
                  long_block_reasons=[], short_block_reasons=[],
                  confirmation_conflict=False, asof=None,
                  notes=["Kutatási/paper szabályok; a küszöbök még nem auditáltak."])
    if not cfg.enabled:
        result["notes"].append("Transition/whipsaw réteg kikapcsolva.")
        return result
    df = _prepare(df_1m, cfg)
    if df.empty:
        raise ValueError("Empty transition input.")
    latest = df.index[-1]
    cutoff = latest + pd.Timedelta(minutes=1) if metadata.get("last_bar_is_complete", False) else latest
    df = _active_minutes(df.loc[df.index < cutoff], cfg)
    if df.empty:
        raise ValueError("No known completed active-session minute.")
    labels = _session_keys(df.index, cfg)
    tp = (df.high + df.low + df.close) / 3
    vwap = ((tp * df.volume).groupby(labels).cumsum()
            / df.volume.groupby(labels).cumsum().replace(0, np.nan))
    # A truncated CSV must not masquerade as a full session VWAP.
    for day, part in df.groupby(labels, sort=False):
        start, _ = _session_bounds(day, cfg)
        if part.index[0] != start:
            vwap.loc[part.index] = np.nan
    five = _bars(df, 5, cutoff, cfg)
    atr5 = _atr(five) if len(five) else pd.Series(dtype=float, index=five.index)
    atr1 = _atr(df)
    htf_bars = {}
    for minutes in (60, 120, 240):
        b = _bars(df, minutes, cutoff, cfg)
        htf_bars[minutes] = b
        result["timeframes"][f"{minutes // 60}h"] = _timeframe(
            b, five, vwap, atr5, df, cutoff, minutes, cfg
        )
    whip = _whipsaw(df, vwap, atr1, cutoff, cfg)
    result.update(whipsaw=whip, asof=cutoff)
    result["ict_structure"] = evaluate_market_structure(
        df, config=replace(cfg.structure, timezone=cfg.timezone,
                           session_open_hour=cfg.session_open_hour,
                           session_close_hour=cfg.session_close_hour),
        data_context={**metadata, "last_bar_is_complete": True},
        prepared_bars=htf_bars, prepared_five=five)
    result["primary_structure_layer"] = ("ICT_PRICE_STRUCTURE_OBSERVATION" if cfg.structure.enabled
                                        else "LEGACY_VWAP_TRANSITION")
    long_reasons, short_reasons = [], []
    if (whip["state"] == "HIGH"
            or (whip["state"] == "UNKNOWN" and cfg.block_unknown_whipsaw)
            or (whip["state"] == "ELEVATED" and cfg.block_elevated_whipsaw)):
        reason = f"1m WHIPSAW: {whip['state']}"
        long_reasons.append(reason)
        short_reasons.append(reason)
    directions = set()
    for name, tf in result["timeframes"].items():
        confirmed, pending = tf["confirmed_direction"], tf["pending_direction"]
        if confirmed:
            directions.add(confirmed)
        if confirmed and cfg.block_opposite_confirmed:
            target = short_reasons if confirmed == 1 else long_reasons
            target.append(f"{name} megerősített {'bullish' if confirmed == 1 else 'bearish'} VWAP-visszateszt")
        if pending:
            if cfg.block_against_pending_turn:
                target = short_reasons if pending == 1 else long_reasons
                target.append(f"{name} ellenirányú, még nem visszatesztelt VWAP-forduló")
            if cfg.block_pending_turn_entry:
                target = long_reasons if pending == 1 else short_reasons
                target.append(f"{name} forduló: belépés előtt későbbi visszateszt szükséges")
    result.update(
        confirmation_conflict=len(directions) > 1,
        block_new_longs=bool(long_reasons), block_new_shorts=bool(short_reasons),
        long_block_reasons=long_reasons, short_block_reasons=short_reasons,
    )
    result["notes"] += [
        "A session VWAP nem óránként újrainduló VWAP. Az órás close a saját lezáráskori VWAP-jához mérődik.",
        "A forduló nem automatikus LONG/SHORT jel és nem kezel meglévő pozíciót.",
        "Session-nyitásra horgonyzott HTF gyertyák; a platformod gyertyahatáraival egyeztetni kell.",
        "Hiányzó session-nyitó perc esetén a session VWAP N/A, nem helyettesítjük részleges VWAP-pal.",
    ]
    return result


def _transition_state_text(state: str) -> str:
    return {
        "BULLISH_CROSS_PENDING": "FELFELÉ OLDALVÁLTÁS, VISSZATESZTRE VÁR",
        "BEARISH_CROSS_PENDING": "LEFELÉ OLDALVÁLTÁS, VISSZATESZTRE VÁR",
        "BULLISH_RETEST_CONFIRMED": "FELFELÉ FORDULÓ, VISSZATESZTTEL MEGERŐSÍTVE",
        "BEARISH_RETEST_CONFIRMED": "LEFELÉ FORDULÓ, VISSZATESZTTEL MEGERŐSÍTVE",
        "FAILED_BULLISH_RECLAIM": "A FELFELÉ FORDULÓ ÉRVÉNYTELENÍTŐDÖTT",
        "FAILED_BEARISH_REJECTION": "A LEFELÉ FORDULÓ ÉRVÉNYTELENÍTŐDÖTT",
        "TURN_EXPIRED": "A KORÁBBI FORDULÓ ÉRVÉNYESSÉGE LEJÁRT",
        "NO_TURN": "NINCS ÉSZLELT FORDULÓ",
        "UNKNOWN": "NEM ÉRTÉKELHETŐ BIZTONSÁGOSAN",
        "DISABLED": "VIZSGÁLAT KIKAPCSOLVA",
    }.get(state, state)


def _transition_rows_readable(checks: list[dict[str, Any]], *,
                              whipsaw: bool = False) -> list[str]:
    """Presentation only. Use existing passed flags; never recompute decisions."""
    meanings = {
        "Elég lezárt gyertya az aktuális sessionben":
            "Adatfeltétel: a darabszám a vizsgált 1h/2h/4h lezárt gyertyák száma, nem fordulók száma.",
        "Lezárt close távolsága a session VWAP-tól":
            "A negatív érték VWAP alatti, a pozitív VWAP feletti zárást jelent. Nem valószínűség.",
        "Történeti bearish -> bullish VWAP oldalváltás":
            "Esemény: a lezárt HTF zárás a VWAP alsó oldaláról a felsőre került. Lejárt esemény is szerepelhet.",
        "Történeti bullish -> bearish VWAP oldalváltás":
            "Esemény: a lezárt HTF zárás a VWAP felső oldaláról az alsóra került. Lejárt esemény is szerepelhet.",
        "Külön későbbi lezárt 5m visszateszt":
            "Megerősítés: egy későbbi 5m gyertya érintette a VWAP-zónát, és az új oldalon zárt, elfogadott mérettel.",
        "Értékelt aktív ablakban nincs ellenoldali 5m zárás":
            "Érvényességi ellenőrzés: csak a kiírt értékelési végpontig vizsgált, nem a falióra szerinti jelenig.",
        "Forduló most is aktív":
            "Az elemzési időpontban még nem lejárt és nem érvénytelenített. A visszateszt ettől még hiányozhat.",
        "Perces adatlefedettség":
            "Adatfeltétel, nem veszélyjel: a gyertyaszámon túl a hiányokat, frissességet és számíthatóságot is ellenőrzi.",
        "Kétoldali kanócos gyertyák aránya":
            "Az olyan perces gyertyák aránya, amelyeknek egyszerre jelentős felső és alsó kanóca van.",
        "Kis testű gyertyák aránya":
            "A kis nyitó-záró különbségű gyertyák aránya a teljes high-low mérethez viszonyítva.",
        "Close-irányváltások aránya":
            "A nem nulla perces záróármozgások közötti iránycserék aránya; nem a bearish gyertyák aránya.",
        "Nettó mozgás / teljes close-út":
            "A nettó elmozdulás a perces záróárak összes abszolút lépéséhez képest. Kicsi érték = sok oda-vissza út.",
        "Nagy range gyertyák aránya":
            "Az átlagosnál nagyobb high-low méretű gyertyák aránya. A küszöb az előző perces ATR-hez mér.",
        "Extrém egyszeri range / előző ATR1m":
            "A vizsgált ablak legnagyobb gyertyamérete az előző ATR többszöröseként; nem százalék.",
        "Pufferelt VWAP oldalváltások száma":
            "A VWAP két oldala közötti záróárváltások száma, kis zajszűrő pufferrel. Nem minden VWAP-érintés számít.",
    }
    ratio_labels = {
        "Kétoldali kanócos gyertyák aránya", "Kis testű gyertyák aránya",
        "Close-irányváltások aránya", "Nettó mozgás / teljes close-út",
        "Nagy range gyertyák aránya",
    }
    lines = []
    for row in checks:
        label, value, flag = row["label"], row["value"], row["passed"]
        data_condition = label == "Perces adatlefedettség"
        if flag is None:
            outcome, measured = "NEM ÉRTÉKELHETŐ / NEM ALKALMAZHATÓ", "N/A"
        else:
            outcome = (
                ("KOCKÁZATI FELTÉTEL JELEN VAN" if flag else "NEM ÉRI EL A KOCKÁZATI KÜSZÖBÖT")
                if whipsaw and not data_condition else
                ("FELTÉTEL TELJESÜL" if flag else "FELTÉTEL NEM TELJESÜL")
            )
            if isinstance(value, (bool, np.bool_)):
                # The verdict already states the boolean: avoid IGEN | IGEN duplication.
                measured = "esemény / érvényességi feltétel, nincs külön számszerű mérés"
            elif label in ratio_labels:
                measured = f"{float(value):.1%}"
            elif label == "Lezárt close távolsága a session VWAP-tól":
                measured = f"{float(value):+.3f} x ATR(5m)"
            elif label == "Extrém egyszeri range / előző ATR1m":
                measured = f"{float(value):.3f} x előző ATR(1m)"
            elif label in ("Perces adatlefedettség", "Elég lezárt gyertya az aktuális sessionben"):
                measured = f"{value} lezárt gyertya"
            elif label == "Pufferelt VWAP oldalváltások száma":
                measured = f"{value} oldalváltás"
            else:
                measured = str(value)
        threshold = row["threshold"]
        if label == "Nettó mozgás / teljes close-út":
            threshold += " (0.25 = 25%; az aránynál a kisebb érték a zajosabb)"
        lines += [f"- {label}",
                  f"  Értékelés: {outcome}",
                  f"  Mért adat: {measured} | Szabály: {threshold}"]
        if label in meanings:
            lines.append(f"  Jelentése: {meanings[label]}")
    return lines


def format_transition_risk(result: dict[str, Any], *, instrument: str = "GC") -> list[str]:
    """Readable view of the unchanged transition and whipsaw decisions."""
    asof = result.get("asof")
    structure_lines = (format_market_structure(result["ict_structure"], instrument=instrument)
                       if "ict_structure" in result else [])
    lines = structure_lines + ["", "=" * 100, f"{instrument} ÓRÁS VWAP-FORDULÓK ÉS PERCES ODA-VISSZA MOZGÁS",
             "=" * 100, f"Elemzési időpont (nem feltétlenül valós idő): {asof}",
             "HTF = magasabb idősík (1h/2h/4h); close = gyertyazáró ár; retest = visszateszt.",
             "ATR = átlagos gyertyamozgás mérőszáma; itt távolsági egység, nem százalék.",
             "GYORS ÖSSZEFOGLALÓ"]
    for name, tf in result.get("timeframes", {}).items():
        lines.append(f"- {name.upper()}: {_transition_state_text(tf['state'])}")
    whip = result.get("whipsaw", {})
    whip_text = {
        "LOW": "ALACSONY mért oda-vissza mozgási kockázat",
        "ELEVATED": "EMELKEDETT mért oda-vissza mozgási kockázat",
        "HIGH": "MAGAS mért oda-vissza mozgási kockázat",
        "UNKNOWN": "NEM ÉRTÉKELHETŐ (adatkorlát)",
        "DISABLED": "KIKAPCSOLVA",
    }.get(whip.get("state"), whip.get("state", "N/A"))
    lines += [f"- 1m whipsaw: {whip_text}",
              "A forduló iránya és a belépés engedélyezése két külön kérdés."]
    for name, tf in result.get("timeframes", {}).items():
        state = tf["state"]
        context = tf.get("current_context", "UNKNOWN")
        context_text = {
            "ABOVE_SESSION_VWAP": "Az utolsó lezárt HTF zárás a VWAP fölötti pufferen túl van.",
            "BELOW_SESSION_VWAP": "Az utolsó lezárt HTF zárás a VWAP alatti pufferen túl van.",
            "NEAR_OR_UNKNOWN_VWAP": "VWAP-pufferzóna vagy nem számítható távolság; lásd a mérést.",
            "UNKNOWN": "A lezárt VWAP-helyzet nem értékelhető.",
        }.get(context, context)
        lines += ["", "-" * 100, f"{name.upper()} FORDULÓ CHECKLIST",
                  f"Magyar állapot: {_transition_state_text(state)}",
                  f"Technikai kód: {state} | {context}",
                  f"Lezárt árhelyzet: {context_text}"]
        if state.endswith("CROSS_PENDING"):
            lines.append("Lényeg: az oldalváltás már megtörtént, de nincs igazolt későbbi 5m visszateszt.")
        elif state.endswith("RETEST_CONFIRMED"):
            lines.append("Lényeg: az oldalváltást későbbi visszateszt megerősítette; ettől még nem automatikus belépés.")
        elif state == "TURN_EXPIRED":
            lines.append("Lényeg: a cross történeti esemény. Az aktív fordulókapu már nem ezt az epizódot követi.")
        elif state.startswith("FAILED_"):
            lines.append("Lényeg: egy későbbi ellenoldali zárás érvénytelenítette ezt a fordulóepizódot.")
        lines += [f"Utolsó lezárt {name} gyertya vége: {tf.get('last_closed_at') or 'N/A'}",
                  f"Oldalváltás időpontja: {tf.get('cross_at') or 'Nincs észlelt esemény'}",
                  f"Igazolt visszateszt: {tf.get('retest_at') or 'Nincs igazolt esemény; az értékelhetőséget lásd lent'}",
                  f"Érvénytelenítő zárás: {tf.get('failed_at') or 'Nincs igazolt esemény; az értékelhetőséget lásd lent'}",
                  f"Eddig vizsgált fordulóablak: {tf.get('evaluated_until') or 'Nem alkalmazható'}",
                  f"Forduló érvényességének vége: {tf.get('expires_at') or 'Nem alkalmazható'}"]
        if tf.get("active_turn") and tf.get("expires_at") is not None and asof is not None:
            minutes = max(0, (pd.Timestamp(tf["expires_at"])-pd.Timestamp(asof)).total_seconds()/60)
            lines.append(f"Hátralévő érvényesség az elemzési időponttól: {minutes:.0f} perc")
        forming = tf.get("forming_state", "NONE")
        forming_text = {
            "NONE": "Nincs észlelt ellenirányú jel a formálódó HTF gyertyán.",
            "BULLISH_RECLAIM_IN_PROGRESS": "Felfelé ellenirányú oldalváltás formálódik; még NEM lezárt megerősítés.",
            "BEARISH_REJECTION_IN_PROGRESS": "Lefelé ellenirányú oldalváltás formálódik; még NEM lezárt megerősítés.",
        }.get(forming, forming)
        lines += [forming_text, "RÉSZLETES ELLENŐRZÉSEK",
                  *_transition_rows_readable(tf.get("checks", []))]
        lines += [f"- {note}" for note in tf.get("notes", [])]
    lines += ["", "-" * 100, "1m WHIPSAW-KOCKÁZAT: PERCES ODA-VISSZA MOZGÁS",
              f"Összesített minősítés: {whip_text} | Technikai kód: {whip.get('state', 'N/A')}",
              "Egy kockázati feltétel teljesülése itt figyelmeztetés, nem pozitív trade-jel.",
              "A LOW nem jelent kanócmentes piacot. Az összesítés feltételcsoportokat kombinál, nem IGEN-eket számol.",
              *_transition_rows_readable(whip.get("checks", []), whipsaw=True)]
    lines += [f"- {note}" for note in whip.get("notes", [])]
    lines += ["", "-" * 100, "BELÉPÉSI KAPUK: MIÉRT LEHET MINDKÉT IRÁNY TILTVA?",
              f"Új LONG kapu: {'TILTVA' if result.get('block_new_longs') else 'NINCS ÚJ TILTÁS EBBŐL A RÉTEGBŐL'}",
              f"Új SHORT kapu: {'TILTVA' if result.get('block_new_shorts') else 'NINCS ÚJ TILTÁS EBBŐL A RÉTEGBŐL'}",
              f"Ellentétes MEGERŐSÍTETT HTF fordulók: {'IGEN' if result.get('confirmation_conflict') else 'NEM'}"]
    pending = {tf.get("pending_direction") for tf in result.get("timeframes", {}).values()} - {0, None}
    if result.get("block_new_longs") and result.get("block_new_shorts"):
        if pending == {-1}:
            lines.append("Lefelé forduló vár visszatesztre: LONG = ellene mennél; SHORT = még nincs visszateszt.")
        elif pending == {1}:
            lines.append("Felfelé forduló vár visszatesztre: SHORT = ellene mennél; LONG = még nincs visszateszt.")
        lines.append("A két tiltás nem feltétlen iránykonfliktus. A tényleges okok:")
    lines += [f"- LONG tiltás oka: {reason}" for reason in result.get("long_block_reasons", [])]
    lines += [f"- SHORT tiltás oka: {reason}" for reason in result.get("short_block_reasons", [])]
    lines += ["A NINCS ÚJ TILTÁS nem belépési engedély; a többi rezsim- és risk-feltétel továbbra is számít.",
              "A tiltások új belépési jelöltekre vonatkoznak, nem meglévő pozíció zárási utasításai."]
    lines += [f"- {note}" for note in result.get("notes", [])]
    return lines


def transition_trade_block(result: dict[str, Any], direction: int) -> dict[str, Any]:
    side = "LONG" if direction == 1 else "SHORT"
    reasons = result["long_block_reasons"] if direction == 1 else result["short_block_reasons"]
    return dict(trade_allowed=False,
                reason=f"Új {side} tiltva: HTF-forduló / 1m whipsaw adatkapu.",
                details="; ".join(reasons),
                next_setup="Várj a whipsaw enyhülésére és az órás oldalváltás "
                           "külön későbbi visszatesztjére. Ne lépj be csak a checklist miatt.")
