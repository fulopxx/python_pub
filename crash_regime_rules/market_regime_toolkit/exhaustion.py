"""GC / ES / NQ multi-timeframe exhaustion overlay, version 0.1.3.

Research / paper analysis only. Thresholds are unvalidated starting values,
not fitted probabilities of a crash. Input: 1m OHLCV, minute-start timestamps.
The latest input minute is discarded unless explicitly marked complete.
Optional daily_history uses COMPLETION timestamps of closed, same-contract
daily bars. No wall-clock reads, downloads, orders or implicit disk state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

VERSION = "0.1.3"
OHLCV = ["open", "high", "low", "close", "volume"]
AGG = dict(open="first", high="max", low="min", close="last", volume="sum")


@dataclass(frozen=True)
class ExhaustionConfig:
    symbol: str
    vwap_extension_atr: float = 4.0
    hourly_extension_atr: float = 2.5
    daily_extension_atr: float = 2.0
    acceleration_ratio: float = 1.5
    acceleration_min_slope: float = 0.10
    acceleration_min_r2: float = 0.80
    volume_climax_ratio: float = 2.0
    volume_min_sessions: int = 5
    upper_wick_fraction: float = 0.60
    atr_expansion_ratio: float = 1.5
    momentum_rsi: float = 55.0
    min_5m_bars: int = 50
    min_hourly_bars: int = 30
    min_daily_bars: int = 25
    swing_width: int = 2
    breakout_lookback: int = 12
    evidence_window_bars: int = 6
    replay_bars: int = 96
    extension_memory_minutes: int = 60
    latch_minutes: int = 240
    break_buffer_atr: float = 0.10
    retest_tolerance_atr: float = 0.25
    reset_buffer_atr: float = 0.25
    reset_bars: int = 2
    block_unknown_longs: bool = True
    block_parabolic_longs: bool = False
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.symbol not in ("GC", "ES", "NQ"):
            raise ValueError("ExhaustionConfig symbol must be GC, ES or NQ.")
        if self.replay_bars * 5 <= self.latch_minutes + self.extension_memory_minutes:
            raise ValueError("Replay window must exceed both latch lifetimes together.")
        if self.reset_bars < 1 or self.swing_width < 1:
            raise ValueError("reset_bars and swing_width must be positive.")


# Separate configurable profiles, NOT empirically calibrated.
GC_EXHAUSTION = ExhaustionConfig("GC", daily_extension_atr=2.5)
ES_EXHAUSTION = ExhaustionConfig("ES")
NQ_EXHAUSTION = ExhaustionConfig(
    "NQ", vwap_extension_atr=5.0,
    hourly_extension_atr=3.0, daily_extension_atr=2.5,
)
PROFILES = {"GC": GC_EXHAUSTION, "ES": ES_EXHAUSTION, "NQ": NQ_EXHAUSTION}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("OHLCV index must be DatetimeIndex.")
    if not set(OHLCV).issubset(df.columns):
        raise ValueError("Missing OHLCV columns.")
    out = df[OHLCV].apply(pd.to_numeric, errors="coerce").copy()
    out.index = pd.to_datetime(out.index, utc=True).tz_convert("America/New_York")
    out = out.sort_index()
    if out.index.has_duplicates:
        raise ValueError("Duplicate input timestamps: deduplicate before analysis.")
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("Non-finite OHLCV input.")
    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Prices must be positive.")
    if ((out.high < out[["open", "close", "low"]].max(axis=1))
            | (out.low > out[["open", "close", "high"]].min(axis=1))
            | (out.volume < 0)).any():
        raise ValueError("Invalid OHLC or negative volume.")
    utc_index = out.index.tz_convert("UTC")
    if len(out) and (utc_index != utc_index.floor("min")).any():
        raise ValueError("1m timestamps must mark exact minute starts.")
    return out


def _atr(df: pd.DataFrame) -> pd.Series:
    prev = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - prev).abs(),
                    (df.low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False,
                                  min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False,
                                     min_periods=period).mean()
    result = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    result = result.mask((loss == 0) & (gain > 0), 100)
    return result.mask((loss == 0) & (gain == 0), 50)


def _closed_bars(df: pd.DataFrame, minutes: int,
                 cutoff: pd.Timestamp) -> pd.DataFrame:
    # UTC binning is unambiguous through New York DST changes.
    src = df.tz_convert("UTC")
    rule = f"{minutes}min"
    out = src.resample(rule, label="left", closed="left").agg(AGG)
    count = src.close.resample(rule).count()
    end = out.index + pd.Timedelta(minutes=minutes)
    out = out.loc[(count == minutes) & (end <= cutoff.tz_convert("UTC"))]
    out.index = (out.index + pd.Timedelta(minutes=minutes)).tz_convert(
        "America/New_York"
    )
    return out


def _session_labels(idx: pd.DatetimeIndex) -> pd.Index:
    # Calendar arithmetic in local wall time, not a DST-sensitive 18h UTC shift.
    # floor("D") is the vectorized equivalent of element-wise .date labels.
    local = idx.tz_localize(None)
    return pd.Index((local - pd.Timedelta(hours=18)).floor("D"))


def _daily_from_minutes(df: pd.DataFrame,
                        cutoff: pd.Timestamp) -> pd.DataFrame:
    records, completions = [], []
    for date, part in df.groupby(_session_labels(df.index)):
        start = (pd.Timestamp(date) + pd.Timedelta(hours=18)).tz_localize(
            "America/New_York"
        )
        close_date = date + timedelta(days=1)
        end = (pd.Timestamp(close_date) + pd.Timedelta(hours=17)).tz_localize(
            "America/New_York"
        )
        expected_minutes = int((end - start).total_seconds() / 60)
        # Omit current session, partial first session, and low-coverage sessions.
        if (end > cutoff or part.index.min() > start
                or part.index.max() < end - pd.Timedelta(minutes=1)
                or len(part) < 0.90 * expected_minutes):
            continue
        records.append(dict(open=part.open.iloc[0], high=part.high.max(),
                            low=part.low.min(), close=part.close.iloc[-1],
                            volume=part.volume.sum()))
        completions.append(end)
    return pd.DataFrame(records, columns=OHLCV,
                        index=pd.DatetimeIndex(completions,
                                               tz="America/New_York"))


def _slope(series: pd.Series, atr: float) -> tuple[float, float]:
    y = series.to_numpy(float)
    # Timestamp resolution can be ns or us; timedelta division is unit-safe.
    x = np.asarray((series.index - series.index[0]) / pd.Timedelta(hours=1))
    if len(y) < 2 or atr <= 0 or np.ptp(x) == 0:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    residual = np.sum((y - (slope * x + intercept)) ** 2)
    variance = np.sum((y - y.mean()) ** 2)
    return slope / atr, 1 - residual / variance if variance > 0 else 0.0


def _slope_arrays(x: np.ndarray, y: np.ndarray, atr: float) -> tuple[float, float]:
    """_slope with precomputed x/y arrays; same math and edge handling.

    x is rebased to the slice's own origin, matching the original
    _slope(series, ...) coordinates exactly (series.index - series.index[0]).
    """
    if len(y) < 2 or atr <= 0 or np.ptp(x) == 0:
        return np.nan, np.nan
    x = x - x[0]
    slope, intercept = np.polyfit(x, y, 1)
    residual = np.sum((y - (slope * x + intercept)) ** 2)
    variance = np.sum((y - y.mean()) ** 2)
    return slope / atr, 1 - residual / variance if variance > 0 else 0.0


def _htf_features(df: pd.DataFrame, minimum: int) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    if df.empty:
        return f
    atr = _atr(df)
    ema = df.close.ewm(span=20, adjust=False, min_periods=20).mean()
    f["extension"] = ((df.close - ema) / atr.replace(0, np.nan)).where(
        np.arange(len(df)) >= minimum - 1
    )
    f["acceleration"] = np.nan
    f["slope"] = np.nan
    f["r2"] = np.nan
    acceleration = np.full(len(df), np.nan)
    slopes = np.full(len(df), np.nan)
    fits = np.full(len(df), np.nan)
    close = df.close.to_numpy(float)
    atr_values = atr.to_numpy(float)
    # Timestamp resolution can be ns or us; timedelta division is unit-safe.
    x = np.asarray((df.index - df.index[0]) / pd.Timedelta(hours=1))
    for i in range(max(minimum - 1, 11), len(df)):
        a = atr_values[i]
        old, _ = _slope_arrays(x[i - 11:i - 5], close[i - 11:i - 5], a)
        new, r2 = _slope_arrays(x[i - 5:i + 1], close[i - 5:i + 1], a)
        acceleration[i] = (
            new / old if old > 0 and new > 0 else 0.0
        )
        slopes[i] = new
        fits[i] = r2
    f["acceleration"], f["slope"], f["r2"] = acceleration, slopes, fits
    return f


def _align(source: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    if source.empty:
        return pd.DataFrame(index=index, columns=source.columns, dtype=float)
    # Source indices are COMPLETION times; forward filling cannot see future bars.
    return source.reindex(index, method="ffill")


def _slot_baseline(values: np.ndarray, slots: np.ndarray, labels: np.ndarray,
                   min_sessions: int) -> np.ndarray:
    """Prior-session expanding-median baseline per time-of-day slot.

    Vectorized equivalent of the previous per-slot groupby transform: for
    every 5m slot, the baseline is the median of that slot's first-per-
    session volumes over the sessions before the current one. A repeated
    local slot on DST fall-back counts once and uses its first occurrence.
    """
    out = np.full(len(values), np.nan)
    first_per_slot: dict = {}
    position_of: dict = {}
    seen = set()
    for row in range(len(values)):
        key = (slots[row], labels[row])
        if key in seen:
            continue
        seen.add(key)
        slot = slots[row]
        if slot not in first_per_slot:
            first_per_slot[slot] = []
            position_of[slot] = {}
        first_per_slot[slot].append(values[row])
        position_of[slot][labels[row]] = len(first_per_slot[slot]) - 1
    prior = {}
    for slot, volumes in first_per_slot.items():
        session_values = np.asarray(volumes)
        for k in range(min_sessions, len(session_values)):
            prior[(slot, k)] = float(np.median(session_values[:k]))
    for row in range(len(values)):
        slot = slots[row]
        position = position_of.get(slot, {}).get(labels[row])
        if position is not None:
            value = prior.get((slot, position))
            if value is not None:
                out[row] = value
    return out


def _features(df: pd.DataFrame, bars: pd.DataFrame, hourly: pd.DataFrame,
              daily: pd.DataFrame, cfg: ExhaustionConfig) -> pd.DataFrame:
    f = bars.copy()
    f["atr"] = _atr(bars)
    f["rsi"] = _rsi(bars.close)
    labels = _session_labels(df.index)
    typical = (df.high + df.low + df.close) / 3
    vwap = ((typical * df.volume).groupby(labels).cumsum()
            / df.volume.groupby(labels).cumsum().replace(0, np.nan))
    # The completed 5m close corresponds to its last constituent minute start.
    f["vwap"] = vwap.reindex(bars.index - pd.Timedelta(minutes=1)).to_numpy()
    f["vwap_ext"] = (f.close - f.vwap) / f.atr.replace(0, np.nan)
    h = _align(_htf_features(hourly, cfg.min_hourly_bars), bars.index)
    d = _align(_htf_features(daily, cfg.min_daily_bars), bars.index)
    for key in ("extension", "acceleration", "slope", "r2"):
        f["h_" + key] = h[key] if key in h else np.nan
    f["d_extension"] = d["extension"] if "extension" in d else np.nan
    # A completed 60m bar older than 2h does not represent current HTF context.
    h_times = pd.Series(hourly.index, index=hourly.index)
    if len(h_times):
        aligned_times = h_times.reindex(bars.index, method="ffill")
        stale = (pd.Series(bars.index, index=bars.index) - aligned_times
                 > pd.Timedelta(hours=2))
        f.loc[stale, [c for c in f if c.startswith("h_")]] = np.nan
    if len(daily):
        dtimes = pd.Series(daily.index, index=daily.index).reindex(
            bars.index, method="ffill"
        )
        dstale = (pd.Series(bars.index, index=bars.index) - dtimes
                  > pd.Timedelta(days=4))
        f.loc[dstale, "d_extension"] = np.nan
    previous_high = f.high.shift(1).rolling(cfg.breakout_lookback).max()
    f["failed_breakout"] = (f.high > previous_high) & (f.close < previous_high)
    fraction = (f.high - f[["open", "close"]].max(axis=1)) / (
        f.high - f.low
    ).replace(0, np.nan)
    f["upper_wick"] = (
        (fraction >= cfg.upper_wick_fraction) & (f.high >= previous_high)
        & (f.close < f.open)
    )
    f["wick_fraction"] = fraction
    # A time-of-day baseline, with one completed 5m bar per day and slot.
    slots = f.index.hour * 60 + f.index.minute
    labels = np.asarray(_session_labels(f.index - pd.Timedelta(minutes=1)))
    volume_base = pd.Series(
        _slot_baseline(f.volume.to_numpy(float), np.asarray(slots), labels,
                       cfg.volume_min_sessions),
        index=f.index)
    f["volume_ratio_slot"] = f.volume / volume_base.replace(0, np.nan)
    f["atr_expansion"] = f.atr / f.atr.shift(1).rolling(
        48, min_periods=24
    ).median().replace(0, np.nan)
    width = cfg.swing_width
    divergence = np.zeros(len(f), dtype=bool)
    higher_low = np.full(len(f), np.nan)
    highs, lows, rsis = (f[c].to_numpy(float) for c in ("high", "low", "rsi"))
    previous_peak = None
    previous_low = None
    known_higher_low = np.nan
    for i in range(len(f)):
        candidate = i - width
        if candidate >= width:
            peak_window = highs[candidate - width:candidate + width + 1]
            low_window = lows[candidate - width:candidate + width + 1]
            if (highs[candidate] == peak_window.max()
                    and np.count_nonzero(peak_window == highs[candidate]) == 1):
                if previous_peak is not None:
                    divergence[i] = bool(
                        highs[candidate] > highs[previous_peak]
                        and rsis[candidate] < rsis[previous_peak]
                    )
                previous_peak = candidate
            if (lows[candidate] == low_window.min()
                    and np.count_nonzero(low_window == lows[candidate]) == 1):
                if previous_low is not None and lows[candidate] > lows[previous_low]:
                    known_higher_low = float(lows[candidate])
                else:
                    known_higher_low = np.nan
                previous_low = candidate
        higher_low[i] = known_higher_low
    f["divergence"], f["higher_low"] = divergence, higher_low
    return f


def _check(label: str, value: Any, threshold: str, timeframe: str,
           passed: Any, available: bool = True) -> dict[str, Any]:
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        available = False
    return dict(label=label, value=value, threshold=threshold, timeframe=timeframe,
                passed=bool(passed) if available else None)


def _gates(f: pd.DataFrame, i: int, cfg: ExhaustionConfig) -> dict[str, Any]:
    row = f.iloc[i]
    window = f.iloc[max(0, i - cfg.evidence_window_bars + 1):i + 1]
    window = window.loc[
        window.index >= f.index[i] - pd.Timedelta(
            minutes=5 * (cfg.evidence_window_bars - 1)
        )
    ]
    ext = [
        _check("Session VWAP extension", row.vwap_ext,
               f">= {cfg.vwap_extension_atr:g} ATR(5m)", "5m",
               row.vwap_ext >= cfg.vwap_extension_atr),
        _check("EMA20 extension", row.h_extension,
               f">= {cfg.hourly_extension_atr:g} ATR(60m)", "60m",
               row.h_extension >= cfg.hourly_extension_atr),
        _check("EMA20 extension", row.d_extension,
               f">= {cfg.daily_extension_atr:g} ATR(D)", "D",
               row.d_extension >= cfg.daily_extension_atr),
        _check("Emelkedési gyorsulás",
               (f"{row.h_acceleration:.2f}x; slope={row.h_slope:.3f}; R2={row.h_r2:.2f}"
                if all(np.isfinite(row[k]) for k in
                       ("h_acceleration", "h_slope", "h_r2")) else None),
               f">= {cfg.acceleration_ratio:g}; slope >= "
               f"{cfg.acceleration_min_slope:g} ATR/óra; R2 >= "
               f"{cfg.acceleration_min_r2:g}", "60m",
               (row.h_acceleration >= cfg.acceleration_ratio
                and row.h_slope >= cfg.acceleration_min_slope
                and row.h_r2 >= cfg.acceleration_min_r2),
               available=all(np.isfinite(row[k]) for k in
                             ("h_acceleration", "h_slope", "h_r2"))),
    ]
    structure = bool(window.failed_breakout.any() or window.upper_wick.any())
    max_volume = window.volume_ratio_slot.max()
    divergence = bool(window.divergence.any())
    # Momentum loss is separate from RSI extreme extension.
    momentum = bool(row.rsi < cfg.momentum_rsi and row.close < row.open)
    exhaustion = [
        _check("Failed breakout / felső kanócos elutasítás", structure,
               "Legalább 1 lezárt árstruktúra-jel", "5m", structure),
        _check("Idősávhoz mért volumenklimax", max_volume,
               f">= {cfg.volume_climax_ratio:g}x; "
               f"legalább {cfg.volume_min_sessions} korábbi session", "5m",
               max_volume >= cfg.volume_climax_ratio),
        _check("Megerősített bearish RSI(14) divergencia", divergence,
               "Magasabb ár-csúcs, alacsonyabb RSI; swing már megerősített",
               "5m", divergence),
        _check("Momentumvesztés", row.rsi,
               f"RSI(14) < {cfg.momentum_rsi:g} és bearish close", "5m", momentum),
    ]
    ext_count = sum(c["passed"] is True for c in ext)
    high_tf = ext[1]["passed"] is True or ext[2]["passed"] is True
    parabolic = ext_count >= 2 and high_tf
    exhaustion_count = sum(c["passed"] is True for c in exhaustion)
    has_htf = np.isfinite(row.h_extension) or np.isfinite(row.d_extension)
    local_ready = i >= cfg.min_5m_bars - 1 and np.isfinite(row.atr) and row.atr > 0
    return dict(extension_checks=ext, exhaustion_checks=exhaustion,
                parabolic=parabolic,
                exhaustion=parabolic and structure and exhaustion_count >= 2,
                exhaustion_evidence=structure and exhaustion_count >= 2,
                sufficient=bool(has_htf and local_ready))


def advance_state(memory: dict[str, Any], timestamp: pd.Timestamp,
                  row: dict[str, Any], gates: dict[str, Any],
                  cfg: ExhaustionConfig) -> dict[str, Any]:
    """One CLOSED 5m event. Pure causal state transition; duplicate-safe."""
    m = dict(memory)
    ts = pd.Timestamp(timestamp)
    last = m.get("last_bar")
    if last is not None and ts <= pd.Timestamp(last):
        return m
    m["last_bar"] = ts.isoformat()
    expiry = m.get("expires_at")
    if expiry is not None and ts >= pd.Timestamp(expiry):
        m = {"last_bar": ts.isoformat(), "state": "NORMAL"}
    if gates["sufficient"] and gates["parabolic"]:
        m["extension_until"] = (
            ts + pd.Timedelta(minutes=cfg.extension_memory_minutes)
        ).isoformat()
    extension_until = m.get("extension_until")
    recent_extension = bool(
        extension_until is not None and ts < pd.Timestamp(extension_until)
    )
    state = m.get("state", "NORMAL")
    valid_atr = np.isfinite(row["atr"]) and row["atr"] > 0
    if state in ("EXHAUSTION_RISK", "BREAKDOWN_CONFIRMED") and valid_atr:
        above_peak = row["close"] > m["peak"] + cfg.reset_buffer_atr * row["atr"]
        m["reset_count"] = m.get("reset_count", 0) + 1 if above_peak else 0
        if m["reset_count"] >= cfg.reset_bars:
            # A successful reset must not immediately relatch the OLD six-bar
            # evidence on the same candle.
            return {
                "last_bar": ts.isoformat(), "state_since": ts.isoformat(),
                "state": "PARABOLIC_EXTENSION" if gates["parabolic"] else "NORMAL",
            }
    exhaustion_qualified = (
        gates["exhaustion"]
        or (recent_extension and gates.get("exhaustion_evidence", False))
    )
    if gates["sufficient"] and exhaustion_qualified:
        if state not in ("EXHAUSTION_RISK", "BREAKDOWN_CONFIRMED"):
            m.update(state="EXHAUSTION_RISK", state_since=ts.isoformat(),
                     peak=float(row["peak"]), support=float(row["higher_low"]),
                     broken_at=None, retest_at=None, reset_count=0)
        # Renewal only when new closed-bar evidence qualifies, never on reruns.
        m["expires_at"] = (ts + pd.Timedelta(minutes=cfg.latch_minutes)).isoformat()
    state = m.get("state", "NORMAL")
    support = m.get("support", np.nan)
    if state in ("EXHAUSTION_RISK", "BREAKDOWN_CONFIRMED") and valid_atr:
        if np.isfinite(support):
            buffer = cfg.break_buffer_atr * row["atr"]
            broken_at = m.get("broken_at")
            if row["close"] > support + buffer:
                m["broken_at"] = None
                m["retest_at"] = None
                m["state"] = "EXHAUSTION_RISK"
            elif (broken_at is None and ts > pd.Timestamp(m["state_since"])
                  and row["close"] < support - buffer):
                m["broken_at"] = ts.isoformat()
            elif broken_at is not None and ts > pd.Timestamp(broken_at):
                touched = row["high"] >= support - cfg.retest_tolerance_atr * row["atr"]
                if touched and row["close"] < support - buffer and row["close"] < row["open"]:
                    m["state"] = "BREAKDOWN_CONFIRMED"
                    m["retest_at"] = ts.isoformat()
    if m.get("state", "NORMAL") not in ("EXHAUSTION_RISK", "BREAKDOWN_CONFIRMED"):
        m["state"] = (
            "UNKNOWN" if not gates["sufficient"]
            else "PARABOLIC_EXTENSION" if recent_extension else "NORMAL"
        )
        if m["state"] != state:
            m["state_since"] = ts.isoformat()
    return m


def _context_checks(symbol: str, context: dict[str, Any] | None,
                    asof: pd.Timestamp) -> list[dict[str, Any]]:
    # Explicit bool evidence with its own observation timestamp, not invented feeds.
    specs = [
        ("bearish_gap_rejection", "Közeli NDOG/NWOG bearish rejection", "5m"),
    ]
    if symbol == "GC":
        specs += [("dxy_strength", "DXY erősödés", "macro"),
                  ("real_yield_rising", "Valódi intraday reálhozam emelkedés", "macro")]
    else:
        specs += [("peer_non_confirmation", "Másik index nem erősíti meg a csúcsot", "cross"),
                  ("relative_strength_falling", "NQ/ES relatív erő romlása", "cross"),
                  ("vix_stress", "VIX/VIX3M stressz (NQ-nál széles piaci)", "macro"),
                  ("breadth_weakening", "Piaci breadth romlása", "cross")]
    checks = []
    for key, label, frame in specs:
        item = (context or {}).get(key, {})
        ok = isinstance(item, dict) and isinstance(item.get("value"), (bool, np.bool_))
        observed = item.get("timestamp") if isinstance(item, dict) else None
        if observed is None:
            ok = False
        else:
            observed = pd.to_datetime(observed, utc=True, errors="coerce")
            age = asof.tz_convert("UTC") - observed
            ok = ok and pd.notna(observed) and pd.Timedelta(0) <= age <= pd.Timedelta(minutes=15)
        value = item.get("value") if ok else None
        checks.append(_check(label, value, "Timestampolt, max. 15 perces input",
                             frame, value, available=bool(ok)))
    return checks


def build_index_cross_context(
    es_1m: pd.DataFrame,
    nq_1m: pd.DataFrame,
    *,
    vix_context: dict[str, Any] | None = None,
    vix_stress_ratio: float = 1.10,
) -> dict[str, dict[str, Any]]:
    """Optional runner adapter: synchronized, CLOSED 5m ES/NQ evidence.

    No peer data is fabricated when a runner only passes a single instrument.
    A timestamped VIX input is required for the new VIX context row.
    This context is informative only and never substitutes for a local gate.
    """
    es, nq = _clean(es_1m), _clean(nq_1m)
    result: dict[str, dict[str, Any]] = {"ES": {}, "NQ": {}}
    if es.empty or nq.empty:
        return result
    cutoff = min(es.index[-1], nq.index[-1])
    e5 = _closed_bars(es.loc[es.index < cutoff], 5, cutoff)
    n5 = _closed_bars(nq.loc[nq.index < cutoff], 5, cutoff)
    common = e5.index.intersection(n5.index)
    if len(common) < 13:
        return result
    common = common[-13:]
    if cutoff - common[-1] > pd.Timedelta(minutes=10):
        return result
    e5, n5 = e5.loc[common], n5.loc[common]
    stamp = common[-1]
    enew = bool(e5.high.iloc[-1] > e5.high.iloc[:-1].max())
    nnew = bool(n5.high.iloc[-1] > n5.high.iloc[:-1].max())
    result["ES"]["peer_non_confirmation"] = dict(
        value=enew and not nnew, timestamp=stamp
    )
    result["NQ"]["peer_non_confirmation"] = dict(
        value=nnew and not enew, timestamp=stamp
    )
    # Do not call a sparse four-bar slice a 15m relative-strength comparison.
    if common[-1] - common[-4] == pd.Timedelta(minutes=15):
        ratio = n5.close / e5.close
        falling = bool(ratio.iloc[-1] < ratio.iloc[-4])
        for symbol in result:
            result[symbol]["relative_strength_falling"] = dict(
                value=falling, timestamp=stamp
            )
    vx = vix_context or {}
    if vx.get("timestamp") is not None:
        vix, vix3m = vx.get("vix", np.nan), vx.get("vix3m", np.nan)
        if np.isfinite(vix) and np.isfinite(vix3m) and vix3m > 0:
            for symbol in result:
                result[symbol]["vix_stress"] = dict(
                    value=bool(vix / vix3m >= vix_stress_ratio),
                    timestamp=vx["timestamp"],
                )
    return result


def evaluate_exhaustion(
    df_1m: pd.DataFrame,
    symbol: str,
    *,
    config: ExhaustionConfig | None = None,
    daily_history: pd.DataFrame | None = None,
    cross_market_context: dict[str, Any] | None = None,
    previous_state: dict[str, Any] | None = None,
    data_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or PROFILES[symbol]
    if cfg.symbol != symbol:
        raise ValueError("Exhaustion profile does not match instrument.")
    metadata = data_context or {}
    notes = [
        "Kutatási / paper réteg; küszöbök még nem auditáltak.",
        "A BREAKDOWN_CONFIRMED helyi lefordulás, nem crash-előrejelzés.",
    ]
    mode = str(metadata.get("mode", "ANALYSIS_ONLY")).upper()
    result = dict(symbol=symbol, version=VERSION, state="UNKNOWN",
                  block_new_longs=cfg.block_unknown_longs, short_setup_candidate=False,
                  execution_ready=False, data_mode=mode, extension_checks=[],
                  exhaustion_checks=[], breakdown_checks=[], context_checks=[],
                  memory={}, notes=notes, data_quality="INSUFFICIENT",
                  hourly_extension_available=False, daily_extension_available=False,
                  state_since=None, asof=None, input_timestamp=None)
    if not cfg.enabled:
        result.update(state="DISABLED", block_new_longs=False, data_quality="DISABLED")
        return result
    df = _clean(df_1m)
    if df.empty:
        return result
    input_timestamp = df.index[-1]
    result["input_timestamp"] = input_timestamp
    complete = bool(metadata.get("last_bar_is_complete", False))
    cutoff = input_timestamp + pd.Timedelta(minutes=1) if complete else input_timestamp
    df = df[df.index < cutoff]
    bars = _closed_bars(df, 5, cutoff)
    hourly = _closed_bars(df, 60, cutoff)
    if bars.empty:
        notes.append("Nincs teljes, lezárt 5m gyertya.")
        return result
    if daily_history is None:
        daily = _daily_from_minutes(df, cutoff)
        notes.append("Napi OHLCV az inputból, 18:00-17:00 ET session szerint.")
    else:
        intraday_contract = metadata.get("contract_id", df_1m.attrs.get("contract_id"))
        daily_contract = daily_history.attrs.get("contract_id")
        if (intraday_contract is not None and daily_contract is not None
                and intraday_contract != daily_contract):
            raise ValueError("Daily and intraday contract_id mismatch.")
        daily = _clean(daily_history)
        daily = daily.loc[daily.index <= cutoff]
        notes.append("Külső napi OHLCV: index = lezárás ideje, azonos kontraktus szükséges.")
        if daily_history.attrs.get("timestamp_basis") == "SESSION_CLOSE_PLUS_ASSUMED_AVAILABILITY_LAG":
            notes[-1] = "Külső GC napi OHLCV: index = session-close + konzervatív elérhetőségi késleltetés."
            notes.append(
                "A napi close settlement is lehet; a tényleges publikálási idő nincs igazolva. "
                "A cache önmagában nem point-in-time auditadat."
            )
        if daily_contract is None or intraday_contract is None:
            notes.append("Napi/intraday kontraktusazonosság nincs metaadatból igazolva.")
    f = _features(df, bars, hourly, daily, cfg)
    cfg_key = repr(asdict(cfg))
    memory: dict[str, Any] = {}
    if previous_state and previous_state.get("last_bar") is not None:
        if (previous_state.get("symbol") != symbol
                or previous_state.get("config_key") != cfg_key):
            raise ValueError("previous_state symbol/config mismatch.")
        last = pd.Timestamp(previous_state["last_bar"])
        if last > bars.index[-1]:
            raise ValueError("Future previous_state is not allowed.")
        # If input does not contain the intermediate bars, replaying would skip events.
        if bars.index[0] > last + pd.Timedelta(minutes=5):
            notes.append("Korábbi állapot túl régi az inputhoz; újrajátszás az ablakból.")
        else:
            memory = dict(previous_state)
    start = max(cfg.min_5m_bars - 1, len(f) - cfg.replay_bars)
    for i in range(start, len(f)):
        gates = _gates(f, i, cfg)
        r = f.iloc[i]
        row = dict(atr=r.atr, close=r.close, open=r.open, high=r.high,
                   higher_low=r.higher_low,
                   peak=f.high.iloc[max(0, i - cfg.breakout_lookback + 1):i + 1].max())
        memory = advance_state(memory, f.index[i], row, gates, cfg)
    latest = f.iloc[-1]
    gates = _gates(f, len(f) - 1, cfg)
    state = memory.get("state", "UNKNOWN")
    fresh_5m = cutoff - f.index[-1] <= pd.Timedelta(minutes=10)
    sufficient = gates["sufficient"] and fresh_5m
    if metadata.get("roll_detected", False):
        sufficient = False
        notes.append("Jelzett kontraktus-roll: az új jelzéseket nem tekintjük igazoltnak.")
    # Retain a known risk latch, but never call incomplete data safe.
    if not sufficient and state not in ("EXHAUSTION_RISK", "BREAKDOWN_CONFIRMED"):
        state = "UNKNOWN"
    context = _context_checks(symbol, cross_market_context, cutoff)
    support = memory.get("support", np.nan)
    broken = memory.get("broken_at") is not None
    retest = memory.get("retest_at") is not None
    breakdown = [
        _check("Előre ismert, megerősített higher-low szint", support,
               "Két megerősített swing low közül a második magasabb", "5m",
               np.isfinite(support)),
        _check("Későbbi close a tárolt higher low alatt", broken,
               f"Close < szint - {cfg.break_buffer_atr:g} ATR(5m)", "5m", broken,
               available=bool(np.isfinite(support))),
        _check("Külön későbbi gyertyán sikertelen visszateszt", retest,
               "Szintérintés + bearish close a törött szint alatt", "5m", retest,
               available=broken),
        _check("Volatilitási tágulás (kiegészítő)", latest.atr_expansion,
               f">= {cfg.atr_expansion_ratio:g}x korábbi ATR-medián", "5m",
               latest.atr_expansion >= cfg.atr_expansion_ratio),
    ]
    if not np.isfinite(latest.d_extension):
        notes.append("Napi EMA/ATR extension: N/A (kevés vagy elavult lezárt napi adat).")
        notes.append(
            "A napi chart túlfeszítettsége nem értékelt. "
            "A NORMAL nem jelent többhetes tető- vagy crash-kockázatmentességet."
        )
    if not np.isfinite(latest.h_extension):
        notes.append("60m EMA/ATR extension: N/A (kevés vagy elavult teljes óra).")
    if not np.isfinite(latest.volume_ratio_slot):
        notes.append("Idősávos volumen: N/A (kevés korábbi session).")
    if memory.get("extension_until"):
        notes.append(f"Túlfeszítettség memóriája eddig él: {memory['extension_until']}.")
    if memory.get("expires_at"):
        notes.append(f"Kifáradási tiltás lejárata: {memory['expires_at']}.")
    if not np.isfinite(support):
        notes.append(
            "A breakdown kapu nem élesített: nincs kifáradási epizódhoz tárolt "
            "higher-low szint. Ez nem bizonyítja, hogy a charton nincs swing low."
        )
    block = (state in ("EXHAUSTION_RISK", "BREAKDOWN_CONFIRMED")
             or (state == "UNKNOWN" and cfg.block_unknown_longs)
             or (state == "PARABOLIC_EXTENSION" and cfg.block_parabolic_longs))
    if metadata.get("roll_detected", False):
        block = True
    # Only tagged fresh REALTIME data can be described as execution-ready.
    age_seconds = None
    clock = metadata.get("as_of")
    if clock is not None:
        clock = pd.to_datetime(clock, utc=True, errors="coerce")
        if pd.notna(clock):
            age_seconds = (clock - cutoff.tz_convert("UTC")).total_seconds()
    ready = bool(mode == "REALTIME" and age_seconds is not None
                 and 0 <= age_seconds <= 180 and sufficient)
    if mode != "REALTIME":
        notes.append(f"Adatmód: {mode}; elemzés, nem valós idejű belépési engedély.")
    elif not ready:
        notes.append("REALTIME címke mellett sincs friss, teljes adatokkal igazolt jel.")
        block = True
    memory.update(symbol=symbol, config_key=cfg_key)
    result.update(
        state=state, block_new_longs=bool(block),
        short_setup_candidate=bool(state == "BREAKDOWN_CONFIRMED" and sufficient),
        execution_ready=ready, age_seconds=age_seconds,
        extension_checks=gates["extension_checks"],
        exhaustion_checks=gates["exhaustion_checks"], breakdown_checks=breakdown,
        context_checks=context, memory=memory, data_quality="OK" if sufficient else "PARTIAL",
        state_since=memory.get("state_since"), asof=f.index[-1],
        hourly_extension_available=bool(np.isfinite(latest.h_extension)),
        daily_extension_available=bool(np.isfinite(latest.d_extension)),
        input_timestamp=input_timestamp,
        metrics=dict(atr_5m=float(latest.atr), vwap_extension_atr=float(latest.vwap_ext),
                     hourly_extension_atr=float(latest.h_extension),
                     daily_extension_atr=float(latest.d_extension),
                     hourly_acceleration=float(latest.h_acceleration),
                     hourly_slope_atr_per_hour=float(latest.h_slope),
                     hourly_acceleration_r2=float(latest.h_r2),
                     volume_ratio_slot=float(latest.volume_ratio_slot)),
    )
    return result


def format_exhaustion(result: dict[str, Any]) -> list[str]:
    daily_available = result.get("daily_extension_available", False)
    hourly_available = result.get("hourly_extension_available", False)
    scope = (
        "INTRADAY + NAPI EXTENSION (nem többnapos crash-modell)"
        if daily_available
        else "INTRADAY 5m/60m; NAPI VIZSGÁLAT HIÁNYOS"
        if hourly_available
        else "MAGASABB IDŐSÍKÚ VIZSGÁLAT HIÁNYOS"
    )
    lines = ["", "=" * 100,
             f"{result['symbol']} BLOW-OFF / PARABOLIKUS KIFÁRADÁS - v{VERSION}",
             "=" * 100, f"Állapot: {result['state']} | Adatminőség: {result['data_quality']}",
             f"Vizsgálati kör: {scope}",
             "Adatminőség: a szükséges core adatokra vonatkozik, nem minden feed teljességére.",
             f"Utolsó lezárt 5m időpont: {result.get('asof')}",
             f"Kockázati állapot kezdete: {result.get('state_since') or 'Nincs aktivált állapot'}",
             f"Új long kapu: {'TILTVA' if result['block_new_longs'] else 'NINCS ÚJ TILTÁS'}",
             f"Breakdown short környezet-jelölt: "
             f"{'IGEN' if result['short_setup_candidate'] else 'NEM'} (nem entry)",
             f"Valós idejű adatkapu: {'OK' if result['execution_ready'] else 'NEM IGAZOLT'}"]
    for key, title in [("extension_checks", "TÚLFESZÍTETTSÉG"),
                       ("exhaustion_checks", "KIFÁRADÁS"),
                       ("breakdown_checks", "STRUKTÚRATÖRÉS"),
                       ("context_checks", "KERESZTPIACI KONTEXTUS (nem önálló kapu)")]:
        lines += ["", title,
                  f"{'ÁLLAPOT':<8} | {'IDŐSÍK':<6} | {'MÉRT':<10} | FELTÉTEL / KÜSZÖB"]
        for c in result[key]:
            status = "N/A" if c["passed"] is None else "IGEN" if c["passed"] else "NEM"
            value = c["value"]
            if c["passed"] is None or value is None or (isinstance(value, (float, np.floating))
                                 and not np.isfinite(value)):
                text = "N/A"
            elif isinstance(value, (bool, np.bool_)):
                text = "IGEN" if value else "NEM"
            elif isinstance(value, (float, np.floating)):
                text = f"{value:.3f}"
            else:
                text = str(value)
            lines.append(f"{status:<8} | {c['timeframe']:<6} | {text:<10} | "
                         f"{c['label']}: {c['threshold']}")
    lines += [""] + [f"- {n}" for n in result["notes"]]
    return lines


def exhaustion_long_block(result: dict[str, Any]) -> dict[str, Any]:
    evidence = "; ".join(c["label"] for c in
                         result["exhaustion_checks"] if c["passed"] is True)
    data_gate_only = result["state"] == "NORMAL"
    return dict(
        trade_allowed=False,
        reason=("Új LONG tiltva: a friss REALTIME adatkapu nincs igazolva."
                if data_gate_only
                else f"Új LONG tiltva: exhaustion állapot = {result['state']}."),
        details=evidence or (
            f"Új long adatkapu zárt. Data quality={result.get('data_quality')}; "
            f"mode={result.get('data_mode')}; age={result.get('age_seconds')} mp."
        ),
        next_setup="Várj a kockázati állapot dokumentált resetjére, majd új setupra. "
                   "A long tiltás önmagában nem shortjel.",
        exhaustion_state=result["state"],
    )
