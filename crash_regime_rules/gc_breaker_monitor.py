"""GC 4h automatic breaker view of the shared ICT-inspired structure engine."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from market_structure import DEFAULT_STRUCTURE, evaluate_market_structure

VERSION = "0.1.1"


def summarize_gc_breakers(structure: dict) -> dict:
    tf = structure.get("timeframes", {}).get("4h", {})
    zones = [dict(z) for z in tf.get("zones", []) if z["kind"] == "BREAKER"]
    candidates = [dict(z) for z in tf.get("breaker_candidates", [])]
    if structure.get("state") == "DISABLED":
        state = "DISABLED"
    elif structure.get("state") == "DATA_UNAVAILABLE":
        state = "DATA_UNAVAILABLE"
    elif tf.get("quality") in (None, "INSUFFICIENT_HISTORY"):
        state = "INSUFFICIENT_4H_HISTORY"
    elif any(z["lifecycle"] == "ACTIVE" for z in zones):
        state = "ACTIVE_BREAKER_ZONES"
    elif any(z["lifecycle"] == "ACTIVE" for z in candidates):
        state = "WEAKER_BREAKER_CANDIDATES_ONLY"
    else:
        state = "NO_ACTIVE_QUALIFIED_BREAKER"
    return dict(version=VERSION, state=state, asof=structure.get("asof"), timeframe="4h",
                data_mode=structure.get("data_mode", "UNVERIFIED"),
                zones=zones, breaker_candidates=candidates,
                active_candidates=sum(z["lifecycle"] == "ACTIVE" for z in candidates),
                active_bullish=sum(z["lifecycle"] == "ACTIVE" and z["direction"] == 1 for z in zones),
                active_bearish=sum(z["lifecycle"] == "ACTIVE" and z["direction"] == -1 for z in zones),
                data_quality=tf.get("quality", "N/A"), source_quality=structure.get("source_quality"),
                execution_allowed=False, gate_effect="OBSERVATION_ONLY",
                notes=["BREAKER = szabályokkal kvalifikált failed OB, nem kézzel rajzolt ICT-zóna azonossága.",
                       "A zóna idősíkja 4h, a reakcióé 5m. Nem automatikus entry.",
                       "A gyengébb jelölt külön hipotézis; rövidített eredetgyertya nem teljes 4h bar.",
                       "Nincs felismert zóna nem egyenlő a charton lévő zóna hiányával."])


def evaluate_gc_breaker_monitor(df_1m: pd.DataFrame, *, data_context=None, config=None) -> dict:
    cfg = replace(config or DEFAULT_STRUCTURE, timeframes_minutes=(240,))
    return summarize_gc_breakers(evaluate_market_structure(df_1m, config=cfg, data_context=data_context))


def format_gc_breaker_monitor(result: dict) -> list[str]:
    lines = ["", "="*100, "GC AUTOMATIKUS 4H BREAKER-ZÓNA MONITOR", "="*100,
             f"Állapot: {result['state']} | végpont: {result['asof']} | adat: {result['data_quality']}",
             f"Adatmód: {result['data_mode']} | forráskor ellenőrzése: {result['source_quality']}",
             f"Aktív bullish: {result['active_bullish']} | aktív bearish: {result['active_bearish']}"]
    if not result["zones"]:
        lines.append("Nincs a meghatározott feltételekkel kvalifikált 4h breaker; ez nem VWAP-retest.")
    lines.append(f"Külön gyengébb aktív jelöltek: {result.get('active_candidates', 0)}")
    for z in sorted(result.get("breaker_candidates", []),
                    key=lambda z: (z["lifecycle"] != "ACTIVE", z["distance_usd"]))[:6]:
        ev = z["evidence"]
        lines += [
            f"GYENGÉBB {'BEARISH' if z['direction'] == -1 else 'BULLISH'} BREAKER-JELÖLT: "
            f"{z['low']:.2f} - {z['high']:.2f} | {z['lifecycle']}",
            f"  Eredet: {ev['source_start']} | {ev['source_scheduled_minutes']} perc; "
            f"{'teljes HTF' if ev['source_is_full_timeframe'] else 'RÖVIDÍTETT SESSION-VÉGI GYERTYA'}",
            f"  Ismert: {z['known_at']} | nem kvalifikált failed OB; OBSERVATION_ONLY",
            f"  5m helyzet: {z['current_position']} | {z['current_position_at'] or 'N/A'} | "
            f"reakció: {z['last_reaction']} | {z['last_reaction_at'] or 'Nincs'}",
            f"  Aktuális érintés: {z['current_interaction']} | reakcióadat: {z['reaction_quality']} | "
            f"végponthoz friss: {z['reaction_fresh']}"]
    for z in result["zones"][:6]:
        side = "BULLISH BREAKER / TÁMASZ-JELÖLT" if z["direction"] == 1 else "BEARISH BREAKER / ELLENÁLLÁS-JELÖLT"
        lines += [f"{side}: {z['low']:.2f} - {z['high']:.2f} | {z['lifecycle']}",
                  f"  Ismert ettől: {z['known_at']} | eredeti OB gyertya: {z['origin_at']}",
                  f"  Utolsó 5m helyzet: {z['current_position']} | {z['current_position_at'] or 'N/A'} | "
                  f"érinti: {'IGEN' if z['current_interaction'] else 'NEM / NEM IGAZOLT'}",
                  f"  5m reakció: {z['last_reaction']} | {z['last_reaction_at'] or 'Nincs'} | "
                  f"esemény friss: {'IGEN' if z['reaction_fresh'] else 'NEM'} | "
                  f"minőség: {z['reaction_quality']}",
                  f"  Érintő barok: {z['touch_count']} | külön visszatérések: {z['visit_count']} | "
                  f"{z['mitigation_status']} | ismételt retest: {z['repeated_retest_warning']}",
                  f"  4h invalidáció: {z['invalidated_at'] or 'Nincs igazolt'} | "
                  f"lejárat: {z['expires_at']}"]
    lines += [f"- {n}" for n in result["notes"]]
    return lines


def export_gc_breaker_monitor(result: dict, folder: str | Path) -> None:
    """Atomic strict JSON. Includes evidence timestamps; no orders or accounts."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=folder,
                                         suffix=".tmp", delete=False) as stream:
            tmp = Path(stream.name)
            stream.write(payload)
        os.replace(tmp, folder/"gc_breaker_latest.json")
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()
    pd.json_normalize(result["zones"]).to_csv(folder/"gc_breaker_zones.csv", index=False)
    pd.json_normalize(result.get("breaker_candidates", [])).to_csv(
        folder/"gc_breaker_candidates.csv", index=False)


def demo_htf_bars() -> pd.DataFrame:
    """Synthetic 100-price example, not a market quote or profitability test."""
    from market_structure import _expected_ends
    cfg = DEFAULT_STRUCTURE
    ends = _expected_ends(pd.Timestamp("2026-10-05 22:00", tz=cfg.timezone),
                          pd.Timestamp("2026-10-14 14:00", tz=cfg.timezone), 240, cfg)[:26]
    rows = []
    for i in range(18):
        close = 100+.20*np.sin(i*1.2)
        rows.append([close-.1, close+.5, close-.5, close, 100])
    rows += [
        [100.2, 102., 100., 101.7, 100],  # last up candle => later bearish OB
        [101.7, 101.9, 95., 96., 100],   # displaced bearish swing break
        [96., 97., 94., 95., 100],       # confirms bearish FVG + OB
        [95., 96., 93., 94., 100],
        [94., 98., 94., 97., 100],
        [97., 100., 95., 99., 100],      # confirms swing low at 93
        [97., 98., 92., 94., 100],       # sellside sweep and reclaim of 93
        [94., 105., 94., 104.5, 100],    # displaced bullish shift, fails old OB
    ]
    b = pd.DataFrame(rows, index=ends, columns=["open", "high", "low", "close", "volume"])
    b["start"] = b.index-pd.Timedelta(hours=4)
    b["scheduled_minutes"] = 240
    return b


def demo_history() -> pd.DataFrame:
    b = demo_htf_bars()
    end = b.index[-1]
    all_minutes = pd.date_range(b.start.iloc[0], end+pd.Timedelta(minutes=29), freq="min")
    day, hour = all_minutes.weekday, all_minutes.hour
    active = ((hour >= 18) & np.isin(day, [6, 0, 1, 2, 3])) | (
        (hour < 17) & np.isin(day, [0, 1, 2, 3, 4]))
    index = all_minutes[active]
    out = pd.DataFrame(index=index, columns=["open", "high", "low", "close", "volume"], dtype=float)
    for stamp, row in b.iterrows():
        idx = pd.date_range(stamp-pd.Timedelta(hours=4), stamp-pd.Timedelta(minutes=1), freq="min")
        edges = np.linspace(row.open, row.close, 241)
        opens, closes = edges[:-1], edges[1:]
        highs = np.minimum(row.high, np.maximum(opens, closes)+.4)
        lows = np.maximum(row.low, np.minimum(opens, closes)-.4)
        highs[60], lows[180] = row.high, row.low
        out.loc[idx] = np.column_stack([opens, highs, lows, closes, np.full(240, 100.)])
    out["close"] = out.close.ffill()
    missing = out.open.isna()
    out.loc[missing, "open"] = out.loc[missing, "close"]
    out.loc[missing, "high"] = out.loc[missing, "close"]+.4
    out.loc[missing, "low"] = out.loc[missing, "close"]-.4
    out.loc[missing, "volume"] = 100
    reactions = [
        [104.5, 104.8, 103.3, 103.5],
        [103.5, 103.7, 101.5, 102.5],
        [101.8, 103.5, 101.6, 103.4],
        [103.4, 103.8, 103.1, 103.5],
        [103.5, 104.0, 103.2, 103.7],
        [103.7, 104.0, 103.3, 103.6],
    ]
    for i, (op, hi, lo, cl) in enumerate(reactions):
        idx = pd.date_range(end+pd.Timedelta(minutes=5*i), periods=5, freq="min")
        edges = np.linspace(op, cl, 6)
        highs, lows = np.maximum(edges[:-1], edges[1:])+.1, np.minimum(edges[:-1], edges[1:])-.1
        highs[1], lows[3] = hi, lo
        out.loc[idx] = np.column_stack([edges[:-1], highs, lows, edges[1:], np.full(5, 100.)])
    return out
