"""Timestamped VX-monitor context + index price action, research/paper only.

Does not infer participant positioning, generate orders, or choose direction
from a VIX headline. A missing monitor file never implies NORMAL.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def _stamp(value: Any) -> pd.Timestamp:
    if value is None:
        raise ValueError("Missing timestamp.")
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Timezone-aware timestamp required.")
    return stamp.tz_convert("UTC")


def _number(value: Any) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def validate_vx_context(packet: dict[str, Any] | None, analysis: pd.Timestamp,
                        wall: Any = None) -> dict[str, Any]:
    result = dict(available=False, quality="MISSING", state="N/A", asof=None,
                  age_to_index_minutes=None, wall_age_minutes=None,
                  stress_candidate=None, stress_confirmed=None,
                  confirmation_count=None, metrics={}, reason="Nincs VX-monitor input.")
    if not packet:
        return result
    try:
        if not isinstance(packet, dict):
            raise ValueError("VX input must be a JSON object.")
        result["state"] = packet.get("candidate_state", "N/A")
        asof, observed = _stamp(packet.get("asof")), _stamp(packet.get("observed_at"))
        analysis = _stamp(analysis)
        clock = _stamp(wall) if wall is not None else analysis
        result.update(asof=asof, age_to_index_minutes=(analysis-asof).total_seconds()/60,
                      wall_age_minutes=(clock-asof).total_seconds()/60)
        if packet.get("data_mode") == "SYNTHETIC_DEMO_NOT_MARKET_DATA":
            raise ValueError("Szintetikus demo nem piaci kontextus.")
        if observed < asof or observed > clock:
            raise ValueError("A monitor publikálási/futási időpontja nem használható.")
        if asof > analysis:
            result.update(quality="FUTURE_INPUT",
                          reason="A VX forrásadata későbbi az index elemzési idejénél.")
            return result
        if result["age_to_index_minutes"] > 15 or result["wall_age_minutes"] > 45:
            result.update(quality="STALE_INPUT", reason="A VX kontextus túl régi az indexhez vagy a faliórához.")
            return result
        sources = packet.get("sources", {})
        first, second = sources.get("F1", {}), sources.get("F2", {})
        if (first.get("tradingClass") != "VX" or second.get("tradingClass") != "VX"
                or int(first.get("conId", 0)) <= 0 or int(second.get("conId", 0)) <= 0
                or int(first["conId"]) == int(second["conId"])):
            raise ValueError("Két külön validált havi VX futures szükséges.")
        first_expiry = pd.Timestamp(first.get("expiry"))
        second_expiry = pd.Timestamp(second.get("expiry"))
        if (pd.isna(first_expiry) or pd.isna(second_expiry)
                or not asof.date() < first_expiry.date() < second_expiry.date()):
            raise ValueError("Lejárt vagy hibás sorrendű VX lejáratok.")
        if packet.get("candidate_state") == "DATA_UNAVAILABLE":
            result.update(quality="CORE_UNAVAILABLE", reason="A VX-monitor core feedje nem értékelhető.")
            return result
        if packet.get("candidate_state") not in (
                "NORMAL", "WATCH", "STRESS_BUILDING", "BACKWARDATION_STRESS"):
            raise ValueError("Ismeretlen VX-monitor állapot.")
        metrics = packet.get("metrics", {})
        required = ("f1_price", "f2_price", "curve_pct", "f1_change_15m_pct",
                    "f1_change_30m_pct", "curve_change_30m_pp", "front_outperformance_30m_pp")
        numbers = {key: _number(metrics.get(key)) for key in required}
        if not all(np.isfinite(x) for x in numbers.values()):
            raise ValueError("Hiányos VX core metrikák.")
        if min(numbers["f1_price"], numbers["f2_price"]) <= 0:
            raise ValueError("Érvénytelen VX ár.")
        computed_curve = 100*(numbers["f2_price"]/numbers["f1_price"]-1)
        if abs(computed_curve-numbers["curve_pct"]) > .05:
            raise ValueError("A VX árak és a görbemérték nem konzisztensek.")
        count = _number(packet.get("confirmation_count", 0))
        if not np.isfinite(count) or count < 0 or int(count) != count:
            raise ValueError("Érvénytelen megerősítési számláló.")
        candidate = packet["candidate_state"] in ("STRESS_BUILDING", "BACKWARDATION_STRESS")
        result.update(available=True, quality="OK", metrics=numbers,
                      confirmation_count=int(count), stress_candidate=candidate,
                      stress_confirmed=candidate and count >= 2,
                      reason="Timestampolt VX-monitor, legfeljebb 15 perc indexeltéréssel; két megerősítés szükséges.")
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        result.update(quality="INVALID_OR_UNVERIFIED", reason=str(error))
    return result


def evaluate_index_short_context(snapshot: dict[str, Any], packet: dict[str, Any] | None,
                                 *, analysis_timestamp: pd.Timestamp, wall_clock: Any,
                                 config: Any) -> dict[str, Any]:
    vx = validate_vx_context(packet, analysis_timestamp, wall_clock)
    transition = snapshot.get("transition_risk", {})
    exhaustion = snapshot.get("exhaustion", {})
    below = bool(snapshot["close"] < snapshot["vwap"])
    bearish_di = bool(snapshot["minus_di_5m"] > snapshot["plus_di_5m"])
    adx = bool(snapshot["adx_5m"] > config.adx_trend_min)
    volume = bool(snapshot["volume_ratio"] >= config.volume_expansion_ratio)
    oversold = bool(snapshot["rsi_2"] < config.rsi2_extreme_oversold
                    or snapshot["rsi_5"] < config.rsi5_extreme_oversold)
    extension = bool(snapshot["vwap_distance_atr"] < config.max_short_extension_below_vwap_atr)
    bullish_gap = any(snapshot["gap_context"][key].get("reaction") == "BULLISH_RECLAIM"
                      for key in ("ndog_reaction", "nwog_reaction"))
    structure = any(tf.get("confirmed_direction") == -1 and tf.get("active_turn", False)
                    for tf in transition.get("timeframes", {}).values())
    structure = structure or bool(exhaustion.get("short_setup_candidate", False))
    blocked = bool(transition.get("block_new_shorts", False) or oversold or extension
                   or bullish_gap or snapshot.get("data_quality") != "OK")
    price_confirmed = below and bearish_di and adx and volume and structure
    confirmed = bool(vx["stress_confirmed"] and price_confirmed)
    if not vx["available"]:
        state = "VX_CONTEXT_NA"
    elif confirmed and blocked:
        state = "SHORT_ENVIRONMENT_BUT_ENTRY_BLOCKED"
    elif confirmed:
        state = "PRICE_AND_VX_STRESS_CONFIRMED"
    elif vx["stress_candidate"]:
        state = "VX_STRESS_WAIT_FOR_PRICE_OR_CONFIRMATION"
    elif price_confirmed:
        state = "BEARISH_PRICE_WITHOUT_VX_STRESS"
    else:
        state = "NO_COMBINED_SHORT_ENVIRONMENT"
    return dict(
        state=state, vx=vx, environment_candidate=confirmed,
        paper_setup_candidate=confirmed and not blocked, execution_allowed=False,
        short_covering_confirmed=False,
        checks={
            "VX input időbélyeg és adatminőség rendben": vx["available"],
            "VX stresszjelölt jelen van": vx["stress_candidate"],
            "VX stressz legalább két forrásmegfigyeléssel": vx["stress_confirmed"],
            "Index a session VWAP alatt": below,
            "Index -DI dominál +DI felett": bearish_di,
            f"Index ADX(5m) > {config.adx_trend_min:g}": adx,
            f"Index volumen >= {config.volume_expansion_ratio:g}x": volume,
            "Megerősített bearish HTF retest vagy exhaustion breakdown": structure,
            "Nincs transition/whipsaw short tiltás": not transition.get("block_new_shorts", False),
            "Index nem extrém túladott": not oversold,
            "Index nincs túl messze a VWAP alatt": not extension,
            "Nincs bullish gap reclaim": not bullish_gap,
            "Index core adatminőség rendben": snapshot.get("data_quality") == "OK",
        },
        notes=[
            "VX-stressz és indexgyengülés együtt vizsgált környezet, nem igazolt oksági kapcsolat.",
            "VIX-short kitettség kivétele / szereplői shortzárás: NEM IGAZOLT.",
            "Napi OI és szereplői/COT input: N/A; ár/volumenből nem helyettesítjük.",
            "Kutatási/paper checklist; nem crash-valószínűség és nem automatikus entry.",
            "Az együttmozgó checklist-sorok nem független statisztikai bizonyítékok.",
        ])


def format_index_short_context(result: dict[str, Any], symbol: str) -> list[str]:
    labels = {
        "VX_CONTEXT_NA": "VX KONFIRMÁCIÓ NEM ÉRTÉKELHETŐ",
        "SHORT_ENVIRONMENT_BUT_ENTRY_BLOCKED": "ÁR + VX STRESSZ IGAZOLT, DE ÚJ BELÉPÉS TILTVA",
        "PRICE_AND_VX_STRESS_CONFIRMED": "ÁR + VX STRESSZ EGYÜTT MEGERŐSÍTVE (paper)",
        "VX_STRESS_WAIT_FOR_PRICE_OR_CONFIRMATION": "VX STRESSZ LÁTSZIK, INDEXSTRUKTÚRA / TARTÓSSÁG MÉG KELL",
        "BEARISH_PRICE_WITHOUT_VX_STRESS": "BEARISH ÁRSTRUKTÚRA, VX STRESSZ NÉLKÜL",
        "NO_COMBINED_SHORT_ENVIRONMENT": "NINCS EGYÜTTES SHORT-KÖRNYEZET",
    }
    vx = result["vx"]
    def age_text(value: Any) -> str:
        return f"{value:.1f} perc" if value is not None and np.isfinite(value) else "N/A"
    lines = ["", "="*100, f"{symbol} SHORT-KÖRNYEZET / VX FUTURES STRESSZ",
             labels.get(result["state"], result["state"]),
             f"VX adatstátusz: {vx['quality']} | Monitor állapot: {vx['state']}",
             f"VX forrás-idő: {vx['asof'] or 'N/A'} | Megerősítések: {vx['confirmation_count'] if vx['confirmation_count'] is not None else 'N/A'}/2",
             f"VX indexeltérés: {age_text(vx['age_to_index_minutes'])} | Falióra szerinti kor: {age_text(vx['wall_age_minutes'])}",
             f"Adatmagyarázat: {vx['reason']}",
             "A VIX/VIX3M indexarány és a VX F1/F2 futures-görbe külön mérés.",
             "VIX = széles piaci stressz; NQ-nál nem saját Nasdaq-volatilitás."]
    metric_labels = {
        "f1_price": ("Első havi VX futures ára", "pont"),
        "f2_price": ("Második havi VX futures ára", "pont"),
        "curve_pct": ("VX F1/F2 görbe: pozitív=contango, negatív=backwardation", "%"),
        "f1_change_15m_pct": ("Első VX futures 15m változása", "%"),
        "f1_change_30m_pct": ("Első VX futures 30m változása", "%"),
        "curve_change_30m_pp": ("Görbe 30m változása: negatív=kompresszió", "százalékpont"),
        "front_outperformance_30m_pp": ("Első futures relatív erősödése a másodikhoz, 30m", "százalékpont"),
    }
    for key, value in vx["metrics"].items():
        label, unit = metric_labels.get(key, (key, ""))
        lines.append(f"- {label}: {value:.4f} {unit}")
    for name, flag in result["checks"].items():
        lines.append(f"[{'N/A' if flag is None else 'TELJESÜL' if flag else 'NEM TELJESÜL'}] {name}")
    lines += [f"- {note}" for note in result["notes"]]
    lines += [f"Paper setup jelölt: {'IGEN' if result['paper_setup_candidate'] else 'NEM'}",
              "Valós végrehajtási engedély: NEM; a delayed feed nem élő belépési jel."]
    return lines
