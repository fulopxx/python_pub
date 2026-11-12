"""Exploratory GC audit conclusions, not causal ablation or trading permission."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd

VERSION = "0.1.0"
QUESTIONS = {
    "Q1": "Jobb kimenetet mutatnak-e az engedett candidate eventek a tiltott eventeknél?",
    "Q2": "Milyen arányban enged belépési jelöltet a kapurendszer?",
    "Q3": "Kedvezőbb-e a long kimenet bull rezsimben, mint más ismert rezsimekben?",
    "Q4": "Kedvezőbb-e a short kimenet bear rezsimben, mint más ismert rezsimekben?",
    "Q5": "Kedvezőbbek-e a nem túlmély short helyzetek a VWAP alatti szélsőségeknél?",
    "Q6": "Kedvezőbbek-e a nem túlmagas long helyzetek a VWAP feletti szélsőségeknél?",
    "Q7": "Kedvezőbb-e shortnál az RSI2-túladottság kerülése, mint az extrém alacsony RSI2?",
    "Q8": "Jobb-e az irányos kimenet az ADX-küszöb felett, mint alatta?",
    "Q9": "Jobb-e az irányos kimenet a volumenküszöb felett, mint alatta?",
    "Q10L": "Bull rezsimben jobb-e a long kimenet bullish gap-reclaim mellett?",
    "Q10S": "Bear rezsimben jobb-e a short kimenet bearish gap-rejection mellett?",
    "Q11": "Eldönthető-e a jelenlegi forward/MAE adatokból a stop hasznossága?",
    "Q12": "Jobb-e a long kimenet, amikor a kifáradási kapu nem tilt?",
    "Q13L": "Jobb-e a long kimenet, amikor a fordulókapu nem tilt?",
    "Q13S": "Jobb-e a short kimenet, amikor a fordulókapu nem tilt?",
    "Q14": "Jobb-e az irányos kimenet LOW whipsaw mellett, mint HIGH mellett?",
}


@dataclass(frozen=True)
class ConclusionConfig:
    horizons: tuple[int, ...] = (15, 30, 60)
    min_samples: int = 20
    min_days_per_group: int = 5
    min_total_days: int = 10
    min_effect_atr: float = 0.05
    bootstrap_samples: int = 2000
    seed: int = 20261019
    adx_threshold: float = 30.0
    volume_threshold: float = 1.30
    rsi_threshold: float = 10.0
    vwap_threshold: float = 5.0
    stop_atr: float = 1.0

    def __post_init__(self):
        if (not self.horizons or len(set(self.horizons)) != len(self.horizons)
                or any(not isinstance(h, int) or h < 1 for h in self.horizons)):
            raise ValueError("Positive, distinct integer horizons required.")
        if min(self.min_samples, self.min_days_per_group, self.min_total_days,
               self.bootstrap_samples) < 1:
            raise ValueError("Positive sample counts required.")
        if any(not np.isfinite(x) or x <= 0 for x in (
                self.min_effect_atr, self.adx_threshold, self.volume_threshold,
                self.rsi_threshold, self.vwap_threshold, self.stop_atr)):
            raise ValueError("Positive finite thresholds required.")


def _num(frame, column):
    return pd.to_numeric(
        frame[column] if column in frame else pd.Series(np.nan, index=frame.index),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)


def _bool(frame, column):
    """Unknown/missing is not False. Handles read_csv string boolean columns."""
    if column not in frame:
        return pd.Series(pd.NA, index=frame.index, dtype="boolean")
    labels = frame[column].astype("string").str.strip().str.lower()
    return labels.map({
        "true": True, "1": True, "1.0": True, "igen": True,
        "false": False, "0": False, "0.0": False, "nem": False,
    }).astype("boolean")


def _code(frame, column):
    if column not in frame:
        return pd.Series("", index=frame.index, dtype="string")
    return (frame[column].astype("string").str.upper()
            .str.replace(r"[^A-Z0-9]", "", regex=True).fillna(""))


def _subset(frame, mask):
    return frame.loc[mask.fillna(False)].copy()


def _sample(frame, metric):
    result = pd.DataFrame({"value": _num(frame, metric)})
    if "timestamp" not in frame:
        result["day"] = pd.NaT
    else:
        # Explicit UTC parsing also handles mixed DST offsets in legacy CSVs.
        stamp = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        result["day"] = stamp.dt.tz_convert("America/New_York").dt.date
        result = result.loc[stamp.notna()]
    return result.dropna(subset=["value"])


def _stats(sample):
    values = sample["value"]
    return dict(n=len(values), days=int(sample["day"].nunique()),
                mean=float(values.mean()) if len(values) else np.nan,
                median=float(values.median()) if len(values) else np.nan,
                win_rate=float((values > 0).mean()) if len(values) else np.nan)


def _difference_ci(a, b, cfg):
    """Paired resampling of observed NY calendar days, not iid observations."""
    if a.empty or b.empty or a.day.isna().any() or b.day.isna().any():
        return np.nan, np.nan
    days = pd.Index(sorted(set(a.day) | set(b.day)))
    if (len(days) < cfg.min_total_days or a.day.nunique() < cfg.min_days_per_group
            or b.day.nunique() < cfg.min_days_per_group):
        return np.nan, np.nan
    def totals(sample):
        return sample.groupby("day")["value"].agg(["sum", "count"]).reindex(
            days, fill_value=0
        ).to_numpy(float)
    aa, bb = totals(a), totals(b)
    rng = np.random.default_rng(cfg.seed)
    # Bound temporary memory for long histories.
    parts = []
    for offset in range(0, cfg.bootstrap_samples, 256):
        draws = rng.integers(0, len(days),
                            size=(min(256, cfg.bootstrap_samples-offset), len(days)))
        av, bv = aa[draws].sum(axis=1), bb[draws].sum(axis=1)
        valid = (av[:, 1] > 0) & (bv[:, 1] > 0)
        parts.append(av[valid, 0]/av[valid, 1] - bv[valid, 0]/bv[valid, 1])
    diffs = np.concatenate(parts)
    if len(diffs) < 0.9 * cfg.bootstrap_samples:
        return np.nan, np.nan
    return tuple(float(x) for x in np.quantile(diffs, [0.025, 0.975]))


def _comparison(identifier, label, a, b, a_label, b_label, template, note, cfg):
    rows = []
    for horizon in cfg.horizons:
        metric = template.format(h=horizon)
        left, right = _sample(a, metric), _sample(b, metric)
        sa, sb = _stats(left), _stats(right)
        low, high = _difference_ci(left, right, cfg)
        delta = sa["mean"] - sb["mean"]
        median_delta = sa["median"] - sb["median"]
        win_delta = sa["win_rate"] - sb["win_rate"]
        enough = (min(sa["n"], sb["n"]) >= cfg.min_samples
                  and min(sa["days"], sb["days"]) >= cfg.min_days_per_group
                  and len(set(left.day.dropna()) | set(right.day.dropna()))
                  >= cfg.min_total_days)
        if metric not in a or metric not in b:
            verdict = "NINCS ADAT"
        elif not sa["n"] or not sb["n"]:
            verdict = "NINCS ADAT"
        elif not enough:
            verdict = "KIS MINTA"
        elif not np.isfinite(low) or not np.isfinite(high):
            verdict = "NEM IGAZOLT"
        elif (abs(delta) >= cfg.min_effect_atr
              and (delta * median_delta < 0 or delta * win_delta < 0)):
            verdict = "VEGYES MUTATÓK"
        elif delta >= cfg.min_effect_atr and low > 0:
            verdict = "HASZNOSNAK TŰNIK"
        elif delta <= -cfg.min_effect_atr and high < 0:
            verdict = "RONTHAT"
        else:
            verdict = "NEM IGAZOLT"
        rows.append(dict(
            id=identifier, parameter=label, horizon_bars=horizon, metric=metric,
            group_a=a_label, group_b=b_label, a_rows=len(a), b_rows=len(b),
            **{f"a_{k}": v for k, v in sa.items()},
            **{f"b_{k}": v for k, v in sb.items()},
            delta_mean_atr=delta, delta_median_atr=median_delta,
            delta_win_pp=win_delta*100, ci_low_atr=low, ci_high_atr=high,
            conclusion=verdict, comment=note,
        ))
    return rows


def build_conclusions(observations, events, config=None):
    """Return one row per question plus machine-readable per-horizon evidence."""
    cfg = config or ConclusionConfig()
    obs, ev = observations.copy().reset_index(drop=True), events.copy().reset_index(drop=True)
    bull = _subset(obs, _bool(obs, "is_bull_regime"))
    bear = _subset(obs, _bool(obs, "is_bear_regime"))
    directional = _subset(obs, _num(obs, "direction").isin([1, -1]))
    comparisons = []

    def add(identifier, label, a, b, a_label, b_label, template, note):
        comparisons.extend(_comparison(identifier, label, a, b, a_label, b_label,
                                       template, note, cfg))

    event_types = _code(ev, "event_type")
    add("Q1", "Összes belépési kapu",
        _subset(ev, event_types.eq("TRADECANDIDATE")),
        _subset(ev, event_types.eq("BLOCKEDTRADE")),
        "Engedett candidate event", "Tiltott event", "forward_directional_{h}m_atr",
        "A teljes kapurendszer összefüggése; nem egyetlen filter önálló hatása.")
    known = _bool(obs, "trade_allowed")
    add("Q3", "Bull rezsim / long",
        bull, _subset(obs, _bool(obs, "is_bull_regime").eq(False)),
        "Bull rezsim", "Ismert nem-bull rezsim", "forward_long_{h}m_atr",
        "Rezsimkörnyezet-összevetés; nem izolált szűrőhatás.")
    add("Q4", "Bear rezsim / short",
        bear, _subset(obs, _bool(obs, "is_bear_regime").eq(False)),
        "Bear rezsim", "Ismert nem-bear rezsim", "forward_short_{h}m_atr",
        "Rezsimkörnyezet-összevetés; nem izolált szűrőhatás.")

    def split_numeric(identifier, label, base, column, pass_mask,
                      a_label, b_label, metric, note=""):
        valid = _num(base, column).notna()
        add(identifier, label, _subset(base, valid & pass_mask),
            _subset(base, valid & ~pass_mask), a_label, b_label, metric,
            note or "Más kapuk is aktívak lehetnek; ez nem ki/be kapcsolásos ablation.")

    split_numeric("Q5", "VWAP short anti-chasing", bear, "vwap_distance_atr",
                  _num(bear, "vwap_distance_atr") >= -cfg.vwap_threshold,
                  f"VWAP >= -{cfg.vwap_threshold:g} ATR",
                  f"VWAP < -{cfg.vwap_threshold:g} ATR", "forward_short_{h}m_atr")
    split_numeric("Q6", "VWAP long anti-chasing", bull, "vwap_distance_atr",
                  _num(bull, "vwap_distance_atr") <= cfg.vwap_threshold,
                  f"VWAP <= {cfg.vwap_threshold:g} ATR",
                  f"VWAP > {cfg.vwap_threshold:g} ATR", "forward_long_{h}m_atr")
    split_numeric("Q7", "RSI2 short túladottság", bear, "rsi_2",
                  _num(bear, "rsi_2") >= cfg.rsi_threshold,
                  f"RSI2 >= {cfg.rsi_threshold:g}",
                  f"RSI2 < {cfg.rsi_threshold:g}", "forward_short_{h}m_atr")
    split_numeric("Q8", "ADX küszöb", directional, "adx_5m",
                  _num(directional, "adx_5m") > cfg.adx_threshold,
                  f"ADX > {cfg.adx_threshold:g}", f"ADX <= {cfg.adx_threshold:g}",
                  "forward_directional_{h}m_atr",
                  "Long/short rezsimek keveréke; összetételhatás lehetséges.")
    split_numeric("Q9", "Volumen küszöb", directional, "volume_ratio",
                  _num(directional, "volume_ratio") >= cfg.volume_threshold,
                  f"Volume >= {cfg.volume_threshold:g}x",
                  f"Volume < {cfg.volume_threshold:g}x", "forward_directional_{h}m_atr")
    for identifier, label, base, reaction, template in (
        ("Q10L", "NDOG/NWOG bullish reclaim", bull, "BULLISHRECLAIM", "forward_long_{h}m_atr"),
        ("Q10S", "NDOG/NWOG bearish rejection", bear, "BEARISHREJECTION", "forward_short_{h}m_atr"),
    ):
        ndog, nwog = _code(base, "ndog_reaction"), _code(base, "nwog_reaction")
        recognized = {"BULLISHRECLAIM", "BEARISHREJECTION", "NEARLEVEL", "NORECENTINTERACTION"}
        hit = ndog.eq(reaction) | nwog.eq(reaction)
        no_hit = ndog.isin(recognized) & nwog.isin(recognized) & ~hit
        add(identifier, label, _subset(base, hit), _subset(base, no_hit),
            "Megerősítő gap-reakció", "Mindkét gap ismert; nincs ilyen reakció", template,
            "Az N/A/UNKNOWN nem számít reakció nélküli kontrollnak.")
    for identifier, label, base, column, template in (
        ("Q12", "Kifáradási long kapu", bull, "exhaustion_block_new_longs", "forward_long_{h}m_atr"),
        ("Q13L", "Forduló long kapu", bull, "transition_block_new_longs", "forward_long_{h}m_atr"),
        ("Q13S", "Forduló short kapu", bear, "transition_block_new_shorts", "forward_short_{h}m_atr"),
    ):
        blocked = _bool(base, column)
        add(identifier, label, _subset(base, blocked.eq(False)), _subset(base, blocked),
            "A kapu nem tilt", "A kapu tilt", template,
            "A teljes kapuflagot vizsgálja; más szűrők hatását nem választja le.")
    whipsaw = _code(directional, "whipsaw_state")
    add("Q14", "Whipsaw alacsony / magas",
        _subset(directional, whipsaw.eq("LOW")),
        _subset(directional, whipsaw.eq("HIGH")),
        "LOW zaj", "HIGH zaj", "forward_directional_{h}m_atr",
        "Az ELEVATED és UNKNOWN kimarad; ez nem az összes whipsaw-tiltás ablationje.")

    evidence = pd.DataFrame(comparisons)
    rows = []
    comments = {
        "HASZNOSNAK TŰNIK": "A megtartott/megerősített csoport jobb a mintában.",
        "RONTHAT": "A megtartott csoport gyengébb; a filter ronthat a mintabeli kimeneten.",
        "NEM IGAZOLT": "Nincs elég következetes bizonyíték az előnyre vagy hátrányra.",
        "KIS MINTA": "Legalább egy csoportban kevés observation vagy különböző nap.",
        "NINCS ADAT": "Hiányzó mező vagy üres összehasonlító csoport.",
        "VEGYES EREDMÉNY": "Az időtávok vagy az átlag/medián/találati arány ellentmondanak.",
        "KORLÁTOZOTT ADAT": "Néhány időtávon hiányos/kicsi a minta; nincs összesített ítélet.",
    }
    for identifier, group in evidence.groupby("id", sort=False):
        states = set(group.conclusion)
        if states == {"NINCS ADAT"}:
            verdict = "NINCS ADAT"
        elif states == {"KIS MINTA"}:
            verdict = "KIS MINTA"
        elif states & {"NINCS ADAT", "KIS MINTA"}:
            verdict = "KORLÁTOZOTT ADAT"
        elif "VEGYES MUTATÓK" in states or {"HASZNOSNAK TŰNIK", "RONTHAT"} <= states:
            verdict = "VEGYES EREDMÉNY"
        elif states == {"HASZNOSNAK TŰNIK"}:
            verdict = "HASZNOSNAK TŰNIK"
        elif states == {"RONTHAT"}:
            verdict = "RONTHAT"
        elif states - {"NEM IGAZOLT"}:
            verdict = "VEGYES EREDMÉNY"
        else:
            verdict = "NEM IGAZOLT"
        first = group.iloc[0]
        row = dict(id=identifier, parameter=first.parameter, conclusion=verdict,
                   group_a=first.group_a, group_b=first.group_b,
                   comment=comments[verdict] + " " + first.comment)
        for _, item in group.iterrows():
            h = int(item.horizon_bars)
            for col in ("a_n", "b_n", "a_days", "b_days", "delta_mean_atr",
                        "ci_low_atr", "ci_high_atr"):
                row[f"{col}_{h}b"] = item[col]
        rows.append(row)

    allowed_count = int(known.eq(True).sum())
    known_count = int(known.notna().sum())
    rows.insert(1, dict(
        id="Q2", parameter="Belépési engedély aránya", conclusion="CSAK LEÍRÓ",
        group_a="Engedett observation", group_b="Ismert trade_allowed observation",
        comment=f"{allowed_count}/{known_count} = "
                f"{100*allowed_count/known_count:.1f}%. Az alacsony arány önmagában nem hiba."
                if known_count else "Nincs ismert trade_allowed adat; az arány nem számolható.",
    ))
    stop_parts = []
    for h in cfg.horizons:
        mae = _num(directional, f"mae_{h}m_atr").dropna()
        if len(mae):
            stop_parts.append(
                f"{h} bar: {(mae <= -cfg.stop_atr).mean()*100:.1f}% elérte a "
                f"-{cfg.stop_atr:g} ATR szintet (N={len(mae)})"
            )
    rows.insert(11, dict(
        id="Q11", parameter=f"Stop {cfg.stop_atr:g} ATR", conclusion="NEM ELDÖNTHETŐ",
        group_a="Irányos observationök", group_b="Nincs stop nélküli PnL-kontroll",
        comment=("; ".join(stop_parts) + ". " if stop_parts else "Nincs MAE-adat. ")
                + "A MAE-érintés nem fill/PnL; célár-sorrend, költség és slippage nélkül "
                  "a stop hasznossága nem dönthető el.",
    ))
    summary = pd.DataFrame(rows)
    for frame in (summary, evidence):
        frame["question"] = frame["id"].map(QUESTIONS)
        frame["analysis_version"] = VERSION
        frame["min_samples_per_group"] = cfg.min_samples
        frame["min_days_per_group"] = cfg.min_days_per_group
        frame["min_total_days"] = cfg.min_total_days
        frame["min_effect_atr"] = cfg.min_effect_atr
        frame["bootstrap_samples"] = cfg.bootstrap_samples
        frame["bootstrap_seed"] = cfg.seed
        frame["ci_method"] = "paired_NY_calendar_day_bootstrap_exploratory"
        frame["adx_threshold"] = cfg.adx_threshold
        frame["volume_threshold"] = cfg.volume_threshold
        frame["rsi_threshold"] = cfg.rsi_threshold
        frame["vwap_threshold"] = cfg.vwap_threshold
        frame["stop_atr"] = cfg.stop_atr
    return summary, evidence


def _fmt(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:+.3f}"


def _table(headers, rows, widths):
    border = "+" + "+".join("-"*(w+2) for w in widths) + "+"
    out = [border]
    for row in [headers] + list(rows):
        chunks = [textwrap.wrap(str(x), width=w, break_long_words=True,
                               break_on_hyphens=False) or [""]
                  for x, w in zip(row, widths)]
        for line in range(max(map(len, chunks))):
            out.append("| " + " | ".join(
                (chunk[line] if line < len(chunk) else "").ljust(width)
                for chunk, width in zip(chunks, widths)) + " |")
        out.append(border)
    return out


def render_report(summary, evidence, cfg, *, metadata=""):
    lines = [
        "GC AUDIT: AUTOMATIKUS PARAMÉTER- ÉS FILTERKONKLÚZIÓK",
        metadata,
        "Minden delta = A (megtartott/megerősített) mínusz B (kizárt/kontroll).",
        "Pozitív: A jobb; negatív: B jobb. A csoportnevek a részletes táblákban.",
        "A 15/30/60 címkék forward sorlépést jelentenek, nem garantált eltelt perceket.",
        f"Min.: {cfg.min_samples} véges minta és {cfg.min_days_per_group} NY naptári nap"
        f" csoportonként; összesen {cfg.min_total_days} nap.",
        f"Szabályalapú döntési küszöb: {cfg.min_effect_atr:g} ATR; "
        f"{cfg.bootstrap_samples} napblokk-bootstrap, exploratív 95%-os intervallum.",
        f"Vizsgált küszöbök: ADX>{cfg.adx_threshold:g}; volumen>={cfg.volume_threshold:g}x; "
        f"RSI2>={cfg.rsi_threshold:g}; VWAP +/-{cfg.vwap_threshold:g} ATR; "
        f"stop={cfg.stop_atr:g} ATR. Az összevetés ezeket a besorolásokat teszteli.",
        "A 'HASZNOSNAK TŰNIK' / 'RONTHAT' minden vizsgált időtávon következetes "
        "előjelet, nullát kizáró CI-t és nem ellentmondó mediánt/találati arányt kíván.",
        "Nem okozati ablation, nem költség utáni PnL, nem többszörös tesztelésre "
        "korrigált vagy out-of-sample bizonyíték. A napok közötti függés is megmaradhat.",
        "Semmit nem kapcsol ki/be, és nem módosít paramétert a motorban.",
        "",
    ]
    table_rows = []
    for _, r in summary.iterrows():
        a, b = r.get("a_n_15b", np.nan), r.get("b_n_15b", np.nan)
        counts = f"{int(a)}/{int(b)}" if pd.notna(a) and pd.notna(b) else "N/A"
        table_rows.append([r.id, r.parameter, r.conclusion, counts,
                           _fmt(r.get("delta_mean_atr_15b", np.nan)),
                           _fmt(r.get("delta_mean_atr_30b", np.nan)), r.comment])
    summary_table = _table(
        ["ID", "Paraméter / filter", "Konklúzió", "N15 A/B", "D15 ATR", "D30 ATR",
         "Megjegyzés"], table_rows, [5, 26, 19, 11, 8, 8, 53])
    lines += ["", "RÉSZLETES ELLENŐRZŐ KÉRDÉSEK ÉS ÖSSZEHASONLÍTÁSOK"]
    for identifier, group in evidence.groupby("id", sort=False):
        item = group.iloc[0]
        lines += ["", f"{identifier} | {item.parameter}", QUESTIONS[identifier],
                  f"A = {item.group_a}; B = {item.group_b}", item.comment]
        rows = []
        for _, r in group.iterrows():
            rows.append([
                f"{int(r.horizon_bars)} bar", f"{r.a_n}/{r.b_n}",
                f"{r.a_days}/{r.b_days}", f"{_fmt(r.a_mean)}/{_fmt(r.b_mean)}",
                _fmt(r.delta_mean_atr), f"[{_fmt(r.ci_low_atr)}; {_fmt(r.ci_high_atr)}]",
                _fmt(r.delta_median_atr), _fmt(r.delta_win_pp), r.conclusion,
            ])
        lines.extend(_table(
            ["Horizont", "N A/B", "Nap A/B", "Átlag ATR A/B", "D átlag", "95% CI ATR",
             "D medián", "D találat pp", "Konklúzió"], rows,
            [8, 11, 9, 17, 9, 20, 9, 12, 19]))
    lines += ["", "LEÍRÓ KÉRDÉSEK"]
    for _, r in summary[summary.id.isin(["Q2", "Q11"])].iterrows():
        lines += [f"{r.id} | {r.parameter} | {r.conclusion}", QUESTIONS[r.id], r.comment]
    lines += ["", "VÉGSŐ ÖSSZEGZŐ TÁBLÁZAT"] + summary_table
    return "\n".join(lines) + "\n"


def save_conclusion_outputs(summary, evidence, cfg, output_dir=".", *, metadata="", preamble=""):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    text = preamble + render_report(summary, evidence, cfg, metadata=metadata)
    paths = [
        root/"gc_audit_conclusions.csv", root/"gc_audit_comparisons.csv",
        root/"gc_audit_report.txt", root/"gc_audit_report.md",
    ]
    summary.to_csv(paths[0], index=False, encoding="utf-8-sig")
    evidence.to_csv(paths[1], index=False, encoding="utf-8-sig")
    paths[2].write_text(text, encoding="utf-8")
    # Same renderer and data as console/TXT, no separately inferred judgments.
    paths[3].write_text("# GC audit: paraméter- és filterkonklúziók\n\n"
                        "Azonos tartalom a konzolos riporttal. A minősítések csak "
                        "mintabeli, exploratív összefüggéseket jelentenek.\n\n"
                        "```text\n" + text + "```\n", encoding="utf-8")
    return text, paths
