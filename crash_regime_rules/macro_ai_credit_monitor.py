"""Console macro/AI-credit thesis monitor. Research only; no TWS, orders or positions.

Default: attempts a new SEC filing/facts + BLS fetch on every run, while keeping
the separately dated provider earnings/consensus snapshot. --offline disables HTTP.
No consensus, AI attribution, concealment proof or GDP contribution is invented.
Stdlib only, Python 3.10+. All rate inputs use PERCENT, contributions use pp.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import tempfile
import os
from typing import Any

VERSION = "0.3.3"
BASE = Path(__file__).resolve().parent


def number(x):
    if x is None or isinstance(x, bool):
        return None
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError):
        return None


def rate(x, name):
    y = number(x)
    if y is None or y <= -100:
        raise ValueError(f"{name}: véges, -100%-nál nagyobb érték kell.")
    return y / 100


def timestamp(text):
    t = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError("Timezone-os ISO időpont szükséges.")
    return t.astimezone(timezone.utc)


def annual_to_quarter(pct):
    return ((1 + rate(pct, "annual"))**.25 - 1)*100


def quarter_to_annual(pct):
    return ((1 + rate(pct, "quarter"))**4 - 1)*100


def gdp_counterfactual(nominal_pct, official_deflator_pct, alt_deflator_pct,
                       ai_real_contribution_pp, basis="ANNUAL"):
    """All rates same basis: ANNUAL, YOY or QOQ_SAAR. No CPI substitution.

    AI pp is a real-growth contribution under the baseline price system.
    Reprice the SAME no-AI nominal counterfactual with the alternative deflator.
    Chained-volume component subtraction is an approximation, not a BEA account.
    """
    if basis not in ("ANNUAL", "YOY", "QOQ_SAAR"):
        raise ValueError("GDP időalap nem igazolt vagy nem támogatott.")
    n = rate(nominal_pct, "nominal")
    d = rate(official_deflator_pct, "baseline deflator")
    alt = rate(alt_deflator_pct, "alternative deflator")
    ai = number(ai_real_contribution_pp)
    if ai is None or ai < 0:
        raise ValueError("AI hozzájárulás: nemnegatív, véges százalékpont.")
    real_factor = (1+n)/(1+d)
    no_ai_factor = real_factor-ai/100
    if no_ai_factor <= 0:
        raise ValueError("Az AI feltevés nempozitív GDP-szintet eredményez.")
    return dict(
        basis=basis, reported_real_pct=(real_factor-1)*100,
        no_ai_baseline_pct=(no_ai_factor-1)*100,
        alternative_real_pct=((1+n)/(1+alt)-1)*100,
        no_ai_alternative_pct=(no_ai_factor*(1+d)/(1+alt)-1)*100,
        required_ai_pp_for_minus_one=(real_factor-.99)*100,
        nominal_pct_for_minus_ten_at_alt_deflator=(.90*(1+alt)-1)*100,
        method="SCENARIO_APPROXIMATION_NOT_CAUSAL_GDP_ESTIMATE",
    )


def equity_shock(eps_change_pct, multiple_change_pct, reference_level=None):
    eps, multiple = rate(eps_change_pct, "EPS"), rate(multiple_change_pct, "multiple")
    factor = (1+eps)*(1+multiple)
    level = number(reference_level)
    if level is not None and level <= 0:
        raise ValueError("Pozitív referenciaár kell.")
    return dict(change_pct=(factor-1)*100,
                scenario_level=level*factor if level is not None else None,
                fair_value_identified=False, horizon="UNSPECIFIED_NOT_NEXT_MONDAY")


def gold_shock(real_yield_change_pp, dollar_change_pct, safe_haven_pct,
               liquidity_pct, beta_real_yield=-.08, beta_dollar=-1.0):
    """Uncalibrated log-factor stress, NOT an estimated gold forecasting model."""
    values = [real_yield_change_pp, dollar_change_pct, safe_haven_pct,
              liquidity_pct, beta_real_yield, beta_dollar]
    if any(number(v) is None for v in values):
        raise ValueError("Véges gold stresszparaméterek szükségesek.")
    if abs(real_yield_change_pp) > 10 or abs(dollar_change_pct) > 100:
        raise ValueError("Gold stresszparaméter tartományon kívül.")
    log_return = (beta_real_yield*real_yield_change_pp +
                  beta_dollar*dollar_change_pct/100 +
                  safe_haven_pct/100 + liquidity_pct/100)
    if abs(log_return) > 10:
        raise ValueError("Extrém loghozam nem támogatott.")
    return dict(change_pct=math.expm1(log_return)*100, calibrated=False,
                method="ILLUSTRATIVE_LOG_FACTOR_STRESS")


def credit_loss(exposure_usd, pd_pct, lgd_pct):
    ead, pd, lgd = number(exposure_usd), number(pd_pct), number(lgd_pct)
    if ead is None or ead < 0 or pd is None or lgd is None or not 0 <= pd <= 100 or not 0 <= lgd <= 100:
        raise ValueError("EAD >= 0; PD és LGD 0..100% kell.")
    return ead * pd/100 * lgd/100


def surprise(actual, consensus, actual_period, consensus_period,
             actual_unit, consensus_unit, actual_basis=None, consensus_basis=None):
    a, c = number(actual), number(consensus)
    if a is None or c is None:
        return dict(status="CONSENSUS_OR_ACTUAL_NA", difference=None)
    if actual_period != consensus_period or actual_unit != consensus_unit:
        return dict(status="PERIOD_OR_UNIT_MISMATCH", difference=None)
    if actual_basis is None or consensus_basis is None or actual_basis != consensus_basis:
        return dict(status="RATE_BASIS_UNVERIFIED_OR_MISMATCH", difference=None)
    return dict(status="OK", difference=a-c)


def apply_public_consensus(prices, consensus_rows, asof):
    """Optional user/provider consensus. No proof of an actual release timestamp."""
    for p in prices:
        matches = [c for c in consensus_rows if c.get("series_id") == p["series_id"] and
                   c.get("period") == p["period"]]
        if len(matches) > 1:
            raise ValueError("Több consensus ugyanahhoz a BLS sorozathoz/időszakhoz.")
        p["surprise"] = dict(status="CONSENSUS_NA", difference=None)
        if not matches:
            continue
        c = matches[0]
        if (not c.get("source") or c.get("basis") != p["basis"] or
                c.get("unit") != "percent" or not c.get("recorded_at") or
                not c.get("scheduled_release_at")):
            p["surprise"]["status"] = "CONSENSUS_METADATA_MISSING_OR_MISMATCH"
            continue
        recorded, scheduled = timestamp(c["recorded_at"]), timestamp(c["scheduled_release_at"])
        if recorded >= scheduled or recorded > asof or scheduled > asof:
            p["surprise"]["status"] = "CONSENSUS_OR_RELEASE_TIME_NOT_ELIGIBLE"
            continue
        actual, expected = number(p.get("change_pct")), number(c.get("value"))
        if actual is None or expected is None:
            continue
        p["surprise"] = dict(status="DESCRIPTIVE_MATCH_RELEASE_TIME_UNVERIFIED",
                             difference=actual-expected, unit="percentage_points",
                             consensus_value=expected, source=c["source"],
                             recorded_at=c["recorded_at"])
    return prices


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as handle:
            temp = Path(handle.name)
            json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(temp, path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def default_scenarios():
    return dict(
        label="ILLUSTRATIVE_ASSUMPTIONS_NOT_OBSERVED_DATA",
        basis="ANNUAL", nominal_gdp_pct=6.0, baseline_deflator_pct=3.7,
        alternative_deflators_pct=[3.7, 5, 7, 10],
        ai_contributions_pp=[0, 1, 2, 3.2, 4],
        equity_scenarios=[
            dict(name="ENYHE", eps_change_pct=-5, multiple_change_pct=-5),
            dict(name="KETTOS_SOKK", eps_change_pct=-10, multiple_change_pct=-20),
            dict(name="SULYOS", eps_change_pct=-20, multiple_change_pct=-30)],
        gold_scenarios=[
            dict(name="REALHOZAM_USD_LIKVIDACIO", real_yield_change_pp=.75,
                 dollar_change_pct=4, safe_haven_pct=2, liquidity_pct=-8),
            dict(name="MENEDÉKKERESLET", real_yield_change_pp=-.75,
                 dollar_change_pct=-4, safe_haven_pct=4, liquidity_pct=0)],
        credit=dict(exposure_usd=100_000_000_000, baseline_pd_pct=2,
                    baseline_lgd_pct=50, stress_pd_pct=15, stress_lgd_pct=60),
        # The following are questions, never conclusions.
        reference_levels={"NQ": None, "DJI": None},
        hypotheses=[
            "Az AI-hitelezés romlása okozza a DJI esését.",
            "Az AI nélkül negatív lenne a GDP növekedése.",
            "A magasabb infláció kizárólag az AI következménye.",
            "2026-10-19 hétfőn crash lesz."])


def bank_earnings(snapshot, asof):
    rows = []
    for bank in snapshot.get("banks", []):
        latest = sorted(bank.get("earnings", []), key=lambda r: timestamp(r["release_at"]), reverse=True)
        eligible = [r for r in latest if timestamp(r["release_at"]) <= asof]
        if not eligible:
            rows.append(dict(ticker=bank["ticker"], status="NOT_AVAILABLE_AT_CUTOFF",
                             period=None, actual_eps=None, consensus_eps=None,
                             eps_surprise_pct=None, revenue_surprise_pct=None))
            continue
        r = eligible[0]
        ae, ce = number(r.get("actual_eps")), number(r.get("consensus_eps"))
        ar, cr = number(r.get("actual_revenue")), number(r.get("consensus_revenue"))
        rows.append(dict(ticker=bank["ticker"], status="DATED_PROVIDER_SNAPSHOT",
                         period=r["period"], release_at=r["release_at"],
                         actual_eps=ae, consensus_eps=ce,
                         eps_surprise_pct=(ae-ce)/abs(ce)*100 if ae is not None and ce not in (None, 0) else None,
                         revenue_surprise_pct=(ar-cr)/abs(cr)*100 if ar is not None and cr not in (None, 0) else None))
    return rows


def evaluate(snapshot, scenarios, asof):
    asof = timestamp(asof) if isinstance(asof, str) else asof
    if asof.tzinfo is None:
        raise ValueError("Timezone-os asof kell.")
    collected = timestamp(snapshot["collected_at"])
    cache_age = (asof-collected).total_seconds()/3600
    cache_status = "FUTURE_SNAPSHOT" if cache_age < 0 else "STALE" if cache_age > 24 else "DATED_NOT_REALTIME"
    banks = bank_earnings(snapshot, asof) if cache_age >= 0 else []
    macro = []
    for item in snapshot.get("macro", []):
        r = dict(item)
        period = datetime.fromisoformat(r["period_end"]).date()
        if period > asof.date():
            continue
        r["observation_age_days"] = (asof.date()-period).days
        r["quality"] = ("FUTURE_PERIOD" if period > asof.date() else
                        "OBSERVATION_LAG_PUBLICATION_TIME_UNKNOWN")
        # previous != consensus; never infer surprise from previous_value.
        r["surprise"] = surprise(r.get("value"), r.get("consensus"),
                                r["period_end"], r.get("consensus_period"),
                                r["unit"], r.get("consensus_unit"),
                                r.get("rate_basis"), r.get("consensus_basis"))
        macro.append(r)
    if cache_age < 0:
        macro = []
    grid = []
    for d in scenarios["alternative_deflators_pct"]:
        for a in scenarios["ai_contributions_pp"]:
            row = gdp_counterfactual(scenarios["nominal_gdp_pct"],
                    scenarios["baseline_deflator_pct"], d, a, scenarios["basis"])
            row.update(alternative_deflator_pct=d, ai_contribution_pp=a)
            grid.append(row)
    equities = []
    for asset, level in scenarios.get("reference_levels", {}).items():
        if asset not in ("NQ", "DJI", "NDX", "QQQ"):
            raise ValueError(f"Nem támogatott index referencia: {asset}")
        # Levels are user assumptions, not silently reused historical market quotes.
        for s in scenarios["equity_scenarios"]:
            equities.append(dict(asset=asset, **s, **equity_shock(
                s["eps_change_pct"], s["multiple_change_pct"], level)))
    gold = [dict(**s, **gold_shock(**{k: v for k, v in s.items() if k != "name"}))
            for s in scenarios["gold_scenarios"]]
    c = scenarios["credit"]
    baseline_loss = credit_loss(c["exposure_usd"], c["baseline_pd_pct"], c["baseline_lgd_pct"])
    stress_loss = credit_loss(c["exposure_usd"], c["stress_pd_pct"], c["stress_lgd_pct"])
    credit_facts = []
    if cache_age >= 0:
        for fact in snapshot.get("credit_facts", []):
            if fact.get("period_end") and fact["period_end"] > asof.date().isoformat():
                continue
            if fact.get("available_at") and timestamp(fact["available_at"]) > asof:
                continue
            credit_facts.append(fact)
    return dict(
        version=VERSION, asof=asof.isoformat(), mode="RESEARCH_PAPER",
        cache_status=cache_status, cache_age_hours=cache_age,
        earnings=banks, macro=macro, credit_facts=credit_facts,
        public_refresh=snapshot.get("public_refresh"),
        calendar_context=snapshot.get("calendar_context"),
        public_consensus=snapshot.get("public_consensus", []),
        ai_credit_evidence="NOT_ESTABLISHED_FROM_GENERAL_BANK_PROVISIONS",
        ai_caused_inflation="NOT_IDENTIFIED",
        crash_date_probability=None, monday_crash="NOT_PREDICTABLE_BY_THIS_MODEL",
        verified_no_ai_gdp=None, justified_market_decline_pct=None,
        source_conflicts=snapshot.get("source_conflicts", []),
        data_sources=snapshot.get("sources", []),
        gdp_scenarios=grid, equity_scenarios=equities, gold_scenarios=gold,
        credit_scenario=dict(**c, baseline_loss_usd=baseline_loss,
                             stress_loss_usd=stress_loss,
                             incremental_loss_usd=stress_loss-baseline_loss,
                             observed_ai_exposure=False),
        scenario_assumptions=scenarios, execution_allowed=False,
        notes=[
            "A reál GDP már árhatástól tisztított; újabb CPI-levonás kettős deflálás lenne.",
            "CPI, PPI, GDP-deflátor külön mutató. Éves/YoY és negyedéves annualizált adat nem keverhető.",
            "AI capex != hazai hozzáadott érték: import, dupla számolás, kiszorítás és bázis számít.",
            "AI nélküli GDP: statikus pp-érzékenység, nem azonosított oksági ellenforgatókönyv.",
            "A GDP és a részvényindex árszintje között nincs közvetlen azonosság.",
            "NQ futures != NDX cash index; referenciaár csak az adott instrumentumhoz használható.",
            "Gold: reálhozam, USD, menedékkereslet és likvidáció ellenkező irányba is hathat.",
            "Banki céltartalék nem automatikusan rossz AI-hitel; provision != nettó charge-off.",
            "A snapshot nem point-in-time adatbázis; revíziók és a konszenzus korábbi vintage-a nem igazolt.",
            "A stresszforgatókönyvek nem kalibráltak, nincs hozzájuk megbízható valószínűség.",
            "Nincs order, számlaegyenleg, pozíciókezelés, exercise vagy automatikus shortjel."])


def fmt(x, places=2):
    y = number(x)
    return "N/A" if y is None else f"{y:+.{places}f}"


def console(result, ascii_only=False, details=False):
    from macro_console import render
    return render(result, ascii_only=ascii_only, details=details)


def _legacy_console(result):
    r, lines = result, ["="*104, f"AI / BANKI HITEL / INFLÁCIÓ STRESSZ ELEMZŐ {VERSION}",
                        "KUTATÁSI / PAPER RENDSZER; NEM CRASH-JÓSLAT", "="*104]
    lines += [f"Értékelés: {r['asof']} | snapshot: {r['cache_status']} | kor: {r['cache_age_hours']:.1f} óra",
              "Earnings: publikálás utáni snapshot; nem valós idejű megbízási adat.",
              "", "BANKI EARNINGS / NEM AI-HITELVESZTESÉG-BIZONYÍTÉK",
              "BANK | IDŐSZAK | EPS TÉNY / VÁRT | EPS ELTÉRÉS % | BEVÉTEL ELTÉRÉS %"]
    for b in r["earnings"]:
        lines.append(f"{b['ticker']:<4} | {b['period'] or 'N/A':<8} | "
                     f"{fmt(b['actual_eps'])} / {fmt(b['consensus_eps'])} | "
                     f"{fmt(b['eps_surprise_pct'])}% | {fmt(b['revenue_surprise_pct'])}%")
    lines += ["", "ÁLTALÁNOS HITELADATOK: forrásidő / összehasonlíthatóság külön ellenőrzendő"]
    for c in r["credit_facts"]:
        def money(key):
            v = number(c.get(key))
            return "N/A" if v is None else f"{v/1e9:.3f} Mrd USD"
        lines.append(f"{c['ticker']} | {c['period']} | provision={money('provision_usd')} | "
                     f"bruttó hitel={money('gross_loans_usd')} | allowance abs={money('allowance_usd')} | "
                     f"{c.get('quality', 'PUBLICATION_TIME_UNVERIFIED')}")
        lines.append(f"  Provision QoQ={fmt(c.get('provision_qoq_pct'))}% | "
                     f"allowance / gross={fmt(c.get('allowance_to_gross_pct'))}% | "
                     "AI-specifikus kitettség, NPL, NCO, CET1: N/A")
    if r["public_refresh"]:
        lines += ["", "NYILVÁNOS API FRISSÍTÉS", *r["public_refresh"].get("messages", [])]
        for b in r["public_refresh"].get("banks", []):
            lines.append(f"{b['ticker']} | SEC: {b['status']} | {b.get('period', 'N/A')} | "
                         f"provision USD={b.get('provision_usd')} | AI-bontás=N/A | "
                         f"legutóbbi earnings 8-K/NPL/NCO/CET1=NEM KINYERT")
        for m in r["public_refresh"].get("prices", []):
            lines.append(f"BLS {m['series']} | {m['period']} | index={m['value']} | "
                         f"azonos sorozatból változás={fmt(m['change_pct'])}% | {m['basis']} | konszenzus=N/A")
    lines += ["", "MAKROADATOK: REFERENCIA-IDŐSZAK ÉS MEGLEPETÉS"]
    for m in r["macro"]:
        lines.append(f"{m['name']} | {m['value']} {m['unit']} | {m['period_end']} | "
                     f"megfigyelés kora={m['observation_age_days']} nap | "
                     f"surprise={m['surprise']['status']}")
    lines += ["", "ADATÜTKÖZÉSEK / ELLENŐRIZENDŐ ÁLLÍTÁSOK"]
    if r.get("calendar_context"):
        calendar = r["calendar_context"]
        lines += [f"Release-naptár referencia: {calendar['reference_date']} | {calendar['classification']}",
                  f"CPI: {calendar['cpi_scheduled_at']} | PPI: {calendar['ppi_scheduled_at']}",
                  "A naptár nem bizonyítja a tényleges publikálást vagy meglepetést."]
    lines += [f"- {c}" for c in r["source_conflicts"]]
    lines += ["AI-specifikus tényleges veszteség: NEM IGAZOLT.",
              "AI okozta infláció: NEM AZONOSÍTOTT.",
              "Hétfői crash: NINCS SZÁMÍTHATÓ VALÓSZÍNŰSÉG / IDŐZÍTÉS.",
              "Jelenlegi, igazolt AI nélküli GDP: N/A; indokolt index-esés: N/A.",
              "", "FORRÁSFÁJLOK (a bemeneti JSON-ban teljesebb proveniencia)",
              *[f"- {s['tool']}: {s['filename']}" for s in r["data_sources"]],
              "", "GDP ÉRZÉKENYSÉG: KIZÁRÓLAG FELTEVÉSEK, NEM A VALÓS GDP ÚJRABECSLÉSE"]
    s = r["scenario_assumptions"]
    lines.append(f"Nominális={s['nominal_gdp_pct']}% | alap GDP-deflátor={s['baseline_deflator_pct']}% | "
                 f"időalap={s['basis']} | AI-hatás: reálnövekedési százalékpont")
    lines.append("ALT DEFLÁTOR % | AI PP | ALAP REÁL % | AI NÉLKÜL ALAP % | ALT REÁL % | AI NÉLKÜL ALT %")
    for g in r["gdp_scenarios"]:
        lines.append(f"{g['alternative_deflator_pct']:>14.1f} | {g['ai_contribution_pp']:>5.1f} | "
                     f"{fmt(g['reported_real_pct']):>11} | {fmt(g['no_ai_baseline_pct']):>16} | "
                     f"{fmt(g['alternative_real_pct']):>10} | {fmt(g['no_ai_alternative_pct']):>15}")
    lines += ["", "RÉSZVÉNYINDEX STRESSZ: EPS × ÉRTÉKELTSÉGI MULTIPLIKÁTOR, NEM FAIR VALUE",
              "INDEX | FELTEVÉS | EPS % | MULTIPLE % | FELTÉTELES ÁRVÁLTOZÁS % | FELTÉTELES SZINT"]
    for e in r["equity_scenarios"]:
        lines.append(f"{e['asset']} | {e['name']} | {fmt(e['eps_change_pct'])} | "
                     f"{fmt(e['multiple_change_pct'])} | {fmt(e['change_pct'])} | {fmt(e['scenario_level'])}")
    lines += ["Nem következik GDP-ből; nem hétfőre szól. A szintekhez saját referencia kell.",
              "", "GOLD: NEM KALIBRÁLT LOG-FAKTOR FORGATÓKÖNYVEK"]
    for g in r["gold_scenarios"]:
        lines.append(f"{g['name']} | reálhozam pp={g['real_yield_change_pp']} | USD %={g['dollar_change_pct']} | "
                     f"menedék log-pp={g['safe_haven_pct']} | likvidáció log-pp={g['liquidity_pct']} | "
                     f"feltételes ár %={fmt(g['change_pct'])}")
    c = r["credit_scenario"]
    lines += ["", "AI-HITEL STRESSZ: MINDEN BEMENET FELTEVÉS, NEM MÉRT BANKI AI-KITETTSÉG",
              f"EAD={c['exposure_usd']/1e9:.1f} Mrd USD | "
              f"alap EL={c['baseline_loss_usd']/1e9:.2f} Mrd | stressz EL={c['stress_loss_usd']/1e9:.2f} Mrd | "
              f"többlet={c['incremental_loss_usd']/1e9:.2f} Mrd USD",
              "EL=EAD×PD×LGD; nem adható automatikusan a banki provision/allowance mellé.",
              "", "KORLÁTOK", *[f"- {n}" for n in r["notes"]],
              "Végrehajtási engedély: NEM.", "="*104]
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, default=BASE/"macro_provider_snapshot.json")
    p.add_argument("--scenarios", type=Path, default=BASE/"macro_scenarios.json")
    p.add_argument("--asof", help="Timezone-os ISO időpont; alap: UTC most.")
    p.add_argument("--output-dir", type=Path, default=BASE/"macro_ai_output")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--refresh-public", action="store_true", help="Alapértelmezett: új SEC/BLS lekérés.")
    mode.add_argument("--offline", action="store_true", help="Nincs hálózat; korábbi sikeres cache/snapshot.")
    p.add_argument("--ascii", action="store_true", help="Emoji/Unicode táblakeret helyett ASCII.")
    p.add_argument("--details", action="store_true", help="Minden szövegtalálat és makróreferencia.")
    p.add_argument("--legacy-console", action="store_true", help="Régi 0.1.x szövegformátum.")
    p.add_argument("--finance-export-dir", type=Path, help="Új provider CSV-export importálása futáskor.")
    p.add_argument("--credit-details", type=Path, help="Normalizált, forrással ellátott negyedéves hitel-JSON.")
    p.add_argument("--disclosures", type=Path, help="Opcionális helyi jelentéskivonatok JSON-listája.")
    p.add_argument("--consensus", type=Path, default=BASE/"macro_consensus.json",
                   help="Opcionális időbélyegzett CPI/PPI konszenzus, alapból üres.")
    args = p.parse_args(argv)
    try:
        snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
        scenarios = json.loads(args.scenarios.read_text(encoding="utf-8")) if args.scenarios.exists() else default_scenarios()
        asof = timestamp(args.asof) if args.asof else datetime.now(timezone.utc)
        if args.asof:
            args.output_dir = args.output_dir/"replay"
        if args.finance_export_dir:
            from build_macro_provider_snapshot import build
            calendar = snapshot.get("calendar_context")
            snapshot = build(args.finance_export_dir, datetime.now(timezone.utc).isoformat())
            snapshot["calendar_context"] = calendar
        from macro_monitor_state import update_state, merge_public_cache
        cache_path = args.output_dir/"macro_public_cache.json"
        public_cache = {}
        if cache_path.exists():
            try:
                public_cache = json.loads(cache_path.read_text(encoding="utf-8"))
                if not isinstance(public_cache, dict):
                    public_cache = {}
            except ValueError:
                print("FIGYELEM: public cache nem olvasható; nincs csendes fallback.")
        if not args.offline:
            from macro_public_data import refresh_public
            print("Friss SEC earnings/filing + BLS CPI/PPI lekérés indul; részleges hiba külön látható.")
            snapshot["public_refresh"] = refresh_public(asof, source_dir=args.output_dir/"sec_sources")
            if not any("NONCURRENT_ASOF" in m for m in snapshot["public_refresh"]["messages"]):
                atomic_json(cache_path, merge_public_cache(public_cache, snapshot["public_refresh"]))
        elif public_cache and (not args.asof or timestamp(public_cache["collected_at"]) <= asof):
            snapshot["public_refresh"] = public_cache
        result = evaluate(snapshot, scenarios, asof)
        result["refresh_mode"] = "OFFLINE_CACHE_NOT_CURRENT" if args.offline else "NEW_PUBLIC_FETCH_ATTEMPT"
        if args.asof:
            result["refresh_mode"] = "ASOF_REPLAY_NO_LIVE_STATE"
        consensus = json.loads(args.consensus.read_text(encoding="utf-8")) if args.consensus.exists() else []
        if not isinstance(consensus, list):
            raise ValueError("A consensus fájl JSON-lista legyen.")
        if result.get("public_refresh"):
            apply_public_consensus(result["public_refresh"].get("prices", []), consensus, asof)
        from macro_credit_forensics import evaluate_forensics
        documents = (result.get("public_refresh") or {}).get("disclosures", [])
        if args.disclosures:
            documents = documents+json.loads(args.disclosures.read_text(encoding="utf-8"))
        documents = [d for d in documents if d.get("accepted_at") and
                     timestamp(d["accepted_at"]) <= asof]
        from macro_credit_autodata import build_automatic_credit, extraction_coverage
        normalized = build_automatic_credit(
            result.get("public_refresh") or {}, result["credit_facts"], asof)
        if args.credit_details:
            manual = json.loads(args.credit_details.read_text(encoding="utf-8"))
            if not isinstance(manual, list):
                raise ValueError("A credit-details negyedéves sorok JSON-listája legyen.")
            combined = {(c["ticker"], c["period_end"]): c for c in normalized}
            for c in manual:
                if c["period_end"] > asof.date().isoformat() or not c.get("available_at") or timestamp(c["available_at"]) > asof:
                    continue
                if c.get("normalization_verified") and not c.get("source"):
                    raise ValueError("Normalizáltnak jelölt hiteladathoz forrás szükséges.")
                combined[(c["ticker"], c["period_end"])] = c
            normalized = list(combined.values())
        normalized.sort(key=lambda r: (r["ticker"], r["period_end"]))
        result["forensics"] = evaluate_forensics(documents, normalized)
        result["normalized_credit_facts"] = normalized
        result["credit_coverage"] = extraction_coverage(result.get("public_refresh") or {}, normalized, asof)
        result["forensics"]["automatic_extraction"] = True
        result["forensics"]["extraction_diagnostics"] = (
            result.get("public_refresh") or {}).get("credit_diagnostics", [])
        result["run_changes"] = (update_state(args.output_dir/"macro_ai_state.json", result)
                                 if not args.asof else dict(changes=[], warning="ASOF replay: live state nem frissül."))
        if hasattr(__import__("sys").stdout, "reconfigure"):
            __import__("sys").stdout.reconfigure(encoding="utf-8", errors="replace")
        lines = (_legacy_console(result) if args.legacy_console else
                 console(result, ascii_only=args.ascii, details=args.details))
        for line in lines:
            print(line)
        # Keep source URLs and detected excerpts, not megabytes of raw HTML text, in history.
        if result.get("public_refresh"):
            result["public_refresh"] = dict(result["public_refresh"], disclosures=[
                {k: v for k, v in d.items() if k != "text"}
                for d in result["public_refresh"].get("disclosures", [])])
        atomic_json(args.output_dir/"macro_ai_latest.json", result)
        atomic_json(args.output_dir/"macro_credit_normalized.json", normalized)
        atomic_json(args.output_dir/"macro_credit_diagnostics.json", result["credit_coverage"])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir/"macro_ai_console.txt").write_text("\n".join(lines), encoding="utf-8")
        history = args.output_dir/"history"
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        atomic_json(history/f"macro_ai_{run_id}.json", result)
        for old in sorted(history.glob("macro_ai_*.json"))[:-30]:
            old.unlink()
        print(f"Export: {args.output_dir.resolve()}")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.exit(1, f"Makró elemző hiba: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
