"""Convert finance-connector CSV exports to portable console input, offline.

No credentials and no HTTP. --finance-dir points at already-downloaded CSVs.
If duplicate exports exist, rejects ambiguity rather than combining vintages.
"""
from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime, timezone
from pathlib import Path

from macro_ai_credit_monitor import atomic_json, number

BANKS = ("JPM", "BAC", "C", "WFC", "GS", "MS")


def read_one(root, pattern):
    paths = list(root.rglob(pattern))
    if len(paths) > 1:
        raise ValueError(f"Több export: {pattern}; adj meg tiszta finance-dir mappát.")
    if not paths:
        return [], None
    with paths[0].open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle)), paths[0].name


def build(root, collected_at):
    snapshot = dict(schema_version="0.1.0", collected_at=collected_at,
                    source_kind="DATED_FINANCE_CONNECTOR_EXPORT_NOT_LIVE_FEED",
                    banks=[], credit_facts=[], macro=[], sources=[],
                    source_conflicts=[
                        "CPI/PPI aktuális release-actual és release előtti konszenzus együtt nem áll rendelkezésre.",
                        "A makróexport GDP Growth Rate sora nem rögzíti egyértelműen a SAAR/egyszerű QoQ bázist; "
                        "nem használjuk automatikusan a számpélda alapjaként.",
                        "AI-kitettség és AI-hoz rendelt tényleges charge-off nincs a standard adatmezőkben.",
                        "A transcript-megállapítás a lekért kivonatra vonatkozik, nem teljes körű banki audit."])
    for ticker in BANKS:
        earnings, source = read_one(root, ticker+"_quarterly_earnings_history.csv")
        parsed = [dict(period=r["period"], release_at=r["date"],
                       actual_eps=number(r.get("actualEps")), consensus_eps=number(r.get("estimatedEps")),
                       actual_revenue=number(r.get("actualRevenue")),
                       consensus_revenue=number(r.get("estimatedRevenue")),
                       source_file=source) for r in earnings]
        snapshot["banks"].append(dict(ticker=ticker, earnings=parsed))
        transcript, transcript_file = read_one(root, ticker+"_Q*_earnings_transcript.csv")
        for row in transcript:
            matches = re.findall(r"\bEPS of \$(\d+(?:\.\d+)?)", row.get("text", ""))
            period = f"{row.get('fiscal_quarter')} {row.get('fiscal_year')}"
            event = next((e for e in parsed if e["period"] == period), None)
            if event and matches and abs(float(matches[0])-(event["actual_eps"] or 0)) > .005:
                snapshot["source_conflicts"].append(
                    f"{ticker} {period}: earnings feed és call EPS eltér; bázis/provider reconciliation szükséges.")
        if transcript_file:
            snapshot["sources"].append(dict(tool="finance_earnings_transcript", filename=transcript_file))
        if source:
            snapshot["sources"].append(dict(tool="finance_earnings_history", filename=source))
        income, inc_source = read_one(root, ticker+"_quarter_*_income_statement_*.csv")
        balance, bal_source = read_one(root, ticker+"_quarter_*_balance_sheet_*.csv")
        for r in income:
            end = r["date"]
            b = next((v for v in balance if v["date"] == end), {})
            previous = max((v for v in income if v["date"] < end), key=lambda x: x["date"], default={})
            provision = number(r.get("income_statement_provision_for_credit_losses"))
            prior = number(previous.get("income_statement_provision_for_credit_losses"))
            gross = number(b.get("balance_sheet_gross_loans"))
            allowance = number(b.get("balance_sheet_less_allowance_for_loan_losses"))
            allowance = abs(allowance) if allowance is not None else None
            # Income and earnings period strings use different order.
            year, q = r["period"].split()
            event = next((e for e in parsed if e["period"] == f"{q} {year}"), {})
            snapshot["credit_facts"].append(dict(
                ticker=ticker, period=r["period"], period_end=end,
                available_at=event.get("release_at"), publication_basis="EARNINGS_DATE_PROXY_NOT_FILING_TIME",
                provision_usd=provision, provision_qoq_pct=(provision/prior-1)*100
                if provision is not None and prior not in (None, 0) else None,
                gross_loans_usd=gross, allowance_usd=allowance,
                allowance_to_gross_pct=allowance/gross*100 if allowance is not None and gross not in (None, 0) else None,
                equity_usd=number(b.get("balance_sheet_total_shareholders_equity")),
                quality="PUBLICATION_PROXY_UNVERIFIED"+("" if b else "|NO_SAME_PERIOD_BALANCE"),
                income_source_file=inc_source, balance_source_file=bal_source if b else None))
        for source in (inc_source, bal_source):
            if source:
                snapshot["sources"].append(dict(tool="finance_company_financials", filename=source))
    macro, source = read_one(root, "macro_snapshot_GDP_growth*United_States*.csv")
    for r in macro:
        name = r["category"]
        basis = ("MOM" if "MoM" in name else "YOY" if "YoY" in name
                 or name in ("Inflation Rate", "Core Inflation Rate", "GDP Annual Growth Rate")
                 else "ANNUAL" if name == "Full Year GDP Growth"
                 else "LEVEL" if r["unit"] == "points" else None)
        snapshot["macro"].append(dict(name=r["category"], series_id=r["series_id"],
            value=number(r["latest_value"]), previous_value=number(r["previous_value"]),
            period_end=r["latest_value_date"], unit=r["unit"], frequency=r["frequency"],
            adjustment=r["adjustment"], source_name=r["source"],
            source_url=r["source_url"], source_file=source, consensus=None, rate_basis=basis))
    if source:
        snapshot["sources"].append(dict(tool="finance_macro_snapshot", filename=source))
    snapshot["calendar_context"] = None
    if not any(b["earnings"] for b in snapshot["banks"]):
        raise ValueError("Nincs használható banki earnings export.")
    return snapshot


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--finance-dir", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent/"macro_provider_snapshot.json")
    p.add_argument("--calendar", type=Path, help="Külön ellenőrzött release-naptár JSON, nem actual.")
    args = p.parse_args(argv)
    snapshot = build(args.finance_dir, datetime.now(timezone.utc).isoformat())
    if args.calendar:
        import json
        snapshot["calendar_context"] = json.loads(args.calendar.read_text(encoding="utf-8"))
    atomic_json(args.output, snapshot)
    print(f"Snapshot: {args.output}")


if __name__ == "__main__":
    main()
