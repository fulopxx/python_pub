"""Optional direct public SEC/BLS adapter, no credentials/account/orders.

Runs on user's machine by default (or --refresh-public); --offline disables it.
Here tested with mocks;
actual initial values come from finance connector, not this HTTP adapter.
SEC facts != same-day earnings release. BLS observation month != release time.
"""
from __future__ import annotations

import json
import os
import time
import re
import hashlib
from pathlib import Path
from html.parser import HTMLParser
from datetime import datetime, timezone, timedelta
from urllib.request import Request, urlopen
from urllib.parse import urlparse, quote

from macro_ai_credit_monitor import number

# Explicit known large-bank watchlist, not a complete systemic-bank universe.
BANKS = {"JPM": "0000019617", "BAC": "0000070858", "C": "0000831001",
         "WFC": "0000072971", "GS": "0000886982", "MS": "0000895421"}
BLS_SERIES = {"CPI_YOY_NSA": ("CUUR0000SA0", 12),
              "CPI_MOM_SA": ("CUSR0000SA0", 1),
              "PPI_FINAL_DEMAND_YOY_NSA": ("WPUFD4", 12),
              "PPI_FINAL_DEMAND_MOM_SA": ("WPSFD4", 1)}
TAGS = {
    "provision_usd": ("ProvisionForLoanLeaseAndOtherLosses", "ProvisionForLoanAndLeaseLosses",
                      "ProvisionForCreditLosses"),
    "gross_loans_usd": ("LoansAndLeasesReceivableNetReportedAmount",),
    "allowance_usd": ("FinancingReceivableAllowanceForCreditLosses", "AllowanceForLoanAndLeaseLosses"),
    "equity_usd": ("StockholdersEquity",)}


def request_json(url, payload=None, user_agent=None):
    return json.loads(_request(url, payload, user_agent))


def _request(url, payload=None, user_agent=None):
    if urlparse(url).scheme != "https" or urlparse(url).hostname not in ("data.sec.gov", "api.bls.gov", "www.sec.gov"):
        raise ValueError("Csak a megadott SEC/BLS HTTPS host engedélyezett.")
    headers = {"Accept": "application/json", "User-Agent": user_agent or "MacroPaperResearch/0.1"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode()
    req = Request(url, data=body, headers=headers)
    time.sleep(.25)  # sequential public access, modest request rate
    with urlopen(req, timeout=25) as response:
        data = response.read(30_000_001)
    if len(data) > 30_000_000:
        raise ValueError("Túl nagy API-válasz.")
    return data


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1
        if tag in ("p", "div", "tr", "br"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.hidden:
            self.hidden -= 1
        if tag in ("p", "div", "tr"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def request_text(url, user_agent):
    raw = _request(url, user_agent=user_agent).decode("utf-8", errors="replace")
    parser = PlainText()
    parser.feed(raw)
    # Textual context only, never infer table-column financial values from it.
    return re.sub(r"\s+", " ", " ".join(parser.parts))[:2_000_000]


def request_document(url, user_agent, document, asof):
    from macro_credit_autodata import inline_facts
    from macro_credit_tables import table_facts
    raw = _request(url, user_agent=user_agent).decode("utf-8", errors="replace")
    parser = PlainText()
    parser.feed(raw)
    candidates = []
    facts, diagnostics = inline_facts(raw, document, asof, candidates)
    table_values, table_issues = table_facts(raw, document, asof)
    facts.extend(table_values)
    diagnostics.extend(table_issues)
    source_file = None
    if document.get("source_dir"):
        try:
            folder = Path(document["source_dir"])
            folder.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(raw.encode()).hexdigest()
            path = folder/f"{document['ticker']}_{digest}.html"
            # Binary write: Windows text newline conversion would invalidate
            # the content-addressed hash and double CR in existing CRLF HTML.
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                path.write_bytes(raw.encode("utf-8"))
            source_file = dict(ticker=document["ticker"], path=str(path),
                               source=url, sha256=digest, kind="SEC_FILING_HTML",
                               collected_at=asof.isoformat())
        except OSError as exc:
            diagnostics.append(dict(reason="SOURCE_CACHE_WRITE_ERROR", error=type(exc).__name__,
                                    source=url))
    return dict(text=re.sub(r"\s+", " ", " ".join(parser.parts))[:2_000_000],
                credit_raw_facts=facts, credit_diagnostics=diagnostics,
                credit_candidate_facts=candidates, source_file=source_file)


def recent_filings(payload, ticker, cik, asof):
    recent = payload.get("filings", {}).get("recent", {})
    eligible = []
    for i, form in enumerate(recent.get("form", [])):
        def cell(key):
            values = recent.get(key, [])
            return values[i] if i < len(values) else ""
        if form not in ("8-K", "8-K/A", "10-Q", "10-Q/A", "10-K", "10-K/A"):
            continue
        acceptance, filed = cell("acceptanceDateTime"), cell("filingDate")
        try:
            known = datetime.fromisoformat(acceptance.replace("Z", "+00:00"))
            if known.tzinfo is None:
                raise ValueError("Timezone unknown")
        except ValueError:
            try:
                known = datetime.fromisoformat(filed).replace(tzinfo=timezone.utc)+timedelta(days=1)
            except ValueError:
                continue
        if known > asof or (asof-known).days > 240:
            continue
        accession, document, items = cell("accessionNumber"), cell("primaryDocument"), cell("items")
        if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession) or not document or "/" in document or ".." in document:
            continue
        if form.startswith("8-K") and "2.02" not in items:
            continue
        root = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"
        eligible.append(dict(ticker=ticker, form=form, items=items, accession=accession,
                             accepted_at=known.isoformat(), filed=filed, report_period=cell("reportDate"),
                             primary_document=document, source=root+quote(document), archive_root=root,
                             status="EARNINGS_EVENT_2_02" if form.startswith("8-K") else "PERIODIC_REPORT"))
    eligible.sort(key=lambda r: r["accepted_at"], reverse=True)
    # One earnings event + two distinct periodic reports. The prior quarter's
    # filing is needed for comparisons and bank-specific extension contexts.
    chosen = []
    for group in ("EARNINGS_EVENT_2_02", "PERIODIC_REPORT"):
        periods = set()
        for row in (r for r in eligible if r["status"] == group):
            key = row.get("report_period") or row["filed"]
            if key not in periods:
                chosen.append(row)
                periods.add(key)
            if len(periods) >= (2 if group == "PERIODIC_REPORT" else 1):
                break
    return chosen


def fetch_disclosures(filings, user_agent, asof=None, source_dir=None):
    asof = asof or datetime.now(timezone.utc)
    documents = []
    for filing in filings:
        targets = [(filing["source"], "PRIMARY_DOCUMENT")]
        if filing["status"] == "EARNINGS_EVENT_2_02":
            try:
                listing = request_json(filing["archive_root"]+"index.json", user_agent=user_agent)
                names = [r.get("name", "") for r in listing.get("directory", {}).get("item", [])]
                candidates = [n for n in names if re.search(r"99|earnings|release|supplement|financial", n, re.I) and
                              re.search(r"\.html?$", n, re.I) and "/" not in n and ".." not in n]
                targets += [(filing["archive_root"]+quote(n), "EXHIBIT_FILENAME_CANDIDATE_TYPE_UNVERIFIED")
                            for n in sorted(candidates)[:6] if filing["archive_root"]+quote(n) != filing["source"]]
                if not candidates:
                    documents.append(dict(ticker=filing["ticker"], accession=filing["accession"],
                                          source=filing["archive_root"]+"index.json",
                                          status="NO_SUPPORTED_HTML_EXHIBIT_FILENAME"))
            except Exception:
                documents.append(dict(ticker=filing["ticker"], accession=filing["accession"],
                                      source=filing["archive_root"]+"index.json", status="EXHIBIT_INDEX_UNAVAILABLE"))
        for url, quality in targets:
            try:
                doc = dict(filing, source=url, cik=BANKS.get(filing["ticker"]),
                           source_dir=str(source_dir) if source_dir else None)
                extracted = request_document(url, user_agent, doc, asof)
                text = extracted["text"]
                documents.append(dict(ticker=filing["ticker"], accession=filing["accession"],
                                      accepted_at=filing["accepted_at"], source=url,
                                      status="TEXT_EXTRACTED", document_quality=quality+
                                      ("|PREFIX_TRUNCATED" if len(text) >= 2_000_000 else ""),
                                      text=text, normalized_financials_extracted=bool(extracted["credit_raw_facts"]),
                                      credit_raw_facts=extracted["credit_raw_facts"],
                                      credit_diagnostics=extracted["credit_diagnostics"],
                                      credit_candidate_facts=extracted.get("credit_candidate_facts", []),
                                      source_file=extracted.get("source_file")))
            except Exception as exc:
                documents.append(dict(ticker=filing["ticker"], accession=filing["accession"],
                                      source=url, status="TEXT_FETCH_ERROR", error=type(exc).__name__))
    return documents


def _filed_eligible(row, asof):
    # Only filing DATE is supplied: conservative next UTC midnight availability.
    try:
        known = datetime.fromisoformat(row["filed"]).replace(tzinfo=timezone.utc)+timedelta(days=1)
        return known <= asof and row["end"] <= asof.date().isoformat()
    except (KeyError, ValueError):
        return False


def extract_sec(payload, ticker, asof):
    facts = payload.get("facts", {}).get("us-gaap", {})
    candidates = []
    # An obsolete provision tag must not select a 2019 "current" bank quarter.
    latest_ends = [r["end"] for tags in TAGS.values() for tag in tags
                   for r in facts.get(tag, {}).get("units", {}).get("USD", [])
                   if _filed_eligible(r, asof) and r.get("form") in ("10-Q", "10-K", "10-Q/A", "10-K/A")]
    latest_end = max(latest_ends, default=None)
    for tag in TAGS["provision_usd"]:
        for row in facts.get(tag, {}).get("units", {}).get("USD", []):
            if row.get("form") not in ("10-Q", "10-K", "10-Q/A", "10-K/A") or not _filed_eligible(row, asof):
                continue
            try:
                duration = (datetime.fromisoformat(row["end"])-datetime.fromisoformat(row["start"])).days
            except (KeyError, ValueError):
                continue
            # Never treat YTD / annual flow as quarterly; no blind subtraction.
            if 70 <= duration <= 105 and row["end"] == latest_end and number(row.get("val")) is not None:
                candidates.append((row["end"], row["filed"], tag, row))
    if not candidates:
        return dict(ticker=ticker, period=latest_end, status="NO_DIRECT_QUARTER_PROVISION_TAG",
                    ai_exposure_usd=None)
    _, _, tag, row = max(candidates, key=lambda x: (x[0], x[1]))
    out = dict(ticker=ticker, status="FILED_QUARTER_FACTS_ONLY_NOT_LATEST_EARNINGS",
               period=row["end"], filed=row["filed"], provision_usd=float(row["val"]),
               provision_tag=tag, gross_loans_usd=None, allowance_usd=None, equity_usd=None,
               ai_exposure_usd=None, net_chargeoffs_usd=None, npl_usd=None, cet1_pct=None)
    for name in ("gross_loans_usd", "allowance_usd", "equity_usd"):
        records = []
        for t in TAGS[name]:
            for r in facts.get(t, {}).get("units", {}).get("USD", []):
                if r.get("start") or r.get("end") != row["end"] or not _filed_eligible(r, asof):
                    continue
                if number(r.get("val")) is not None:
                    records.append((r["filed"], t, float(r["val"])))
        if records:
            _, t, value = max(records, key=lambda x: x[0])
            if name == "gross_loans_usd":
                # Tag explicitly NET; do not silently present it as gross loans.
                out["net_loans_usd"] = value
                out["net_loans_tag"] = t
            else:
                out[name] = abs(value) if name == "allowance_usd" else value
    out["balance_period_aligned"] = True
    return out


def extract_bls(payload, asof):
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise ValueError("BLS API nem sikeres: "+str(payload.get("message")))
    series = {r["seriesID"]: r.get("data", []) for r in payload.get("Results", {}).get("series", [])}
    out = []
    for label, (sid, lag) in BLS_SERIES.items():
        points = {}
        for r in series.get(sid, []):
            if r.get("period", "").startswith("M") and r["period"][1:].isdigit():
                month = int(r["period"][1:])
                if 1 <= month <= 12 and number(r.get("value")) is not None:
                    year = int(r["year"])
                    # Monthly data for current/future month cannot be completed.
                    if (year, month) < (asof.year, asof.month):
                        points[year*12+month-1] = number(r["value"])
        if not points:
            continue
        last = max(points)
        previous = points.get(last-lag)
        year, month = divmod(last, 12)
        out.append(dict(series=label, series_id=sid, period=f"{year:04d}-{month+1:02d}",
                        value=points[last],
                        change_pct=(points[last]/previous-1)*100 if previous not in (None, 0) else None,
                        basis="YOY_NSA" if lag == 12 else "MOM_SA",
                        release_at=None, consensus=None, source="https://www.bls.gov",
                        quality="CURRENT_VINTAGE_RELEASE_TIMESTAMP_UNVERIFIED"))
    return out


def refresh_public(asof, source_dir=None):
    """Partial failures retained; no fallback to invented last known 'OK'."""
    result = dict(collected_at=datetime.now(timezone.utc).isoformat(), banks=[], prices=[], messages=[],
                  filings=[], disclosures=[], bank_fetch_status={},
                  credit_raw_facts=[], credit_diagnostics=[], credit_candidate_facts=[], source_files=[])
    now = datetime.now(timezone.utc)
    if abs((asof-now).total_seconds()) > 300:
        result["messages"].append("NONCURRENT_ASOF: friss HTTP lekérés tiltva; nincs point-in-time/future vintage.")
        return result
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in ua or len(ua) < 12 or "\n" in ua or "\r" in ua:
        result["messages"].append("SEC SKIPPED: állíts be valós kapcsolati SEC_USER_AGENT értéket; nem API-kulcs.")
    else:
        for ticker, cik in BANKS.items():
            try:
                payload = request_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", user_agent=ua)
                from macro_credit_autodata import company_facts
                raw, issues = company_facts(payload, ticker,
                    f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", asof,
                    result["credit_candidate_facts"])
                if source_dir:
                    try:
                        from macro_ai_credit_monitor import atomic_json
                        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                        path = Path(source_dir)/f"{ticker}_companyfacts_{digest}.json"
                        if not path.exists():
                            atomic_json(path, payload)
                        result["source_files"].append(dict(
                            ticker=ticker, path=str(path), source=f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                            sha256=digest, kind="SEC_COMPANYFACTS", collected_at=asof.isoformat()))
                    except OSError as exc:
                        result["messages"].append(f"{ticker}: forráscache nem írható: {type(exc).__name__}")
                result["credit_raw_facts"].extend(raw)
                result["credit_diagnostics"].extend(dict(i, ticker=ticker) for i in issues)
                row = extract_sec(payload, ticker, asof)
                row["source"] = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
                result["banks"].append(row)
            except Exception as exc:
                result["banks"].append(dict(ticker=ticker, status="FETCH_OR_PARSE_ERROR",
                                            error=type(exc).__name__))
            try:
                submissions = request_json(f"https://data.sec.gov/submissions/CIK{cik}.json", user_agent=ua)
                filings = recent_filings(submissions, ticker, cik, asof)
                result["filings"].extend(filings)
                docs = fetch_disclosures(filings, ua, asof, source_dir)
                result["disclosures"].extend(docs)
                for doc in docs:
                    result["credit_raw_facts"].extend(doc.pop("credit_raw_facts", []))
                    result["credit_diagnostics"].extend(
                        dict(i, ticker=ticker) for i in doc.pop("credit_diagnostics", []))
                    result["credit_candidate_facts"].extend(doc.pop("credit_candidate_facts", []))
                    if doc.get("source_file"):
                        result["source_files"].append(doc.pop("source_file"))
                result["bank_fetch_status"][ticker] = "SUBMISSIONS_FETCHED" if filings else "NO_MATCHING_RECENT_FILING"
            except Exception as exc:
                result["bank_fetch_status"][ticker] = f"SUBMISSIONS_ERROR:{type(exc).__name__}"
            time.sleep(.25)
    try:
        payload = request_json("https://api.bls.gov/publicAPI/v2/timeseries/data/",
                               dict(seriesid=[v[0] for v in BLS_SERIES.values()],
                                    startyear=str(asof.year-1), endyear=str(asof.year)))
        result["prices"] = extract_bls(payload, asof)
        if len(result["prices"]) < len(BLS_SERIES):
            result["messages"].append("BLS PARTIAL: néhány CPI/PPI sorozat nem érkezett meg.")
    except Exception as exc:
        result["messages"].append(f"BLS FETCH_OR_PARSE_ERROR: {type(exc).__name__}")
    result["messages"] += [
        "SEC: companyfacts + inline-XBRL hiteladatkinyerés; nem címkézett szövegből nem találunk ki számot.",
        "BLS: indexváltozás, nem konszenzus-meglepetés; publikálási idő nincs igazolva.",
        "A friss API-adat külön blokk; nem írja át csendben a dated earnings snapshotot."]
    return result
