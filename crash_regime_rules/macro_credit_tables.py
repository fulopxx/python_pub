"""Reviewed earnings HTML adapters. Stdlib only; fail closed on layout drift.

Not a free-text number scraper. Bank, table fingerprint, explicit date headers,
units, row labels and column geometry must all match. Missing is never zero.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
import hashlib
import re
from urllib.parse import urlparse

VERSION = "0.3.3"
CIKS = {"JPM": "19617", "BAC": "70858", "C": "831001", "WFC": "72971",
        "GS": "886982", "MS": "895421"}
PROFILES = {bank: f"REVIEWED_EARNINGS_{bank}_FUNDED_CREDIT_V1" for bank in CIKS}
# The meaning is deliberately bank-specific. Not a cross-bank NPL league table.
PROFILE_LABELS = {
    "JPM": "Megtartott hitelek; nonaccrual = consumer + wholesale retained, HFS/FV nélkül.",
    "BAC": "Folyósított hitelek és lízing; unfunded reserve nélkül.",
    "C": "ACLL: folyósított hitelek; ACLUC és egyéb eszközök nélkül.",
    "WFC": "Folyósított hitelek/lízing; unfunded commitments nélkül.",
    "GS": "Funded loan allowance / loan NCO; nem teljes credit-loss provision.",
    "MS": "HFI loan ACL és loan provision; lending commitments nélkül.",
}
KINDS = {"npl_usd": "instant", "allowance_usd": "instant",
         "provision_usd": "flow", "net_chargeoffs_usd": "flow",
         "acl_other_changes_usd": "flow"}
TABLE_CONCEPTS = {
    f"Reviewed{bank}{field}": (field, kind)
    for bank in CIKS for field, kind in KINDS.items()
}


def clean(text):
    return re.sub(r"\s+", " ", text.replace("\u200b", "").replace("\u00ad", "")).strip()


class Tables(HTMLParser):
    """Keep direct cells and span geometry; never flatten nested financial tables."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self.stack = [], []
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("script", "style"):
            self.skip += 1
        if tag == "table":
            table = dict(rows=[], row=None, cell=None, nested=False,
                         context=clean(" ".join(self.parts[-100:]))[-1200:])
            if self.stack:
                self.stack[-1]["nested"] = True
            self.tables.append(table)
            self.stack.append(table)
        if not self.stack:
            return
        t = self.stack[-1]
        if tag == "tr":
            t["row"] = []
            t["rows"].append(t["row"])
        if tag in ("td", "th") and t["row"] is not None:
            try:
                spans = [int(attrs.get(a, "1")) for a in ("colspan", "rowspan")]
                if any(not 1 <= n <= 300 for n in spans):
                    raise ValueError("Invalid table span")
            except ValueError:
                spans = [1, 1]
                t["nested"] = True  # untrusted geometry, reject whole table
            t["cell"] = dict(parts=[], colspan=spans[0], rowspan=spans[1])
            t["row"].append(t["cell"])
        if tag in ("br", "p", "div") and t["cell"] is not None:
            t["cell"]["parts"].append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip-1)
        if self.stack:
            if tag in ("td", "th"):
                self.stack[-1]["cell"] = None
            elif tag == "tr":
                self.stack[-1]["row"] = None
            elif tag == "table":
                self.stack.pop()

    def handle_data(self, text):
        if self.skip:
            return
        self.parts.append(text)
        if self.stack and self.stack[-1]["cell"] is not None:
            self.stack[-1]["cell"]["parts"].append(text)


def grid(table):
    """Expanded columns contain (text, origin-column), including merged headers."""
    out, pending = [], {}
    for raw in table["rows"]:
        old = pending
        row = dict(old)
        pending = {c: (item, n-1) for c, (item,n) in old.items() if n > 1}
        # Pending entries are (cell, rows left).
        row = {c: v[0] for c, v in row.items()}
        col = 0
        for cell in raw:
            while col in row:
                col += 1
            text = clean("".join(cell["parts"]))
            for k in range(col, col+cell["colspan"]):
                if k in row:
                    raise ValueError("Overlapping table spans")
                row[k] = (text, col)
                if cell["rowspan"] > 1:
                    pending[k] = ((text, col), cell["rowspan"]-1)
            col += cell["colspan"]
        out.append(row)
    return out


def row_text(row):
    return clean(" ".join(text for c, (text, origin) in sorted(row.items()) if c == origin))


def label(row):
    cells = [(c, text) for c, (text, origin) in sorted(row.items()) if c == origin and text]
    return cells[0][1] if cells else ""


MONTHS = {v: i for i, v in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}


def parse_period(text, year=None):
    text = clean(text)
    if year and re.fullmatch(r"[1-4]Q", text, re.I):
        text += str(year)
    q = re.fullmatch(r"([1-4])Q\s*['’]?\s*(\d{2}|\d{4})", text, re.I)
    if q:
        y = int(q[2]); y = y+2000 if y < 100 else y
        m = int(q[1])*3
        return (date(y+1, 1, 1) if m == 12 else date(y, m+1, 1))-timedelta(days=1)
    q = re.fullmatch(r"(First|Second|Third|Fourth) Quarter\s+(\d{4})", text, re.I)
    if q:
        return parse_period(f"{('first','second','third','fourth').index(q[1].lower())+1}Q{q[2]}")
    d = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2}),?\s*(\d{4})?", text)
    if d and d[1][:3].lower() in MONTHS and (d[3] or year):
        value = date(int(d[3] or year), MONTHS[d[1][:3].lower()], int(d[2]))
        if (value.month, value.day) in ((3,31), (6,30), (9,30), (12,31)):
            return value
    return None


def headers(rows):
    """Use the first dated row, never later percent-change/YTD headings."""
    for idx, row in enumerate(rows[:14]):
        found = []
        for c, (text, origin) in sorted(row.items()):
            if origin != c:
                continue
            year_text = rows[idx+1].get(c, ("",0))[0] if idx+1 < len(rows) else ""
            year = int(year_text) if re.fullmatch(r"20\d{2}", year_text) else None
            value = parse_period(text, year)
            if value:
                found.append((c, value.isoformat()))
        if len(found) >= 2:
            # Repeat dates occur only after the historical columns (change/YTD).
            unique = []
            for item in found:
                if item[1] in [p for _, p in unique]:
                    unique.append((item[0], None))  # change-column boundary, not data
                    break
                unique.append(item)
            if unique[-1][1] is not None:
                boundary = next((c for c,(t,o) in sorted(row.items())
                                 if c==o and c>unique[-1][0] and t and
                                 any(w in t.lower() for w in ("change", "decrease", "increase"))), None)
                if boundary is not None:
                    unique.append((boundary, None))
            return unique, idx
    return [], None


def amount(text, dash_zero=False):
    text = clean(text).replace("$", "").strip()
    if text in ("—", "–", "-"):
        return Decimal(0) if dash_zero else None
    if not re.fullmatch(r"\(?-?\d[\d,]*(?:\.\d+)?\)?", text):
        return None
    if text.startswith("(") != text.endswith(")"):
        return None
    return Decimal(text.strip("()").replace(",", "")) * (-1 if text.startswith("(") else 1)


def values(row, cols, dash_zero=False):
    """One amount per date group; ignore only explicit $, %, footnote cells."""
    result = []
    for i, (start, period) in enumerate(cols):
        if period is None:
            continue
        stop = cols[i+1][0] if i+1 < len(cols) else max(row, default=0)+1
        # Last historical date may be followed by change columns; callers normally
        # supply a bound. Never use last group if it contains several amounts.
        tokens = []
        for c, (text, origin) in sorted(row.items()):
            if c != origin or not start <= c < stop or not text or text == "$":
                continue
            if re.fullmatch(r"\([a-z](?:,\s*[a-z])*\)", text, re.I):
                continue
            tokens.append(text)
        nums = [amount(t, dash_zero) for t in tokens]
        if len(nums) != 1 or nums[0] is None:
            result.append((period, None))
        else:
            result.append((period, nums[0]))
    return result


def table_facts(html, document, asof):
    """Return accepted native-shaped facts plus explicit extraction diagnostics."""
    ticker = document.get("ticker")
    issues, facts = [], []
    if ticker not in CIKS:
        return facts, issues
    url = urlparse(document.get("source", ""))
    try:
        accepted = datetime.fromisoformat(document["accepted_at"].replace("Z", "+00:00"))
        accession = document.get("accession", "")
        if (url.scheme != "https" or url.netloc != "www.sec.gov" or
                not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession) or
                not url.path.startswith(f"/Archives/edgar/data/{CIKS[ticker]}/{accession.replace('-', '')}/") or
                not accepted.tzinfo or accepted > asof):
            return facts, [dict(ticker=ticker, source=document.get("source"),
                                reason="TABLE_SOURCE_OR_AVAILABILITY_NOT_VERIFIED")]
    except (KeyError, ValueError, TypeError):
        return facts, issues
    parser = Tables()
    parser.feed(html)
    digest = hashlib.sha256(html.encode("utf-8")).hexdigest()

    def emit(field, period, value, table_index, row_index, row, cols, method=None, components=None,
             decimals="-6"):
        if value is None or period > asof.date().isoformat():
            return
        kind = KINDS[field]
        d = date.fromisoformat(period)
        start = date(d.year, ((d.month-1)//3)*3+1, 1).isoformat() if kind == "flow" else None
        tag = f"Reviewed{ticker}{field}"
        facts.append(dict(
            ticker=ticker, tag=tag, concept="reviewed-table:"+tag, end=period, start=start,
            value=float(value*1_000_000), unit="USD", dimensions=[], entity=ticker,
            available_at=accepted.isoformat(), accession=document["accession"],
            source=document["source"], transport="SEC_REVIEWED_EARNINGS_HTML",
            decimals=decimals, scope_id=PROFILES[ticker], scope_description=PROFILE_LABELS[ticker],
            method=method or "REVIEWED_TABLE_ROW_AND_DATE_COLUMN", adapter_version=VERSION,
            source_sha256=digest, table_index=table_index, row_index=row_index,
            row_label=label(row), row_text=row_text(row), date_columns=cols,
            components=components or []))

    for ti, table in enumerate(parser.tables):
        if table["nested"]:
            continue
        try:
            rows = grid(table)
            text = " ".join(row_text(r) for r in rows)
            # Table unit, not a nearby segment headline or global document text.
            unit_context = text[:1000]+" "+table["context"][-800:]
            if not re.search(r"(?:in |dollars in |[$] )millions", unit_context, re.I):
                continue
            if ticker == "MS" and "Consolidated Loans and Lending Commitments" in text[:300]:
                match = re.search(r"Allowance for Credit Losses \(ACL\) as of ([A-Za-z]+ \d+, \d{4})", text)
                if match:
                    period = parse_period(match[1])
                    q = re.search(r"\bQ([1-4]) Provision\b", text)
                    if not period or not q or int(q[1])*3 != period.month:
                        raise ValueError("MS ACL matrix date/provision quarter mismatch")
                    heading = next((row_text(r) for r in rows
                                    if "Loans and Lending Commitments" in row_text(r) and
                                    "ACL (1)" in row_text(r) and "Provision" in row_text(r)), "")
                    if not re.search(r"Loans and Lending Commitments ACL \(1\) ACL % Q[1-4] Provision", heading):
                        raise ValueError("MS ACL matrix column order changed")
                    for ri,row in enumerate(rows):
                        if label(row) != "Total Loans":
                            continue
                        numbers = [amount(t) for c,(t,o) in sorted(row.items())
                                   if c==o and c>0 and t and t!="$"]
                        if len(numbers) != 3 or any(n is None for n in numbers):
                            raise ValueError("MS Total Loans columns changed")
                        emit("allowance_usd",period.isoformat(),numbers[1],ti,ri,row,[])
                        emit("provision_usd",period.isoformat(),numbers[2],ti,ri,row,[])
                    continue
            cols, hi = headers(rows)
            if not cols:
                continue

            def read(pattern, field, subset=None, negate=False, dash_zero=False, limit=None):
                for ri, row in subset if subset is not None else enumerate(rows):
                    if not re.fullmatch(pattern, label(row), re.I):
                        continue
                    used = [c for c in cols if c[1]] if limit is None else cols[:limit]
                    # Preserve the next header as a sentinel bounding the last group.
                    vals = values(row, cols, dash_zero)
                    for period, value in vals[:len(used)]:
                        emit(field, period, -value if negate and value is not None else value,
                             ti, ri, row, cols)

            if ticker == "JPM" and "SUMMARY OF CHANGES IN THE ALLOWANCES" in text:
                lo = next(i for i,r in enumerate(rows) if label(r) == "ALLOWANCE FOR LOAN LOSSES")
                stop = next(i for i,r in enumerate(rows) if label(r) == "ALLOWANCE FOR LENDING-RELATED COMMITMENTS")
                segment = list(enumerate(rows))[lo+1:stop]
                # Two latest periods have an unambiguous next-column boundary.
                read(r"Ending balance", "allowance_usd", segment, limit=2)
                read(r"Provision for loan losses", "provision_usd", segment, limit=2)
                read(r"Net charge-offs", "net_chargeoffs_usd", segment, limit=2)
                # In these reviewed roll-forwards a printed em dash denotes zero.
                read(r"Other", "acl_other_changes_usd", segment, dash_zero=True, limit=2)
            elif ticker == "JPM" and "NONPERFORMING ASSETS (a)" in text and "Wholesale nonaccrual loans" in text:
                retained = [(ri,r) for ri,r in enumerate(rows) if label(r) == "Loans retained"]
                if len(retained) != 2:
                    raise ValueError("JPM retained consumer/wholesale layout changed")
                pairs = [values(r, cols) for _,r in retained]
                for i, (_, period) in enumerate(cols[:2]):
                    vv = [v[i][1] for v in pairs]
                    if all(v is not None for v in vv):
                        emit("npl_usd", period, sum(vv), ti, retained[0][0], retained[0][1], cols,
                             "SUM_CONSUMER_WHOLESALE_RETAINED_NONACCRUAL",
                             [dict(row_index=ri, row_text=row_text(r), value_millions=float(vv[k]))
                              for k,(ri,r) in enumerate(retained)])
            elif ticker == "BAC" and "Nonperforming Loans, Leases and Foreclosed Properties" in text[:500]:
                read(r"Total nonperforming loans and leases", "npl_usd", limit=2)
            elif ticker == "BAC" and "Quarterly Net Charge-offs and Net Charge-off Ratios" in text[:500]:
                # Each group contains an amount AND a ratio: choose only the amount
                # with the explicit dollar marker, not every second arbitrary number.
                for ri,row in enumerate(rows):
                    if label(row) != "Total net charge-offs":
                        continue
                    for k,(start,period) in enumerate(cols[:2]):
                        stop = cols[k+1][0]
                        cells = [(c,t) for c,(t,o) in sorted(row.items())
                                 if c==o and start<=c<stop and t]
                        dollars = [j for j,(_,t) in enumerate(cells) if t=="$"]
                        if len(dollars)==1 and dollars[0]+1<len(cells):
                            emit("net_chargeoffs_usd", period, amount(cells[dollars[0]+1][1]),
                                 ti,ri,row,cols)
            elif ticker == "BAC" and "Allocation of the Allowance for Credit Losses by Product Type" in text[:500]:
                # Amount headers identify the amount column, distinct from ratios.
                for ri,row in enumerate(rows):
                    if label(row) != "Allowance for loan and lease losses":
                        continue
                    for k,(start,period) in enumerate(cols[:2]):
                        stop = cols[k+1][0]
                        amount_cols = [c for r in rows[:7] for c,(t,o) in r.items()
                                       if c==o and start<=c<stop and t=="Amount"]
                        if len(amount_cols) != 1:
                            continue
                        ac = amount_cols[0]
                        candidates = [t for c,(t,o) in sorted(row.items())
                                      if c==o and ac<=c<ac+4 and t and t!="$"]
                        if len(candidates)==1:
                            emit("allowance_usd", period, amount(candidates[0]),ti,ri,row,cols)
            elif ticker == "C" and "Total Citigroup" in text and "ACLL at end of period (a)" in text:
                stop = next(i for i,r in enumerate(rows) if label(r) == "Consumer")
                segment = list(enumerate(rows))[:stop]
                # Citigroup columns run oldest-to-newest. Latest date may not have
                # the next historical date as sentinel: header geometry handles it.
                read(r"ACLL at end of period \(a\)", "allowance_usd", segment)
                read(r"Provision for credit losses on loans \(PCLL\)", "provision_usd", segment)
                read(r"Net credit \(losses\) / recoveries on loans \(NCLs\)", "net_chargeoffs_usd", segment, negate=True)
                read(r"Other, net(?:\s*\(\d+\))*", "acl_other_changes_usd", segment, dash_zero=True)
            elif ticker == "C" and "Corporate non-accrual loans by region" in text:
                read(r"Total non-accrual loans \(NAL\)", "npl_usd")
            elif ticker == "WFC" and "Allowance for loan losses" in text and "Net loan charge-offs:" in text:
                read(r"Allowance for loan losses", "allowance_usd", limit=2)
                read(r"Net loan charge-offs", "net_chargeoffs_usd", negate=True, limit=2)
                issues.append(dict(ticker=ticker, source=document["source"],
                                   reason="WFC_PROVISION_AND_OTHER_INCLUDE_UNFUNDED_NOT_FUNDED_ONLY"))
            elif ticker == "WFC" and "Nonaccrual loans:" in text and "Total nonperforming assets" in text:
                # Percent columns present; release's simpler risk table is used below.
                continue
            elif ticker == "WFC" and "Total nonaccrual loans" in text and "Net loan charge-offs as a %" in text:
                read(r"Total nonaccrual loans", "npl_usd", limit=2)
        except (ValueError, InvalidOperation, StopIteration, IndexError) as exc:
            # Reject partial extraction from this table on any layout problem.
            facts[:] = [f for f in facts if f.get("table_index") != ti]
            issues.append(dict(ticker=ticker, source=document["source"],
                               reason="EARNINGS_TABLE_LAYOUT_NOT_ACCEPTED", table_index=ti,
                               detail=str(exc)))
    if ticker == "GS":
        # This exhibit is exported as PDF-style positioned divs, not HTML tables.
        # Only two reviewed, uniquely worded disclosures are admitted. The total
        # (funded + commitments) allowance and total provision are NOT substituted.
        text = clean(" ".join(parser.parts))
        funded = list(re.finditer(
            r"Total allowance for loan losses and losses on lending commitments was "
            r"\$\s*[\d.]+\s*billion\s*\(\$\s*([\d.]+)\s*billion for funded loans\)", text))
        nco = list(re.finditer(
            r"Net charge-offs for ([1-4]Q\d{2}) of \$\s*([\d.]+)\s*million", text))
        if len(funded) == len(nco) == 1:
            period = parse_period(nco[0][1])
            if period:
                dummy = {0: ("Funded loans allowance / loan NCO disclosure", 0)}
                emit("allowance_usd", period.isoformat(), Decimal(funded[0][1])*1000,
                     -1, 0, dummy, [], "REVIEWED_EXACT_FUNDED_LOAN_DISCLOSURE", decimals="-7")
                emit("net_chargeoffs_usd", period.isoformat(), Decimal(nco[0][2]),
                     -1, 1, dummy, [], "REVIEWED_EXACT_QUARTER_LOAN_NCO_DISCLOSURE")
                facts[-2]["row_text"] = funded[0][0]
                facts[-1]["row_text"] = nco[0][0]
        issues.append(dict(ticker=ticker, source=document["source"],
                           reason="GS_FUNDED_ONLY_PROVISION_NPL_NOT_IDENTIFIED_IN_REVIEWED_LAYOUT"))
    if ticker == "BAC" and facts:
        issues.append(dict(ticker=ticker, source=document["source"],
                           reason="BAC_FUNDED_ONLY_PROVISION_AND_ACL_OTHER_NOT_IDENTIFIED"))
    return facts, issues
