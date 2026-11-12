"""Conservative, automatic SEC credit normalization. No network or table guesses.

Companyfacts (USD, entity-wide) and inline-XBRL numeric facts are supported.
Exact concept allowlist, dimensionless entity contexts, per-metric provenance.
Unsupported extensions are diagnostics, not silently interpreted financial facts.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
import re
from macro_credit_tables import TABLE_CONCEPTS, PROFILES, PROFILE_LABELS

METRICS = {
    "npl_usd": ("instant", (
        "FinancingReceivableRecordedInvestmentNonaccrualStatus",
        "FinancingReceivableNonaccrualRecordedInvestment",
        "FinancingReceivableExcludingAccruedInterestNonaccrual",
    )),
    "allowance_usd": ("instant", (
        "FinancingReceivableAllowanceForCreditLosses",
        "AllowanceForLoanAndLeaseLosses",
        "FinancingReceivableAllowanceForCreditLossExcludingAccruedInterest",
    )),
    "provision_usd": ("flow", (
        "ProvisionForLoanAndLeaseLosses",
        "FinancingReceivableAllowanceForCreditLossesProvision",
        "FinancingReceivableExcludingAccruedInterestCreditLossExpenseReversal",
    )),
    "chargeoffs_usd": ("flow", (
        "FinancingReceivableAllowanceForCreditLossesWriteOffs",
        "AllowanceForLoanAndLeaseLossesWriteOffs",
        "FinancingReceivableAllowanceForCreditLossWriteOff",
        "FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossWriteoff",
    )),
    "recoveries_usd": ("flow", (
        "FinancingReceivableAllowanceForCreditLossesRecoveries",
        "AllowanceForLoanAndLeaseLossesRecoveries",
        "AllowanceForLoanAndLeaseLossesRecoveryOfBadDebts",
        "FinancingReceivableAllowanceForCreditLossRecovery",
        "FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossRecovery",
    )),
    "net_chargeoffs_usd": ("flow", (
        "FinancingReceivableAllowanceForCreditLossesNetWriteOffs",
        "FinancingReceivableAllowanceForCreditLossWriteoffAfterRecovery",
        "FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossWriteoffAfterRecovery",
    )),
    "modified_loans_usd": ("instant", (
        "FinancingReceivableRecordedInvestmentTroubledDebtRestructuring",
    )),
    "pik_interest_usd": ("flow", ("InterestIncomePaymentInKind",)),
    "acl_other_changes_usd": ("flow", (
        "FinancingReceivableAllowanceForCreditLossesOtherChanges",
        "AllowanceForLoanAndLeaseLossesOtherChanges",
    )),
}
FIELDS = ("npl_usd", "allowance_usd", "provision_usd", "net_chargeoffs_usd",
          "modified_loans_usd", "pik_interest_usd", "acl_other_changes_usd")
CONCEPTS = {name: (metric, kind) for metric, (kind, names) in METRICS.items() for name in names}
PROFILE = "SEC_LOAN_ALLOWANCE_ENTITY_USD_V1"
RULES_VERSION = "0.3.3"
BANK_CIK = {"JPM": "19617", "BAC": "70858", "C": "831001", "WFC": "72971",
            "GS": "886982", "MS": "895421"}
LEASE_PROFILE = "SEC_LOANS_AND_NET_LEASE_INVESTMENT_ENTITY_USD_V1"
# Explicit WFC loan + net lease profile, never mixed with loan-only facts.
WFC_CONCEPTS = {
    "FinancingReceivableAndNetInvestmentInLeaseExcludingAccruedInterestNonaccrual":
        ("npl_usd", "instant"),
    "FinancingReceivableAndNetInvestmentInLeaseAllowanceForCreditLossExcludingAccruedInterest":
        ("allowance_usd", "instant"),
    "FinancingReceivableAndNetInvestmentInLeaseExcludingAccruedInterestCreditLossExpenseReversal":
        ("provision_usd", "flow"),
    "FinancingReceivableAndNetInvestmentInLeaseExcludingAccruedInterestAllowanceForCreditLossWriteoff":
        ("chargeoffs_usd", "flow"),
    "FinancingReceivableAndNetInvestmentInLeaseExcludingAccruedInterestAllowanceForCreditLossRecovery":
        ("recoveries_usd", "flow"),
    "FinancingReceivableAndNetInvestmentInLeaseExcludingAccruedInterestAllowanceForCreditLossWriteoffRecoveryTotal":
        ("net_chargeoffs_usd", "flow"),
}
CREDIT_CANDIDATE = re.compile(
    r"nonaccrual|nonperform|financingreceivable|allowancefor|charge.?off|"
    r"provision.*(?:credit|loan)|recover.*(?:loan|loss|debt)|paymentinkind|"
    r"pikinterest|modifiedloan|loanmodification|capitalizedinterest", re.I)


def quarter_window(asof, count=8):
    end = date.fromisoformat(expected_quarter(asof))
    out = []
    for _ in range(count):
        out.append(end.isoformat())
        start = date(end.year, ((end.month-1)//3)*3+1, 1)
        end = start-timedelta(days=1)
    return sorted(out)


def compact_diagnostics(items):
    """Occurrences != independent concepts or adverse bank observations."""
    groups = {}
    for row in items:
        key = tuple(row.get(k) for k in ("ticker", "concept", "reason", "source"))
        item = groups.setdefault(key, dict(row, occurrences=0))
        item["occurrences"] += row.get("occurrences", 1)
    return list(groups.values())


def retained_candidate_facts(refresh, asof):
    """Exact reviewed concepts, SEC source entity, USD and empty dimensions.

    Restores saved values, never guesses from text snippets or tag similarity.
    Qualified names in old reports have no URI: this limitation is explicit.
    """
    from urllib.parse import urlparse
    docs = {d.get("source"): d for d in refresh.get("disclosures", [])
            if d.get("status") == "TEXT_EXTRACTED"}
    out = []
    for item in refresh.get("credit_candidate_facts", []):
        ticker = item.get("ticker")
        prefix, _, tag = item.get("concept", "").partition(":")
        standard = prefix == "us-gaap" and tag in CONCEPTS
        wfc = ticker == "WFC" and prefix == "wfc" and tag in WFC_CONCEPTS
        if ticker not in BANK_CIK or not (standard or wfc):
            continue
        parsed = urlparse(item.get("source", ""))
        company = "value" in item
        try:
            if company:
                if (not standard or parsed.scheme != "https" or parsed.netloc != "data.sec.gov" or
                        parsed.path != f"/api/xbrl/companyfacts/CIK{BANK_CIK[ticker].zfill(10)}.json" or
                        item.get("unit") != "USD"):
                    continue
                row = dict(item)
                value = Decimal(str(item["value"]))
                decimals = None
            else:
                doc = docs.get(item.get("source"))
                if (not doc or doc.get("ticker") != ticker or parsed.scheme != "https" or
                        parsed.netloc != "www.sec.gov" or
                        not parsed.path.startswith(f"/Archives/edgar/data/{BANK_CIK[ticker]}/")):
                    continue
                ctx, unit, attrs = item.get("context", {}), item.get("unit", {}), item.get("attributes", {})
                if (ctx.get("dimensions") or ctx.get("identifier", "").lstrip("0") != BANK_CIK[ticker] or
                        unit.get("divided") or len(unit.get("measures", [])) != 1 or
                        unit["measures"][0].split(":")[-1] != "USD" or
                        attrs.get("xsi:nil") in ("true", "1") or attrs.get("name") != item["concept"]):
                    continue
                if attrs.get("format", "").split(":")[-1].lower() not in ("", "num-dot-decimal", "numdotdecimal"):
                    continue
                text = item.get("raw_text", "").replace(",", "").replace("\u00a0", "").strip()
                if not re.fullmatch(r"\d+(?:\.\d+)?", text):
                    continue
                scale = int(attrs.get("scale", "0"))
                if not -12 <= scale <= 12:
                    continue
                value = Decimal(text) * Decimal(10)**scale
                if attrs.get("sign") == "-":
                    value = -value
                decimals = attrs.get("decimals")
                row = dict(item, start=ctx.get("startdate"),
                           end=ctx.get("instant") or ctx.get("enddate"),
                           available_at=item.get("accepted_at") or doc.get("accepted_at"),
                           accession=item.get("accession") or doc.get("accession"))
            known = _known(row, asof)
            if not known or not value.is_finite():
                continue
            out.append(dict(
                ticker=ticker, tag=tag, concept=item["concept"], start=row.get("start"),
                end=row["end"], value=float(value), unit="USD", dimensions=[], entity=ticker,
                available_at=known.isoformat(), accession=row.get("accession"), source=item["source"],
                transport="RETAINED_COMPANYFACTS_CANDIDATE" if company else "RETAINED_INLINE_CANDIDATE",
                decimals=decimals, scope_id=LEASE_PROFILE if wfc else PROFILE,
                namespace_validation="QUALIFIED_NAME_SEC_SOURCE_AND_ENTITY;URI_NOT_RETAINED",
                admission_method="EXACT_REVIEWED_CONCEPT_V032"))
        except (KeyError, TypeError, ValueError, InvalidOperation):
            continue
    return out


def _known(row, asof):
    try:
        value = row.get("available_at") or row.get("accepted_at")
        known = (datetime.fromisoformat(value.replace("Z", "+00:00")) if value else
                 datetime.fromisoformat(row["filed"]).replace(tzinfo=timezone.utc)+timedelta(days=1))
        return known if known.tzinfo and known <= asof and row["end"] <= asof.date().isoformat() else None
    except (ValueError, KeyError, TypeError):
        return None


def _quarter_end(value):
    d = date.fromisoformat(value)
    return (d.month, d.day) in ((3, 31), (6, 30), (9, 30), (12, 31))


def _period_start(end):
    d = date.fromisoformat(end)
    return date(d.year, ((d.month-1)//3)*3+1, 1).isoformat()


def expected_quarter(asof):
    """Most recently completed calendar quarter, NOT assumed publicly filed."""
    d = asof.date()
    return (date(d.year, ((d.month-1)//3)*3+1, 1)-timedelta(days=1)).isoformat()


def company_facts(payload, ticker, source, asof, candidates=None):
    facts, diagnostics = [], []
    for namespace, tags in payload.get("facts", {}).items():
        for tag, info in tags.items():
            if tag not in CONCEPTS or namespace != "us-gaap":
                if CREDIT_CANDIDATE.search(tag):
                    diagnostics.append(dict(concept=f"{namespace}:{tag}",
                                            source=source, ticker=ticker,
                                            reason="UNMAPPED_CONCEPT_NOT_AUTO_APPROVED"))
                    if candidates is not None:
                        for unit, records in info.get("units", {}).items():
                            for row in records:
                                known = _known(row, asof)
                                if known and row["end"] >= quarter_window(asof)[0]:
                                    candidates.append(dict(
                                        ticker=ticker, concept=f"{namespace}:{tag}",
                                        unit=unit, value=row.get("val"), start=row.get("start"),
                                        end=row["end"], available_at=known.isoformat(),
                                        accession=row.get("accn"), source=source,
                                        status="UNVERIFIED_CANDIDATE_NOT_NORMALIZED"))
                continue
            for row in info.get("units", {}).get("USD", []):
                known = _known(row, asof)
                if row.get("form") not in ("10-Q", "10-Q/A", "10-K", "10-K/A") or not known:
                    continue
                try:
                    value = float(row["val"])
                    if not Decimal(str(value)).is_finite():
                        continue
                except (ValueError, TypeError, KeyError):
                    continue
                facts.append(dict(ticker=ticker, concept="us-gaap:"+tag, tag=tag,
                                  start=row.get("start"), end=row["end"], value=value,
                                  unit="USD", dimensions=[], entity=ticker,
                                  available_at=known.isoformat(), filed=row.get("filed"),
                                  accession=row.get("accn"), source=source,
                                  transport="SEC_COMPANYFACTS", decimals=row.get("decimals")))
    return facts, diagnostics


class InlineFacts(HTMLParser):
    """Read contexts/units/nonFraction, not visual neighboring table cells."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.contexts, self.units, self.raw = {}, {}, []
        self.ctx, self.unit, self.fact = None, None, None
        self.capture = None
        self.namespaces = {}
        self.exclude_depth = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.namespaces.update({k.split(":", 1)[1]: v for k, v in attrs if k.startswith("xmlns:")})
        local = tag.split(":")[-1]
        if tag.endswith(":exclude") and self.fact is not None:
            self.exclude_depth += 1
        if local == "context":
            self.ctx = dict(id=a.get("id"), dimensions=[], entity=None)
        elif local == "unit":
            self.unit = dict(id=a.get("id"), measures=[])
        elif self.ctx is not None and local in ("instant", "startdate", "enddate", "identifier",
                                                "explicitmember", "typedmember"):
            self.capture = (local, [])
            if local in ("explicitmember", "typedmember"):
                self.ctx["dimensions"].append(a.get("dimension", "UNKNOWN_DIMENSION"))
        elif self.unit is not None and local == "measure":
            self.capture = ("measure", [])
        elif self.unit is not None and local == "divide":
            self.unit["divided"] = True
        elif tag.endswith(":nonfraction"):
            self.fact = dict(attrs=a, parts=[])

    def handle_data(self, data):
        if self.capture:
            self.capture[1].append(data)
        if self.fact is not None and not self.exclude_depth:
            self.fact["parts"].append(data)

    def handle_endtag(self, tag):
        local = tag.split(":")[-1]
        if tag.endswith(":exclude") and self.exclude_depth:
            self.exclude_depth -= 1
        if self.capture and local == self.capture[0]:
            key, parts = self.capture
            text = "".join(parts).strip()
            if key == "measure" and self.unit is not None:
                self.unit["measures"].append(text)
            elif self.ctx is not None:
                self.ctx[key] = text
            self.capture = None
        if local == "context" and self.ctx is not None:
            self.contexts[self.ctx["id"]] = self.ctx
            self.ctx = None
        elif local == "unit" and self.unit is not None:
            self.units[self.unit["id"]] = self.unit
            self.unit = None
        elif tag.endswith(":nonfraction") and self.fact is not None:
            self.raw.append(self.fact)
            self.fact = None


def inline_facts(html, document, asof, candidates=None):
    parser = InlineFacts()
    parser.feed(html)
    out, diagnostics = [], []
    for raw in parser.raw:
        a = raw["attrs"]
        concept = a.get("name", "")
        prefix, _, tag = concept.partition(":")
        ns = parser.namespaces.get(prefix, "")
        if tag not in CONCEPTS or "fasb.org/us-gaap/" not in ns.lower():
            if CREDIT_CANDIDATE.search(tag):
                diagnostics.append(dict(concept=concept, reason="UNMAPPED_INLINE_CONCEPT",
                                        source=document.get("source")))
                if candidates is not None:
                    candidates.append(dict(
                        ticker=document["ticker"], concept=concept,
                        source=document.get("source"), accession=document.get("accession"),
                        accepted_at=document.get("accepted_at") or document.get("available_at"),
                        context=parser.contexts.get(a.get("contextref"), {}),
                        unit=parser.units.get(a.get("unitref"), {}),
                        attributes=a, raw_text="".join(raw["parts"]),
                        status="UNVERIFIED_CANDIDATE_NOT_NORMALIZED"))
            continue
        ctx = parser.contexts.get(a.get("contextref"), {})
        unit = parser.units.get(a.get("unitref"), {})
        rejection = ("DIMENSIONED_CONTEXT_NOT_BANK_TOTAL" if ctx.get("dimensions") else
                     "UNIT_NOT_SINGLE_USD" if unit.get("divided") or
                     len(unit.get("measures", [])) != 1 or
                     unit["measures"][0].split(":")[-1] != "USD" else None)
        if rejection:
            diagnostics.append(dict(concept=concept, reason=rejection, source=document.get("source")))
            if candidates is not None:
                candidates.append(dict(ticker=document["ticker"], concept=concept, context=ctx,
                                       unit=unit, attributes=a, raw_text="".join(raw["parts"]),
                                       source=document.get("source"), reason=rejection,
                                       status="UNVERIFIED_CANDIDATE_NOT_NORMALIZED"))
            continue
        if a.get("xsi:nil") in ("true", "1"):
            continue
        fmt = a.get("format", "").split(":")[-1].lower()
        # Do not guess comma-decimal/dash/word transforms.
        if fmt not in ("", "num-dot-decimal", "numdotdecimal"):
            diagnostics.append(dict(concept=concept, reason="UNSUPPORTED_NUMERIC_TRANSFORM", format=fmt,
                                    source=document.get("source")))
            continue
        text = "".join(raw["parts"]).replace(",", "").replace("\u00a0", "").strip()
        if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
            continue
        try:
            scale = int(a.get("scale", "0"))
            if not -12 <= scale <= 12:
                continue
            value = Decimal(text)*(Decimal(10)**scale)
            if a.get("sign") == "-":
                value = -value
            if not value.is_finite():
                continue
        except (ValueError, InvalidOperation):
            continue
        row = dict(document, end=ctx.get("instant") or ctx.get("enddate"))
        known = _known(row, asof)
        if not known or not ctx.get("identifier"):
            continue
        # Do not read another entity's tagged subsidiary facts as bank totals.
        cik = str(document.get("cik", "")).lstrip("0")
        if not cik or ctx["identifier"].lstrip("0") != cik:
            continue
        out.append(dict(ticker=document["ticker"], concept="us-gaap:"+tag, tag=tag,
                        start=ctx.get("startdate"), end=row["end"], value=float(value),
                        unit="USD", dimensions=[], entity=document["ticker"],
                        available_at=known.isoformat(), source=document["source"],
                        accession=document.get("accession"), transport="SEC_INLINE_XBRL",
                        decimals=a.get("decimals")))
    return out, diagnostics


def _select(items):
    """Latest disclosure vintage; conflicts at same vintage remain unavailable."""
    if not items:
        return None, None
    latest = max(f["available_at"] for f in items)
    same = [f for f in items if f["available_at"] == latest]
    def radius(f):
        try:
            d = int(f.get("decimals"))
            return .5*10**(-d) if -12 <= d <= 12 else 0
        except (ValueError, TypeError):
            return 0
    # Example: 1.4bn rounded to 0.1bn is compatible with 1.412bn reported
    # to 1m. Prefer precision; genuinely inconsistent intervals still fail.
    chosen = min(same, key=radius)
    if any(abs(f["value"]-chosen["value"]) > radius(f)+radius(chosen) for f in same):
        return None, "CONFLICTING_FACT_VALUES"
    return chosen, None


def normalize_credit(facts, ticker, asof, diagnostics=None, profile=PROFILE):
    """Build last 8 calendar quarters plus expected-quarter missing diagnostics.

    Flows: direct quarter preferred. YTD subtraction only same concept, fiscal
    year start, accession and disclosure vintage; no derived zero 'other'.
    """
    groups, periods = defaultdict(list), set()
    for f in facts:
        try:
            known = _known(f, asof)
            if f.get("ticker") != ticker or not known or not _quarter_end(f["end"]):
                continue
            if f.get("dimensions") or f.get("unit") != "USD" or f.get("entity") != ticker:
                continue
            if f.get("scope_id", PROFILE) != profile:
                continue
            if not Decimal(str(f["value"])).is_finite():
                continue
            if profile == PROFILES.get(ticker):
                if f.get("transport") != "SEC_REVIEWED_EARNINGS_HTML":
                    continue
                metric, kind = TABLE_CONCEPTS[f["tag"]]
                if f["tag"] != f"Reviewed{ticker}{metric}":
                    continue
            else:
                metric, kind = (WFC_CONCEPTS[f["tag"]] if profile == LEASE_PROFILE and ticker == "WFC"
                                else CONCEPTS[f["tag"]])
            if (kind == "instant") != (not f.get("start")):
                continue
            if kind == "flow" and f["start"] > f["end"]:
                continue
            groups[(metric, f["end"])].append(f)
            periods.add(f["end"])
        except (KeyError, ValueError, TypeError):
            continue
    # Calendar window, not "last 8 nonempty observations": never bring 2021
    # back as the predecessor to 2026 Q3 when recent tags are absent.
    ends = quarter_window(asof)
    out = []
    for end in ends:
        values, evidence, issues = {}, {}, {}
        for metric, (kind, _) in METRICS.items():
            items = groups.get((metric, end), [])
            latest_vintage = max((f["available_at"] for f in items), default=None)
            current_items = [f for f in items if f["available_at"] == latest_vintage]
            selected, error = _select([f for f in current_items if kind == "instant" or
                                      f["start"] == _period_start(end)])
            if kind == "flow" and selected is None and not error:
                # Same filing YTD minus prior YTD, with exact calendar-year start.
                derived = []
                for annual in current_items:
                    if annual.get("start") != end[:4]+"-01-01" or not annual.get("accession"):
                        continue
                    prev_end = (date.fromisoformat(_period_start(end))-timedelta(days=1)).isoformat()
                    for before in groups.get((metric, prev_end), []):
                        if (before.get("start") == annual["start"] and
                                before["concept"] == annual["concept"] and
                                before.get("accession") == annual["accession"] and
                                before["available_at"] == annual["available_at"]):
                            derived.append(dict(annual, value=annual["value"]-before["value"],
                                                start=_period_start(end), method="SAME_FILING_YTD_DIFFERENCE",
                                                components=[annual, before]))
                selected, error = _select(derived)
            if selected:
                # No abs() for signed flows; negative provision is meaningful.
                if kind == "instant" and selected["value"] < 0:
                    issues[metric] = "NEGATIVE_STOCK_VALUE_NOT_NORMALIZED"
                    continue
                values[metric] = selected["value"]
                evidence[metric] = dict(selected, definition_id=selected["concept"],
                                        scope_id=profile, method=selected.get("method", "DIRECT_REPORTED_FACT"))
            else:
                issues[metric] = error or ("QUARTER_FLOW_NOT_AVAILABLE" if items else
                                          "NO_ACCEPTED_INPUT_FACT_NOT_PROOF_OF_NONDISCLOSURE")
                if not items and profile == PROFILES.get(ticker):
                    issues[metric] = {
                        "modified_loans_usd": "NO_COMPARABLE_MODIFIED_LOAN_STOCK_IN_REVIEWED_EARNINGS",
                        "pik_interest_usd": "NO_SEPARATELY_IDENTIFIED_QUARTER_PIK_IN_REVIEWED_EARNINGS",
                        "provision_usd": "FUNDED_ONLY_QUARTER_PROVISION_NOT_IDENTIFIED",
                        "acl_other_changes_usd": "FUNDED_ONLY_OTHER_CHANGES_NOT_IDENTIFIED_NOT_ASSUMED_ZERO",
                        "npl_usd": "FUNDED_NONACCRUAL_STOCK_NOT_IDENTIFIED",
                        "net_chargeoffs_usd": "QUARTER_NET_NOT_GROSS_CHARGEOFFS_NOT_IDENTIFIED",
                    }.get(metric, "FIELD_NOT_IDENTIFIED_IN_REVIEWED_EARNINGS_LAYOUT")
        if "net_chargeoffs_usd" not in values and all(k in values for k in ("chargeoffs_usd", "recoveries_usd")):
            co, rec = evidence["chargeoffs_usd"], evidence["recoveries_usd"]
            if (co.get("accession") and co.get("accession") == rec.get("accession") and
                    co["start"] == rec["start"] and co["available_at"] == rec["available_at"] and
                    values["chargeoffs_usd"] >= 0 and values["recoveries_usd"] >= 0):
                values["net_chargeoffs_usd"] = values["chargeoffs_usd"]-values["recoveries_usd"]
                evidence["net_chargeoffs_usd"] = dict(
                    co, definition_id="DERIVED_SAME_SCOPE_WRITEOFFS_MINUS_RECOVERIES",
                    method="WRITEOFFS_MINUS_RECOVERIES", components=[co, rec])
                issues.pop("net_chargeoffs_usd", None)
        if all(k in values for k in ("net_chargeoffs_usd", "chargeoffs_usd", "recoveries_usd")):
            ev = [evidence[k] for k in ("net_chargeoffs_usd", "chargeoffs_usd", "recoveries_usd")]
            if len({(e.get("accession"), e.get("start")) for e in ev}) == 1:
                def rounding(e):
                    try:
                        d = int(e.get("decimals"))
                        return .5*10**(-d) if -12 <= d <= 12 else 0
                    except (TypeError, ValueError):
                        return 0
                if abs(values["net_chargeoffs_usd"] -
                       (values["chargeoffs_usd"]-values["recoveries_usd"])) > sum(map(rounding, ev)):
                    values.pop("net_chargeoffs_usd")
                    evidence.pop("net_chargeoffs_usd")
                    issues["net_chargeoffs_usd"] = "REPORTED_NET_WRITEOFF_MISMATCH_WITH_COMPONENTS"
        public_at = max((e["available_at"] for e in evidence.values()), default=None)
        out.append(dict(ticker=ticker, period_end=end, scope_id=profile,
                        normalization_verified=bool(evidence), flow_basis="QUARTER",
                        normalization_method=("AUTO_SEC_REVIEWED_EARNINGS" if profile == PROFILES.get(ticker)
                                              else "AUTO_SEC_EXACT_CONCEPTS"),
                        scope_description=PROFILE_LABELS.get(ticker) if profile == PROFILES.get(ticker) else None,
                        normalization_version=RULES_VERSION,
                        available_at=public_at, source=next((e["source"] for e in evidence.values()), None),
                        metric_provenance=evidence, extraction_issues=issues,
                        expected_latest_quarter=end == expected_quarter(asof),
                        **{k: values.get(k) for k in FIELDS}))
    return out


def extraction_coverage(refresh, rows, asof):
    """Diagnose collection separately from financial rule outcomes."""
    diagnostics = compact_diagnostics(refresh.get("credit_diagnostics", []))
    raw = refresh.get("credit_raw_facts", [])
    latest = {r["ticker"]: r for r in rows if r["period_end"] == expected_quarter(asof)}
    coverage = []
    for ticker in sorted(latest):
        r = latest[ticker]
        available = [x for x in rows if x["ticker"] == ticker and x.get("available_at")]
        old = [f for f in raw if f.get("ticker") == ticker and f.get("end", "") < quarter_window(asof)[0]]
        coverage.append(dict(
            ticker=ticker, target_period=r["period_end"],
            current_fields=[k for k in FIELDS if r.get(k) is not None],
            latest_accepted_period=max((x["period_end"] for x in available), default=None),
            excluded_old_fact_count=len(old),
            diagnostic_occurrences=sum(d["occurrences"] for d in diagnostics if d.get("ticker") == ticker),
            unique_diagnostic_groups=sum(d.get("ticker") == ticker for d in diagnostics),
            scope_id=r["scope_id"],
            target_candidate_facts=sum(
                x.get("ticker") == ticker and
                (x.get("end") or x.get("context", {}).get("instant") or
                 x.get("context", {}).get("enddate")) == r["period_end"]
                for x in refresh.get("credit_candidate_facts", [])),
            status="CURRENT_PARTIAL_OR_COMPLETE" if r.get("available_at") else "CURRENT_NUMERIC_INPUT_MISSING"))
    return dict(banks=coverage, diagnostics=diagnostics,
                reviewed_earnings_fact_count=sum(f.get("transport") == "SEC_REVIEWED_EARNINGS_HTML"
                                                for f in raw),
                restored_candidate_fact_count=len(retained_candidate_facts(refresh, asof)),
                occurrence_count=sum(d["occurrences"] for d in diagnostics),
                window_start=quarter_window(asof)[0], window_end=expected_quarter(asof),
                source_values_retained=bool(refresh.get("credit_candidate_facts") or refresh.get("source_files")),
                warning="Elutasított tag nem bizonyítja az adat közzétételének hiányát.")


def build_automatic_credit(refresh, fallback_rows, asof):
    """Fresh tagged facts win over incomplete dated snapshot; no snapshot mixing."""
    facts = refresh.get("credit_raw_facts", [])+retained_candidate_facts(refresh, asof)
    facts = [f for f in facts if _known(f, asof)]
    tickers = sorted({r["ticker"] for r in fallback_rows} |
                     set(refresh.get("bank_fetch_status", {})) |
                     {f["ticker"] for f in facts} | {"JPM", "BAC", "C", "WFC", "GS", "MS"})
    auto = []
    for ticker in tickers:
        # No silent cross-scope stitching. A reviewed table profile uses its own
        # comparative periods, not unrelated tagged historical definitions.
        table_end = max((f["end"] for f in facts if f.get("ticker") == ticker and
                         f.get("scope_id") == PROFILES.get(ticker)), default="")
        native_end = max((f["end"] for f in facts if f.get("ticker") == ticker and
                          f.get("scope_id", PROFILE) in (PROFILE, LEASE_PROFILE)), default="")
        if table_end and table_end >= native_end:
            profile = PROFILES[ticker]
        elif ticker == "WFC" and any(f.get("scope_id") == LEASE_PROFILE and
                                   f.get("ticker") == ticker for f in facts):
            profile = LEASE_PROFILE
        else:
            profile = PROFILE
        auto.extend(normalize_credit(facts, ticker, asof, profile=profile))
    return auto
