"""Per-observation change tracking: failed/partial requests never erase good history."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from macro_ai_credit_monitor import atomic_json


def observations(result):
    out = {}
    for b in result.get("earnings", []):
        if b.get("status") == "DATED_PROVIDER_SNAPSHOT" and b.get("period"):
            out[f"earnings:{b['ticker']}:{b['period']}"] = {
                k: b.get(k) for k in ("actual_eps", "consensus_eps", "revenue_surprise_pct")}
    refresh = result.get("public_refresh") or {}
    for b in result.get("normalized_credit_facts", []):
        if b.get("available_at"):
            # Scope changes are a new baseline, not a financial deterioration
            # or revision. Do not retain old non-null fields from another scope.
            scope = ":"+b["scope_id"] if b.get("scope_id") else ""
            out[f"credit:{b['ticker']}:{b['period_end']}{scope}"] = {k: b.get(k) for k in (
                "npl_usd", "allowance_usd", "provision_usd", "net_chargeoffs_usd",
                "modified_loans_usd", "pik_interest_usd", "acl_other_changes_usd")}
    for b in refresh.get("banks", []):
        if b.get("status") != "FILED_QUARTER_FACTS_ONLY_NOT_LATEST_EARNINGS":
            continue
        out[f"sec:{b['ticker']}:{b['period']}"] = {k: b.get(k) for k in (
            "provision_usd", "allowance_usd", "equity_usd", "net_loans_usd", "filed")}
    for p in refresh.get("prices", []):
        if p.get("value") is not None:
            out[f"bls:{p['series_id']}:{p['period']}"] = {
                "value": p["value"], "change_pct": p.get("change_pct")}
    for f in refresh.get("filings", []):
        if f.get("accession"):
            out[f"filing:{f['ticker']}:{f['accession']}"] = {
                k: f.get(k) for k in ("form", "items", "accepted_at", "primary_document")}
    return out


def update_state(path, result):
    path = Path(path)
    previous = {}
    warning = None
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            previous = state["observations"]
            if state.get("schema_version") != 1 or not isinstance(previous, dict):
                raise ValueError("Inkompatibilis state")
        except (ValueError, KeyError, TypeError):
            # Preserve the corrupt original; explicit timestamped backup.
            backup = path.with_name(path.name+".invalid."+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
            path.replace(backup)
            previous = {}
            warning = f"State érvénytelen, megőrzött backup: {backup.name}; új baseline."
    current = observations(result)
    changes = []
    merged = dict(previous)
    for key, value in current.items():
        if isinstance(value, dict):
            old = previous.get(key, {}) if isinstance(previous.get(key), dict) else {}
            value = {**old, **{k: v for k, v in value.items() if v is not None}}
        if key not in previous:
            changes.append(dict(key=key, kind="FIRST_SEEN_OBSERVATION", before=None, after=value))
        elif value != previous[key]:
            changes.append(dict(key=key, kind="VALUE_OR_REVISION_CHANGED", before=previous[key], after=value))
        merged[key] = value
    atomic_json(path, dict(schema_version=1, last_run_at=result["asof"], observations=merged))
    return dict(changes=changes, first_baseline=not previous,
                observed_this_run=len(current), remembered_total=len(merged),
                warning=warning, vanished_is_not_deletion=True)


def merge_public_cache(previous, current):
    """Cache for explicitly OFFLINE use. Live reports must show current failures."""
    out = dict(current)
    specs = {"banks": ("ticker", lambda x: x.get("status") == "FILED_QUARTER_FACTS_ONLY_NOT_LATEST_EARNINGS"),
             "prices": ("series_id", lambda x: x.get("value") is not None),
             "filings": ("accession", lambda x: bool(x.get("accession"))),
             "disclosures": ("source", lambda x: x.get("status") == "TEXT_EXTRACTED")}
    for group, (key, usable) in specs.items():
        items = {row[key]: row for row in previous.get(group, []) if key in row and usable(row)}
        for row in current.get(group, []):
            if key in row and usable(row):
                items[row[key]] = dict(row, fetched_at=current["collected_at"])
        out[group] = list(items.values())[-100:]
    # Cache tagged source facts, never re-use them silently in live reports.
    raw = {}
    for row in previous.get("credit_raw_facts", [])+current.get("credit_raw_facts", []):
        key = tuple(row.get(k) for k in ("ticker", "concept", "start", "end", "accession",
                                        "source", "available_at", "value"))
        raw[key] = row
    out["credit_raw_facts"] = list(raw.values())[-20000:]
    candidates = {}
    for row in previous.get("credit_candidate_facts", [])+current.get("credit_candidate_facts", []):
        candidates[json.dumps(row, sort_keys=True, ensure_ascii=False)] = row
    out["credit_candidate_facts"] = list(candidates.values())[-30000:]
    source_files = {}
    for row in previous.get("source_files", [])+current.get("source_files", []):
        source_files[row.get("path")] = row
    out["source_files"] = list(source_files.values())[-100:]
    # Diagnostic occurrences count source observations, not repeated runs.
    out["credit_diagnostics"] = current.get("credit_diagnostics") or previous.get("credit_diagnostics", [])
    out["messages"] = ["OFFLINE CACHE: korábban sikeres megfigyelések, most nem ellenőrizve."]
    out["bank_fetch_status"] = {r["ticker"]: "OFFLINE_LAST_SUCCESS_NOT_REFETCHED" for r in out["banks"]}
    out["cache_last_attempt_at"] = current.get("collected_at")
    return out
