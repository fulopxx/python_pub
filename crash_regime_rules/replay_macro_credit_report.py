"""Re-evaluate a saved monitor report without network, orders or live-state changes."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, BadZipFile
from urllib.parse import urlparse

from macro_ai_credit_monitor import VERSION, atomic_json, console
from macro_credit_autodata import (
    build_automatic_credit, company_facts, compact_diagnostics, extraction_coverage, inline_facts,
)
from macro_credit_forensics import quantitative_checks
from macro_public_data import BANKS
from macro_credit_tables import table_facts


def import_earnings_zip(refresh, bundle, asof):
    """In-memory, bounded, hash-verified public HTML import. Never extract paths."""
    pending, issues = [], []
    with ZipFile(bundle) as z:
        infos = z.infolist()
        if (len(infos) > 100 or len({i.filename for i in infos}) != len(infos) or
                sum(i.file_size for i in infos) > 80_000_000):
            raise ValueError("Duplicate entries / excessive ZIP size or member count.")
        manifest = json.loads(z.read("manifest.json"))
        if manifest.get("public_sec_documents_only") is not True:
            raise ValueError("Public SEC manifest required.")
        declared = {"manifest.json"}
        for meta in manifest.get("files", []):
            name, ticker = meta["filename"], meta["ticker"]
            if (ticker not in BANKS or Path(name).name != name or
                    "/" in name or "\\" in name or not name.endswith(".html")):
                raise ValueError("Invalid source filename or bank.")
            member = "html/"+name
            if member in declared:
                raise ValueError("Duplicate source in manifest.")
            declared.add(member)
            data = z.read(member)
            if hashlib.sha256(data).hexdigest() != meta["sha256"]:
                raise ValueError("Source SHA256 mismatch: "+name)
            parsed = urlparse(meta["source"])
            cik = BANKS[ticker].lstrip("0")
            accession = meta["accession"]
            if (parsed.scheme != "https" or parsed.netloc != "www.sec.gov" or
                    not parsed.path.startswith(f"/Archives/edgar/data/{cik}/{accession.replace('-', '')}/")):
                raise ValueError("Source CIK/accession mismatch.")
            ff, dd = table_facts(data.decode("utf-8"), meta, asof)
            pending.extend(ff)
            issues.extend(dd)
        if set(z.namelist()) != declared:
            raise ValueError("ZIP contains undeclared files.")
    # Commit only after the whole manifest validated; a corrupt last file cannot
    # leave an apparently successful half-import.
    refresh.setdefault("credit_raw_facts", []).extend(pending)
    refresh.setdefault("credit_diagnostics", []).extend(issues)
    refresh["earnings_bundle_import"] = dict(
        files=len(declared)-1, sha256_verified=True, extracted_facts=len(pending),
        network_used=False, normalization_version=VERSION)


def replay(report, source_dir=None, earnings_zip=None):
    result = deepcopy(report)
    asof = datetime.fromisoformat(result["asof"].replace("Z", "+00:00"))
    if asof.tzinfo is None:
        raise ValueError("Replay requires timezone-aware saved asof.")
    refresh = result.setdefault("public_refresh", {})
    messages = refresh.setdefault("messages", [])
    if source_dir:
        # Never follow arbitrary local paths from a supplied report. User selects
        # a source folder explicitly; content hashes must match report metadata.
        folder = Path(source_dir)
        docs = {d.get("source"): d for d in refresh.get("disclosures", [])}
        for meta in refresh.get("source_files", []):
            ticker = meta.get("ticker")
            if ticker not in BANKS:
                continue
            name = str(meta.get("path", "")).replace("\\", "/").rsplit("/", 1)[-1]
            if not name or name in (".", ".."):
                continue
            path = folder/name
            try:
                if path.resolve().parent != folder.resolve():
                    raise ValueError("Source path outside supplied folder.")
                data = path.read_bytes()
                raw = data.decode("utf-8")
                if meta["kind"] == "SEC_COMPANYFACTS":
                    payload = json.loads(raw)
                    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                    if str(payload.get("cik", "")).lstrip("0") != BANKS[ticker].lstrip("0"):
                        raise ValueError("Source entity CIK mismatch.")
                else:
                    digest = hashlib.sha256(data).hexdigest()
                    if digest != meta.get("sha256"):
                        restored = data.replace(b"\r\n", b"\n")
                        if hashlib.sha256(restored).hexdigest() == meta.get("sha256"):
                            raw = restored.decode("utf-8")
                            digest = meta["sha256"]
                if digest != meta.get("sha256"):
                    raise ValueError("Source content hash mismatch.")
                candidates = []
                if meta["kind"] == "SEC_COMPANYFACTS":
                    facts, issues = company_facts(payload, ticker, meta["source"], asof, candidates)
                elif meta["kind"] == "SEC_FILING_HTML" and meta["source"] in docs:
                    doc = dict(docs[meta["source"]], ticker=ticker, cik=BANKS[ticker])
                    facts, issues = inline_facts(raw, doc, asof, candidates)
                    ff, dd = table_facts(raw, doc, asof)
                    facts.extend(ff)
                    issues.extend(dd)
                else:
                    continue
                refresh.setdefault("credit_raw_facts", []).extend(facts)
                refresh.setdefault("credit_diagnostics", []).extend(dict(i, ticker=ticker) for i in issues)
                refresh.setdefault("credit_candidate_facts", []).extend(candidates)
            except (OSError, ValueError, KeyError) as exc:
                messages.append(f"REPLAY_SOURCE_ERROR {ticker}: {type(exc).__name__}: {exc}")
    if earnings_zip:
        import_earnings_zip(refresh, earnings_zip, asof)
    rows = build_automatic_credit(refresh, result.get("credit_facts", []), asof)
    quant = quantitative_checks(rows)
    audit = result.setdefault("forensics", {})
    audit.update(quantitative=quant, automatic_extraction=True,
                 review_flags=sum(c["status"] == "REVIEW_FLAG" for r in quant for c in r["checks"]),
                 not_testable=sum(c["status"] == "NOT_TESTABLE" for r in quant for c in r["checks"]),
                 flag_count=sum(c["status"] == "REVIEW_FLAG" for r in quant for c in r["checks"]),
                 not_testable_count=sum(c["status"] == "NOT_TESTABLE" for r in quant for c in r["checks"]),
                 extraction_diagnostics=compact_diagnostics(refresh.get("credit_diagnostics", [])))
    result.update(version=VERSION, replay=True, original_version=report["version"],
                  refresh_mode="SAVED_REPORT_REPLAY_NO_NETWORK",
                  run_changes=dict(changes=[], remembered_total=0, first_baseline=False),
                  normalized_credit_facts=rows, credit_coverage=extraction_coverage(refresh, rows, asof))
    result["credit_coverage"]["replay_limitation"] = (
        "Az eredeti asof marad; nincs új hálózat vagy friss mai adat. "
        "A nem azonosított mező továbbra is hiányzik, nem nulla.")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--source-dir", type=Path)
    p.add_argument("--earnings-zip", type=Path, help="Hash-ellenőrzött nyilvános earnings HTML-csomag.")
    p.add_argument("--output-dir", type=Path, default=Path("macro_credit_replay"))
    p.add_argument("--ascii", action="store_true")
    args = p.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = replay(report, args.source_dir, args.earnings_zip)
    atomic_json(args.output_dir/"macro_ai_replay.json", result)
    atomic_json(args.output_dir/"macro_credit_normalized.json", result["normalized_credit_facts"])
    atomic_json(args.output_dir/"macro_credit_diagnostics.json", result["credit_coverage"])
    text = "\n".join(console(result, ascii_only=args.ascii))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"macro_ai_console.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
