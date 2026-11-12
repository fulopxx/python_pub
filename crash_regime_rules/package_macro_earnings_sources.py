"""Bundle only locally cached public earnings HTML, no API calls or credentials."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from zipfile import ZipFile, ZIP_DEFLATED


def package(report, source_dir, output):
    source_dir, output = Path(source_dir), Path(output)
    exhibits = {d["source"]: d for d in report.get("public_refresh", {}).get("disclosures", [])
                if d.get("status") == "TEXT_EXTRACTED" and
                d.get("document_quality") == "EXHIBIT_FILENAME_CANDIDATE_TYPE_UNVERIFIED"}
    accepted, files, errors = [], {}, []
    for item in report.get("public_refresh", {}).get("source_files", []):
        if item.get("kind") != "SEC_FILING_HTML" or item.get("source") not in exhibits:
            continue
        url = urlparse(item["source"])
        if url.scheme != "https" or url.netloc != "www.sec.gov" or not url.path.startswith("/Archives/edgar/data/"):
            continue
        name = str(item.get("path", "")).replace("\\", "/").rsplit("/", 1)[-1]
        path = source_dir/name
        try:
            if not name.endswith(".html") or path.resolve().parent != source_dir.resolve():
                raise ValueError("Not an HTML file in the explicit source directory")
            data = path.read_bytes()
            repair = None
            if hashlib.sha256(data).hexdigest() != item.get("sha256"):
                restored = data.replace(b"\r\n", b"\n")
                if hashlib.sha256(restored).hexdigest() != item.get("sha256"):
                    raise ValueError("Content hash mismatch")
                data = restored
                repair = "UNDONE_WINDOWS_TEXT_NEWLINE_TRANSLATION_HASH_VERIFIED"
            doc = exhibits[item["source"]]
            accepted.append(dict(ticker=item["ticker"], source=item["source"],
                                 accession=doc.get("accession"), accepted_at=doc.get("accepted_at"),
                                 sha256=item["sha256"], filename=name, newline_repair=repair))
            files[name] = data
        except (OSError, ValueError) as exc:
            errors.append(dict(filename=name, error=type(exc).__name__+": "+str(exc)))
    if not files:
        raise FileNotFoundError("Nincs becsomagolható, hash szerint egyező earnings HTML a megadott sec_sources mappában.")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(asof=report.get("asof"), public_sec_documents_only=True, files=accepted, errors=errors)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        for name, data in files.items():
            archive.writestr("html/"+name, data)
    return manifest


def main(argv=None):
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, default=root/"macro_ai_output"/"macro_ai_latest.json")
    p.add_argument("--source-dir", type=Path, default=root/"macro_ai_output"/"sec_sources")
    p.add_argument("--output", type=Path, default=root/"macro_ai_output"/"macro_earnings_sources.zip")
    args = p.parse_args(argv)
    r = json.loads(args.report.read_text(encoding="utf-8"))
    manifest = package(r, args.source_dir, args.output)
    print(f"Nyilvános SEC earnings HTML-források: {len(manifest['files'])}; kimaradt fájl: {len(manifest['errors'])}")
    print("Nincs új hálózati lekérés; SEC-kapcsolattartót és konfigurációs JSON-okat nem csomagol.")
    print(f"Feltölthető csomag: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
