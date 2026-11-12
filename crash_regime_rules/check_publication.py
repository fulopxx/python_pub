"""Read-only check of the Git index, not a complete secret scanner or history audit."""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

DOCS = {
    "README.md", "CHANGELOG.md", ".gitignore", "requirements.txt",
    "GC_audit_1_3_12_felbontasi_profilok.md",
    "GC_1_0_19_NDOG_NWOG_celarak.md",
    "QQQ_options_advisor_utmutato.md", "ES_NQ_regime_v12_utmutato.md",
    "ICT_market_transition_es_GC_breaker_utmutato.md",
    "VIX_stress_monitor_utmutato.md", "Makro_AI_bankhitel_console_utmutato.md",
}
SENSITIVE_NAMES = re.compile(
    r"(?i)(^local_.*\.py$|.*_local\.py$|.*secret.*\.py$|"
    r".*credential.*\.py$|.*_(?:backup|old)\.py$)"
)
PATTERNS = (
    ("private-key material", re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")),
    ("GitHub token-like value", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    ("AWS access-key-like value", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("API token-like value", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b")),
    ("literal credential assignment", re.compile(
        r"""(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|client[_-]?secret)\b["']?\s*[:=]\s*["']([^"'\r\n]{8,})["']""")),
    ("broker account-like identifier", re.compile(r"\b(?:DU|U)\d{6,}\b")),
    ("URL containing username/password", re.compile(r"https?://[^/\s:@]+:[^/\s@]+@")),
    ("personal workstation path", re.compile(r"(?i)(?:[A-Z]:[\\/]Users[\\/]|/(?:home|Users)/[^/\s]+/)")),
)
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)


def findings(text: str) -> list[tuple[int, str]]:
    result = []
    for line_no, line in enumerate(text.splitlines(), 1):
        for label, pattern in PATTERNS:
            for match in pattern.finditer(line):
                if label == "literal credential assignment":
                    value = match.group(1).lower()
                    if value in {"example", "placeholder", "dummy", "test"} or re.fullmatch(
                        r"(?:<[^>]+>|(?:your|replace|example|dummy|test)[_-][a-z0-9_-]+)", value
                    ):
                        continue
                result.append((line_no, label))
        for match in EMAIL.finditer(line):
            domain = match.group().rsplit("@", 1)[1].lower()
            if domain in {"example.com", "example.org", "example.net"} or domain.endswith((".test", ".invalid")):
                continue
            result.append((line_no, "email address: review privacy"))
    return result


def allowed(relative: str) -> bool:
    p = PurePosixPath(relative)
    return (
        len(p.parts) == 1
        and not SENSITIVE_NAMES.fullmatch(p.name)
        and (p.name in DOCS or p.suffix == ".py")
    )


def check(repo: Path) -> int:
    root = Path(git(repo, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    prefix = "crash_regime_rules/"
    issues = []
    entries = {}
    for record in git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        meta, raw_path = record.split(b"\t", 1)
        mode, oid, stage = meta.decode("ascii").split()
        path = raw_path.decode("utf-8")
        if path.startswith(prefix) or path in {"README.md", ".gitignore"}:
            if stage != "0":
                issues.append((path, 0, "unmerged index entry"))
            entries[path] = (mode, oid)
    if not any(p.startswith(prefix) and p.endswith(".py") for p in entries):
        issues.append((prefix, 0, "no toolkit Python files are staged/tracked"))
    for required in ("README.md", ".gitignore", prefix + "README.md", prefix + ".gitignore", prefix + "check_publication.py"):
        if required not in entries:
            issues.append((required, 0, "required publication file missing from index"))
    changed = git(root, "diff", "--cached", "--name-only", "-z").split(b"\0")
    for raw_path in changed:
        if not raw_path:
            continue
        path = raw_path.decode("utf-8")
        if not path.startswith(prefix) and path not in {"README.md", ".gitignore"}:
            issues.append((path, 0, "unrelated staged change: review separately"))
    # Existing tracked toolkit files are checked as well as newly staged files.
    # This uses current worktree ignore rules, so the guide also requires diffing
    # the staged ignore file and rerunning immediately before committing.
    for raw_path in git(root, "ls-files", "-ci", "--exclude-standard", "-z").split(b"\0"):
        if raw_path:
            path = raw_path.decode("utf-8")
            if path.startswith(prefix):
                issues.append((path, 0, "tracked despite current ignore rules"))
    total = 0
    for path, (mode, oid) in sorted(entries.items()):
        if mode != "100644":
            issues.append((path, 0, "non-regular file, executable, symlink or embedded repository"))
            continue
        if path.startswith(prefix) and not allowed(path[len(prefix):]):
            issues.append((path, 0, "outside the initial publication allowlist"))
            continue
        size = int(git(root, "cat-file", "-s", oid))
        if size > 1024 * 1024:
            issues.append((path, 0, "over 1 MiB: review necessity before publication"))
            continue
        raw = git(root, "cat-file", "blob", oid)
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            issues.append((path, 0, "not UTF-8 text"))
            continue
        if b"\0" in raw:
            issues.append((path, 0, "binary content"))
            continue
        total += 1
        issues.extend((path, n, label) for n, label in findings(text))
    for path, line_no, label in issues:
        # Never print the matched value or full line.
        print(f"REVIEW {path}:{line_no}: {label}")
    print(f"Checked {total} index blobs; {len(issues)} review item(s).")
    print("Read-only. No network, file edits, staging, commits or pushes performed.")
    print("Heuristic only: review ALL selected diffs and outgoing commits manually.")
    return 1 if issues else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        return check(args.repo)
    except (subprocess.CalledProcessError, UnicodeError, ValueError, OSError):
        print("ERROR: cannot safely read Git index; check Git installation, path and repository state.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
