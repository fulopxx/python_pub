"""Publication boundary tests; Git operations use isolated synthetic temp repos."""
from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest

import check_publication as publication


ROOT = Path(__file__).resolve().parent


class PublicationAllowlistTests(unittest.TestCase):
    def test_existing_root_source_policy_is_preserved(self):
        for name in ("README.md", ".gitignore", "check_publication.py", "gc_engine.py"):
            with self.subTest(name=name):
                self.assertTrue(publication.allowed(name))
        for name in ("local_config.py", "account_local.py", "credentials.py", "x_old.py"):
            with self.subTest(name=name):
                self.assertFalse(publication.allowed(name))

    def test_only_two_exact_package_files_are_added(self):
        self.assertEqual(publication.PACKAGE_FILES, {
            "market_regime_toolkit/__init__.py",
            "market_regime_toolkit/exhaustion.py",
        })
        for name in publication.PACKAGE_FILES:
            with self.subTest(name=name):
                self.assertTrue(publication.allowed(name))

    def test_unreviewed_nested_source_and_data_stay_blocked(self):
        for name in (
            "market_regime_toolkit/extra.py",
            "market_regime_toolkit/local_config.py",
            "market_regime_toolkit/.env",
            "market_regime_toolkit/data.json",
            "market_regime_toolkit/history.csv",
            "market_regime_toolkit/cache/entry.py",
            "market_regime_toolkit/__pycache__/exhaustion.pyc",
            "market_regime_toolkit/exhaustion.py/extra.py",
            "tests/fixtures/input.json",
            "venv/config.py",
        ):
            with self.subTest(name=name):
                self.assertFalse(publication.allowed(name))


@unittest.skipUnless(shutil.which("git"), "Git is required for index-policy tests")
class PublicationGitIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.toolkit = self.repo / "crash_regime_rules"
        self.toolkit.mkdir()
        # Do not inherit GIT_DIR, GIT_INDEX_FILE or other caller-repo overrides.
        self.env = {key: value for key, value in os.environ.items()
                    if not key.upper().startswith("GIT_")}
        self.env.update(
            GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
            GIT_TERMINAL_PROMPT="0",
        )
        self.git("init", "--quiet")
        (self.repo / "README.md").write_text("# Synthetic test repository\n", encoding="utf-8")
        (self.repo / ".gitignore").write_text("", encoding="utf-8")
        shutil.copyfile(ROOT / ".gitignore", self.toolkit / ".gitignore")
        (self.toolkit / "README.md").write_text("# Synthetic toolkit\n", encoding="utf-8")
        shutil.copyfile(ROOT / "check_publication.py", self.toolkit / "check_publication.py")
        self.package = self.toolkit / "market_regime_toolkit"
        self.package.mkdir()
        for name in ("__init__.py", "exhaustion.py"):
            (self.package / name).write_text("# Synthetic reviewed source\n", encoding="utf-8")

    def git(self, *args, input=None):
        return subprocess.run(
            ["git", "-C", str(self.repo), *args], input=input, env=self.env,
            capture_output=True, text=True, check=True,
        ).stdout

    def stage_reviewed_fixture(self):
        self.git(
            "add", "--", "README.md", ".gitignore",
            "crash_regime_rules/README.md", "crash_regime_rules/.gitignore",
            "crash_regime_rules/check_publication.py",
            "crash_regime_rules/market_regime_toolkit/__init__.py",
            "crash_regime_rules/market_regime_toolkit/exhaustion.py",
        )

    def check_index(self):
        with patch_git_environment(self.env), contextlib.redirect_stdout(io.StringIO()) as output:
            result = publication.check(self.repo)
        return result, output.getvalue()

    def test_gitignore_keeps_package_allowlist_narrow(self):
        allowed = (
            "crash_regime_rules/market_regime_toolkit/__init__.py",
            "crash_regime_rules/market_regime_toolkit/exhaustion.py",
        )
        blocked = (
            "crash_regime_rules/market_regime_toolkit/extra.py",
            "crash_regime_rules/market_regime_toolkit/local_config.py",
            "crash_regime_rules/market_regime_toolkit/data.json",
            "crash_regime_rules/market_regime_toolkit/history.csv",
            "crash_regime_rules/market_regime_toolkit/nested/module.py",
            "crash_regime_rules/market_regime_toolkit/__pycache__/x.pyc",
            "crash_regime_rules/venv/x.py",
            "crash_regime_rules/macro_provider_snapshot.json",
        )
        result = subprocess.run(
            ["git", "-C", str(self.repo), "check-ignore", "--no-index", "--stdin", "-z"],
            # Binary NUL framing avoids Windows text-mode LF -> CRLF conversion
            # and Git's quoting of paths that accidentally contain a CR byte.
            input=b"\0".join(path.encode("utf-8") for path in allowed + blocked) + b"\0",
            capture_output=True, text=False, env=self.env, check=True,
        )
        ignored = {path.decode("utf-8") for path in result.stdout.split(b"\0") if path}
        self.assertEqual(ignored, set(blocked))

    def test_reviewed_nested_index_passes_full_checker(self):
        self.stage_reviewed_fixture()
        result, output = self.check_index()
        self.assertEqual(result, 0, output)
        self.assertIn("0 review item(s)", output)

    def test_unreviewed_nested_index_blob_is_rejected(self):
        self.stage_reviewed_fixture()
        # Deliberately construct a bad index entry in this disposable test repo.
        # No ignored data is added to the user's repository and no push occurs.
        oid = self.git("hash-object", "-w", "--stdin", input="# Synthetic unreviewed source\n").strip()
        bad_path = "crash_regime_rules/market_regime_toolkit/extra.py"
        self.git("update-index", "--add", "--cacheinfo", "100644", oid, bad_path)
        result, output = self.check_index()
        self.assertEqual(result, 1, output)
        self.assertIn("outside the initial publication allowlist", output)


def patch_git_environment(environment):
    from unittest.mock import patch
    return patch.dict(os.environ, environment, clear=True)


if __name__ == "__main__":
    unittest.main()
