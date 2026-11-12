"""Rerun state, adversarial review, SEC filing discovery and aligned output."""
from __future__ import annotations

import io
import json
import tempfile
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

import macro_ai_credit_monitor as m
import macro_credit_forensics as f
import macro_public_data as api
import macro_console as ui
from macro_monitor_state import update_state, merge_public_cache
from test_macro_ai_credit_monitor import snapshot, ASOF


def result():
    return m.evaluate(snapshot(), m.default_scenarios(), ASOF)


def rows():
    common = dict(ticker="TEST", scope_id="LOAN_ACL_SAME_PORTFOLIO", normalization_verified=True,
                  flow_basis="QUARTER", source="https://example.test/synthetic")
    return [dict(common, period_end="2026-06-30", allowance_usd=100_000_000, npl_usd=10_000_000,
                 provision_usd=20_000_000, net_chargeoffs_usd=15_000_000, modified_loans_usd=3,
                 pik_interest_usd=1, acl_other_changes_usd=0),
            dict(common, period_end="2026-09-30", allowance_usd=90_000_000, npl_usd=20_000_000,
                 provision_usd=10_000_000, net_chargeoffs_usd=20_000_000, modified_loans_usd=4,
                 pik_interest_usd=2, acl_other_changes_usd=0)]


class ForensicTests(TestCase):
    def test_review_patterns_not_concealment_proof(self):
        r = f.evaluate_forensics([], rows())
        self.assertGreater(r["review_flags"], 0)
        self.assertFalse(r["concealment_proven"])
        self.assertFalse(r["ai_specific_loss_proven"])

    def test_unknown_is_not_pass(self):
        rr = rows()
        del rr[1]["npl_usd"]
        r = f.quantitative_checks(rr)[1]
        self.assertEqual(r["checks"][0]["status"], "NOT_TESTABLE")

    def test_unnormalized_provider_data_not_confirmed(self):
        rr = rows()
        rr[1]["normalization_verified"] = False
        self.assertTrue(all(c["status"] == "NOT_TESTABLE"
                            for c in f.quantitative_checks(rr)[1]["checks"]))

    def test_different_portfolios_not_compared(self):
        rr = rows()
        rr[1]["scope_id"] = "OTHER"
        self.assertFalse(f.quantitative_checks(rr)[1]["comparable"])

    def test_missing_quarter_not_bridged(self):
        rr = rows()
        rr[1]["period_end"] = "2027-03-31"
        self.assertFalse(f.quantitative_checks(rr)[1]["comparable"])

    def test_acl_rollforward_exact(self):
        self.assertEqual(f.quantitative_checks(rows())[1]["checks"][-1]["status"], "NO_TRIGGER")

    def test_other_not_assumed_zero(self):
        rr = rows()
        del rr[1]["acl_other_changes_usd"]
        self.assertEqual(f.quantitative_checks(rr)[1]["checks"][-1]["status"], "NOT_TESTABLE")

    def test_acl_residual_is_review_only(self):
        rr = rows()
        rr[1]["allowance_usd"] = 80_000_000
        self.assertEqual(f.quantitative_checks(rr)[1]["checks"][-1]["status"], "REVIEW_FLAG")

    def test_ytd_flow_rejected(self):
        rr = rows()
        rr[1]["flow_basis"] = "YTD"
        self.assertEqual(f.quantitative_checks(rr)[1]["checks"][1]["status"], "NOT_TESTABLE")

    def test_numeric_strings_converted_before_comparison(self):
        rr = rows()
        rr[0]["npl_usd"], rr[1]["npl_usd"] = "9000000", "10000000"
        self.assertEqual(f.quantitative_checks(rr)[1]["checks"][0]["status"], "REVIEW_FLAG")

    def test_negative_statement_is_only_text_mention(self):
        docs = [dict(ticker="TEST", status="TEXT_EXTRACTED", text="We have no material nonaccrual loans.",
                     source="https://example.test/synthetic")]
        r = f.scan_disclosures(docs)
        self.assertEqual(r[0]["status"], "TEXT_MENTION_NOT_ADVERSE_CONCLUSION")
        self.assertTrue(r[0]["negation_or_boilerplate_not_resolved"])
        self.assertIsNone(r[0]["probability"])

    def test_ai_substring_not_ai_exposure(self):
        docs = [dict(ticker="TEST", status="TEXT_EXTRACTED",
                     text="We maintain a CECL model.", source="https://example.test/synthetic")]
        self.assertFalse(f.scan_disclosures(docs)[0]["ai_context_present"])

    def test_ai_keyword_is_context_not_causation(self):
        docs = [dict(ticker="TEST", status="TEXT_EXTRACTED",
                     text="AI data center loan modifications are discussed here.")]
        self.assertTrue(f.scan_disclosures(docs)[0]["ai_context_present"])


class StateTests(TestCase):
    def test_first_seen_not_publication_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = update_state(Path(tmp)/"s.json", result())
            self.assertTrue(r["first_baseline"])
            self.assertEqual(r["changes"][0]["kind"], "FIRST_SEEN_OBSERVATION")

    def test_identical_rerun_no_false_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"s.json"
            update_state(p, result())
            self.assertEqual(update_state(p, result())["changes"], [])

    def test_value_revision_separate_from_new_period(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"s.json"
            r = result()
            update_state(p, r)
            r["earnings"][0]["actual_eps"] = 4
            self.assertEqual(update_state(p, r)["changes"][0]["kind"], "VALUE_OR_REVISION_CHANGED")

    def test_failed_fetch_not_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"s.json"
            update_state(p, result())
            empty = dict(asof=ASOF.isoformat(), earnings=[], public_refresh={"banks": [
                dict(ticker="JPM", status="FETCH_OR_PARSE_ERROR")]})
            second = update_state(p, empty)
            self.assertEqual(second["changes"], [])
            self.assertEqual(second["remembered_total"], 1)

    def test_partial_value_not_erase_good_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"s.json"
            r = result()
            update_state(p, r)
            r["earnings"][0]["actual_eps"] = None
            self.assertEqual(update_state(p, r)["changes"], [])
            self.assertEqual(list(json.loads(p.read_text())["observations"].values())[0]["actual_eps"], 5)

    def test_corrupt_state_backup_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"s.json"
            p.write_text("{invalid")
            r = update_state(p, result())
            self.assertTrue(r["warning"])
            self.assertEqual(len(list(Path(tmp).glob("s.json.invalid.*"))), 1)

    def test_cache_failure_keeps_success_for_explicit_offline(self):
        old = dict(banks=[dict(ticker="JPM", status="FILED_QUARTER_FACTS_ONLY_NOT_LATEST_EARNINGS", period="Q2")])
        new = dict(collected_at=ASOF.isoformat(), banks=[dict(ticker="JPM", status="FETCH_OR_PARSE_ERROR")])
        merged = merge_public_cache(old, new)
        self.assertEqual(merged["banks"][0]["period"], "Q2")
        self.assertIn("OFFLINE", merged["messages"][0])


class FilingTests(TestCase):
    def payload(self):
        return dict(filings=dict(recent=dict(
            form=["8-K", "10-Q", "8-K", "8-K"],
            items=["2.02,9.01", "", "5.02", "2.02"],
            accessionNumber=["0000019617-26-000001", "0000019617-26-000002",
                             "0000019617-26-000003", "0000019617-26-000004"],
            acceptanceDateTime=["2026-10-13T12:00:00Z", "2026-08-01T12:00:00Z",
                                "2026-10-13T15:00:00Z", "2026-10-15T12:00:00Z"],
            filingDate=["2026-10-13", "2026-08-01", "2026-10-13", "2026-10-15"],
            primaryDocument=["earnings.htm", "quarter.htm", "board.htm", "future.htm"])))

    def test_new_earnings_and_periodic_but_not_board_or_future(self):
        r = api.recent_filings(self.payload(), "JPM", "0000019617", ASOF)
        self.assertEqual([x["primary_document"] for x in r], ["earnings.htm", "quarter.htm"])

    def test_bad_document_path_rejected(self):
        p = self.payload()
        p["filings"]["recent"]["primaryDocument"][0] = "../private"
        self.assertEqual(len(api.recent_filings(p, "JPM", "0000019617", ASOF)), 1)

    def test_exhibit_source_quality_is_explicit(self):
        filings = api.recent_filings(self.payload(), "JPM", "0000019617", ASOF)[:1]
        with patch.object(api, "request_json", return_value={"directory": {"item": [{"name": "ex99.htm"}]}}), \
             patch.object(api, "request_document", return_value=dict(
                 text="credit quality", credit_raw_facts=[], credit_diagnostics=[])):
            r = api.fetch_disclosures(filings, "Test research test@example.test")
        self.assertIn("TYPE_UNVERIFIED", r[-1]["document_quality"])
        self.assertFalse(r[-1]["normalized_financials_extracted"])

    def test_html_scripts_not_rendered(self):
        p = api.PlainText()
        p.feed("<p>credit</p><script>evil()</script><style>bad</style><p>quality</p>")
        self.assertNotIn("evil", " ".join(p.parts))
        self.assertIn("credit", " ".join(p.parts))


class PresentationTests(TestCase):
    def test_unicode_table_aligned_by_display_cells(self):
        t = ui.table(["A", "B"], [["✅ rendben", "↑ +2.0%"], ["❌ N/A", "a longwordabcdefghijk"]], [12, 12])
        self.assertEqual(len({ui.width(line) for line in t}), 1)

    def test_ascii_table_has_no_unicode_borders_or_emoji(self):
        r = result()
        rendered = "\n".join(m.console(r, ascii_only=True))
        self.assertNotIn("✅", rendered)
        self.assertNotIn("┌", rendered)
        self.assertIn("[X]", rendered)

    def test_consensus_requires_pre_release_and_same_basis(self):
        prices = [dict(series_id="TEST", period="2026-09", basis="MOM_SA", change_pct=.4)]
        consensus = [dict(series_id="TEST", period="2026-09", basis="MOM_SA", unit="percent",
                          value=.3, source="https://example.test/synthetic", recorded_at="2026-10-13T10:00:00Z",
                          scheduled_release_at="2026-10-13T12:30:00Z")]
        r = m.apply_public_consensus(prices, consensus, ASOF)
        self.assertAlmostEqual(r[0]["surprise"]["difference"], .1)
        self.assertIn("UNVERIFIED", r[0]["surprise"]["status"])
        consensus[0]["basis"] = "YOY_NSA"
        self.assertIsNone(m.apply_public_consensus(prices, consensus, ASOF)[0]["surprise"]["difference"])

    def test_after_release_consensus_not_accepted(self):
        prices = [dict(series_id="TEST", period="2026-09", basis="MOM_SA", change_pct=.4)]
        c = [dict(series_id="TEST", period="2026-09", basis="MOM_SA", unit="percent", value=.3,
                  source="https://example.test/synthetic", recorded_at="2026-10-13T13:00:00Z",
                  scheduled_release_at="2026-10-13T12:30:00Z")]
        self.assertIsNone(m.apply_public_consensus(prices, c, ASOF)[0]["surprise"]["difference"])

    def test_offline_cli_twice_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root/"input.json"
            m.atomic_json(inp, snapshot())
            args = ["--snapshot", str(inp), "--output-dir", str(root/"out"), "--offline", "--ascii"]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(m.main(args), 0)
                self.assertEqual(m.main(args), 0)
            latest = json.loads((root/"out"/"macro_ai_latest.json").read_text())
            self.assertEqual(latest["run_changes"]["changes"], [])
            self.assertFalse(latest["forensics"]["concealment_proven"])
            self.assertEqual(len(list((root/"out"/"history").glob("*.json"))), 2)

    def test_default_cli_attempts_new_fetch_mock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root/"input.json"
            m.atomic_json(inp, snapshot())
            fake = dict(collected_at=ASOF.isoformat(), banks=[], prices=[], filings=[], disclosures=[],
                        bank_fetch_status={}, messages=[])
            with patch.object(api, "refresh_public", return_value=fake) as refresh, redirect_stdout(io.StringIO()):
                self.assertEqual(m.main(["--snapshot", str(inp), "--output-dir", str(root/"out"), "--ascii"]), 0)
            refresh.assert_called_once()


if __name__ == "__main__":
    main()
