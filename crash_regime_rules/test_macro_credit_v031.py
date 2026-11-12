"""Regression tests for tag gaps observed in the user's 0.3.0 exports.

Amounts are synthetic. Tests never call SEC/BLS or send orders.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import macro_credit_autodata as a
import macro_public_data as public
from macro_ai_credit_monitor import atomic_json, console
from replay_macro_credit_report import replay
from test_macro_ai_credit_monitor import snapshot
import test_macro_credit_autodata as samples
from test_macro_credit_autodata import ASOF, fact, inputs


def mini_report():
    from macro_ai_credit_monitor import evaluate
    from pathlib import Path
    scenarios = json.loads(Path("macro_scenarios.json").read_text())
    r = evaluate(snapshot(), scenarios, ASOF)
    r.update(asof=ASOF.isoformat(), version="0.3.0",
             public_refresh=dict(credit_raw_facts=inputs(), credit_diagnostics=[
                 dict(ticker="JPM", concept="us-gaap:FinancingReceivableExcludingAccruedInterestNonaccrual",
                      reason="UNMAPPED_CONCEPT_NOT_AUTO_APPROVED")]*3,
                 prices=[], filings=[], disclosures=[], banks=[], messages=[]))
    r["forensics"] = dict(disclosure_hits=[], method_priorities=[], disclaimer="Test",
                          documents_scanned=0, warning="Synthetic")
    return r


class Credit031Tests(unittest.TestCase):
    def test_modern_nonaccrual_concept_accepted(self):
        raw = [fact("FinancingReceivableExcludingAccruedInterestNonaccrual", "2026-09-30", 17)]
        self.assertEqual(a.normalize_credit(raw, "JPM", ASOF)[-1]["npl_usd"], 17)

    def test_modern_allowance_concept_accepted(self):
        raw = [fact("FinancingReceivableAllowanceForCreditLossExcludingAccruedInterest", "2026-09-30", 19)]
        self.assertEqual(a.normalize_credit(raw, "JPM", ASOF)[-1]["allowance_usd"], 19)

    def test_dimensioned_subset_never_becomes_total(self):
        parser = samples.InlineTests()
        html = parser.html(dimensions='<xbrldi:explicitMember dimension="loan:Type">Consumer</xbrldi:explicitMember>')
        candidates = []
        facts, diag = a.inline_facts(html, parser.doc(), ASOF, candidates)
        self.assertEqual(facts, [])
        self.assertTrue(candidates[0]["context"]["dimensions"])
        self.assertEqual(diag[0]["reason"], "DIMENSIONED_CONTEXT_NOT_BANK_TOTAL")

    def test_8_adjacent_calendar_quarters_even_when_empty(self):
        rows = a.normalize_credit([], "JPM", ASOF)
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["period_end"], "2024-12-31")
        self.assertEqual(rows[-2]["period_end"], "2026-06-30")

    def test_stale_input_not_backfilled_into_current_window(self):
        raw = [fact("FinancingReceivableAllowanceForCreditLosses", "2021-06-30", 20)]
        rows = a.normalize_credit(raw, "JPM", ASOF)
        self.assertFalse(any(r["normalization_verified"] for r in rows))
        self.assertTrue(all(r["period_end"] >= "2024-12-31" for r in rows))

    def test_diagnostic_occurrences_not_concept_count(self):
        items = [dict(ticker="JPM", concept="x", reason="unmapped", source="s")]*10
        compact = a.compact_diagnostics(items)
        self.assertEqual(len(compact), 1)
        self.assertEqual(compact[0]["occurrences"], 10)
        self.assertEqual(a.compact_diagnostics(compact), compact)

    def test_no_cross_bank_diagnostic_merging(self):
        rows = [dict(ticker=t, concept="x", reason="u", source="s") for t in ("JPM", "BAC")]
        self.assertEqual(len(a.compact_diagnostics(rows)), 2)

    def test_missing_current_fact_does_not_claim_not_published(self):
        row = a.normalize_credit([], "JPM", ASOF)[-1]
        self.assertIn("NOT_PROOF_OF_NONDISCLOSURE", row["extraction_issues"]["npl_usd"])

    def test_unmapped_candidate_values_preserved_not_approved(self):
        payload = {"facts": {"jpm": {"NonaccrualLoansSpecialSubset": {"units": {"USD": [
            dict(val=123, end="2026-06-30", filed="2026-08-01", form="10-Q", accn="fake")]}}}}}
        candidates = []
        approved, diag = a.company_facts(payload, "JPM", "https://example.test", ASOF, candidates)
        self.assertEqual(approved, [])
        self.assertEqual(candidates[0]["value"], 123)
        self.assertEqual(candidates[0]["status"], "UNVERIFIED_CANDIDATE_NOT_NORMALIZED")

    def test_broad_provision_not_silently_loan_only(self):
        raw = [fact("ProvisionForCreditLosses", "2026-09-30", 999, "2026-07-01")]
        self.assertIsNone(a.normalize_credit(raw, "JPM", ASOF)[-1]["provision_usd"])

    def test_source_html_retained_with_content_hash(self):
        parser = samples.InlineTests()
        with tempfile.TemporaryDirectory() as tmp:
            doc = dict(parser.doc(), source_dir=tmp)
            with patch.object(public, "_request", return_value=parser.html().encode()):
                r = public.request_document(doc["source"], "research", doc, ASOF)
            path = Path(r["source_file"]["path"])
            self.assertTrue(path.exists())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), r["source_file"]["sha256"])

    def test_replay_never_performs_network(self):
        with patch.object(public, "_request", side_effect=AssertionError("No network")):
            r = replay(mini_report())
        self.assertTrue(r["replay"])
        self.assertEqual(r["asof"], ASOF.isoformat())
        self.assertEqual(r["refresh_mode"], "SAVED_REPORT_REPLAY_NO_NETWORK")

    def test_replay_cannot_invent_missing_candidate_values(self):
        report = mini_report()
        report["public_refresh"]["credit_raw_facts"] = []
        r = replay(report)
        self.assertTrue(all(not row["normalization_verified"] for row in r["normalized_credit_facts"]))
        self.assertFalse(r["credit_coverage"]["source_values_retained"])

    def test_replay_does_not_mutate_original(self):
        report = mini_report()
        before = deepcopy(report)
        replay(report)
        self.assertEqual(report, before)

    def test_console_has_glossary_and_replay_warning(self):
        text = "\n".join(console(replay(mini_report()), ascii_only=True))
        self.assertIn("RÖVIDÍTÉSEK", text)
        self.assertIn("NINCS MOSTANI HÁLÓZATI LEKÉRÉS", text)
        self.assertIn("NEM hányados", text)
        self.assertIn("Célnegyedév:", text)

    def test_coverage_old_facts_explicitly_excluded(self):
        raw = [fact("FinancingReceivableAllowanceForCreditLosses", "2021-06-30", 20)]
        refresh = dict(credit_raw_facts=raw)
        rows = a.build_automatic_credit(refresh, [], ASOF)
        c = a.extraction_coverage(refresh, rows, ASOF)
        jpm = next(r for r in c["banks"] if r["ticker"] == "JPM")
        self.assertEqual(jpm["excluded_old_fact_count"], 1)
        self.assertEqual(jpm["current_fields"], [])

    def test_source_replay_requires_hash_match(self):
        report = mini_report()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"fake.json"
            atomic_json(path, dict(cik=19617, facts={}))
            report["public_refresh"]["source_files"] = [
                dict(ticker="JPM", path=str(path), sha256="wrong", kind="SEC_COMPANYFACTS",
                     source="https://example.test")]
            r = replay(report, tmp)
        self.assertTrue(any("hash mismatch" in m for m in r["public_refresh"]["messages"]))

    def test_tagged_companyfacts_source_can_be_reparsed_offline(self):
        report = mini_report()
        report["public_refresh"]["credit_raw_facts"] = []
        payload = dict(cik=19617, facts={"us-gaap": {
            "FinancingReceivableExcludingAccruedInterestNonaccrual": {"units": {"USD": [
                dict(val=123, end="2026-06-30", filed="2026-08-01", form="10-Q", accn="fake")]}}}})
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"source.json"
            atomic_json(path, payload)
            report["public_refresh"]["source_files"] = [
                dict(ticker="JPM", path=str(path), sha256=digest, kind="SEC_COMPANYFACTS",
                     source="https://example.test")]
            result = replay(report, tmp)
        row = next(r for r in result["normalized_credit_facts"]
                   if r["ticker"] == "JPM" and r["period_end"] == "2026-06-30")
        self.assertEqual(row["npl_usd"], 123)


if __name__ == "__main__":
    unittest.main()
