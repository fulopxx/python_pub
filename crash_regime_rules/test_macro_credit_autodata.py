"""Synthetic tests; no financial figures here are actual bank observations."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

import macro_credit_autodata as auto
import macro_credit_forensics as audit
import macro_ai_credit_monitor as monitor
import macro_public_data as public
from macro_monitor_state import merge_public_cache, update_state

ASOF = datetime(2026, 10, 14, tzinfo=timezone.utc)


def fact(tag, end, value, start=None, **extra):
    return dict(ticker="JPM", tag=tag, concept="us-gaap:"+tag, end=end, start=start,
                value=value, unit="USD", dimensions=[], entity="JPM",
                available_at="2026-10-13T12:00:00+00:00", accession="synthetic",
                source="https://example.test/synthetic", **extra)


def inputs():
    out = []
    tags = {k: v[1][0] for k, v in auto.METRICS.items()}
    for end, start, npl, acl, prov, co in [
        ("2026-06-30", "2026-04-01", 10e6, 100e6, 20e6, 15e6),
        ("2026-09-30", "2026-07-01", 20e6, 90e6, 10e6, 20e6)]:
        for k, v in dict(npl_usd=npl, allowance_usd=acl, provision_usd=prov,
                         net_chargeoffs_usd=co, modified_loans_usd=2e6,
                         pik_interest_usd=3e6, acl_other_changes_usd=0).items():
            out.append(fact(tags[k], end, v, start if auto.METRICS[k][0] == "flow" else None))
    return out


class NormalizationTests(unittest.TestCase):
    def test_complete_auto_input_runs_all_five_checks(self):
        rows = auto.normalize_credit(inputs(), "JPM", ASOF)
        current = audit.quantitative_checks(rows)[-1]
        self.assertTrue(all(c["status"] != "NOT_TESTABLE" for c in current["checks"]))
        self.assertEqual(current["checks"][0]["status"], "REVIEW_FLAG")
        self.assertEqual(current["checks"][1]["status"], "REVIEW_FLAG")
        self.assertEqual(current["checks"][-1]["status"], "NO_TRIGGER")

    def test_missing_fact_not_zero_or_green(self):
        raw = [r for r in inputs() if r["tag"] != "InterestIncomePaymentInKind"]
        rows = auto.normalize_credit(raw, "JPM", ASOF)
        c = audit.quantitative_checks(rows)[-1]["checks"][3]
        self.assertEqual(c["status"], "NOT_TESTABLE")
        self.assertIn("pik_interest_usd", " ".join(c["missing_reasons"]))

    def test_expected_new_quarter_not_assumed_published(self):
        raw = [r for r in inputs() if r["end"] == "2026-06-30"]
        rows = auto.normalize_credit(raw, "JPM", ASOF)
        self.assertEqual(rows[-1]["period_end"], "2026-09-30")
        self.assertFalse(rows[-1]["normalization_verified"])

    def test_entity_dimension_and_currency_filtered(self):
        for key, value in (("dimensions", ["Consumer"]), ("unit", "EUR"), ("entity", "OTHER")):
            raw = inputs()
            for r in raw:
                r[key] = value
            self.assertFalse(any(r["normalization_verified"] for r in auto.normalize_credit(raw, "JPM", ASOF)))

    def test_unknown_concept_not_used(self):
        raw = [fact("UnknownNumber", "2026-09-30", 999)]
        self.assertFalse(auto.normalize_credit(raw, "JPM", ASOF)[-1]["normalization_verified"])

    def test_future_source_excluded(self):
        raw = inputs()
        for r in raw:
            r["available_at"] = "2026-10-15T00:00:00Z"
        self.assertFalse(auto.normalize_credit(raw, "JPM", ASOF)[-1]["normalization_verified"])

    def test_conflict_not_arbitrary_first(self):
        raw = inputs()
        same = deepcopy(raw[0])
        same["value"] += 1
        raw.append(same)
        rows = auto.normalize_credit(raw, "JPM", ASOF)
        row = next(r for r in rows if r["period_end"] == "2026-06-30")
        self.assertIsNone(row["npl_usd"])
        self.assertEqual(row["extraction_issues"]["npl_usd"], "CONFLICTING_FACT_VALUES")

    def test_negative_provision_preserved(self):
        raw = [fact("ProvisionForLoanAndLeaseLosses", "2026-09-30", -5,
                    "2026-07-01")]
        self.assertEqual(auto.normalize_credit(raw, "JPM", ASOF)[-1]["provision_usd"], -5)

    def test_ytd_not_quarter_without_same_filing_comparator(self):
        raw = [fact("ProvisionForLoanAndLeaseLosses", "2026-09-30", 900, "2026-01-01")]
        self.assertIsNone(auto.normalize_credit(raw, "JPM", ASOF)[-1]["provision_usd"])

    def test_same_filing_ytd_difference(self):
        raw = [fact("ProvisionForLoanAndLeaseLosses", "2026-09-30", 900, "2026-01-01"),
               fact("ProvisionForLoanAndLeaseLosses", "2026-06-30", 500, "2026-01-01")]
        row = auto.normalize_credit(raw, "JPM", ASOF)[-1]
        self.assertEqual(row["provision_usd"], 400)
        self.assertEqual(row["metric_provenance"]["provision_usd"]["method"], "SAME_FILING_YTD_DIFFERENCE")

    def test_ytd_different_accessions_not_subtracted(self):
        raw = [fact("ProvisionForLoanAndLeaseLosses", "2026-09-30", 900, "2026-01-01"),
               fact("ProvisionForLoanAndLeaseLosses", "2026-06-30", 500, "2026-01-01")]
        raw[1]["accession"] = "old-vintage"
        self.assertIsNone(auto.normalize_credit(raw, "JPM", ASOF)[-1]["provision_usd"])

    def test_net_chargeoff_from_reported_components(self):
        raw = [fact("FinancingReceivableAllowanceForCreditLossesWriteOffs", "2026-09-30", 12, "2026-07-01"),
               fact("FinancingReceivableAllowanceForCreditLossesRecoveries", "2026-09-30", 2, "2026-07-01")]
        self.assertEqual(auto.normalize_credit(raw, "JPM", ASOF)[-1]["net_chargeoffs_usd"], 10)

    def test_other_not_inferred_from_acl_identity(self):
        raw = [r for r in inputs() if "OtherChanges" not in r["tag"]]
        row = auto.normalize_credit(raw, "JPM", ASOF)[-1]
        self.assertIsNone(row["acl_other_changes_usd"])
        self.assertEqual(audit.quantitative_checks(auto.normalize_credit(raw, "JPM", ASOF))[-1]["checks"][-1]["status"],
                         "NOT_TESTABLE")

    def test_definition_change_blocks_comparison(self):
        rows = auto.normalize_credit(inputs(), "JPM", ASOF)
        rows[-1]["metric_provenance"]["npl_usd"]["definition_id"] = "new-meaning"
        c = audit.quantitative_checks(rows)[-1]["checks"][0]
        self.assertEqual(c["status"], "NOT_TESTABLE")
        self.assertIn("definíció", " ".join(c["missing_reasons"]))

    def test_rollforward_mixed_filings_blocked(self):
        rows = auto.normalize_credit(inputs(), "JPM", ASOF)
        rows[-1]["metric_provenance"]["provision_usd"]["accession"] = "other"
        self.assertEqual(audit.quantitative_checks(rows)[-1]["checks"][-1]["status"], "NOT_TESTABLE")

    def test_old_tag_not_latest_quarter(self):
        payload = {"facts": {"us-gaap": {
            "ProvisionForLoanAndLeaseLosses": {"units": {"USD": [
                dict(start="2019-07-01", end="2019-09-30", val=123, form="10-Q", filed="2019-11-01")]}},
            "StockholdersEquity": {"units": {"USD": [
                dict(end="2026-06-30", val=456, form="10-Q", filed="2026-08-01")]}}}}}
        row = public.extract_sec(payload, "JPM", ASOF)
        self.assertEqual(row["period"], "2026-06-30")
        self.assertEqual(row["status"], "NO_DIRECT_QUARTER_PROVISION_TAG")

    def test_companyfacts_filters_future_and_extension(self):
        payload = {"facts": {"us-gaap": {"ProvisionForLoanAndLeaseLosses": {"units": {"USD": [
            dict(start="2026-07-01", end="2026-09-30", val=7, form="10-Q", filed="2026-10-14")]}}},
            "jpm": {"NonaccrualLoans": {"units": {"USD": []}}}}}
        facts, diag = auto.company_facts(payload, "JPM", "https://example.test", ASOF)
        self.assertEqual(facts, [])
        self.assertTrue(diag)

    def test_current_report_can_supersede_an_old_direct_flow(self):
        old = fact("ProvisionForLoanAndLeaseLosses", "2026-09-30", 100, "2026-07-01")
        old["available_at"] = "2026-10-12T12:00:00Z"
        newer = fact("ProvisionForLoanAndLeaseLosses", "2026-09-30", 900, "2026-01-01")
        self.assertIsNone(auto.normalize_credit([old, newer], "JPM", ASOF)[-1]["provision_usd"])

    def test_quarter_boundary_and_calendar(self):
        self.assertEqual(auto.expected_quarter(datetime(2026, 1, 5, tzinfo=timezone.utc)), "2025-12-31")
        self.assertEqual(auto.expected_quarter(datetime(2026, 7, 1, tzinfo=timezone.utc)), "2026-06-30")

    def test_net_chargeoff_components_different_filings_rejected(self):
        raw = [fact("FinancingReceivableAllowanceForCreditLossesWriteOffs", "2026-09-30", 12, "2026-07-01"),
               fact("FinancingReceivableAllowanceForCreditLossesRecoveries", "2026-09-30", 2, "2026-07-01")]
        raw[1]["accession"] = "not-the-same"
        self.assertIsNone(auto.normalize_credit(raw, "JPM", ASOF)[-1]["net_chargeoffs_usd"])

    def test_rounding_used_in_rollforward_tolerance(self):
        rows = auto.normalize_credit(inputs(), "JPM", ASOF)
        rows[-1]["allowance_usd"] += 2_000_000
        for row in rows:
            for ev in row["metric_provenance"].values():
                ev["decimals"] = "-6"
        self.assertEqual(audit.quantitative_checks(rows)[-1]["checks"][-1]["status"], "NO_TRIGGER")


class InlineTests(unittest.TestCase):
    def html(self, value="1,234", scale="6", sign="", dimensions="", unit="iso4217:USD"):
        return f"""<html xmlns:us-gaap="http://fasb.org/us-gaap/2026"
          xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">
          <xbrli:context id="c"><xbrli:entity><xbrli:identifier>0000019617</xbrli:identifier>
          {dimensions}</xbrli:entity><xbrli:period><xbrli:instant>2026-09-30</xbrli:instant>
          </xbrli:period></xbrli:context><xbrli:unit id="u"><xbrli:measure>{unit}</xbrli:measure></xbrli:unit>
          <ix:nonFraction name="us-gaap:FinancingReceivableAllowanceForCreditLosses"
           contextRef="c" unitRef="u" scale="{scale}" sign="{sign}" decimals="-6"
           format="ixt:num-dot-decimal">{value}</ix:nonFraction></html>"""

    def doc(self):
        return dict(ticker="JPM", cik="0000019617", available_at="2026-10-13T12:00:00Z",
                    accession="synthetic", source="https://example.test/synthetic")

    def test_scaling_and_context(self):
        facts, _ = auto.inline_facts(self.html(), self.doc(), ASOF)
        self.assertEqual(facts[0]["value"], 1_234_000_000)
        self.assertEqual(facts[0]["end"], "2026-09-30")

    def test_negative_sign(self):
        facts, _ = auto.inline_facts(self.html(sign="-"), self.doc(), ASOF)
        self.assertLess(facts[0]["value"], 0)

    def test_dimensioned_fact_not_bank_total(self):
        html = self.html(dimensions='<xbrldi:explicitMember dimension="a">Consumer</xbrldi:explicitMember>')
        self.assertEqual(auto.inline_facts(html, self.doc(), ASOF)[0], [])

    def test_wrong_currency_not_usd(self):
        self.assertEqual(auto.inline_facts(self.html(unit="iso4217:EUR"), self.doc(), ASOF)[0], [])

    def test_other_entity_rejected(self):
        doc = self.doc()
        doc["cik"] = "0000000001"
        self.assertEqual(auto.inline_facts(self.html(), doc, ASOF)[0], [])

    def test_nil_not_zero(self):
        html = self.html().replace('sign=""', 'sign="" xsi:nil="true"')
        self.assertEqual(auto.inline_facts(html, self.doc(), ASOF)[0], [])

    def test_unknown_format_not_guessed(self):
        html = self.html().replace("num-dot-decimal", "num-comma-decimal")
        self.assertEqual(auto.inline_facts(html, self.doc(), ASOF)[0], [])

    def test_same_day_actual_acceptance_not_next_midnight(self):
        doc = self.doc()
        del doc["available_at"]
        doc["accepted_at"] = "2026-10-13T12:00:00Z"
        doc["filed"] = "2026-10-13"
        asof = datetime(2026, 10, 13, 17, tzinfo=timezone.utc)
        self.assertTrue(auto.inline_facts(self.html(), doc, asof)[0])

    def test_ix_exclude_not_numeric_value(self):
        html = self.html(value="1<ix:exclude>footnote99</ix:exclude>,234")
        self.assertEqual(auto.inline_facts(html, self.doc(), ASOF)[0][0]["value"], 1_234_000_000)

    def test_unlabeled_visible_number_not_inferred(self):
        html = "<table><tr><td>Allowance</td><td>1234000000</td></tr></table>"
        self.assertEqual(auto.inline_facts(html, self.doc(), ASOF)[0], [])

    def test_request_document_reuses_one_html_download(self):
        with patch.object(public, "_request", return_value=self.html().encode()) as request:
            result = public.request_document("https://example.test/synthetic", "research", self.doc(), ASOF)
        request.assert_called_once()
        self.assertEqual(len(result["credit_raw_facts"]), 1)


class IntegrationTests(unittest.TestCase):
    def test_all_watchlist_banks_shown_when_fetch_missing(self):
        rows = auto.build_automatic_credit({}, [], ASOF)
        self.assertEqual(len(rows), 48)
        self.assertTrue(all(not r["normalization_verified"] for r in rows))

    def test_cache_keeps_tagged_inputs_for_offline(self):
        cached = merge_public_cache(dict(credit_raw_facts=inputs()), dict(collected_at=ASOF.isoformat()))
        self.assertEqual(len(cached["credit_raw_facts"]), len(inputs()))

    def test_auto_without_manual_json(self):
        refresh = dict(credit_raw_facts=inputs(), bank_fetch_status={"JPM": "SUBMISSIONS_FETCHED"})
        rows = auto.build_automatic_credit(refresh, [], ASOF)
        self.assertTrue([r for r in rows if r["ticker"] == "JPM"][-1]["normalization_verified"])

    def test_default_pipeline_exports_auto_rows_and_diagnostics(self):
        from test_macro_ai_credit_monitor import snapshot
        fake = dict(collected_at=ASOF.isoformat(), banks=[], prices=[], filings=[], disclosures=[],
                    bank_fetch_status={"JPM": "SUBMISSIONS_FETCHED"}, messages=[], credit_raw_facts=inputs())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            monitor.atomic_json(root/"snapshot.json", snapshot())
            with patch.object(public, "refresh_public", return_value=fake), redirect_stdout(io.StringIO()):
                self.assertEqual(monitor.main(["--snapshot", str(root/"snapshot.json"),
                                               "--output-dir", str(root/"out"), "--ascii"]), 0)
            rows = json.loads((root/"out"/"macro_credit_normalized.json").read_text())
            self.assertTrue([r for r in rows if r["ticker"] == "JPM"][-1]["normalization_verified"])
            text = (root/"out"/"macro_ai_console.txt").read_text()
            self.assertNotIn("[X] N/A", text)
            self.assertIn("ADATHIÁNY", text)
            result = json.loads((root/"out"/"macro_ai_latest.json").read_text())
            self.assertTrue(result["forensics"]["automatic_extraction"])

    def test_empty_inputs_precise_missing_explanation(self):
        rows = auto.normalize_credit([], "JPM", ASOF)
        check = audit.quantitative_checks(rows)[0]["checks"][0]
        self.assertIn("Nincs előző", " ".join(check["missing_reasons"]))
        self.assertIn("npl_usd", " ".join(check["missing_reasons"]))

    def test_state_tracks_actual_credit_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"state.json"
            result = dict(asof=ASOF.isoformat(), normalized_credit_facts=auto.normalize_credit(inputs(), "JPM", ASOF))
            update_state(path, result)
            self.assertEqual(update_state(path, result)["changes"], [])
            result["normalized_credit_facts"][-1]["npl_usd"] += 1
            self.assertEqual(len(update_state(path, result)["changes"]), 1)


if __name__ == "__main__":
    unittest.main()
