"""Synthetic regressions shaped like the user's 0.3.1 retained candidates."""
from copy import deepcopy
from datetime import datetime, timezone
import unittest
import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile

import macro_credit_autodata as a
import macro_public_data as public
from test_macro_credit_autodata import ASOF, fact

ROOT = "https://www.sec.gov/Archives/edgar/data/72971/000007297126000302/"
WFC_TAG = "FinancingReceivableAndNetInvestmentInLeaseExcludingAccruedInterestNonaccrual"
NCO = "FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossWriteoffAfterRecovery"


def inline_candidate(tag=WFC_TAG, prefix="wfc", ticker="WFC", value="123"):
    return dict(ticker=ticker, concept=f"{prefix}:{tag}", raw_text=value,
                source=ROOT+"wfc-20260630.htm", accession="0000072971-26-000302",
                accepted_at="2026-07-29T00:46:31+00:00",
                context=dict(dimensions=[], identifier="0000072971", instant="2026-06-30"),
                unit=dict(measures=["iso4217:USD"]),
                attributes=dict(name=f"{prefix}:{tag}", scale="6", decimals="-6"))


def refresh(candidate):
    return dict(credit_candidate_facts=[candidate], disclosures=[
        dict(ticker=candidate["ticker"], source=candidate["source"], status="TEXT_EXTRACTED",
             accepted_at="2026-07-29T00:46:31+00:00")])


def cf_candidate(tag=NCO):
    return dict(ticker="BAC", concept="us-gaap:"+tag, unit="USD", value=100,
                start="2026-04-01", end="2026-06-30", available_at="2026-08-02T00:00:00+00:00",
                accession="synthetic", source="https://data.sec.gov/api/xbrl/companyfacts/CIK0000070858.json")


class Credit032Tests(unittest.TestCase):
    def bundle_input(self, tmp):
        raw = b"<html>synthetic public document</html>"
        path = Path(tmp)/"source.html"
        path.write_bytes(raw)
        source = ROOT+"ex991.htm"
        return dict(public_refresh=dict(
            disclosures=[dict(source=source, status="TEXT_EXTRACTED",
                              document_quality="EXHIBIT_FILENAME_CANDIDATE_TYPE_UNVERIFIED")],
            source_files=[dict(kind="SEC_FILING_HTML", path=r"C:\old\source.html",
                               source=source, ticker="WFC", sha256=hashlib.sha256(raw).hexdigest())]))

    def test_public_earnings_bundle_uses_explicit_folder(self):
        from package_macro_earnings_sources import package
        with tempfile.TemporaryDirectory() as tmp:
            report = self.bundle_input(tmp)
            out = Path(tmp)/"out.zip"
            manifest = package(report, tmp, out)
            self.assertEqual(len(manifest["files"]), 1)
            with ZipFile(out) as z:
                self.assertEqual(set(z.namelist()), {"manifest.json", "html/source.html"})

    def test_bundle_rejects_changed_source(self):
        from package_macro_earnings_sources import package
        with tempfile.TemporaryDirectory() as tmp:
            report = self.bundle_input(tmp)
            (Path(tmp)/"source.html").write_text("tampered")
            with self.assertRaises(FileNotFoundError):
                package(report, tmp, Path(tmp)/"out.zip")

    def test_windows_newline_repair_keeps_original_hash(self):
        from package_macro_earnings_sources import package
        with tempfile.TemporaryDirectory() as tmp:
            report = self.bundle_input(tmp)
            original = b"<html>\nsynthetic\n</html>"
            meta = report["public_refresh"]["source_files"][0]
            meta["sha256"] = hashlib.sha256(original).hexdigest()
            (Path(tmp)/"source.html").write_bytes(original.replace(b"\n", b"\r\n"))
            out = Path(tmp)/"out.zip"
            manifest = package(report, tmp, out)
            self.assertTrue(manifest["files"][0]["newline_repair"])
            with ZipFile(out) as z:
                self.assertEqual(z.read("html/source.html"), original)

    def test_original_crlf_windows_double_conversion_repaired(self):
        from package_macro_earnings_sources import package
        with tempfile.TemporaryDirectory() as tmp:
            report = self.bundle_input(tmp)
            original = b"<html>\r\nsynthetic\r\n</html>"
            report["public_refresh"]["source_files"][0]["sha256"] = hashlib.sha256(original).hexdigest()
            (Path(tmp)/"source.html").write_bytes(original.replace(b"\n", b"\r\n"))
            out = Path(tmp)/"out.zip"
            package(report, tmp, out)
            with ZipFile(out) as z:
                self.assertEqual(z.read("html/source.html"), original)

    def test_valid_crlf_is_not_normalized_unnecessarily(self):
        from package_macro_earnings_sources import package
        with tempfile.TemporaryDirectory() as tmp:
            report = self.bundle_input(tmp)
            original = b"<html>\r\nsynthetic\r\n</html>"
            report["public_refresh"]["source_files"][0]["sha256"] = hashlib.sha256(original).hexdigest()
            (Path(tmp)/"source.html").write_bytes(original)
            manifest = package(report, tmp, Path(tmp)/"out.zip")
            self.assertIsNone(manifest["files"][0]["newline_repair"])

    def test_bundle_never_follows_source_symlink_outside_folder(self):
        from package_macro_earnings_sources import package
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            report = self.bundle_input(tmp)
            file = Path(tmp)/"source.html"
            target = Path(other)/"secret.html"
            target.write_bytes(file.read_bytes())
            file.unlink()
            try:
                file.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("Symlink creation unavailable on this platform")
            with self.assertRaises(FileNotFoundError):
                package(report, tmp, Path(tmp)/"out.zip")

    def test_standard_retained_company_fact_recovers_nco(self):
        raw = a.retained_candidate_facts(refresh(cf_candidate()), ASOF)
        self.assertEqual(raw[0]["value"], 100)
        rows = a.build_automatic_credit(refresh(cf_candidate()), [], ASOF)
        row = next(r for r in rows if r["ticker"] == "BAC" and r["period_end"] == "2026-06-30")
        self.assertEqual(row["net_chargeoffs_usd"], 100)

    def test_no_future_values_restored(self):
        c = cf_candidate()
        c["end"] = "2026-12-31"
        self.assertEqual(a.retained_candidate_facts(refresh(c), ASOF), [])

    def test_wrong_cik_source_rejected(self):
        c = cf_candidate()
        c["source"] = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000019617.json"
        self.assertEqual(a.retained_candidate_facts(refresh(c), ASOF), [])

    def test_companyfact_non_usd_rejected(self):
        c = cf_candidate()
        c["unit"] = "USD/shares"
        self.assertEqual(a.retained_candidate_facts(refresh(c), ASOF), [])

    def test_wfc_lease_profile_kept_separate(self):
        rows = a.build_automatic_credit(refresh(inline_candidate()), [], ASOF)
        r = next(r for r in rows if r["ticker"] == "WFC" and r["period_end"] == "2026-06-30")
        self.assertEqual(r["scope_id"], a.LEASE_PROFILE)
        self.assertEqual(r["npl_usd"], 123e6)
        self.assertIn("URI_NOT_RETAINED", r["metric_provenance"]["npl_usd"]["namespace_validation"])

    def test_wfc_subsegment_not_total(self):
        c = inline_candidate()
        c["context"]["dimensions"] = ["Consumer"]
        self.assertEqual(a.retained_candidate_facts(refresh(c), ASOF), [])

    def test_wrong_inline_entity_not_accepted(self):
        c = inline_candidate()
        c["context"]["identifier"] = "0000019617"
        self.assertEqual(a.retained_candidate_facts(refresh(c), ASOF), [])

    def test_untrusted_host_not_accepted(self):
        c = inline_candidate()
        c["source"] = "https://example.test/fake.htm"
        self.assertEqual(a.retained_candidate_facts(refresh(c), ASOF), [])

    def test_wfc_alias_not_used_for_other_bank(self):
        c = inline_candidate()
        c["ticker"] = "JPM"
        self.assertEqual(a.retained_candidate_facts(refresh(c), ASOF), [])

    def test_unreviewed_custom_tag_remains_candidate(self):
        c = inline_candidate(tag="SomeOtherLoanNumber")
        self.assertEqual(a.retained_candidate_facts(refresh(c), ASOF), [])

    def test_zero_value_preserved(self):
        c = cf_candidate()
        c["value"] = 0
        self.assertEqual(a.retained_candidate_facts(refresh(c), ASOF)[0]["value"], 0)

    def test_rounding_difference_not_false_conflict(self):
        precise = fact(NCO, "2026-06-30", 1412e6, "2026-04-01")
        rounded = dict(precise, value=1400e6, decimals="-8")
        precise["decimals"] = "-6"
        selected, error = a._select([rounded, precise])
        self.assertIsNone(error)
        self.assertEqual(selected["value"], 1412e6)

    def test_true_conflict_still_fails(self):
        f = fact(NCO, "2026-06-30", 1412e6, "2026-04-01", decimals="-6")
        self.assertEqual(a._select([f, dict(f, value=1500e6)])[1], "CONFLICTING_FACT_VALUES")

    def test_same_scope_net_and_components_must_reconcile(self):
        raw = [fact(NCO, "2026-06-30", 80, "2026-04-01"),
               fact("FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossWriteoff",
                    "2026-06-30", 100, "2026-04-01"),
               fact("FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossRecovery",
                    "2026-06-30", 10, "2026-04-01")]
        row = a.normalize_credit(raw, "JPM", ASOF)[-2]
        self.assertIsNone(row["net_chargeoffs_usd"])
        self.assertIn("MISMATCH", row["extraction_issues"]["net_chargeoffs_usd"])

    def test_no_ytd_backfill_as_quarter(self):
        c = cf_candidate()
        c["start"] = "2026-01-01"
        rows = a.build_automatic_credit(refresh(c), [], ASOF)
        r = next(r for r in rows if r["ticker"] == "BAC" and r["period_end"] == "2026-06-30")
        self.assertIsNone(r["net_chargeoffs_usd"])

    def test_no_loan_and_lease_mixing(self):
        w = a.retained_candidate_facts(refresh(inline_candidate()), ASOF)[0]
        normal = fact("FinancingReceivableAllowanceForCreditLossExcludingAccruedInterest",
                      "2026-06-30", 500)
        normal.update(ticker="WFC", entity="WFC")
        rows = a.normalize_credit([w, normal], "WFC", ASOF, profile=a.LEASE_PROFILE)
        self.assertIsNone(rows[-2]["allowance_usd"])

    def test_previous_periodic_report_collected(self):
        payload = dict(filings=dict(recent=dict(
            form=["10-Q", "10-Q"], reportDate=["2026-06-30", "2026-03-31"],
            filingDate=["2026-08-01", "2026-05-01"],
            acceptanceDateTime=["2026-08-01T12:00:00Z", "2026-05-01T12:00:00Z"],
            accessionNumber=["0000070858-26-000001", "0000070858-26-000002"],
            primaryDocument=["q2.htm", "q1.htm"], items=["", ""])))
        rows = public.recent_filings(payload, "BAC", "0000070858", ASOF)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["report_period"], "2026-03-31")

    def test_amended_same_period_not_counted_twice(self):
        payload = dict(filings=dict(recent=dict(
            form=["10-Q/A", "10-Q", "10-Q"],
            reportDate=["2026-06-30", "2026-06-30", "2026-03-31"],
            filingDate=["2026-08-02", "2026-08-01", "2026-05-01"],
            acceptanceDateTime=["2026-08-02T12:00:00Z", "2026-08-01T12:00:00Z", "2026-05-01T12:00:00Z"],
            accessionNumber=["0000070858-26-000001", "0000070858-26-000002", "0000070858-26-000003"],
            primaryDocument=["amend.htm", "q2.htm", "q1.htm"], items=["", "", ""])))
        rows = public.recent_filings(payload, "BAC", "0000070858", ASOF)
        self.assertEqual([r["primary_document"] for r in rows], ["amend.htm", "q1.htm"])


if __name__ == "__main__":
    unittest.main()
