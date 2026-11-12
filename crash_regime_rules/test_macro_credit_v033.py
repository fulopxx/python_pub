"""Real reduced source regressions + synthetic safety cases; no network."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from macro_credit_autodata import build_automatic_credit
from macro_credit_forensics import quantitative_checks
from macro_credit_tables import (
    PROFILES, Tables, amount, grid, parse_period, table_facts, row_text,
)
from replay_macro_credit_report import import_earnings_zip

FIXTURES = Path(__file__).parent/"tests"/"fixtures"/"earnings_033"
MANIFEST = json.loads((FIXTURES/"manifest.json").read_text(encoding="utf-8"))
ASOF = datetime.fromisoformat(MANIFEST["asof"])


class Credit033Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = {}
        cls.docs = {}
        cls.facts = []
        for doc in MANIFEST["files"]:
            key = doc["fixture"].split(".")[0]
            cls.docs[key] = doc
            cls.raw[key] = (FIXTURES/doc["fixture"]).read_text(encoding="utf-8")
            facts, _ = table_facts(cls.raw[key], doc, ASOF)
            cls.facts.extend(facts)
        cls.rows = build_automatic_credit(dict(credit_raw_facts=cls.facts), [], ASOF)
        cls.target = {r["ticker"]: r for r in cls.rows if r["period_end"] == "2026-09-30"}
        cls.before = {r["ticker"]: r for r in cls.rows if r["period_end"] == "2026-06-30"}
        cls.checks = {r["ticker"]: r for r in quantitative_checks(cls.rows)
                      if r["period_end"] == "2026-09-30"}

    def reparse(self, key, transform):
        return table_facts(transform(self.raw[key]), self.docs[key], ASOF)[0]

    def test_all_six_banks_have_current_values(self):
        self.assertEqual(set(self.target), {"JPM","BAC","C","WFC","GS","MS"})
        self.assertTrue(all(r["normalization_verified"] for r in self.target.values()))

    def test_twenty_current_fields_not_forty_two_fabricated(self):
        from macro_credit_autodata import FIELDS
        self.assertEqual(sum(r.get(k) is not None for r in self.target.values() for k in FIELDS),20)

    def test_jpm_nonaccrual_is_retained_not_total_including_hfs(self):
        self.assertEqual(self.target["JPM"]["npl_usd"],8_053_000_000)
        self.assertEqual(self.before["JPM"]["npl_usd"],8_034_000_000)
        ev = self.target["JPM"]["metric_provenance"]["npl_usd"]
        self.assertEqual(len(ev["components"]),2)

    def test_jpm_loan_provision_not_total_credit_provision(self):
        self.assertEqual(self.target["JPM"]["provision_usd"],2_327_000_000)
        self.assertNotEqual(self.target["JPM"]["provision_usd"],2_743_000_000)

    def test_jpm_net_chargeoffs(self):
        self.assertEqual(self.target["JPM"]["net_chargeoffs_usd"],2_204_000_000)
        self.assertEqual(self.before["JPM"]["net_chargeoffs_usd"],2_366_000_000)

    def test_jpm_actual_reported_other(self):
        self.assertEqual(self.target["JPM"]["acl_other_changes_usd"],-3_000_000)
        self.assertEqual(self.before["JPM"]["acl_other_changes_usd"],0)

    def test_jpm_rollforward_reconciles(self):
        check = self.checks["JPM"]["checks"][-1]
        self.assertEqual(check["status"],"NO_TRIGGER")
        self.assertEqual(check["residual_usd"],0)

    def test_citi_headers_oldest_to_newest(self):
        self.assertEqual(self.target["C"]["allowance_usd"],20_091_000_000)
        self.assertEqual(self.before["C"]["allowance_usd"],19_961_000_000)

    def test_citi_signed_ncl_to_positive_loss(self):
        self.assertEqual(self.target["C"]["net_chargeoffs_usd"],2_217_000_000)

    def test_citi_other_signed_not_balancing_plug(self):
        self.assertEqual(self.target["C"]["acl_other_changes_usd"],-69_000_000)
        self.assertEqual(self.before["C"]["acl_other_changes_usd"],126_000_000)

    def test_citi_nonaccrual_total_not_segment(self):
        self.assertEqual(self.target["C"]["npl_usd"],3_254_000_000)
        self.assertEqual(self.before["C"]["npl_usd"],3_229_000_000)

    def test_citi_rollforward_reconciles(self):
        self.assertEqual(self.checks["C"]["checks"][-1]["residual_usd"],0)

    def test_bac_npl_excludes_foreclosed_assets(self):
        self.assertEqual(self.target["BAC"]["npl_usd"],5_544_000_000)

    def test_bac_allowance_amount_not_percent_or_total_acl(self):
        self.assertEqual(self.target["BAC"]["allowance_usd"],13_005_000_000)
        self.assertEqual(self.before["BAC"]["allowance_usd"],13_114_000_000)

    def test_bac_nco_amount_not_ratio(self):
        self.assertEqual(self.target["BAC"]["net_chargeoffs_usd"],1_349_000_000)

    def test_bac_provision_not_guessed_from_total(self):
        self.assertIsNone(self.target["BAC"]["provision_usd"])

    def test_wfc_funded_allowance_excludes_unfunded(self):
        self.assertEqual(self.target["WFC"]["allowance_usd"],13_819_000_000)
        self.assertEqual(self.before["WFC"]["allowance_usd"],13_905_000_000)
        self.assertNotEqual(self.before["WFC"]["allowance_usd"],14_407_000_000)

    def test_wfc_nco_vs_npl(self):
        self.assertEqual(self.target["WFC"]["npl_usd"],7_060_000_000)
        self.assertEqual(self.target["WFC"]["net_chargeoffs_usd"],983_000_000)

    def test_wfc_broader_provision_and_other_rejected(self):
        self.assertIsNone(self.target["WFC"]["provision_usd"])
        self.assertIsNone(self.target["WFC"]["acl_other_changes_usd"])

    def test_ms_current_loan_matrix(self):
        self.assertEqual(self.target["MS"]["allowance_usd"],1_244_000_000)
        self.assertEqual(self.target["MS"]["provision_usd"],103_000_000)

    def test_ms_no_gross_chargeoff_substitution(self):
        self.assertIsNone(self.target["MS"]["net_chargeoffs_usd"])

    def test_ms_changed_column_order_rejected(self):
        fs = self.reparse("MS_5a5",lambda s:s.replace("ACL (1)","WRONG HEADER"))
        self.assertEqual(fs,[])

    def test_gs_funded_only_narrative(self):
        self.assertEqual(self.target["GS"]["allowance_usd"],2_310_000_000)
        self.assertEqual(self.target["GS"]["net_chargeoffs_usd"],154_000_000)
        self.assertEqual(self.target["GS"]["metric_provenance"]["allowance_usd"]["decimals"],"-7")

    def test_gs_ambiguous_duplicate_paragraph_rejected(self):
        fs = self.reparse("GS_506",lambda s:s+s)
        self.assertEqual(fs,[])

    def test_no_pik_or_modification_invented(self):
        for row in self.target.values():
            self.assertIsNone(row["pik_interest_usd"])
            self.assertIsNone(row["modified_loans_usd"])

    def test_eight_of_thirty_checks_runnable(self):
        checks = [c for r in self.checks.values() for c in r["checks"]]
        self.assertEqual(sum(c["status"]!="NOT_TESTABLE" for c in checks),8)
        self.assertEqual(sum(c["status"]=="REVIEW_FLAG" for c in checks),0)

    def test_table_and_native_scopes_not_stitched(self):
        for bank in self.target:
            self.assertEqual(self.target[bank]["scope_id"],PROFILES[bank])
        self.assertIsNone(self.before["MS"]["allowance_usd"])

    def test_availability_future_document_rejected(self):
        doc = dict(self.docs["JPM_74a"],accepted_at="2026-10-20T00:00:00+00:00")
        self.assertEqual(table_facts(self.raw["JPM_74a"],doc,ASOF)[0],[])

    def test_naive_timestamp_rejected(self):
        doc = dict(self.docs["JPM_74a"],accepted_at="2026-10-13T14:30:38")
        self.assertEqual(table_facts(self.raw["JPM_74a"],doc,ASOF)[0],[])

    def test_wrong_source_bank_rejected(self):
        doc = dict(self.docs["JPM_74a"],ticker="BAC")
        self.assertEqual(table_facts(self.raw["JPM_74a"],doc,ASOF)[0],[])

    def test_source_accession_mismatch_rejected(self):
        doc = dict(self.docs["JPM_74a"],accession="0001628280-26-000001")
        self.assertEqual(table_facts(self.raw["JPM_74a"],doc,ASOF)[0],[])

    def test_future_native_fact_does_not_hide_known_tables(self):
        future = dict(self.facts[0],scope_id="SEC_LOAN_ALLOWANCE_ENTITY_USD_V1",
                      end="2026-12-31",available_at="2027-01-20T00:00:00+00:00")
        rows = build_automatic_credit(dict(credit_raw_facts=self.facts+[future]),[],ASOF)
        current = next(r for r in rows if r["ticker"]=="JPM" and r["period_end"]=="2026-09-30")
        self.assertEqual(current["scope_id"],PROFILES["JPM"])

    def test_state_scope_change_is_new_baseline_not_loss_revision(self):
        from macro_monitor_state import update_state
        row = deepcopy(self.target["WFC"])
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"state.json"
            update_state(path,dict(asof=ASOF.isoformat(),normalized_credit_facts=[row]))
            row["scope_id"]="DIFFERENT_PORTFOLIO"
            row["allowance_usd"]=None
            changes=update_state(path,dict(asof=ASOF.isoformat(),normalized_credit_facts=[row]))
            self.assertEqual(changes["changes"][0]["kind"],"FIRST_SEEN_OBSERVATION")
            self.assertNotIn("allowance_usd",changes["changes"][0]["after"])

    def test_missing_units_do_not_guess_millions(self):
        fs = self.reparse("JPM_74a",lambda s:s.replace("in millions","unit removed"))
        self.assertEqual(fs,[])

    def test_dash_not_globally_zero(self):
        self.assertIsNone(amount("—"))
        self.assertEqual(amount("—",dash_zero=True),0)
        self.assertIsNone(amount("N/A"))
        self.assertIsNone(amount("2.4%"))

    def test_date_variants(self):
        self.assertEqual(str(parse_period("3Q26")),"2026-09-30")
        self.assertEqual(str(parse_period("3Q",2026)),"2026-09-30")
        self.assertEqual(str(parse_period("Fourth Quarter 2025")),"2025-12-31")
        self.assertIsNone(parse_period("Nine Months"))

    def test_rowspan_geometry(self):
        p=Tables();p.feed('<table><tr><td rowspan="3">label</td><td>A</td></tr>'
                         '<tr><td>B</td></tr><tr><td>C</td></tr></table>')
        rows=grid(p.tables[0])
        self.assertEqual(row_text(rows[2]),"label C")

    def test_nested_table_not_double_counted(self):
        p=Tables();p.feed("<table><tr><td>outer<table><tr><td>inner</td></tr></table></td></tr></table>")
        self.assertTrue(p.tables[0]["nested"])

    def test_fresh_fetch_adapter_integrated(self):
        import macro_public_data
        with patch.object(macro_public_data,"_request",return_value=self.raw["JPM_74a"].encode()):
            data=macro_public_data.request_document(self.docs["JPM_74a"]["source"],"test",
                                                    self.docs["JPM_74a"],ASOF)
        self.assertEqual(sum(f.get("transport")=="SEC_REVIEWED_EARNINGS_HTML"
                             for f in data["credit_raw_facts"]),10)

    def bundle(self, folder, tamper=False, extra=False):
        raw=self.raw["JPM_74a"].encode()
        doc=dict(self.docs["JPM_74a"],filename="source.html",sha256=hashlib.sha256(raw).hexdigest())
        path=Path(folder)/"test.zip"
        with ZipFile(path,"w") as z:
            z.writestr("manifest.json",json.dumps(dict(public_sec_documents_only=True,files=[doc])))
            z.writestr("html/source.html",b"wrong" if tamper else raw)
            if extra:
                z.writestr("../outside.txt","no")
        return path

    def test_bundle_validates_hash_and_no_network(self):
        with TemporaryDirectory() as tmp:
            refresh={}
            import_earnings_zip(refresh,self.bundle(tmp),ASOF)
        self.assertEqual(len(refresh["credit_raw_facts"]),10)
        self.assertFalse(refresh["earnings_bundle_import"]["network_used"])

    def test_bundle_rejects_corrupt_bytes_atomically(self):
        with TemporaryDirectory() as tmp:
            refresh={}
            with self.assertRaises(ValueError):
                import_earnings_zip(refresh,self.bundle(tmp,tamper=True),ASOF)
            self.assertEqual(refresh,{})

    def test_bundle_rejects_undeclared_paths_without_extracting(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                import_earnings_zip({},self.bundle(tmp,extra=True),ASOF)
            self.assertFalse((Path(tmp).parent/"outside.txt").exists())

    def test_missing_reason_is_specific(self):
        reasons=" ".join(self.checks["WFC"]["checks"][1]["missing_reasons"])
        self.assertIn("FUNDED_ONLY_QUARTER_PROVISION_NOT_IDENTIFIED",reasons)

    def test_console_explains_scopes_and_missing(self):
        from macro_console import render
        from macro_ai_credit_monitor import evaluate
        # Use the shipped dated snapshot only as a console schema fixture.
        base=Path(__file__).parent
        snapshot=json.loads((base/"macro_provider_snapshot.json").read_text(encoding="utf-8"))
        scenarios=json.loads((base/"macro_scenarios.json").read_text(encoding="utf-8"))
        from macro_credit_forensics import evaluate_forensics
        result=evaluate(snapshot,scenarios,ASOF)
        result["normalized_credit_facts"]=self.rows
        result["forensics"]=dict(evaluate_forensics([],self.rows),automatic_extraction=True)
        text="\n".join(render(result))
        self.assertIn("MIT TARTALMAZNAK",text)
        self.assertIn("ADATHIÁNY",text)
        self.assertNotIn("FUNDED_ONLY_QUARTER_PROVISION_NOT_IDENTIFIED",text)


if __name__=="__main__":
    unittest.main()
