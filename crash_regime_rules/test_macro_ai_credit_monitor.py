"""Deterministic offline tests. Public HTTP adapter is mocked, never networked."""
from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

import macro_ai_credit_monitor as m
import macro_public_data as api

ASOF = datetime(2026, 10, 14, 0, 0, tzinfo=timezone.utc)


def snapshot():
    return dict(collected_at="2026-10-13T20:00:00+00:00",
                banks=[dict(ticker="JPM", earnings=[
                    dict(period="Q3 2026", release_at="2026-10-13T12:30:00Z",
                         actual_eps=5, consensus_eps=4, actual_revenue=100, consensus_revenue=100),
                    dict(period="Q4 2026", release_at="2027-01-01T00:00:00Z",
                         actual_eps=1, consensus_eps=1)])],
                macro=[dict(name="CPI", value=3.4, previous_value=3.0,
                            period_end="2026-08-31", unit="percent")])


class CalculationTests(TestCase):
    def test_exact_deflation_not_subtraction(self):
        r = m.gdp_counterfactual(6, 3.7, 10, 3.2)
        self.assertAlmostEqual(r["reported_real_pct"], (1.06/1.037-1)*100)
        self.assertAlmostEqual(r["no_ai_baseline_pct"], (1.06/1.037-1)*100-3.2)
        self.assertAlmostEqual(r["alternative_real_pct"], -3.63636363636)
        self.assertNotAlmostEqual(r["alternative_real_pct"], r["reported_real_pct"]-10)

    def test_no_ai_alternative_reprices_same_counterfactual(self):
        r = m.gdp_counterfactual(6, 3.7, 10, 3.2)
        self.assertAlmostEqual(r["no_ai_alternative_pct"], ((1.06-.032*1.037)/1.1-1)*100)

    def test_same_deflator_no_artificial_extra_loss(self):
        r = m.gdp_counterfactual(6, 3.7, 3.7, 3.2)
        self.assertAlmostEqual(r["no_ai_baseline_pct"], r["no_ai_alternative_pct"])

    def test_minus_ten_requires_minus_one_nominal_at_ten_deflator(self):
        r = m.gdp_counterfactual(6, 3.7, 10, 0)
        self.assertAlmostEqual(r["nominal_pct_for_minus_ten_at_alt_deflator"], -1)

    def test_saar_conversion_roundtrip(self):
        self.assertAlmostEqual(m.quarter_to_annual(m.annual_to_quarter(4)), 4)

    def test_unverified_basis_rejected(self):
        with self.assertRaises(ValueError):
            m.gdp_counterfactual(6, 3.7, 10, 2, "UNKNOWN_QOQ")

    def test_invalid_gdp_rates_and_ai(self):
        for params in [(6, -100, 10, 1), (6, 3.7, 10, -1), (6, 3.7, 10, 200),
                       (float("nan"), 3.7, 10, 1)]:
            with self.assertRaises(ValueError):
                m.gdp_counterfactual(*params)

    def test_equity_multiplication_not_addition(self):
        r = m.equity_shock(-10, -20, 100)
        self.assertAlmostEqual(r["change_pct"], -28)
        self.assertAlmostEqual(r["scenario_level"], 72)
        self.assertFalse(r["fair_value_identified"])

    def test_no_reference_no_invented_index_target(self):
        self.assertIsNone(m.equity_shock(-10, -20)["scenario_level"])

    def test_equity_invalid_domain(self):
        with self.assertRaises(ValueError):
            m.equity_shock(-100, 0)

    def test_gold_can_have_both_signs(self):
        self.assertLess(m.gold_shock(.75, 4, 2, -8)["change_pct"], 0)
        self.assertGreater(m.gold_shock(-.75, -4, 4, 0)["change_pct"], 0)

    def test_credit_expected_loss(self):
        self.assertEqual(m.credit_loss(100_000_000_000, 2, 50), 1_000_000_000)

    def test_invalid_credit_inputs(self):
        for params in [(-1, 1, 1), (1, 101, 1), (1, 1, -1), (1, float("nan"), 1)]:
            with self.assertRaises(ValueError):
                m.credit_loss(*params)

    def test_surprise_needs_same_period_units(self):
        self.assertEqual(m.surprise(.4, .3, "Sep", "Sep", "mom", "yoy")["status"],
                         "PERIOD_OR_UNIT_MISMATCH")
        self.assertIsNone(m.surprise(.4, None, "Sep", "Sep", "mom", "mom")["difference"])

    def test_surprise_rejects_same_percent_unit_different_rate_basis(self):
        r = m.surprise(.4, .3, "Sep", "Sep", "percent", "percent", "MOM", "YOY")
        self.assertEqual(r["status"], "RATE_BASIS_UNVERIFIED_OR_MISMATCH")

    def test_surprise_identical_basis(self):
        r = m.surprise(.4, .3, "Sep", "Sep", "percent", "percent", "MOM", "MOM")
        self.assertAlmostEqual(r["difference"], .1)


class IntegrityTests(TestCase):
    def test_future_earnings_excluded(self):
        r = m.bank_earnings(snapshot(), ASOF)
        self.assertEqual(r[0]["period"], "Q3 2026")

    def test_previous_cpi_not_consensus(self):
        r = m.evaluate(snapshot(), m.default_scenarios(), ASOF)
        self.assertEqual(r["macro"][0]["surprise"]["status"], "CONSENSUS_OR_ACTUAL_NA")

    def test_no_crash_or_ai_causation_from_stress_numbers(self):
        r = m.evaluate(snapshot(), m.default_scenarios(), ASOF)
        self.assertIsNone(r["crash_date_probability"])
        self.assertIsNone(r["verified_no_ai_gdp"])
        self.assertIsNone(r["justified_market_decline_pct"])
        self.assertFalse(r["execution_allowed"])
        self.assertEqual(r["ai_caused_inflation"], "NOT_IDENTIFIED")

    def test_future_snapshot_not_displayed_as_observed(self):
        s = snapshot()
        s["collected_at"] = "2026-10-15T00:00:00+00:00"
        r = m.evaluate(s, m.default_scenarios(), ASOF)
        self.assertEqual(r["cache_status"], "FUTURE_SNAPSHOT")
        self.assertEqual(r["earnings"], [])
        self.assertEqual(r["macro"], [])

    def test_stale_snapshot_label(self):
        r = m.evaluate(snapshot(), m.default_scenarios(),
                       datetime(2026, 10, 20, tzinfo=timezone.utc))
        self.assertEqual(r["cache_status"], "STALE")

    def test_future_credit_not_available(self):
        s = snapshot()
        s["credit_facts"] = [dict(ticker="BAC", period="Q3", available_at="2026-10-14T12:30:00Z")]
        self.assertEqual(m.evaluate(s, m.default_scenarios(), ASOF)["credit_facts"], [])

    def test_strict_json_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = m.evaluate(snapshot(), m.default_scenarios(), ASOF)
            m.atomic_json(Path(tmp)/"output.json", r)
            self.assertEqual(json.loads((Path(tmp)/"output.json").read_text())["version"], m.VERSION)
            self.assertIn("NEM CRASH-JÓSLAT", "\n".join(m.console(r)))

    def test_naive_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            m.timestamp("2026-10-14T00:00:00")

    def test_missing_all_bank_actuals_visible(self):
        s = snapshot()
        r = m.bank_earnings(s, datetime(2026, 10, 1, tzinfo=timezone.utc))
        self.assertEqual(r[0]["status"], "NOT_AVAILABLE_AT_CUTOFF")


class AdapterTests(TestCase):
    def sec_payload(self):
        def row(start, end, value, filed="2026-08-01"):
            r = dict(end=end, val=value, filed=filed, form="10-Q")
            if start:
                r["start"] = start
            return r
        return {"facts": {"us-gaap": {
            "ProvisionForCreditLosses": {"units": {"USD": [
                row("2026-01-01", "2026-06-30", 800),  # YTD must NOT be quarterly
                row("2026-04-01", "2026-06-30", 300),
                row("2026-07-01", "2026-09-30", 999, "2026-10-14")]}},
            "StockholdersEquity": {"units": {"USD": [
                row(None, "2026-03-31", 100), row(None, "2026-06-30", 200)]}},
            "LoansAndLeasesReceivableNetReportedAmount": {"units": {"USD": [
                row(None, "2026-06-30", 500)]}}
        }}}

    def test_sec_uses_direct_quarter_not_ytd_or_future_filing(self):
        r = api.extract_sec(self.sec_payload(), "X", ASOF)
        self.assertEqual(r["provision_usd"], 300)
        self.assertEqual(r["equity_usd"], 200)
        self.assertEqual(r["net_loans_usd"], 500)
        self.assertIsNone(r["gross_loans_usd"])

    def test_sec_absent_tag_is_missing_not_zero(self):
        self.assertEqual(api.extract_sec({}, "X", ASOF)["status"], "NO_DIRECT_QUARTER_PROVISION_TAG")

    def test_bls_ignores_annual_average_and_current_month(self):
        data = {"status": "REQUEST_SUCCEEDED", "Results": {"series": [{
            "seriesID": "CUUR0000SA0", "data": [
                dict(year="2026", period="M10", value="999"),
                dict(year="2026", period="M13", value="888"),
                dict(year="2026", period="M09", value="110"),
                dict(year="2025", period="M09", value="100")]}]}}
        r = api.extract_bls(data, ASOF)
        self.assertEqual(r[0]["period"], "2026-09")
        self.assertAlmostEqual(r[0]["change_pct"], 10)
        self.assertIsNone(r[0]["release_at"])

    def test_bls_error_not_silently_ok(self):
        with self.assertRaises(ValueError):
            api.extract_bls({"status": "REQUEST_FAILED"}, ASOF)

    def test_request_host_allowlist(self):
        with self.assertRaises(ValueError):
            api.request_json("https://example.com")

    def test_historical_refresh_never_networks(self):
        with patch.object(api, "request_json") as request:
            api.refresh_public(datetime(2020, 1, 1, tzinfo=timezone.utc))
            request.assert_not_called()

    def test_network_failure_is_partial_not_crash_prediction(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(api, "request_json", side_effect=TimeoutError):
            r = api.refresh_public(datetime.now(timezone.utc))
        self.assertEqual(r["banks"], [])
        self.assertTrue(any("BLS FETCH_OR_PARSE_ERROR" in v for v in r["messages"]))


if __name__ == "__main__":
    main()
