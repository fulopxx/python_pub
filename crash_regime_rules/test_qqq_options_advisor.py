"""Offline QQQ option-scanner tests with a fake market-data-only TWS."""
from __future__ import annotations

import ast
import contextlib
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

import pandas as pd
import qqq_options_advisor as m

NOW = pd.Timestamp("2026-10-13T15:30:00Z")
SOURCE = NOW-pd.Timedelta(minutes=15)
# Preserve the original 300 USD boundary fixtures, independent of user defaults.
CFG = m.Config(premium_budget_usd=300)


def packet(structure="QQQ_LONG_PUT_OR_DEBIT_SPREAD", allowed=True):
    return dict(
        schema_version=1, source="ES_NQ_REGIME_RUNNER",
        observed_at=NOW.isoformat(), asof=SOURCE.isoformat(), execution_allowed=False,
        indices={s: dict(asof=SOURCE.isoformat(), data_quality="OK",
                         short_overlay_blocked=False) for s in ("ES", "NQ")},
        advisory=dict(state="QQQ_PUT_CANDIDATE" if allowed else "WAIT_FOR_FAILED_RECLAIM",
                      entry_scope="PAPER_ONLY", execution_allowed=False,
                      entry_allowed=allowed, trade_candidate=True,
                      recommended_structure=structure, reasons=["synthetic"]))


def underlying():
    return dict(price=600, asof=SOURCE.isoformat(), quality="OK", rth_vwap=603,
                below_rth_vwap=True, paper_price_gate=True)


def ticker(strike=600, delta=-.45, bid=2.4, ask=2.5, oi=500, conid=1):
    contract = types.SimpleNamespace(conId=conid, symbol="QQQ", localSymbol=f"TEST-{conid}",
                                     lastTradeDateOrContractMonth="20261020", strike=strike,
                                     right="P", multiplier="100", currency="USD", tradingClass="QQQ")
    greek = types.SimpleNamespace(delta=delta, impliedVol=.3, undPrice=600,
                                  gamma=.01, vega=.2, theta=-.1)
    return types.SimpleNamespace(contract=contract, bid=bid, ask=ask, bidSize=10, askSize=10,
                                 putOpenInterest=oi, modelGreeks=greek, marketDataType=3)


def row(**kwargs):
    return m.quote_row(ticker(**kwargs), CFG, NOW)


def bars():
    dates = pd.date_range("2026-10-13T13:30:00Z", "2026-10-13T15:14:00Z", freq="min")
    close = [602-2*i/(len(dates)-1) for i in range(len(dates))]
    return [dict(date=d, high=c+.1, low=c-.1, close=c, open=c, volume=100)
            for d, c in zip(dates, close)]


class GateTests(unittest.TestCase):
    def test_valid_paper_not_execution(self):
        r = m.advisory_gate(packet(), NOW, CFG)
        self.assertTrue(r["paper_allowed"])
        self.assertEqual(r["quality"], "OK")

    def test_missing_and_stale_do_not_authorize(self):
        self.assertFalse(m.advisory_gate(None, NOW, CFG)["paper_allowed"])
        self.assertEqual(m.advisory_gate(packet(), NOW+pd.Timedelta(minutes=16), CFG)["quality"], "STALE")

    def test_future_publication_and_source(self):
        for field in ("asof", "observed_at"):
            p = packet()
            p[field] = (NOW+pd.Timedelta(minutes=1)).isoformat()
            self.assertFalse(m.advisory_gate(p, NOW, CFG)["paper_allowed"])

    def test_core_short_veto_and_time_skew(self):
        for key, value in (("data_quality", "STALE_INPUT"), ("short_overlay_blocked", True),
                           ("asof", NOW.isoformat())):
            p = packet()
            p["indices"]["NQ"][key] = value
            self.assertFalse(m.advisory_gate(p, NOW, CFG)["paper_allowed"])

    def test_boolean_strings_do_not_authorize(self):
        p = packet()
        p["advisory"]["entry_allowed"] = "true"
        self.assertFalse(m.advisory_gate(p, NOW, CFG)["paper_allowed"])

    def test_wait_or_no_setup_does_not_authorize(self):
        self.assertFalse(m.advisory_gate(packet(allowed=False), NOW, CFG)["paper_allowed"])

    def test_execution_flag_or_missing_scope_rejected(self):
        p = packet()
        p["execution_allowed"] = True
        self.assertEqual(m.advisory_gate(p, NOW, CFG)["quality"], "INVALID")
        p = packet()
        del p["advisory"]["entry_scope"]
        self.assertFalse(m.advisory_gate(p, NOW, CFG)["paper_allowed"])


class QuoteFilterTests(unittest.TestCase):
    def test_valid_quote_and_greeks(self):
        r = row()
        self.assertEqual(m.liquidity_rejections(r, 600, CFG), [])
        self.assertEqual(r["quote_time_quality"], "UNVERIFIED_SOURCE_TIME")
        self.assertIsNone(r["quote_source_timestamp"])
        self.assertEqual(r["iv"], .3)

    def test_oi_missing_is_never_imputed(self):
        r = row(oi=float("nan"))
        self.assertIn("OPEN_INTEREST_NA", m.liquidity_rejections(r, 600, CFG))

    def test_insufficient_oi(self):
        self.assertIn("OPEN_INTEREST_TOO_LOW", m.liquidity_rejections(row(oi=99), 600, CFG))

    def test_no_delta_or_iv_fallback(self):
        t = ticker()
        t.modelGreeks = None
        t.impliedVolatility = .5
        r = m.quote_row(t, CFG, NOW)
        self.assertIsNone(r["iv"])
        self.assertIn("GREEKS_OR_IV_NA", m.liquidity_rejections(r, 600, CFG))

    def test_invalid_model_can_use_bid_greeks_but_not_last(self):
        t = ticker()
        valid = t.modelGreeks
        t.modelGreeks = types.SimpleNamespace(delta=1.797e308, impliedVol=.3, undPrice=600)
        t.bidGreeks = valid
        r = m.quote_row(t, CFG, NOW)
        self.assertEqual(r["greeks_source"], "bidGreeks")
        t.bidGreeks = None
        t.lastGreeks = valid
        self.assertIsNone(m.quote_row(t, CFG, NOW)["delta"])

    def test_crossed_zero_or_negative_book_rejected(self):
        for bid, ask in ((-1, -1), (0, 2), (3, 2)):
            self.assertIn("BID_ASK_MISSING_OR_INVALID",
                          m.liquidity_rejections(row(bid=bid, ask=ask), 600, CFG))

    def test_wide_spread_and_missing_size(self):
        r = row(bid=1, ask=2)
        self.assertIn("SPREAD_TOO_WIDE", m.liquidity_rejections(r, 600, CFG))
        r = row()
        r["ask_size"] = None
        self.assertIn("QUOTE_SIZE_MISSING", m.liquidity_rejections(r, 600, CFG))

    def test_frozen_and_greek_underlying_mismatch(self):
        r = row()
        r["market_data_type_reported"] = 4
        self.assertIn("FROZEN_QUOTE", m.liquidity_rejections(r, 600, CFG))
        r = row()
        r["greek_underlier"] = 620
        self.assertIn("GREEK_UNDERLIER_MISMATCH", m.liquidity_rejections(r, 600, CFG))

    def test_adjusted_option_and_expired_rejected(self):
        r = row()
        r["multiplier"] = 10
        self.assertIn("NONSTANDARD_OR_INVALID_CONTRACT", m.liquidity_rejections(r, 600, CFG))
        r["dte"] = 0
        self.assertIn("DTE_OUTSIDE_RANGE", m.liquidity_rejections(r, 600, CFG))

    def test_budget_includes_assumed_fee(self):
        r = row(bid=2.9, ask=3.0)
        long, _ = m.screen([r], 600, CFG)
        self.assertEqual(r["long_cost_usd"], 301)
        self.assertFalse(r["long_budget_pass"])
        self.assertEqual(len(long), 1)  # still usable as a spread long leg


class SpreadAndSelectionTests(unittest.TestCase):
    def test_spread_uses_ask_minus_bid_and_net_delta(self):
        long = row(bid=4.1, ask=4.2)
        short = row(strike=590, delta=-.20, bid=1.5, ask=1.6, conid=2)
        longs, shorts = m.screen([long, short], 600, CFG)
        spread = m.build_spreads(longs, shorts, CFG)[0]
        self.assertAlmostEqual(spread["indicative_debit"], 2.7)
        self.assertAlmostEqual(spread["opening_cost_usd"], 272)
        self.assertAlmostEqual(spread["theoretical_max_profit_usd"], 728)
        self.assertAlmostEqual(spread["net_delta"], -.25)
        self.assertFalse(spread["execution_allowed"])

    def test_expiry_mismatch_width_and_time_skew(self):
        for mutation in ("expiry", "width", "time"):
            long = row()
            short = row(strike=590, delta=-.20, bid=1.5, ask=1.6, conid=2)
            if mutation == "expiry":
                short["expiry"] = "20261021"
            elif mutation == "width":
                short["strike"] = 610
            else:
                short["collected_at"] = (NOW+pd.Timedelta(seconds=31)).isoformat()
            longs, shorts = m.screen([long, short], 600, CFG)
            self.assertEqual(m.build_spreads(longs, shorts, CFG), [])

    def test_no_free_or_negative_debit(self):
        long = row(bid=1.4, ask=1.5)
        short = row(strike=590, delta=-.2, bid=1.5, ask=1.6, conid=2)
        longs, shorts = m.screen([long, short], 600, CFG)
        self.assertEqual(m.build_spreads(longs, shorts, CFG), [])

    def test_low_budget_no_match(self):
        r = m.evaluate([row()], underlying(), packet(), NOW, replace(CFG, premium_budget_usd=100))
        self.assertEqual(r["state"], "NO_OPTION_MATCH")

    def test_missing_regime_keeps_only_diagnostics(self):
        r = m.evaluate([row()], underlying(), None, NOW, CFG)
        self.assertEqual(len(r["diagnostic_long_puts"]), 1)
        self.assertFalse(r["paper_long_put_candidates"])
        self.assertFalse(r["execution_allowed"])

    def test_preferred_spread_suppresses_long_put_recommendation(self):
        r = m.evaluate([row()], underlying(), packet("QQQ_PUT_DEBIT_SPREAD_PREFERRED"), NOW, CFG)
        self.assertTrue(r["diagnostic_long_puts"])
        self.assertFalse(r["paper_long_put_candidates"])

    def test_qqq_above_vwap_or_unsynced_blocks(self):
        for bad in (dict(underlying(), paper_price_gate=False), dict(underlying(), asof=(SOURCE-pd.Timedelta(minutes=16)).isoformat())):
            r = m.evaluate([row()], bad, packet(), NOW, CFG)
            self.assertFalse(r["paper_long_put_candidates"])
            self.assertEqual(r["state"], "QQQ_PRICE_OR_TIME_ALIGNMENT_BLOCKS_NEW_PUT")

    def test_final_underlying_age_rechecked(self):
        u = underlying()
        u["asof"] = (NOW-pd.Timedelta(minutes=46)).isoformat()
        r = m.evaluate([row()], u, packet(), NOW, CFG)
        self.assertEqual(r["underlying"]["quality"], "STALE")
        self.assertFalse(r["paper_long_put_candidates"])

    def test_standard_chain_only_and_bounded_specs(self):
        chain = types.SimpleNamespace(exchange="SMART", underlyingConId=1, multiplier="100",
                                      tradingClass="QQQ", expirations={"20261020", "20261021", "20261023", "20261013"},
                                      strikes=set(range(400, 700)))
        adjusted = copy.copy(chain)
        adjusted.tradingClass = "QQQ1"
        self.assertIs(m.select_chain([adjusted, chain], 1), chain)
        specs = m.select_contract_specs(chain, 600, NOW, CFG)
        self.assertLessEqual(len(specs), 36)
        self.assertEqual(specs[0]["dte"], 7)
        self.assertTrue(all(4 <= s["dte"] <= 14 for s in specs))


class UnderlyingAndExportTests(unittest.TestCase):
    def test_closed_rth_data_below_vwap(self):
        u = m.underlying_context(pd.DataFrame(bars()), NOW, CFG)
        self.assertEqual(u["asof"], SOURCE.isoformat())
        self.assertTrue(u["paper_price_gate"])
        self.assertEqual(u["quality"], "OK")

    def test_future_bar_not_used(self):
        data = bars()+[dict(date=NOW, close=999, high=999, low=999, volume=100)]
        self.assertEqual(m.underlying_context(pd.DataFrame(data), NOW, CFG)["price"], 600)

    def test_partial_open_missing_and_gaps(self):
        for data in (bars()[1:], bars()[:40]+bars()[50:]):
            u = m.underlying_context(pd.DataFrame(data), NOW, CFG)
            self.assertFalse(u["paper_price_gate"])

    def test_outside_rth_does_not_allow(self):
        u = m.underlying_context(pd.DataFrame(bars()), NOW.replace(hour=12), CFG)
        self.assertFalse(u["paper_price_gate"])

    def test_json_and_csv_are_strict_and_overwritten_when_empty(self):
        r = m.evaluate([row()], underlying(), packet(), NOW, CFG)
        with TemporaryDirectory() as folder:
            m.save_result(r, Path(folder))
            saved = json.loads((Path(folder)/"qqq_options_latest.json").read_text(encoding="utf-8"))
            self.assertFalse(saved["execution_allowed"])
            empty = m.evaluate([], underlying(), packet(), NOW, CFG)
            m.save_result(empty, Path(folder))
            self.assertEqual(len(pd.read_csv(Path(folder)/"qqq_options_quotes.csv")), 0)

    def test_demo_is_separate_and_no_tws_import(self):
        with TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(m.main(["--demo", "--output-dir", folder]), 0)
            self.assertTrue((Path(folder)/"demo"/"qqq_options_latest.json").exists())
            self.assertFalse((Path(folder)/"qqq_options_latest.json").exists())
            self.assertIn("DEMO_", out.getvalue())

    def test_invalid_config_fails_closed(self):
        for kwargs in (dict(premium_budget_usd=float("nan")), dict(target_dte=0),
                       dict(min_open_interest=-1), dict(target_abs_delta=.9)):
            with self.assertRaises(ValueError):
                m.Config(**kwargs)

    def test_no_order_or_exercise_api_calls_in_source(self):
        tree = ast.parse(Path(m.__file__).read_text(encoding="utf-8"))
        forbidden = {"placeOrder", "exerciseOptions", "cancelOrder", "reqAccountSummary",
                     "reqPositions", "reqExecutions"}
        used = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)}
        self.assertFalse(forbidden & used)


class FakeContract:
    def __init__(self, symbol, *args, **kwargs):
        self.symbol = symbol
        self.conId = kwargs.get("conId", 0)
        # Match unqualified API contracts: symbol is NOT an OCC localSymbol.
        self.localSymbol = kwargs.get("localSymbol", "")
        self.currency = kwargs.get("currency", "USD")
        if len(args) >= 4:
            self.lastTradeDateOrContractMonth, self.strike, self.right, self.exchange = args[:4]
            self.tradingClass = kwargs["tradingClass"]
            self.multiplier = kwargs["multiplier"]
        else:
            self.exchange = args[0] if args else "SMART"
            self.right = ""


class FakeEvent:
    def __init__(self):
        self.handlers = []
    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self
    def __isub__(self, handler):
        self.handlers.remove(handler)
        return self
    def emit(self, *args):
        for handler in list(self.handlers):
            handler(*args)


class FakeIB:
    last = None
    def __init__(self):
        FakeIB.last = self
        self.connected = False
        self.next_id = 100
        self.active = []
        self.cancelled = []
        self.requests = []
        self.disconnected = False
        self.details_requests = []
        self.sleep_calls = []
        self.errorEvent = FakeEvent()
    def connect(self, *args, **kwargs):
        self.connected = True
        self.connect_kwargs = kwargs
    def isConnected(self):
        return self.connected
    def disconnect(self):
        self.connected = False
        self.disconnected = True
    def reqMarketDataType(self, mode):
        self.mode = mode
    def qualifyContracts(self, *contracts):
        for c in contracts:
            c.conId = self.next_id
            self.next_id += 1
        return list(contracts)
    def reqMktData(self, contract, generic, snapshot, regulatory):
        self.active.append(contract)
        self.requests.append((contract, generic, snapshot, regulatory))
        t = ticker(strike=getattr(contract, "strike", 600),
                   delta=-.45 if getattr(contract, "strike", 600) >= 599 else -.20)
        t.contract = contract
        return t
    def cancelMktData(self, contract):
        self.active.remove(contract)
        self.cancelled.append(contract)
    def reqHistoricalData(self, *args, **kwargs):
        return bars()
    def reqSecDefOptParams(self, symbol, exchange, kind, conid):
        return [types.SimpleNamespace(exchange="SMART", underlyingConId=conid,
                                      tradingClass="QQQ", multiplier="100",
                                      expirations={"20261020"}, strikes=set(range(520, 620)))]
    def reqContractDetails(self, query):
        self.details_requests.append(query)
        result = []
        for strike in range(530, 620):
            c = FakeContract("QQQ", query.lastTradeDateOrContractMonth, strike, "P", "SMART",
                             multiplier="100", currency="USD", tradingClass="QQQ",
                             conId=1000+strike)
            result.append(types.SimpleNamespace(contract=c, underConId=100))
        return result
    def sleep(self, value):
        self.sleep_calls.append(value)


FAKE_API = types.ModuleType("ib_insync")
FAKE_API.IB, FAKE_API.Stock, FAKE_API.Option = FakeIB, FakeContract, FakeContract
FAKE_API.util = types.SimpleNamespace(df=lambda data: pd.DataFrame(data))


class IntegrationTests(unittest.TestCase):
    def test_complete_fake_tws_cycle_exports_and_cancels(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/"advisory.json"
            path.write_text(json.dumps(packet()), encoding="utf-8")
            with patch.dict(sys.modules, {"ib_insync": FAKE_API}), \
                    patch.object(pd.Timestamp, "now", return_value=NOW), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(m.main(["--advisory-json", str(path), "--output-dir", folder]), 0)
            ib = FakeIB.last
            self.assertTrue(ib.disconnected)
            self.assertTrue(ib.connect_kwargs["readonly"])
            self.assertEqual(ib.active, [])
            self.assertTrue(all(not snap and not reg for _, _, snap, reg in ib.requests))
            self.assertTrue(all(generic == "101" for c, generic, _, _ in ib.requests if c.right == "P"))
            option_requests = [c for c, _, _, _ in ib.requests if c.right == "P"]
            self.assertEqual(len(option_requests), 12)
            self.assertTrue(all(c.conId > 0 for c in option_requests))
            self.assertEqual(len(ib.details_requests), 1)
            self.assertEqual(ib.errorEvent.handlers, [])
            saved = json.loads((Path(folder)/"qqq_options_latest.json").read_text(encoding="utf-8"))
            self.assertFalse(saved["execution_allowed"])
            self.assertIn("ÖSSZESÍTETT STÁTUSZ", output.getvalue())
            self.assertEqual(len(saved["options"]), 12)

    def test_market_data_failure_disconnects_and_exports_error(self):
        with TemporaryDirectory() as folder, \
                patch.dict(sys.modules, {"ib_insync": FAKE_API}), \
                patch.object(FakeIB, "reqHistoricalData", side_effect=RuntimeError("permission")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main(["--advisory-json", str(Path(folder)/"missing.json"),
                                     "--output-dir", folder]), 1)
            self.assertTrue(FakeIB.last.disconnected)
            self.assertEqual(FakeIB.last.active, [])
            saved = json.loads((Path(folder)/"qqq_options_latest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["state"], "COLLECTION_ERROR")


class ScannerRepairTests(unittest.TestCase):
    def setUp(self):
        self.chain = types.SimpleNamespace(exchange="SMART", underlyingConId=100,
                                          tradingClass="QQQ", multiplier="100",
                                          expirations={"20261020"}, strikes={550, 580, 590, 600, 605})

    def test_definitions_not_global_strike_expiry_product(self):
        ib = FakeIB()
        actual = ib.reqContractDetails(FakeContract("QQQ", "20261020", 0, "P", "SMART",
                                                  multiplier="100", tradingClass="QQQ"))
        actual = [d for d in actual if d.contract.strike in {590, 600}]
        ib.details_requests.clear()
        with patch.object(ib, "reqContractDetails", return_value=actual):
            options, notes, counts = m.discover_contracts(ib, FAKE_API, self.chain, 600, NOW, CFG)
        self.assertEqual({c.strike for c in options}, {590, 600})
        self.assertEqual(counts["20261020"]["selected"], 2)
        self.assertTrue(all(c.conId > 0 for c in options))
        self.assertFalse(notes)

    def test_definition_failure_does_not_guess_options(self):
        ib = FakeIB()
        with patch.object(ib, "reqContractDetails", side_effect=RuntimeError("timeout")):
            options, notes, counts = m.discover_contracts(ib, FAKE_API, self.chain, 600, NOW, CFG)
        self.assertEqual(options, [])
        self.assertEqual(counts["20261020"]["status"], "LOOKUP_FAILED")
        self.assertIn("timeout", notes[0])

    def test_wrong_underlying_or_adjusted_contract_not_used(self):
        ib = FakeIB()
        query = FakeContract("QQQ", "20261020", 0, "P", "SMART",
                             multiplier="100", tradingClass="QQQ")
        details = ib.reqContractDetails(query)
        details[0].underConId = 99
        details[1].contract.tradingClass = "QQQ1"
        details[2].contract.multiplier = "10"
        with patch.object(ib, "reqContractDetails", return_value=details[:3]):
            options, _, _ = m.discover_contracts(ib, FAKE_API, self.chain, 600, NOW, CFG)
        self.assertEqual(options, [])

    def test_discovery_bounded_by_expiry_and_quote_count(self):
        ib = FakeIB()
        chain = copy.copy(self.chain)
        chain.expirations = {"20261019", "20261020", "20261021", "20261023"}
        cfg = replace(CFG, max_expiries=2, max_contracts=15)
        options, _, counts = m.discover_contracts(ib, FAKE_API, chain, 600, NOW, cfg)
        self.assertLessEqual(len(options), 15)
        self.assertEqual(len(ib.details_requests), 2)
        self.assertEqual(len(counts), 2)
        self.assertTrue(all(c.strike == 0 and c.right == "P"
                            and c.lastTradeDateOrContractMonth for c in ib.details_requests))

    def test_wait_complete_fields_uses_base_only(self):
        ib = FakeIB()
        self.assertEqual(m.wait_for_fields(ib, [ticker()], CFG), 8)
        self.assertEqual(ib.sleep_calls, [8])

    def test_wait_missing_oi_is_bounded_without_imputation(self):
        ib, t = FakeIB(), ticker(oi=float("nan"))
        cfg = replace(CFG, max_quote_wait_seconds=10)
        self.assertEqual(m.wait_for_fields(ib, [t], cfg), 10)
        self.assertEqual(sum(ib.sleep_calls), 10)
        self.assertIsNone(m.quote_row(t, CFG, NOW)["open_interest"])

    def test_wait_accepts_late_field_arrival(self):
        ib, t = FakeIB(), ticker(oi=float("nan"))
        def arrive(seconds):
            ib.sleep_calls.append(seconds)
            if len(ib.sleep_calls) == 2:
                t.putOpenInterest = 150
        with patch.object(ib, "sleep", side_effect=arrive):
            self.assertEqual(m.wait_for_fields(ib, [t], CFG), 9)
        self.assertEqual(t.putOpenInterest, 150)

    def test_subscription_warning_retains_partial_fields_and_cleans_handler(self):
        ib = FakeIB()
        original = ib.reqMktData
        def limited(contract, *args):
            t = original(contract, *args)
            if contract.right == "P":
                ib.errorEvent.emit(1, 10091, "additional subscription", contract)
            return t
        with patch.object(ib, "reqMktData", side_effect=limited), \
                patch.object(pd.Timestamp, "now", return_value=NOW):
            u, rows, notes = m.collect(ib, FAKE_API, CFG)
        self.assertTrue(u["collection"]["subscription_warning"])
        self.assertEqual(u["collection"]["api_error_counts"][10091], 12)
        self.assertEqual(rows[0]["api_error_codes"], [10091])
        self.assertIsNotNone(rows[0]["delta"])
        self.assertTrue(any("jogosultság" in n for n in notes))
        self.assertEqual(ib.active, [])
        self.assertEqual(ib.errorEvent.handlers, [])

    def test_all_blockers_visible_not_only_missing_json(self):
        u = dict(underlying(), below_rth_vwap=False, paper_price_gate=False)
        r = m.evaluate([row(ask=5, bid=4.9)], u, None, NOW, CFG)
        self.assertEqual(r["state"], "REGIME_INPUT_NA_OR_STALE")
        self.assertTrue(any("MISSING" in b for b in r["blockers"]))
        self.assertTrue(any("VWAP" in b for b in r["blockers"]))
        self.assertEqual(r["field_coverage"]["long_leg_matches"], 1)
        self.assertEqual(r["field_coverage"]["budget_long_matches"], 0)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            m.show(r)
        self.assertIn("AKADÁLY", out.getvalue())
        self.assertIn("LONG költség: 501.00", out.getvalue())

    def test_missing_advisory_still_exports_diagnostics(self):
        with TemporaryDirectory() as folder, \
                patch.dict(sys.modules, {"ib_insync": FAKE_API}), \
                patch.object(pd.Timestamp, "now", return_value=NOW), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(m.main(["--advisory-json", str(Path(folder)/"missing.json"),
                                     "--output-dir", folder]), 0)
            saved = json.loads((Path(folder)/"qqq_options_latest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["state"], "REGIME_INPUT_NA_OR_STALE")
        self.assertFalse(saved["paper_long_put_candidates"])
        self.assertEqual(len(saved["options"]), 12)
        self.assertIn("Csak adatdiagnosztika", out.getvalue())

    def test_empty_underlying_has_explicit_blocker(self):
        r = m.evaluate([], dict(price=None, quality="MISSING", asof=None), None, NOW, CFG)
        self.assertEqual(r["state"], "DATA_UNAVAILABLE")
        self.assertEqual(len(r["blockers"]), 2)

    def test_max_wait_less_than_base_rejected(self):
        with self.assertRaises(ValueError):
            replace(CFG, quote_wait_seconds=8, max_quote_wait_seconds=7)


class ConfigPrecedenceTests(unittest.TestCase):
    def run_demo(self, base, cli):
        with TemporaryDirectory() as folder, \
                patch.object(m, "Config", return_value=base), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(m.main(["--demo", "--output-dir", folder]+cli), 0)
            saved = json.loads((Path(folder)/"demo"/"qqq_options_latest.json").read_text(encoding="utf-8"))
        return saved, out.getvalue()

    def test_no_cli_preserves_edited_config_and_actual_filter(self):
        base = replace(CFG, premium_budget_usd=500)
        saved, output = self.run_demo(base, [])
        self.assertEqual(saved["config"]["premium_budget_usd"], 500)
        self.assertTrue(saved["diagnostic_long_puts"])  # demo long cost is 421 USD
        self.assertIn("Tényleges prémiumkeret: 500.00", output)
        self.assertIn("forrás: Config.premium_budget_usd", output)

    def test_cli_budget_overrides_config_and_actual_filter(self):
        saved, output = self.run_demo(replace(CFG, premium_budget_usd=500), ["--budget", "300"])
        self.assertEqual(saved["config"]["premium_budget_usd"], 300)
        self.assertFalse(saved["diagnostic_long_puts"])
        self.assertIn("forrás: --budget argumentum", output)

    def test_other_config_defaults_not_silently_overridden(self):
        base = replace(CFG, premium_budget_usd=650, target_dte=10,
                       target_abs_delta=.4, min_open_interest=200,
                       quote_wait_seconds=9, max_quote_wait_seconds=25)
        saved, _ = self.run_demo(base, [])
        for field in ("premium_budget_usd", "target_dte", "target_abs_delta",
                      "min_open_interest", "quote_wait_seconds", "max_quote_wait_seconds"):
            self.assertEqual(saved["config"][field], getattr(base, field))

    def test_explicit_zero_is_validated_not_treated_as_missing(self):
        with TemporaryDirectory() as folder, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                m.main(["--demo", "--output-dir", folder, "--budget", "0"])
        self.assertEqual(error.exception.code, 2)


class DefinitionDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.ib = FakeIB()
        self.chain = types.SimpleNamespace(exchange="SMART", underlyingConId=100,
                                           tradingClass="QQQ", multiplier="100",
                                           expirations={"20261020"}, strikes={590,600,605})
        query = FakeContract("QQQ","20261020",0,"P","SMART",
                             multiplier="100",currency="USD",tradingClass="QQQ")
        self.detail = self.ib.reqContractDetails(query)[70]  # strike 600
        self.ib.details_requests.clear()

    def discover(self, details):
        with patch.object(self.ib,"reqContractDetails",return_value=details):
            return m.discover_contracts(self.ib,FAKE_API,self.chain,600,NOW,CFG)

    def test_numeric_string_underlying_and_contract_id(self):
        self.detail.underConId = "100"
        self.detail.contract.conId = "1600"
        options,_,_=self.discover([self.detail])
        self.assertEqual(options[0].conId,1600)

    def test_definition_local_symbol_survives_quote_contract_reconstruction(self):
        self.detail.contract.localSymbol = "QQQ   261020P00600000"
        options,_,_=self.discover([self.detail])
        self.assertEqual(options[0].localSymbol,"QQQ   261020P00600000")
        t=ticker()
        t.contract=options[0]
        quote=m.quote_row(t,CFG,NOW)
        self.assertEqual(quote["local_symbol"],"QQQ   261020P00600000")

    def test_absent_local_symbol_not_fabricated_from_symbol(self):
        self.detail.contract.localSymbol = ""
        options,_,_=self.discover([self.detail])
        self.assertEqual(options[0].localSymbol,"")
        self.assertEqual(m.parameter_value("local_symbol",options[0].localSymbol),"N/A")

    def test_missing_underlying_not_inferred(self):
        self.detail.underConId = 0
        options,_,counts=self.discover([self.detail])
        self.assertEqual(options,[])
        self.assertEqual(counts["20261020"]["rejection_counts"],
                         {"UNDERLYING_CONID_MISSING_OR_INVALID":1})

    def test_wrong_underlying_remains_blocked(self):
        self.detail.underConId = 101
        self.assertIn("UNDERLYING_CONID_MISMATCH",
                      m.definition_rejections(self.detail,"20261020",100,600))

    def test_invalid_chain_ids_do_not_raise_or_match(self):
        self.chain.underlyingConId = "missing"
        self.assertIsNone(m.select_chain([self.chain],100))
        self.assertIsNone(m.select_chain([self.chain],None))

    def test_numeric_multiplier_chain_normalization(self):
        self.chain.multiplier = "100.0"
        self.chain.underlyingConId = "100"
        self.assertIs(m.select_chain([self.chain],100),self.chain)

    def test_timestamped_expiry_normalized_before_filter(self):
        self.detail.contract.lastTradeDateOrContractMonth = "20261020 16:00:00 US/Eastern"
        options,_,counts=self.discover([self.detail])
        self.assertEqual(len(options),1)
        self.assertEqual(options[0].lastTradeDateOrContractMonth,"20261020")
        self.assertEqual(counts["20261020"]["normalized_expiry_count"],1)

    def test_invalid_suffix_not_just_first_eight_characters(self):
        self.detail.contract.lastTradeDateOrContractMonth = "20261020WRONG"
        self.assertFalse(self.discover([self.detail])[0])

    def test_invalid_calendar_date_and_time_rejected(self):
        for text in ("20260230","202610","20261020 99:00:00 US/Eastern"):
            with self.subTest(text=text):
                self.assertIsNone(m.expiry_date(text))

    def test_real_expiry_disagreement_not_overridden(self):
        self.detail.realExpirationDate = "20261021"
        options,_,counts=self.discover([self.detail])
        self.assertFalse(options)
        self.assertIn("REAL_EXPIRATION_DATE_MISMATCH",
                      counts["20261020"]["rejection_counts"])

    def test_empty_api_response_distinct_from_rejection(self):
        options,_,counts=self.discover([])
        report=counts["20261020"]
        self.assertFalse(options)
        self.assertEqual(report["status"],"EMPTY_CONTRACT_DETAILS")
        self.assertEqual(report["raw_returned"],0)
        self.assertEqual(report["rejected_definitions"],0)

    def test_all_rejected_status_and_all_reasons_counted(self):
        self.detail.contract.tradingClass="QQQ1"
        self.detail.contract.multiplier="10"
        _,_,counts=self.discover([self.detail])
        report=counts["20261020"]
        self.assertEqual(report["status"],"ALL_CONTRACT_DETAILS_REJECTED")
        self.assertEqual(report["raw_returned"],1)
        self.assertEqual(report["rejected_definitions"],1)
        self.assertEqual(len(report["rejection_counts"]),2)

    def test_only_outside_price_band_distinguished(self):
        self.detail.contract.strike=900
        _,_,counts=self.discover([self.detail])
        self.assertEqual(counts["20261020"]["status"],"NO_CONTRACTS_IN_PRICE_BAND")
        self.assertEqual(counts["20261020"]["strike_band"],[510,618])

    def test_missing_contract_gets_diagnostic_not_exception(self):
        _,_,counts=self.discover([types.SimpleNamespace()])
        self.assertEqual(counts["20261020"]["rejection_counts"],{"CONTRACT_MISSING":1})

    def test_non_option_security_rejected(self):
        self.detail.contract.secType="FOP"
        self.assertIn("NOT_OPTION",m.definition_rejections(self.detail,"20261020",100,600))

    def test_samples_bounded_and_json_serializable(self):
        details=[]
        for field in ("symbol","right","currency","tradingClass"):
            d=copy.deepcopy(self.detail)
            setattr(d.contract,field,"OTHER")
            details += [d]*5
        _,_,counts=self.discover(details)
        report=counts["20261020"]
        self.assertEqual(report["raw_returned"],20)
        self.assertEqual(report["rejected_definitions"],20)
        self.assertEqual(len(report["rejected_examples"]),3)
        json.dumps(report,allow_nan=False)

    def test_valid_contract_not_lost_among_rejected_details(self):
        wrong=copy.deepcopy(self.detail)
        wrong.underConId=999
        options,_,counts=self.discover([wrong,self.detail])
        self.assertEqual(len(options),1)
        self.assertEqual(counts["20261020"]["available"],1)
        self.assertEqual(counts["20261020"]["rejected_definitions"],1)

    def test_no_rows_is_not_option_filter_failure(self):
        result=m.evaluate([],underlying(),packet(),NOW,CFG)
        self.assertTrue(any("kontraktusfeloldást" in s for s in result["blockers"]))
        self.assertFalse(any("minden long-put szűrőn" in s for s in result["blockers"]))
        self.assertEqual(result["rejection_counts"],{})

    def test_selected_but_no_quotes_is_separate_failure(self):
        u=dict(underlying(),chain=dict(requested_contract_count=3))
        result=m.evaluate([],u,packet(),NOW,CFG)
        self.assertTrue(any("jegyzéssor sem" in s for s in result["blockers"]))

    def test_closed_regime_keeps_independent_quote_collection(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/"advisory.json"
            path.write_text(json.dumps(packet(allowed=False)),encoding="utf-8")
            with patch.dict(sys.modules,{"ib_insync":FAKE_API}), \
                    patch.object(pd.Timestamp,"now",return_value=NOW), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(m.main(["--advisory-json",str(path),"--output-dir",folder,
                                         "--budget","5000"]),0)
            result=json.loads((Path(folder)/"qqq_options_latest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(result["options"]),12)
        self.assertEqual(result["config"]["premium_budget_usd"],5000)
        self.assertEqual(result["state"],"REGIME_BLOCKS_NEW_PUT")
        self.assertFalse(result["paper_long_put_candidates"])
        self.assertFalse(result["paper_spread_candidates"])
        self.assertTrue(result["diagnostic_long_puts"])
        self.assertFalse(result["execution_allowed"])

    def test_show_raw_count_rejections_and_sample(self):
        self.detail.contract.tradingClass="QQQ1"
        _,_,counts=self.discover([self.detail])
        u=dict(underlying(),chain=dict(discovery_by_expiry=counts))
        result=m.evaluate([],u,packet(),NOW,CFG)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            m.show(result)
        text=out.getvalue()
        self.assertIn("nyers API-válasz: 1",text)
        self.assertIn("TRADING_CLASS_NOT_QQQ",text)
        self.assertIn("underConId=100",text)
        self.assertIn("nem bizonyít előfizetési hiányt",text)


class FullDiagnosticDisplayTests(unittest.TestCase):
    def result(self, allowed=False):
        long = row()
        short = row(strike=590,delta=-.2,bid=1.5,ask=1.6,conid=2)
        long["api_error_codes"]=[10091,10167]
        short["api_error_codes"]=[10091,10167]
        p=packet(allowed=allowed)
        if not allowed:
            p["advisory"]["state"]="CRASH_CONFIRMED_WAIT_FOR_BOUNCE"
        return m.evaluate([long,short],underlying(),p,NOW,CFG)

    def test_blocked_matches_have_two_separate_tables(self):
        r=self.result()
        text="\n".join(m.render_diagnostics(r))
        self.assertIn("DIAGNOSZTIKAI LONG PUT TALÁLATOK",text)
        self.assertIn("DIAGNOSZTIKAI PUT DEBIT SPREAD TALÁLATOK",text)
        self.assertGreaterEqual(text.count("REZSIM MIATT NEM BELÉPHETŐ"),4)
        self.assertEqual(len(r["diagnostic_long_puts"]),1)
        self.assertEqual(len(r["diagnostic_spreads"]),1)
        self.assertFalse(r["paper_long_put_candidates"])
        self.assertFalse(r["paper_spread_candidates"])
        self.assertFalse(r["execution_allowed"])

    def test_all_long_fields_have_display_labels(self):
        r=self.result()
        for key in r["diagnostic_long_puts"][0]:
            self.assertIn(key,m.PARAMETER_NAMES)

    def test_unknown_future_scalar_field_not_hidden(self):
        r=self.result()
        r["diagnostic_long_puts"][0]["future_metric"]="UNIQUE_EXTRA_VALUE"
        text="\n".join(m.render_diagnostics(r))
        self.assertIn("future_metric",text)
        self.assertIn("UNIQUE_EXTRA_VALUE",text)

    def test_spread_has_complete_legs_and_all_scalar_fields_named(self):
        r=self.result()
        s=r["diagnostic_spreads"][0]
        self.assertEqual(s["long_leg"],r["options"][0])
        self.assertEqual(s["short_leg"],r["options"][1])
        for k in s:
            if k not in ("long_leg","short_leg"):
                self.assertIn(k,m.PARAMETER_NAMES)

    def test_render_shows_greeks_modes_timestamps_and_errors(self):
        text="\n".join(m.render_diagnostics(self.result()))
        for expected in ("Gamma","Theta","Vega","IV (%)","Open interest","Bid méret",
                         "Ask méret","Nettó delta","Jelzett adatmód",
                         "modelGreeks","10091","10167","UNVERIFIED_SOURCE_TIME",
                         "SOURCE_TIME_AND_COMBO_NOT_QUOTED","Greeks mögötti underlying"):
            with self.subTest(expected=expected):
                # Wrapped values may split long technical identifiers.
                self.assertIn(expected.replace(" ",""),
                              "".join(text.replace("|","").split()))

    def test_net_greeks_are_long_minus_short(self):
        long=row();short=row(strike=590,delta=-.2,bid=1.5,ask=1.6,conid=2)
        long.update(gamma=.02,theta=-.2,vega=.3)
        short.update(gamma=.004,theta=-.05,vega=.1)
        ls,ss=m.screen([long,short],600,CFG)
        s=m.build_spreads(ls,ss,CFG)[0]
        self.assertAlmostEqual(s["net_gamma"],.016)
        self.assertAlmostEqual(s["net_theta"],-.15)
        self.assertAlmostEqual(s["net_vega"],.2)
        self.assertEqual(s["opening_fee_usd"],2)
        self.assertEqual(s["long_ask_cost_usd"],250)
        self.assertEqual(s["short_bid_credit_usd"],150)

    def test_missing_optional_greek_is_not_zero(self):
        r=self.result()
        long,short=r["options"]
        short["gamma"]=None
        s=m.build_spreads([long],[short],CFG)[0]
        self.assertIsNone(s["net_gamma"])

    def test_spread_can_keep_over_budget_standalone_long_leg(self):
        long=row(bid=4.1,ask=4.2)
        short=row(strike=590,delta=-.2,bid=1.5,ask=1.6,conid=2)
        r=m.evaluate([long,short],underlying(),packet(allowed=False),NOW,CFG)
        self.assertEqual(r["diagnostic_long_puts"],[])
        s=r["diagnostic_spreads"][0]
        self.assertFalse(s["long_leg"]["long_budget_pass"])
        self.assertEqual(s["opening_cost_usd"],272)
        text="\n".join(m.render_diagnostics(r))
        self.assertIn("SHORT láb önálló LONG-szűrője lehet NEM",text)

    def test_all_diagnostics_shown_paper_top_five_unchanged(self):
        quotes=[row(strike=600+i,conid=10+i) for i in range(6)]
        quotes.append(row(strike=590,delta=-.2,bid=1,ask=1.1,conid=2))
        r=m.evaluate(quotes,underlying(),packet(),NOW,CFG)
        self.assertEqual(len(r["diagnostic_long_puts"]),6)
        self.assertEqual(len(r["diagnostic_spreads"]),6)
        self.assertEqual(len(r["paper_long_put_candidates"]),5)
        self.assertEqual(len(r["paper_spread_candidates"]),5)
        self.assertEqual(r["field_coverage"]["spread_matches"],6)
        text="\n".join(m.render_diagnostics(r))
        self.assertIn("LONG #6",text)
        self.assertIn("SPREAD #6",text)

    def test_render_does_not_mutate_or_open_gate(self):
        r=self.result();before=copy.deepcopy(r)
        m.render_diagnostics(r)
        self.assertEqual(r,before)
        self.assertFalse(r["regime_gate"]["paper_allowed"])

    def test_stale_input_not_mislabelled_as_regime(self):
        r=self.result();r["regime_gate"]["quality"]="STALE"
        text="\n".join(m.render_diagnostics(r))
        self.assertIn("INDEXADAT MIATT NEM BELÉPHETŐ",text)
        self.assertNotIn("REZSIM MIATT NEM BELÉPHETŐ",text)

    def test_qqq_time_gate_has_own_label(self):
        r=self.result(allowed=True)
        r["state"]="QQQ_PRICE_OR_TIME_ALIGNMENT_BLOCKS_NEW_PUT"
        self.assertEqual(m.diagnostic_status(r,"SPREAD"),"QQQ/IDŐKAPU MIATT NEM BELÉPHETŐ")

    def test_spread_preference_only_blocks_long_label(self):
        r=self.result(allowed=True)
        r["regime_gate"]["structure"]="QQQ_PUT_DEBIT_SPREAD_PREFERRED"
        self.assertIn("STRUKTÚRA",m.diagnostic_status(r,"LONG"))
        self.assertNotIn("STRUKTÚRA",m.diagnostic_status(r,"SPREAD"))

    def test_percent_and_missing_values_preserved(self):
        self.assertEqual(m.parameter_value("iv",.25),"25% (nyers: 0.25)")
        self.assertEqual(m.parameter_value("theta",None),"N/A")
        self.assertEqual(m.parameter_value("net_delta",0),"0")
        self.assertEqual(m.parameter_value("execution_allowed",False),"NEM")
        self.assertEqual(m.parameter_value("local_symbol","   "),"N/A")

    def test_aligned_table_wraps_long_values(self):
        lines=m.console_table(["A","B"],[["long text that wraps","Árvíztűrő"]],[6,10])
        self.assertEqual(len({len(line) for line in lines}),1)
        self.assertGreater(len(lines),5)

    def test_old_result_resolves_legs_by_conid(self):
        r=self.result()
        s=r["diagnostic_spreads"][0]
        del s["long_leg"];del s["short_leg"]
        text="\n".join(m.render_diagnostics(r))
        self.assertIn("SPREAD #1 LÁBAK",text)
        self.assertIn("TEST-2",text)

    def test_missing_old_legs_explicit_no_guess(self):
        r=self.result()
        s=r["diagnostic_spreads"][0]
        del s["long_leg"];del s["short_leg"];r["options"]=[]
        self.assertIn("ADATHIÁNY","\n".join(m.render_diagnostics(r)))

    def test_exports_have_full_flattened_legs_and_block_status(self):
        r=self.result()
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);m.save_result(r,folder)
            spread=pd.read_csv(folder/"qqq_options_diagnostic_spreads.csv")
            longs=pd.read_csv(folder/"qqq_options_diagnostic_long_puts.csv")
            self.assertEqual(spread.loc[0,"diagnostic_status"],"REZSIM MIATT NEM BELÉPHETŐ")
            self.assertEqual(spread.loc[0,"long_leg_conid"],1)
            self.assertEqual(spread.loc[0,"short_leg_conid"],2)
            self.assertIn("long_leg_gamma",spread.columns)
            self.assertIn("short_leg_api_error_codes",spread.columns)
            self.assertEqual(len(longs),1)
            self.assertFalse(spread.loc[0,"execution_allowed"])
            text=(folder/"qqq_options_diagnostics.txt").read_text(encoding="utf-8")
            self.assertEqual(text,"\n".join(m.render_diagnostics(r))+"\n")

    def test_empty_new_run_overwrites_old_diagnostic_exports(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);m.save_result(self.result(),folder)
            empty=m.evaluate([],underlying(),packet(allowed=False),NOW,CFG)
            m.save_result(empty,folder)
            for name in ("long_puts","spreads"):
                self.assertEqual(len(pd.read_csv(folder/f"qqq_options_diagnostic_{name}.csv")),0)
            self.assertNotIn("SPREAD #1",(folder/"qqq_options_diagnostics.txt").read_text(encoding="utf-8"))

    def test_full_json_keeps_original_numeric_leg_values(self):
        r=self.result()
        encoded=json.loads(json.dumps(r,allow_nan=False))
        s=encoded["diagnostic_spreads"][0]
        self.assertIsInstance(s["long_leg"]["iv"],float)
        self.assertEqual(s["long_leg"]["api_error_codes"],[10091,10167])
        self.assertFalse(s["execution_allowed"])


if __name__ == "__main__":
    unittest.main()
