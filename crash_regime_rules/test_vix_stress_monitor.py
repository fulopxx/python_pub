"""Offline regression suite: python -m unittest test_vix_stress_monitor -v."""
from __future__ import annotations

import contextlib
import io
import json
import types
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import run_vix_stress_monitor_v1_0 as m


NOW = pd.Timestamp("2026-10-13 15:00", tz="UTC")


def frame(base, rise=0., n=181):
    idx = pd.date_range(end=NOW-pd.Timedelta(minutes=15), periods=n, freq="min")
    prices = [base+rise*max(0, i-149)/30 for i in range(n)]
    return pd.DataFrame(dict(open=prices, high=[p+.01 for p in prices],
                             low=[p-.01 for p in prices], close=prices, volume=100.), index=idx)


def result(state="STRESS_BUILDING", offset=0):
    return dict(candidate_state=state,
                asof=(NOW+pd.Timedelta(minutes=offset)).isoformat(),
                observed_at=(NOW+pd.Timedelta(minutes=offset+15)).isoformat())


def contract_details(expiry, conid, trading_class="VX", symbol="VIX"):
    return types.SimpleNamespace(
        realExpirationDate=expiry,
        contract=types.SimpleNamespace(secType="FUT", tradingClass=trading_class,
                                       conId=conid, symbol=symbol,
                                       lastTradeDateOrContractMonth=expiry))


class MonitorTests(unittest.TestCase):
    def test_normal_contango(self):
        r = m.evaluate({"F1": frame(21), "F2": frame(24)}, now=NOW)
        self.assertEqual(r["candidate_state"], "NORMAL")

    def test_joint_acceleration_and_compression(self):
        r = m.evaluate({"F1": frame(21, 2), "F2": frame(24, .2)}, now=NOW)
        self.assertEqual(r["candidate_state"], "STRESS_BUILDING")
        self.assertFalse(r["short_covering_confirmed"])

    def test_backwards_curve_with_rising_front(self):
        r = m.evaluate({"F1": frame(25, 1), "F2": frame(24, .1)}, now=NOW)
        self.assertEqual(r["candidate_state"], "BACKWARDATION_STRESS")

    def test_price_alone_is_watch(self):
        r = m.evaluate({"F1": frame(21, 2), "F2": frame(24, 3)}, now=NOW)
        self.assertEqual(r["candidate_state"], "WATCH")

    def test_missing_core_not_normal(self):
        r = m.evaluate({"UVIX": frame(30)}, now=NOW)
        self.assertEqual(r["candidate_state"], "DATA_UNAVAILABLE")

    def test_optional_missing_keeps_core_signal(self):
        r = m.evaluate({"F1": frame(21, 2), "F2": frame(24, .2)}, now=NOW)
        self.assertEqual(r["candidate_state"], "STRESS_BUILDING")
        self.assertIsNone(r["checks"]["uvix_up_svix_down"])

    def test_stale_core_warning_not_market_stress(self):
        r = m.evaluate({"F1": frame(21, 2), "F2": frame(24, .2)},
                       now=NOW+pd.Timedelta(hours=2))
        self.assertEqual(r["candidate_state"], "DATA_UNAVAILABLE")

    def test_same_source_repoll_does_not_confirm(self):
        emit, memory, _ = m.alert_update(result(), {}, "1:2")
        self.assertFalse(emit)
        emit, memory, _ = m.alert_update(result(), memory, "1:2")
        self.assertFalse(emit)
        self.assertEqual(memory["confirmation_count"], 1)

    def test_advancing_source_confirms(self):
        _, memory, _ = m.alert_update(result(), {}, "1:2")
        emit, memory, _ = m.alert_update(result(offset=5), memory, "1:2")
        self.assertTrue(emit)
        self.assertEqual(memory["confirmation_count"], 2)

    def test_not_enough_elapsed_source_time(self):
        _, memory, _ = m.alert_update(result(), {}, "1:2")
        emit, memory, _ = m.alert_update(result(offset=1), memory, "1:2")
        self.assertFalse(emit)
        self.assertEqual(memory["confirmation_count"], 1)

    def test_cooldown_suppresses_repeat(self):
        _, memory, _ = m.alert_update(result(), {}, "1:2")
        _, memory, _ = m.alert_update(result(offset=5), memory, "1:2")
        emit, _, _ = m.alert_update(result(offset=10), memory, "1:2")
        self.assertFalse(emit)

    def test_roll_resets_confirmation(self):
        _, memory, _ = m.alert_update(result(), {}, "1:2")
        emit, memory, _ = m.alert_update(result(offset=5), memory, "2:3")
        self.assertFalse(emit)
        self.assertEqual(memory["confirmation_count"], 1)

    def test_long_restart_gap_cannot_confirm_old_candidate(self):
        _, memory, _ = m.alert_update(result(), {}, "1:2")
        emit, memory, _ = m.alert_update(result(offset=60), memory, "1:2")
        self.assertFalse(emit)
        self.assertEqual(memory["confirmation_count"], 1)

    def test_normal_resets_confirmation(self):
        _, memory, _ = m.alert_update(result(), {}, "1:2")
        _, memory, _ = m.alert_update(result("NORMAL", 5), memory, "1:2")
        emit, memory, _ = m.alert_update(result(offset=10), memory, "1:2")
        self.assertFalse(emit)
        self.assertEqual(memory["confirmation_count"], 1)

    def test_feed_alert_once_not_every_poll(self):
        r = result("DATA_UNAVAILABLE")
        a, memory, _ = m.alert_update(r, {}, "1:2")
        b, _, _ = m.alert_update(r, memory, "1:2")
        self.assertTrue(a)
        self.assertFalse(b)

    def test_latest_forming_bar_excluded(self):
        x = frame(21, 2)
        a = m.closed_minutes(x)
        x.iloc[-1] = [50, 60, 40, 50, 1000000]
        b = m.closed_minutes(x)
        self.assertEqual(a.close.iloc[-1], b.close.iloc[-1])

    def test_reference_gap_blocks_signal(self):
        a, b = frame(21, 2), frame(24, .2)
        target = a.index[-2]-pd.Timedelta(minutes=30)
        b = b.drop(b.index[(b.index >= target-pd.Timedelta(minutes=3)) &
                           (b.index <= target)])
        r = m.evaluate({"F1": a, "F2": b}, now=NOW)
        self.assertEqual(r["candidate_state"], "DATA_UNAVAILABLE")

    def test_return_is_time_based_not_row_count(self):
        x = frame(21, 2)
        x = x.drop(x.index[-10:-5])
        r = m.evaluate({"F1": x, "F2": frame(24, .2)}, now=NOW)
        self.assertEqual(r["candidate_state"], "STRESS_BUILDING")

    def test_monthly_selection_ignores_weeklies(self):
        details = [contract_details("20261021", 1),
                   contract_details("20261014", 99, "VX42"),
                   contract_details("20261118", 2)]
        a, b = m.select_monthlies(details, NOW)
        self.assertEqual((a.conId, b.conId), (1, 2))

    def test_expiry_day_skipped(self):
        details = [contract_details("20261013", 1),
                   contract_details("20261021", 2), contract_details("20261118", 3)]
        a, b = m.select_monthlies(details, NOW)
        self.assertEqual((a.conId, b.conId), (2, 3))

    def test_explicit_expiries(self):
        details = [contract_details("20261021", 1), contract_details("20261118", 2)]
        a, b = m.select_monthlies(details, NOW, "202610", "202611")
        self.assertEqual((a.conId, b.conId), (1, 2))

    def test_ambiguous_or_one_expiry_rejected(self):
        details = [contract_details("20261021", 1), contract_details("20261118", 2)]
        with self.assertRaises(ValueError):
            m.select_monthlies(details, NOW, "202610", None)

    def test_negative_index_volume_does_not_destroy_price(self):
        x = frame(20)
        x.volume = -1
        data = m.closed_minutes(x)
        self.assertFalse(data.empty)
        self.assertTrue(data.volume.isna().all())

    def test_stale_etfs_not_confirmation(self):
        x = frame(30, 4)
        x.index -= pd.Timedelta(days=1)
        r = m.evaluate({"F1": frame(21, 2), "F2": frame(24, .2),
                        "UVIX": x, "SVIX": x}, now=NOW)
        self.assertIsNone(r["checks"]["uvix_up_svix_down"])

    def test_volume_needs_five_comparable_windows(self):
        x = m.closed_minutes(frame(21))
        self.assertIsNone(m.time_of_day_volume_ratio(x, x.index[-1], m.Config()))

    def test_volume_time_of_day_baseline(self):
        x = m.closed_minutes(frame(21))
        earlier = []
        for days in range(1, 6):
            p = x.copy()
            p.index -= pd.Timedelta(days=days)
            p.volume = 50.
            earlier.append(p)
        data = pd.concat([*earlier, x]).sort_index()
        self.assertAlmostEqual(m.time_of_day_volume_ratio(data, x.index[-1], m.Config()), 2.)

    def test_output_json_and_csv(self):
        r = m.evaluate({"F1": frame(21), "F2": frame(24)}, now=NOW)
        emit, memory, message = m.alert_update(r, {}, "1:2")
        with TemporaryDirectory() as path:
            m.save_outputs(r, memory, Path(path), emit, message)
            packet = json.loads((Path(path)/"vix_monitor_latest.json").read_text())
            self.assertFalse(packet["short_covering_confirmed"])
            self.assertEqual(len(pd.read_csv(Path(path)/"vix_monitor_observations.csv")), 1)

    def test_demo_no_tws_dependency(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            exit_code = m.main(["--demo"])
        self.assertEqual(exit_code, 0)
        self.assertIn("SYNTHETIC_DEMO_NOT_MARKET_DATA", output.getvalue())

    def test_summary_for_every_candidate_state(self):
        cases = [
            ("NORMAL", 0, "NINCS AKTUÁLIS STRESSZTRIGGER"),
            ("WATCH", 0, "FIGYELŐ ÁLLAPOT"),
            ("DATA_UNAVAILABLE", 0, "NEM ÉRTÉKELHETŐ"),
            ("STRESS_BUILDING", 1, "MEGERŐSÍTÉSRE VÁR"),
            ("STRESS_BUILDING", 2, "MEGERŐSÍTVE"),
            ("BACKWARDATION_STRESS", 2, "BACKWARDATION STRESSZ"),
        ]
        for state, count, expected in cases:
            with self.subTest(state=state, count=count):
                status, _ = m.overall_status(result(state), {"confirmation_count": count})
                self.assertIn(expected, status)

    def test_five_minute_lagged_etf_uses_own_timestamp(self):
        etf = frame(30, 3)
        etf.index -= pd.Timedelta(minutes=5)
        r = m.evaluate({"F1": frame(21), "F2": frame(24),
                        "UVIX": etf, "SVIX": etf}, now=NOW)
        q = r["feed_quality"]["UVIX"]
        self.assertEqual(q["status"], "OK_LAGGED")
        self.assertEqual(q["lag_to_vx_minutes"], 5)
        self.assertIsNotNone(r["metrics"]["uvix_change_15m_pct"])
        data = m.closed_minutes(etf)
        expected = m.change(data, data.index[-1], 15, m.Config(), same_rth_day=True)
        self.assertAlmostEqual(r["metrics"]["uvix_change_15m_pct"], expected)
        self.assertEqual(q["used_asof"], data.index[-1].isoformat())
        self.assertEqual(r["candidate_state"], "NORMAL")

    def test_six_minute_lagged_etf_rejected(self):
        etf = frame(30, 3)
        etf.index -= pd.Timedelta(minutes=6)
        r = m.evaluate({"F1": frame(21), "F2": frame(24), "UVIX": etf}, now=NOW)
        self.assertEqual(r["feed_quality"]["UVIX"]["status"], "OUT_OF_ALIGNMENT")
        self.assertIsNone(r["metrics"]["uvix_change_15m_pct"])

    def test_pair_mismatch_does_not_hide_individual_returns(self):
        uvix, svix = frame(30, 3), frame(30, -3)
        uvix.index -= pd.Timedelta(minutes=5)
        r = m.evaluate({"F1": frame(21), "F2": frame(24),
                        "UVIX": uvix, "SVIX": svix}, now=NOW)
        self.assertIsNotNone(r["metrics"]["uvix_change_15m_pct"])
        self.assertIsNotNone(r["metrics"]["svix_change_15m_pct"])
        self.assertIsNone(r["checks"]["uvix_up_svix_down"])
        self.assertEqual(r["etf_pair_status"], "PAIR_TIME_MISMATCH")

    def test_index_ratio_requires_aligned_index_pair(self):
        a, b = frame(20), frame(24)
        a.index -= pd.Timedelta(minutes=5)
        r = m.evaluate({"F1": frame(21), "F2": frame(24), "VIX": a, "VIX3M": b}, now=NOW)
        self.assertEqual(r["metrics"]["vix"], 20)
        self.assertIsNone(r["metrics"]["vix_vix3m_ratio"])
        self.assertEqual(r["index_pair_status"], "PAIR_TIME_MISMATCH")

    def test_missing_feeds_diagnosed_even_without_core(self):
        r = m.evaluate({}, now=NOW)
        self.assertEqual(len(r["feed_quality"]), 6)
        self.assertTrue(all(q["status"] == "MISSING" for q in r["feed_quality"].values()))

    def test_stale_etf_diagnostic(self):
        etf = frame(30)
        etf.index -= pd.Timedelta(days=1)
        r = m.evaluate({"F1": frame(21), "F2": frame(24), "UVIX": etf}, now=NOW)
        self.assertEqual(r["feed_quality"]["UVIX"]["status"], "STALE")

    def test_missing_etf_reference_not_replaced_with_bar_count(self):
        etf = frame(30)
        own_end = m.closed_minutes(etf).index[-1]
        # Remove the elapsed reference and its two tolerance minutes.
        completed = etf.index+pd.Timedelta(minutes=1)
        etf = etf.loc[~completed.isin(pd.date_range(
            own_end-pd.Timedelta(minutes=17), own_end-pd.Timedelta(minutes=15), freq="min"))]
        r = m.evaluate({"F1": frame(21), "F2": frame(24), "UVIX": etf}, now=NOW)
        self.assertEqual(r["feed_quality"]["UVIX"]["status"], "MISSING_REFERENCE")
        self.assertIsNone(r["metrics"]["uvix_change_15m_pct"])

    def test_etf_outside_rth_and_opening_reference(self):
        for hour, minute, expected in [(13, 20, "OUTSIDE_RTH"), (13, 40, "MISSING_REFERENCE")]:
            wall = NOW.replace(hour=hour, minute=minute)+pd.Timedelta(minutes=15)
            shift = wall-NOW
            core1, core2, etf = frame(21), frame(24), frame(30)
            for x in (core1, core2, etf):
                x.index += shift
            r = m.evaluate({"F1": core1, "F2": core2, "UVIX": etf}, now=wall)
            self.assertEqual(r["feed_quality"]["UVIX"]["status"], expected)

    def test_optional_diagnostics_do_not_use_future_of_vx(self):
        etf = frame(30, 3)
        etf.index += pd.Timedelta(minutes=10)
        r = m.evaluate({"F1": frame(21), "F2": frame(24), "UVIX": etf}, now=NOW)
        q = r["feed_quality"]["UVIX"]
        self.assertLessEqual(m.utc(q["used_asof"]), m.utc(r["asof"]))
        self.assertGreater(m.utc(q["latest_closed_at"]), m.utc(q["used_asof"]))

    def test_feed_quality_saved_json_and_csv(self):
        r = m.evaluate({"F1": frame(21), "F2": frame(24)}, now=NOW)
        emit, memory, message = m.alert_update(r, {}, "TEST")
        with TemporaryDirectory() as folder:
            m.save_outputs(r, memory, Path(folder), emit, message)
            packet = json.loads((Path(folder)/"vix_monitor_latest.json").read_text())
            self.assertEqual(packet["feed_quality"]["UVIX"]["status"], "MISSING")
            quality = pd.read_csv(Path(folder)/"vix_monitor_feed_quality.csv")
            self.assertEqual(len(quality), 6)
            m.save_outputs(r, memory, Path(folder), emit, message)
            self.assertEqual(len(pd.read_csv(Path(folder)/"vix_monitor_feed_quality.csv")), 12)

    def test_console_summary_even_without_new_alert(self):
        r = m.evaluate({"F1": frame(21, 2), "F2": frame(24, .2)}, now=NOW)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            m.show(r, False, "Cooldown", {"confirmation_count": 2})
        text = output.getvalue()
        self.assertIn("ÖSSZESÍTETT VIX STRESSZ STÁTUSZ", text)
        self.assertIn("MEGERŐSÍTVE", text)
        self.assertIn("Új alert ezen futásban: NEM", text)

    def test_summary_respects_custom_confirmations(self):
        status, _ = m.overall_status(result(), {"confirmation_count": 2},
                                     m.Config(confirmations=3))
        self.assertIn("MEGERŐSÍTÉSRE VÁR", status)

    def test_missing_data_console_diagnostics(self):
        r = m.evaluate({}, now=NOW)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            m.show(r, True, "FEED", {"confirmation_count": 0})
        self.assertIn("NEM ÉRTÉKELHETŐ", output.getvalue())
        self.assertIn("Core F1/F2 data missing", output.getvalue())

    def test_no_arguments_runs_once_and_prints_summary(self):
        r = m.evaluate({"F1": frame(21), "F2": frame(24)}, now=NOW)
        with patch.object(m, "tws_cycle", return_value=(r, "TEST")) as cycle, \
                patch.object(m, "save_outputs"), \
                patch.object(Path, "mkdir"), \
                patch.object(Path, "read_text", side_effect=FileNotFoundError), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(m.main([]), 0)
        cycle.assert_called_once()
        self.assertIn("EGYSZERI FUTÁS", output.getvalue())
        self.assertIn("Összesített státusz:", output.getvalue())


def long_frame(base, rise=0.0, *, wall=NOW):
    idx = pd.date_range(end=wall-pd.Timedelta(minutes=15), periods=601, freq="min")
    # A longer-horizon move followed by more than an hour of flat prices.
    prices = [base+rise*min(max(i-360, 0), 175)/175 for i in range(601)]
    return pd.DataFrame(dict(open=prices, high=[p+.01 for p in prices],
                             low=[p-.01 for p in prices], close=prices, volume=100), index=idx)


class MultiTimeframeTests(unittest.TestCase):
    def evaluate(self, frames=None, wall=NOW):
        return m.evaluate(frames or {"F1": long_frame(20), "F2": long_frame(30)}, now=wall)

    def test_complete_three_horizon_normal_context(self):
        r = self.evaluate()
        self.assertEqual(r["mtf_available_timeframes"], 3)
        self.assertEqual(r["mtf_context_state"], "MTF_NO_TRIGGER")
        for name, minutes in (("15m", 15), ("1h", 60), ("4h", 240)):
            tf = r["timeframes"][name]
            self.assertEqual(tf["quality"], "OK")
            self.assertEqual(tf["state"], "NORMAL")
            self.assertEqual(tf["actual_elapsed_minutes"], minutes)
            self.assertEqual(tf["coverage"], 1)

    def test_longer_stress_not_overwritten_by_quiet_short_window(self):
        r = self.evaluate({"F1": long_frame(20, 4), "F2": long_frame(30)})
        self.assertEqual(r["candidate_state"], "NORMAL")
        self.assertEqual(r["timeframes"]["15m"]["state"], "NORMAL")
        self.assertEqual(r["timeframes"]["1h"]["state"], "NORMAL")
        self.assertEqual(r["timeframes"]["4h"]["state"], "STRESS_BUILDING")
        self.assertEqual(r["mtf_context_state"], "HTF_STRESS_SHORT_TERM_COOLING")
        self.assertAlmostEqual(r["metrics"]["f1_change_4h_pct"], 20)
        self.assertFalse(r["short_covering_confirmed"])

    def test_absent_four_hour_reference_not_normal(self):
        r = self.evaluate({"F1": frame(21), "F2": frame(24)})
        self.assertEqual(r["timeframes"]["4h"]["quality"], "MISSING_REFERENCE")
        self.assertEqual(r["timeframes"]["4h"]["state"], "DATA_UNAVAILABLE")
        self.assertEqual(r["mtf_context_state"], "MTF_PARTIAL_DATA")
        self.assertEqual(r["mtf_available_timeframes"], 2)

    def test_one_hour_reference_hole_does_not_destroy_four_hour_context(self):
        f1, f2 = long_frame(20), long_frame(30)
        end = m.closed_minutes(f1).index[-1]
        remove = pd.date_range(end-pd.Timedelta(minutes=62), end-pd.Timedelta(minutes=60), freq="min")
        f2 = f2.loc[~(f2.index+pd.Timedelta(minutes=1)).isin(remove)]
        r = self.evaluate({"F1": f1, "F2": f2})
        self.assertEqual(r["timeframes"]["1h"]["quality"], "MISSING_REFERENCE")
        self.assertEqual(r["timeframes"]["4h"]["quality"], "OK")

    def test_large_internal_hole_not_hidden_by_available_endpoints(self):
        f1, f2 = long_frame(20), long_frame(30)
        f2 = f2.drop(f2.index[-150:-130])
        r = self.evaluate({"F1": f1, "F2": f2})
        tf = r["timeframes"]["4h"]
        self.assertEqual(tf["quality"], "PARTIAL_WINDOW")
        self.assertEqual(tf["state"], "DATA_UNAVAILABLE")
        self.assertGreater(tf["max_gap_minutes"], 5)
        self.assertEqual(r["timeframes"]["1h"]["quality"], "OK")

    def test_missing_legacy_thirty_minute_reference_keeps_htf_diagnostic(self):
        f1, f2 = long_frame(20, 4), long_frame(30)
        end = m.closed_minutes(f1).index[-1]
        remove = pd.date_range(end-pd.Timedelta(minutes=32), end-pd.Timedelta(minutes=30), freq="min")
        f2 = f2.loc[~(f2.index+pd.Timedelta(minutes=1)).isin(remove)]
        r = self.evaluate({"F1": f1, "F2": f2})
        self.assertEqual(r["candidate_state"], "DATA_UNAVAILABLE")
        self.assertEqual(r["timeframes"]["4h"]["state"], "STRESS_BUILDING")

    def test_optional_four_hour_before_rth_is_unavailable(self):
        r = self.evaluate({"F1": long_frame(20), "F2": long_frame(30), "UVIX": long_frame(30)})
        feed = r["timeframes"]["4h"]["optional_feeds"]["UVIX"]
        self.assertEqual(feed["status"], "INSUFFICIENT_SAME_RTH_SESSION")
        self.assertIsNone(feed["return_pct"])

    def test_optional_four_hour_afternoon_and_own_lagged_endpoint(self):
        wall = NOW.replace(hour=19)
        etf = long_frame(30, 2, wall=wall)
        etf.index -= pd.Timedelta(minutes=5)
        r = self.evaluate({"F1": long_frame(20, wall=wall),
                           "F2": long_frame(30, wall=wall), "UVIX": etf}, wall)
        feed = r["timeframes"]["4h"]["optional_feeds"]["UVIX"]
        self.assertEqual(feed["status"], "OK_LAGGED")
        self.assertIsNotNone(feed["return_pct"])
        self.assertEqual((m.utc(r["asof"])-m.utc(feed["asof"])).total_seconds()/60, 5)

    def test_optional_future_prices_never_leak_to_mtf(self):
        f1, f2, etf = long_frame(20), long_frame(30), long_frame(30, 2)
        etf.index += pd.Timedelta(minutes=10)
        r = self.evaluate({"F1": f1, "F2": f2, "UVIX": etf})
        for tf in r["timeframes"].values():
            feed = tf["optional_feeds"]["UVIX"]
            self.assertLessEqual(m.utc(feed["asof"]), m.utc(r["asof"]))

    def test_mtf_latest_forming_bar_cannot_change_any_window(self):
        f1, f2 = long_frame(20, 4), long_frame(30)
        before = self.evaluate({"F1": f1, "F2": f2})
        f1.iloc[-1] = [60, 70, 50, 60, 1000000]
        after = self.evaluate({"F1": f1, "F2": f2})
        self.assertEqual(before["timeframes"], after["timeframes"])

    def test_stale_core_does_not_make_htf_classification(self):
        r = self.evaluate({"F1": long_frame(20, 4), "F2": long_frame(30)},
                          NOW+pd.Timedelta(hours=2))
        self.assertEqual(r["candidate_state"], "DATA_UNAVAILABLE")
        self.assertEqual(r["mtf_context_state"], "MTF_DATA_UNAVAILABLE")
        self.assertTrue(all(tf["state"] == "DATA_UNAVAILABLE" for tf in r["timeframes"].values()))

    def test_mtf_alert_requires_advancing_source_and_keeps_core_normal(self):
        r = self.evaluate({"F1": long_frame(20, 4), "F2": long_frame(30)})
        emit, memory, _ = m.alert_update(r, {}, "PAIR")
        self.assertFalse(emit)
        emit, memory, _ = m.alert_update(r, memory, "PAIR")
        self.assertFalse(emit)
        self.assertEqual(memory["timeframe_alerts"]["4h"]["confirmation_count"], 1)
        wall = NOW+pd.Timedelta(minutes=5)
        next_r = self.evaluate({"F1": long_frame(20, 4, wall=wall),
                                "F2": long_frame(30, wall=wall)}, wall)
        emit, memory, label = m.alert_update(next_r, memory, "PAIR")
        self.assertTrue(emit)
        self.assertIn("MTF KONTEXTUS ALERT: 4h", label)
        self.assertEqual(memory["confirmation_count"], 0)
        self.assertEqual(next_r["candidate_state"], "NORMAL")
        status, _ = m.overall_status(next_r, memory)
        self.assertIn("HTF STRESSZ", status)
        self.assertIn("MEGERŐSÍTVE", status)
        with TemporaryDirectory() as folder:
            m.save_outputs(next_r, memory, Path(folder), emit, label)
            saved = json.loads((Path(folder)/"vix_monitor_latest.json").read_text())
            self.assertEqual(saved["alert_scope"], "MTF_CONTEXT")
            self.assertFalse(saved["legacy_alert_emitted"])

    def test_mtf_cooldown_and_roll_reset(self):
        memory = {}
        for offset in (0, 5, 10):
            wall = NOW+pd.Timedelta(minutes=offset)
            r = self.evaluate({"F1": long_frame(20, 4, wall=wall),
                               "F2": long_frame(30, wall=wall)}, wall)
            emit, memory, _ = m.alert_update(r, memory, "PAIR")
        self.assertFalse(emit)  # 10m observation is in cooldown
        emit, memory, _ = m.alert_update(r, memory, "NEW_PAIR")
        self.assertFalse(emit)
        self.assertEqual(memory["timeframe_alerts"]["4h"]["confirmation_count"], 1)

    def test_threshold_edit_resets_only_affected_mtf_confirmation(self):
        r = self.evaluate({"F1": long_frame(20, 4), "F2": long_frame(30)})
        _, memory, _ = m.alert_update(r, {}, "PAIR")
        wall = NOW+pd.Timedelta(minutes=5)
        cfg = m.Config(mtf_f1_rise_4h_pct=9)
        next_r = m.evaluate({"F1": long_frame(20, 4, wall=wall),
                             "F2": long_frame(30, wall=wall)}, cfg, now=wall)
        emit, memory, _ = m.alert_update(next_r, memory, "PAIR", cfg)
        self.assertFalse(emit)
        self.assertEqual(memory["timeframe_alerts"]["4h"]["confirmation_count"], 1)

    def test_mtf_export_new_csv_without_changing_legacy_header(self):
        r = self.evaluate()
        emit, memory, label = m.alert_update(r, {}, "PAIR")
        with TemporaryDirectory() as folder:
            m.save_outputs(r, memory, Path(folder), emit, label)
            packet = json.loads((Path(folder)/"vix_monitor_latest.json").read_text())
            self.assertEqual(set(packet["timeframes"]), {"15m", "1h", "4h"})
            self.assertEqual(packet["mtf_confirmation_counts"]["4h"], 0)
            df = pd.read_csv(Path(folder)/"vix_monitor_mtf_observations.csv")
            self.assertEqual(len(df), 3)
            self.assertEqual(set(df.timeframe), {"15m", "1h", "4h"})
            legacy = pd.read_csv(Path(folder)/"vix_monitor_observations.csv")
            self.assertNotIn("timeframe", legacy.columns)
            m.save_outputs(r, memory, Path(folder), emit, label)
            self.assertEqual(len(pd.read_csv(Path(folder)/"vix_monitor_mtf_observations.csv")), 6)

    def test_hourly_volume_baseline_requires_hourly_history(self):
        data = m.closed_minutes(long_frame(20))
        past = []
        for days in range(1, 6):
            previous = data.copy()
            previous.index -= pd.Timedelta(days=days)
            previous.volume = 50
            past.append(previous)
        combined = pd.concat([*past, data]).sort_index()
        self.assertAlmostEqual(m.time_of_day_volume_ratio(combined, data.index[-1], m.Config(), 60), 2)
        self.assertAlmostEqual(m.time_of_day_volume_ratio(combined, data.index[-1], m.Config(), 240), 2)

    def test_console_has_all_horizons_and_distinct_context(self):
        r = self.evaluate({"F1": long_frame(20, 4), "F2": long_frame(30)})
        emit, memory, label = m.alert_update(r, {}, "PAIR")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            m.show(r, emit, label, memory)
        text = out.getvalue()
        self.assertIn("GÖRDÜLŐ ABLAKOK", text)
        self.assertIn("HTF_STRESS_SHORT_TERM_COOLING", text)
        self.assertIn("4h", text)
        self.assertIn("1/2", text)

    def test_mtf_config_invalid_coverage_rejected(self):
        for kwargs in (dict(mtf_min_coverage=0), dict(mtf_max_gap_minutes=0),
                       dict(mtf_f1_rise_4h_pct=float("nan"))):
            with self.assertRaises(ValueError):
                m.Config(**kwargs)


if __name__ == "__main__":
    unittest.main()
