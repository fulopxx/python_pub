"""Regression tests for descriptive, non-causal filter conclusions."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gc_audit_conclusions import (
    ConclusionConfig, _bool, _difference_ci, _sample,
    build_conclusions, render_report, save_conclusion_outputs,
)

CFG = ConclusionConfig(bootstrap_samples=200)


def fixture(good=0.6, bad=-0.4, days=12, per_day=4):
    records = []
    for day in pd.date_range("2026-09-01 12:00", periods=days, freq="D", tz="UTC"):
        for passing in (True, False):
            for i in range(per_day):
                value = good if passing else bad
                row = dict(
                    timestamp=day + pd.Timedelta(minutes=i + (0 if passing else 30)),
                    is_bull_regime=True, is_bear_regime=False, direction=1,
                    regime="BULLTREND", trade_allowed=passing,
                    adx_5m=40 if passing else 20,
                    volume_ratio=2.0 if passing else 0.5,
                    rsi_2=20 if passing else 5,
                    vwap_distance_atr=0 if passing else 6,
                    ndog_reaction="NORECENTINTERACTION",
                    nwog_reaction="NORECENTINTERACTION",
                    exhaustion_block_new_longs=not passing,
                    transition_block_new_longs=not passing,
                    transition_block_new_shorts=not passing,
                    whipsaw_state="LOW" if passing else "HIGH",
                    event_type="TRADE_CANDIDATE" if passing else "BLOCKED_TRADE",
                )
                for h in (5, 15, 30, 60):
                    for metric in ("forward_long", "forward_short", "forward_directional"):
                        row[f"{metric}_{h}m_atr"] = value
                    row[f"mae_{h}m_atr"] = -1.2
                    row[f"mfe_{h}m_atr"] = 1.3
                records.append(row)
    frame = pd.DataFrame(records)
    return frame.drop(columns="event_type"), frame


def verdict(summary, identifier):
    return summary.set_index("id").loc[identifier, "conclusion"]


class VerdictTests(unittest.TestCase):
    def test_positive_association_is_helpful_on_all_horizons(self):
        obs, ev = fixture()
        summary, details = build_conclusions(obs, ev, CFG)
        for identifier in ("Q1", "Q6", "Q8", "Q9", "Q12", "Q13L", "Q14"):
            self.assertEqual(verdict(summary, identifier), "HASZNOSNAK TŰNIK")
        row = details[details.id.eq("Q8")].iloc[0]
        self.assertAlmostEqual(row.delta_mean_atr, 1.0)
        self.assertGreater(row.ci_low_atr, 0)
        self.assertEqual(row.a_days, 12)

    def test_negative_association_warns_filter_may_worsen(self):
        obs, ev = fixture(good=-0.4, bad=0.6)
        summary, _ = build_conclusions(obs, ev, CFG)
        self.assertEqual(verdict(summary, "Q8"), "RONTHAT")

    def test_no_difference_not_proven(self):
        obs, ev = fixture(good=0.3, bad=0.3)
        summary, _ = build_conclusions(obs, ev, CFG)
        self.assertEqual(verdict(summary, "Q8"), "NEM IGAZOLT")

    def test_many_observations_few_days_do_not_create_confidence(self):
        obs, ev = fixture(days=3, per_day=50)
        summary, _ = build_conclusions(obs, ev, CFG)
        self.assertEqual(verdict(summary, "Q8"), "KIS MINTA")

    def test_many_days_but_small_group(self):
        obs, ev = fixture(days=12, per_day=1)
        summary, _ = build_conclusions(obs, ev, CFG)
        self.assertEqual(verdict(summary, "Q8"), "KIS MINTA")

    def test_conflicting_horizons_are_mixed(self):
        obs, ev = fixture()
        obs["forward_directional_30m_atr"] *= -1
        summary, _ = build_conclusions(obs, ev, CFG)
        self.assertEqual(verdict(summary, "Q8"), "VEGYES EREDMÉNY")

    def test_missing_horizon_blocks_strong_overall_verdict(self):
        obs, ev = fixture()
        obs = obs.drop(columns="forward_directional_60m_atr")
        summary, _ = build_conclusions(obs, ev, CFG)
        self.assertEqual(verdict(summary, "Q8"), "KORLÁTOZOTT ADAT")

    def test_median_conflict_not_green(self):
        obs, ev = fixture()
        passing = obs.adx_5m.gt(30)
        for h in CFG.horizons:
            obs.loc[passing, f"forward_directional_{h}m_atr"] = np.tile(
                [100., -1., -1., -1.], 12
            )
            obs.loc[~passing, f"forward_directional_{h}m_atr"] = 0.1
        summary, _ = build_conclusions(obs, ev, CFG)
        self.assertEqual(verdict(summary, "Q8"), "VEGYES EREDMÉNY")

    def test_stop_is_not_a_pnl_recommendation(self):
        obs, ev = fixture()
        summary, _ = build_conclusions(obs, ev, CFG)
        row = summary.set_index("id").loc["Q11"]
        self.assertEqual(row.conclusion, "NEM ELDÖNTHETŐ")
        self.assertIn("100.0%", row.comment)
        self.assertIn("célár-sorrend", row.comment)

    def test_permission_rate_is_descriptive(self):
        obs, ev = fixture()
        summary, _ = build_conclusions(obs, ev, CFG)
        self.assertEqual(verdict(summary, "Q2"), "CSAK LEÍRÓ")

    def test_empty_data_has_all_questions_without_fake_values(self):
        summary, details = build_conclusions(pd.DataFrame(), pd.DataFrame(), CFG)
        self.assertEqual(len(summary), 16)
        self.assertEqual(len(summary.id.unique()), 16)
        self.assertEqual(verdict(summary, "Q8"), "NINCS ADAT")
        self.assertTrue(details.delta_mean_atr.isna().all())

    def test_missing_filter_column_not_false_control(self):
        obs, ev = fixture()
        summary, _ = build_conclusions(obs.drop(columns="transition_block_new_longs"), ev, CFG)
        self.assertEqual(verdict(summary, "Q13L"), "NINCS ADAT")

    def test_input_frames_unchanged(self):
        obs, ev = fixture()
        before_obs, before_ev = obs.copy(deep=True), ev.copy(deep=True)
        build_conclusions(obs, ev, CFG)
        pd.testing.assert_frame_equal(obs, before_obs)
        pd.testing.assert_frame_equal(ev, before_ev)

    def test_underscore_and_compact_gap_codes_normalized(self):
        obs, ev = fixture()
        obs.loc[obs.adx_5m.gt(30), "ndog_reaction"] = "BULLISH_RECLAIM"
        summary, details = build_conclusions(obs, ev, CFG)
        self.assertEqual(verdict(summary, "Q10L"), "HASZNOSNAK TŰNIK")
        self.assertEqual(details[details.id.eq("Q10L")].iloc[0].a_n, 48)

    def test_unknown_gap_not_treated_as_no_reaction(self):
        obs, ev = fixture()
        obs.loc[obs.adx_5m.gt(30), "ndog_reaction"] = "BULLISHRECLAIM"
        obs.loc[obs.adx_5m.le(30), "nwog_reaction"] = "N/A"
        summary, _ = build_conclusions(obs, ev, CFG)
        self.assertEqual(verdict(summary, "Q10L"), "NINCS ADAT")

    def test_short_vwap_direction_has_correct_sign(self):
        obs, ev = fixture()
        obs["is_bull_regime"], obs["is_bear_regime"], obs["direction"] = False, True, -1
        obs["vwap_distance_atr"] = np.where(obs.adx_5m.gt(30), -1.0, -6.0)
        summary, _ = build_conclusions(obs, ev, CFG)
        self.assertEqual(verdict(summary, "Q5"), "HASZNOSNAK TŰNIK")


class DataAndExportTests(unittest.TestCase):
    def test_boolean_strings_and_unknown(self):
        values = _bool(pd.DataFrame({"x": ["False", "True", None, "unknown", "0", "1"]}), "x")
        self.assertEqual(values.iloc[0], False)
        self.assertEqual(values.iloc[1], True)
        self.assertTrue(pd.isna(values.iloc[2]))
        self.assertTrue(pd.isna(values.iloc[3]))
        self.assertEqual(values.iloc[4], False)

    def test_nonfinite_and_bad_timestamps_excluded(self):
        frame = pd.DataFrame({
            "timestamp": ["2026-10-01T12:00:00Z", "bad", "2026-10-02T12:00:00Z",
                          "2026-10-03T12:00:00Z"],
            "x": [1., 999., np.inf, np.nan],
        })
        sample = _sample(frame, "x")
        self.assertEqual(sample.value.tolist(), [1.])

    def test_bootstrap_repeatable(self):
        obs, _ = fixture()
        a = _sample(obs[obs.adx_5m.gt(30)], "forward_directional_15m_atr")
        b = _sample(obs[obs.adx_5m.le(30)], "forward_directional_15m_atr")
        self.assertEqual(_difference_ci(a, b, CFG), _difference_ci(a, b, CFG))

    def test_exports_reuse_same_conclusions_and_numeric_evidence(self):
        obs, ev = fixture()
        summary, details = build_conclusions(obs, ev, CFG)
        with TemporaryDirectory() as temp:
            text, paths = save_conclusion_outputs(
                summary, details, CFG, temp, metadata="TEST", preamble="BASIC\n"
            )
            self.assertEqual(paths[2].read_text(encoding="utf-8"), text)
            self.assertIn(text, paths[3].read_text(encoding="utf-8"))
            saved = pd.read_csv(paths[0], encoding="utf-8-sig")
            self.assertEqual(saved.conclusion.tolist(), summary.conclusion.tolist())
            self.assertEqual(len(pd.read_csv(paths[1])), len(details))
            self.assertIn("bootstrap_seed", saved)
            self.assertTrue(text.startswith("BASIC\n"))
            self.assertIn("nem garantált eltelt perceket", text)
            self.assertIn("Semmit nem kapcsol ki/be", text)
            self.assertLess(text.index("RÉSZLETES ELLENŐRZŐ KÉRDÉSEK"),
                            text.index("VÉGSŐ ÖSSZEGZŐ TÁBLÁZAT"))
            self.assertIn("question", saved)

    def test_normal_audit_writes_raw_and_structured_reports(self):
        import gc_audit_backtest_v1_3_11 as audit
        obs, ev = fixture()
        history = pd.DataFrame({"close": [4000., 4001.]},
                               index=pd.date_range("2026-09-01", periods=2,
                                                   freq="min", tz="UTC"))
        with TemporaryDirectory() as temp:
            args = ["audit", "--workers", "1", "--output-dir", temp]
            with patch("sys.argv", args), patch.object(
                audit, "load_gc_history", return_value=history
            ), patch.object(audit, "run_audit", return_value=(obs, ev)) as run:
                with redirect_stdout(io.StringIO()) as console:
                    audit.main()
            run.assert_called_once_with(history, worker_override=1)
            for name in ("gc_audit_observations.csv", "gc_audit_events.csv",
                         "gc_audit_summary.csv", "gc_audit_conclusions.csv",
                         "gc_audit_comparisons.csv", "gc_audit_report.txt",
                         "gc_audit_report.md"):
                self.assertTrue((Path(temp)/name).is_file(), name)
            text = (Path(temp)/"gc_audit_report.txt").read_text(encoding="utf-8")
            self.assertIn(text, console.getvalue())

    def test_report_only_rejects_overwriting_source_with_conclusions(self):
        import argparse
        import gc_audit_backtest_v1_3_11 as audit
        obs, ev = fixture()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            op, ep = root/"gc_audit_conclusions.csv", root/"input_events.csv"
            obs.to_csv(op, index=False)
            ev.to_csv(ep, index=False)
            before = op.read_bytes()
            args = argparse.Namespace(observations=str(op), events=str(ep), output_dir=temp)
            with self.assertRaises(ValueError):
                audit._report_from_csv(args)
            self.assertEqual(before, op.read_bytes())

    def test_cli_report_only_never_calls_snapshot_or_overwrites_input(self):
        import gc_audit_backtest_v1_3_11 as audit
        obs, ev = fixture()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            op, ep = root/"input_obs.csv", root/"input_events.csv"
            obs.to_csv(op, index=False)
            ev.to_csv(ep, index=False)
            before = op.read_bytes(), ep.read_bytes()
            args = ["audit", "--report-only", "--observations", str(op),
                    "--events", str(ep), "--output-dir", str(root/"report")]
            with patch("sys.argv", args), patch.object(audit, "build_gold_snapshot") as snap:
                with redirect_stdout(io.StringIO()):
                    audit.main()
                snap.assert_not_called()
            self.assertEqual(before, (op.read_bytes(), ep.read_bytes()))
            self.assertTrue((root/"report"/"gc_audit_conclusions.csv").is_file())

    def test_audit_sampling_loop_unchanged(self):
        import ast
        root = Path(__file__).parent
        old = ast.parse((root/"gc_audit_backtest_v1_3_10.py").read_text(encoding="utf-8"))
        new = ast.parse((root/"gc_audit_backtest_v1_3_11.py").read_text(encoding="utf-8"))
        for name in ("run_audit", "record_forward_metrics", "build_observation_record",
                     "_observation_result_iterator", "_evaluate_single_observation"):
            left = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == name)
            right = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(left), ast.dump(right), name)


if __name__ == "__main__":
    unittest.main()
