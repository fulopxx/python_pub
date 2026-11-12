"""GC audit resolution is parent-side sampling, not engine simplification."""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

import gc_audit_backtest_v1_3_11 as previous
import gc_audit_backtest_v1_3_12 as audit
from test_gc_audit_conclusions import fixture


def mock_observation(frame, index):
    return dict(row_index=index, timestamp=frame.index[index],
                regime="BULLTREND", direction=1, trade_allowed=True)


class ResolutionTests(unittest.TestCase):
    def test_profiles(self):
        for name, step in (("quick", 60), ("standard", 15), ("detailed", 5), ("full", 1)):
            self.assertEqual(audit._resolve_resolution(name)["step_bars"], step)

    def test_default_is_previous_resolution(self):
        self.assertEqual(audit._resolve_resolution()["step_bars"],
                         previous.EVALUATION_STEP_BARS)

    def test_custom_step_wins_and_is_marked_custom(self):
        result = audit._resolve_resolution("quick", 10)
        self.assertEqual(result, dict(requested_profile="quick", profile="custom", step_bars=10))

    def test_invalid_profile_and_step(self):
        for value in (0, -1, 1.5, True, "5"):
            with self.subTest(step=value), self.assertRaises(ValueError):
                audit._resolve_resolution("standard", value)
        with self.assertRaises(ValueError):
            audit._resolve_resolution("unknown")

    def test_grid_exactly_matches_previous_default(self):
        for n in (12061, 13000, 48846):
            start = previous.SNAPSHOT_LOOKBACK_BARS - 1
            start += -start % previous.EVALUATION_STEP_BARS
            expected = list(range(start, n-max(previous.FORWARD_HORIZONS_MINUTES),
                                  previous.EVALUATION_STEP_BARS))
            self.assertEqual(audit._evaluation_grid(n, 15), expected)

    def test_grids_are_nested_on_same_input(self):
        grids = [set(audit._evaluation_grid(48846, s)) for s in (60, 15, 5, 1)]
        for sparse, dense in zip(grids, grids[1:]):
            self.assertTrue(sparse < dense)

    def test_original_log_counts(self):
        expected = {60: 614, 15: 2453, 5: 7358, 1: 36787}
        for step, count in expected.items():
            self.assertEqual(len(audit._evaluation_grid(48846, step)), count)

    def test_no_forward_or_lookback_truncation(self):
        for step in (60, 15, 5, 1, 7):
            grid = audit._evaluation_grid(48846, step)
            self.assertGreaterEqual(min(grid), 11999)
            self.assertLess(max(grid), 48846-60)
            self.assertTrue(all(i % step == 0 for i in grid))

    def test_insufficient_history(self):
        with self.assertRaises(ValueError):
            audit._evaluation_grid(12000, 15)

    def test_standard_audit_frames_identical_to_previous(self):
        df = pd.DataFrame(index=pd.date_range("2026-09-01", periods=12500, freq="min", tz="UTC"))
        with patch.object(previous, "_evaluate_single_observation", mock_observation):
            with redirect_stdout(io.StringIO()):
                old_obs, old_ev = previous.run_audit(df, worker_override=1)
        with patch.object(audit, "_evaluate_single_observation", mock_observation):
            with redirect_stdout(io.StringIO()):
                obs, ev = audit.run_audit(df, worker_override=1, step_override=15)
        pd.testing.assert_frame_equal(old_obs, obs)
        pd.testing.assert_frame_equal(old_ev, ev)

    def test_step_is_used_without_global_mutation(self):
        df = pd.DataFrame(index=pd.date_range("2026-09-01", periods=12500, freq="min", tz="UTC"))
        before = audit.EVALUATION_STEP_BARS
        with patch.object(audit, "_evaluate_single_observation", mock_observation):
            with redirect_stdout(io.StringIO()):
                obs, _ = audit.run_audit(df, worker_override=1, step_override=60)
        self.assertEqual(obs.row_index.tolist(), audit._evaluation_grid(len(df), 60))
        self.assertEqual(audit.EVALUATION_STEP_BARS, before)

    def test_worker_and_engine_calculation_functions_unchanged(self):
        root = Path(__file__).parent
        old = ast.parse((root/"gc_audit_backtest_v1_3_11.py").read_text(encoding="utf-8"))
        new = ast.parse((root/"gc_audit_backtest_v1_3_12.py").read_text(encoding="utf-8"))
        for name in ("_init_worker", "_evaluate_single_observation",
                     "_evaluate_observation_task", "_observation_result_iterator",
                     "record_forward_metrics", "build_observation_record"):
            left = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == name)
            right = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(left), ast.dump(right), name)


class CLITests(unittest.TestCase):
    def test_new_flags_parse(self):
        with patch("sys.argv", ["audit", "--profile", "detailed", "--step-bars", "3",
                                "--workers", "8", "--dry-run"]):
            args = audit._report_arguments()
        self.assertEqual(args.profile, "detailed")
        self.assertEqual(args.step_bars, 3)
        self.assertTrue(args.dry_run)

    def test_bad_args_fail_before_data_loading(self):
        for flags in (["--step-bars", "0"], ["--step-bars", "-3"],
                      ["--workers", "-1"], ["--profile", "bad"],
                      ["--report-only", "--profile", "quick"],
                      ["--report-only", "--step-bars", "5"],
                      ["--report-only", "--dry-run"]):
            with self.subTest(flags=flags), patch("sys.argv", ["audit"]+flags):
                with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                    audit._report_arguments()
                self.assertEqual(error.exception.code, 2)

    def test_dry_run_no_snapshot_no_output(self):
        df = pd.DataFrame(index=pd.date_range("2026-09-01", periods=12500, freq="min", tz="UTC"))
        with TemporaryDirectory() as temp:
            target = Path(temp)/"must_not_be_created"
            args = ["audit", "--profile", "full", "--dry-run", "--output-dir", str(target)]
            with patch("sys.argv", args), patch.object(audit, "load_gc_history", return_value=df):
                with patch.object(audit, "run_audit") as run, redirect_stdout(io.StringIO()) as console:
                    audit.main()
                run.assert_not_called()
            self.assertFalse(target.exists())
            self.assertIn("DRY-RUN VÉGE", console.getvalue())
            self.assertIn("full", console.getvalue())
            # A full grid can have one valid point even if coarser grids have none.
            with patch("sys.argv", args), patch.object(
                audit, "load_gc_history", return_value=df.iloc[:12060]
            ):
                with patch.object(audit, "run_audit") as run, redirect_stdout(io.StringIO()):
                    audit.main()
                run.assert_not_called()

    def test_normal_run_metadata_and_verified_report_only(self):
        obs, ev = fixture()
        df = pd.DataFrame(index=pd.date_range("2026-09-01", periods=12500, freq="min", tz="UTC"))
        with TemporaryDirectory() as temp:
            root = Path(temp)
            out = root/"quick"
            args = ["audit", "--profile", "quick", "--workers", "1", "--output-dir", str(out)]
            with patch("sys.argv", args), patch.object(audit, "load_gc_history", return_value=df):
                with patch.object(audit, "run_audit", return_value=(obs, ev)) as run:
                    with redirect_stdout(io.StringIO()):
                        audit.main()
                run.assert_called_once_with(df, worker_override=1, step_override=60)
            context = json.loads((out/"gc_audit_run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(context["profile"], "quick")
            self.assertEqual(context["step_bars"], 60)
            self.assertEqual(context["lookback_bars"], 12000)
            self.assertEqual(context["planned_observations"], len(audit._evaluation_grid(12500, 60)))
            source_obs, source_ev = out/"gc_audit_observations.csv", out/"gc_audit_events.csv"
            self.assertIsNotNone(audit._source_run_context(source_obs, source_ev))
            csv = pd.read_csv(out/"gc_audit_conclusions.csv")
            self.assertTrue(csv.source_audit_profile.eq("quick").all())
            self.assertTrue(csv.source_audit_step_bars.eq(60).all())
            before = source_obs.read_bytes(), source_ev.read_bytes()
            report_root = root/"report_only"
            args = ["audit", "--report-only", "--observations", str(source_obs),
                    "--events", str(source_ev), "--output-dir", str(report_root)]
            with patch("sys.argv", args), patch.object(audit, "run_audit") as run:
                with redirect_stdout(io.StringIO()):
                    audit.main()
                run.assert_not_called()
            verified = pd.read_csv(report_root/"gc_audit_conclusions.csv")
            self.assertTrue(verified.source_audit_profile.eq("quick").all())
            self.assertEqual(before, (source_obs.read_bytes(), source_ev.read_bytes()))
            source_obs.write_bytes(source_obs.read_bytes()+b"\n")
            self.assertIsNone(audit._source_run_context(source_obs, source_ev))

    def test_legacy_csv_without_sidecar_remains_unverified(self):
        with TemporaryDirectory() as temp:
            op, ep = Path(temp)/"obs.csv", Path(temp)/"events.csv"
            op.write_text("x\n1\n")
            ep.write_text("x\n1\n")
            self.assertIsNone(audit._source_run_context(op, ep))

    def test_damaged_sidecar_does_not_claim_verified_profile(self):
        with TemporaryDirectory() as temp:
            op, ep = Path(temp)/"obs.csv", Path(temp)/"events.csv"
            op.write_text("x\n1\n")
            ep.write_text("x\n1\n")
            path = Path(temp)/"gc_audit_run_config.json"
            path.write_text("{broken json")
            self.assertIsNone(audit._source_run_context(op, ep))
            path.write_text("[]")
            self.assertIsNone(audit._source_run_context(op, ep))


if __name__ == "__main__":
    unittest.main()
