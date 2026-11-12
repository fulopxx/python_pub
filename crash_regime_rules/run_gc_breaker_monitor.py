"""Standalone GC 4h breaker replay from existing CSV. No TWS or orders."""
from __future__ import annotations

import argparse
from pathlib import Path

from gc_breaker_monitor import (
    demo_history, evaluate_gc_breaker_monitor, export_gc_breaker_monitor, format_gc_breaker_monitor,
)
from run_market_exhaustion import load_history

BASE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Automatic GC 4h breaker monitor, paper CSV replay.")
    parser.add_argument("csv_path", nargs="?", default=str(BASE/"gc_1m_history.csv"))
    parser.add_argument("--output-dir", type=Path, default=BASE/"gc_breaker_output")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args(argv)
    try:
        path = Path(args.csv_path).expanduser()
        if not path.is_absolute():
            path = BASE/path
        data = demo_history() if args.demo else load_history(path)
        result = evaluate_gc_breaker_monitor(
            data, data_context={"mode": "SYNTHETIC" if args.demo else "HISTORICAL",
                                "last_bar_is_complete": bool(args.demo)})
        print("\n".join(format_gc_breaker_monitor(result)))
        folder = args.output_dir/"demo" if args.demo else args.output_dir
        export_gc_breaker_monitor(result, folder)
        print(f"Export: {folder.resolve()}")
        print("CSV/szintetikus elemzés; nincs TWS-letöltés, order, exercise vagy pozíciókezelés.")
        return 0
    except Exception as error:
        parser.exit(1, f"GC breaker hiba: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
