from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_exhaustion import evaluate_exhaustion, format_exhaustion
from market_transition import evaluate_transition_risk, format_transition_risk


# ===================================================================
# RUN MARKET EXHAUSTION VERSION 1.1.0
#
# Argumentum nélkül:
#   GC + a script melletti gc_1m_history.csv
#
# Példák:
#   python run_market_exhaustion.py
#   python run_market_exhaustion.py GC gc_1m_history.csv
#   python run_market_exhaustion.py ES es_1m_history.csv
#   python run_market_exhaustion.py NQ nq_1m_history.csv
#
# Nem kapcsolódik TWS-hez, nem tölt le adatot, nem küld ordert.
# ===================================================================

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_SYMBOL = "GC"
DEFAULT_CSV_FILES = {
    "GC": "gc_1m_history.csv",
    "ES": "es_1m_history.csv",
    "NQ": "nq_1m_history.csv",
}
REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GC / ES / NQ exhaustion checklist meglévő 1 perces CSV-ből. "
            "Alapértelmezés: GC és a script melletti gc_1m_history.csv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default=DEFAULT_SYMBOL,
        choices=["GC", "ES", "NQ"],
        type=str.upper,
        help="Elemzendő termék.",
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help=(
            "1 perces OHLCV CSV. Ha nincs megadva, a termékhez tartozó "
            "alapértelmezett fájl. Relatív útvonal a script könyvtárához képest."
        ),
    )
    return parser


def resolve_csv_path(symbol: str, csv_path: str | None) -> Path:
    path = Path(csv_path or DEFAULT_CSV_FILES[symbol]).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIRECTORY / path
    return path.resolve()


def load_history(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Nem található a CSV: {csv_path}\n"
            "Másold a megfelelő történeti CSV-t a runner mellé, "
            "vagy adj meg teljes fájlútvonalat."
        )

    df = pd.read_csv(csv_path)
    df.columns = [str(column).strip().lower() for column in df.columns]
    if df.columns.has_duplicates:
        raise ValueError("Duplikált oszlopnév a CSV-ben.")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"Hiányzó CSV-oszlopok: {missing}. "
            f"Elvárt: {REQUIRED_COLUMNS}"
        )
    if df.empty:
        raise ValueError("A CSV üres.")

    # Az audit CSV timezone-aware időbélyegeit UTC-re konvertáljuk.
    # Timezone-naive időbélyeget UTC-ként kezelünk.
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise")
    if df["date"].isna().any():
        raise ValueError("Hiányzó időbélyeg a CSV-ben.")
    if df["date"].duplicated().any():
        raise ValueError(
            "Duplikált időbélyeg a CSV-ben. A történetet előbb "
            "egyértelműen deduplikálni kell."
        )

    df = df.set_index("date")[REQUIRED_COLUMNS[1:]].sort_index()
    for column in REQUIRED_COLUMNS[1:]:
        df[column] = pd.to_numeric(df[column], errors="raise")
    return df


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    csv_path = resolve_csv_path(args.symbol, args.csv_path)

    print("=" * 100)
    print("MARKET EXHAUSTION CSV RUNNER 1.1.0")
    print("=" * 100)
    print(f"Termék:           {args.symbol}")
    print(f"CSV:              {csv_path}")
    print("Adatmód:          HISTORICAL / meglévő CSV")

    try:
        df = load_history(csv_path)
        print(f"Betöltött barok:   {len(df):,}")
        print(f"Első időbélyeg:    {df.index.min()}")
        print(f"Utolsó időbélyeg:  {df.index.max()}")

        result = evaluate_exhaustion(
            df,
            args.symbol,
            data_context={"mode": "HISTORICAL"},
        )
        print("\n".join(format_exhaustion(result)))
        if args.symbol == "GC":
            transition = evaluate_transition_risk(
                df, data_context={"mode": "HISTORICAL"},
            )
            print("\n".join(format_transition_risk(transition)))
        print(
            "\nEz a CSV utolsó időpontjához tartozó történeti elemzés, "
            "nem valós idejű piaci állapot."
        )
        print(
            "A legutolsó 1m bar alapértelmezetten kimarad, "
            "mert még részleges lehet."
        )
        print("TWS-kapcsolat, adatletöltés és orderküldés nem történt.")
        return 0

    except Exception as error:
        print("\n" + "=" * 100)
        print("MARKET EXHAUSTION FUTTATÁSI HIBA")
        print(f"Típus:  {type(error).__name__}")
        print(f"Üzenet: {error}")
        print("=" * 100)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
