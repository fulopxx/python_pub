from __future__ import annotations

from typing import Any

import pandas as pd

from index_regime import (
    CONFIG,
    NQ_CONFIG,
    IndexRegime,
    IndexRegimeConfig,
    build_index_snapshot,
    calculate_index_risk_plan,
    index_regime_active,
    run_index_checklist,
)


# ===================================================================
# NQ-SPECIFIKUS WRAPPER
# ===================================================================

def build_nq_snapshot(
    df_nq_1m: pd.DataFrame,
    vix_context: dict[str, float] | None = None,
    config: IndexRegimeConfig = CONFIG,
) -> dict[str, Any]:
    """
    NQ 1 perces OHLCV adatból market snapshot.

    A VIX context opcionális:
    - használható széles piaci stresszjelzőnek;
    - de NQ-nál nem NQ-specifikus volatility input.
    """
    return build_index_snapshot(
        df_1m=df_nq_1m,
        instrument=NQ_CONFIG,
        vix_context=vix_context,
        config=config,
    )


def get_nq_regime(
    df_nq_1m: pd.DataFrame,
    vix_context: dict[str, float] | None = None,
    config: IndexRegimeConfig = CONFIG,
) -> dict[str, Any]:
    """
    NQ snapshot + NQ rezhimdetektor eredmény.

    Visszaad:
    {
        "snapshot": ...,
        "regime_result": ...,
        "regime": ...
    }
    """
    snapshot = build_nq_snapshot(
        df_nq_1m=df_nq_1m,
        vix_context=vix_context,
        config=config,
    )

    regime_result = index_regime_active(
        snapshot=snapshot,
        config=config,
    )

    return {
        "snapshot": snapshot,
        "regime_result": regime_result,
        "regime": regime_result["regime"],
    }


def run_nq_checklist(
    df_nq_1m: pd.DataFrame,
    equity_usd: float = 10_000.0,
    vix_context: dict[str, float] | None = None,
    config: IndexRegimeConfig = CONFIG,
) -> str:
    """
    Teljes NQ checklist report.

    Ez ugyanazt a közös index-regime motort futtatja,
    de automatikusan NQ_CONFIG konfigurációval.
    """
    return run_index_checklist(
        df_1m=df_nq_1m,
        instrument=NQ_CONFIG,
        equity_usd=equity_usd,
        vix_context=vix_context,
        config=config,
    )


def get_nq_risk_plan(
    df_nq_1m: pd.DataFrame,
    equity_usd: float = 10_000.0,
    vix_context: dict[str, float] | None = None,
    config: IndexRegimeConfig = CONFIG,
) -> dict[str, Any]:
    """
    NQ/MNQ elméleti kockázati terv.

    Nem küld ordert.
    Nem nyit pozíciót.
    """
    nq_analysis = get_nq_regime(
        df_nq_1m=df_nq_1m,
        vix_context=vix_context,
        config=config,
    )

    snapshot = nq_analysis["snapshot"]
    regime = nq_analysis["regime"]

    return calculate_index_risk_plan(
        snapshot=snapshot,
        regime=regime,
        equity_usd=equity_usd,
        config=config,
    )


def nq_regime_summary(
    df_nq_1m: pd.DataFrame,
    vix_context: dict[str, float] | None = None,
    config: IndexRegimeConfig = CONFIG,
) -> str:
    """
    Rövid, programból is könnyen használható NQ összefoglaló.

    Például később Discord/Telegram/dashboard alerthez használható.
    """
    nq_analysis = get_nq_regime(
        df_nq_1m=df_nq_1m,
        vix_context=vix_context,
        config=config,
    )

    snapshot = nq_analysis["snapshot"]
    regime = nq_analysis["regime"]
    regime_result = nq_analysis["regime_result"]

    return (
        f"NQ={snapshot['close']:.2f} | "
        f"VWAP={snapshot['vwap']:.2f} | "
        f"RSI2={snapshot['rsi_2']:.1f} | "
        f"RSI5={snapshot['rsi_5']:.1f} | "
        f"ADX5m={snapshot['adx_5m']:.1f} | "
        f"DI={snapshot['plus_di_5m']:.1f}/"
        f"{snapshot['minus_di_5m']:.1f} | "
        f"Crash={regime_result['crash_score']}/"
        f"{len(regime_result['crash_checks'])} | "
        f"Regime={regime.value}"
    )


# ===================================================================
# FUTTATÁSI INFORMÁCIÓ
# ===================================================================

if __name__ == "__main__":
    print(
        "\n"
        "Az nq_regime.py NQ-specifikus wrapper sikeresen betöltődött.\n"
        "\n"
        "Ez a fájl nem tölt le TWS adatot és nem futtat önálló elemzést.\n"
        "\n"
        "Használat más scriptből:\n"
        "\n"
        "from nq_regime import run_nq_checklist\n"
        "\n"
        "report = run_nq_checklist(\n"
        "    df_nq_1m=df_nq,\n"
        "    equity_usd=10_000,\n"
        "    vix_context=vix_context,\n"
        ")\n"
        "\n"
        "print(report)\n"
    )