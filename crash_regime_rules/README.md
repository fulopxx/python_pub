# Market Regime Research Toolkit

An explainable, rules-based Python toolkit for market-state research and paper
decision support. The project combines gold and equity-index regime analysis,
market-structure diagnostics, volatility context, options screening and
macro/credit analysis with offline audit tooling.

This is research software, not a proven trading strategy or a production
execution platform. The initial public layout deliberately preserves the
existing `crash_regime_rules` folder and versioned Python filenames.

## Capabilities

| Area | Scope | Representative modules |
|---|---|---|
| Gold regimes | Multi-timeframe context, directional regimes and conditional trade plans | `gc_regime_1_0_19.py`, `gc_daily_context.py` |
| Equity-index regimes | Shared ES/NQ analysis and daily/stress context | `index_regime.py`, `index_daily_context.py`, `index_stress_context.py` |
| Exhaustion and transitions | Exhaustion diagnostics, higher-timeframe transitions and short-term whipsaw context | `market_exhaustion.py`, `market_transition.py` |
| Market structure | FVG/order-block/breaker analysis and a dedicated GC breaker monitor | `market_structure.py`, `gc_breaker_monitor.py` |
| Volatility context | VIX/VX and UVIX/SVIX monitoring with timestamp/data-quality context | `run_vix_stress_monitor_v1_0.py` |
| Options research | QQQ long-put and debit-spread screening, liquidity/Greeks/cost diagnostics and entry gating | `qqq_options_advisor.py` |
| Offline audit | Configurable sampling, worker-based processing, forward outcomes and filter/parameter conclusions | `gc_audit_backtest_v1_3_12.py`, `gc_audit_conclusions.py` |
| Macro and credit | Public-source collection, normalization, provenance, missing-data diagnostics and replay | `macro_ai_credit_monitor.py`, `macro_credit_autodata.py`, `macro_credit_tables.py`, `replay_macro_credit_report.py` |

The options diagnostics can display candidates rejected by the regime gate
without making them eligible for entry. Gap-aware GC trade plans can align
nearby targets with NDOG/NWOG levels without overriding the entry protection.

## Architecture

```text
Broker/public sources or local research inputs
                    |
        Validation and time normalization
                    |
    Regime / structure / context calculations
                    |
       Gates, diagnostics and paper plans
                    |
        Console output and local exports

Offline evaluation:
local history -> repeated snapshots -> forward outcomes -> audit conclusions
```

The implementation emphasizes inspectable intermediate results, explicit missing
data and separation between analytical observations and entry eligibility.
Automated tests cover numerical behavior, diagnostic output, sampling rules and
regressions; coverage differs between modules.

## Quick start: synthetic offline demo

Start in this folder, not the `python_pub` repository root. The example below
uses a new virtual environment so that the existing research environment is not
changed.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe qqq_options_advisor.py --demo --output-dir qqq_options_output
```

The example selects Python 3.11 if it is installed; it is a setup recommendation,
not a claim that every dependency/version combination has been certified.
The initial requirements file lists direct dependencies but is not a lockfile.
Record and validate a known-good environment before publishing reproducibility
claims or changing the existing working environment.

The `--demo` input is synthetic and does not require TWS or an IBKR account.
Generated output is local and excluded from the initial publication policy.
Screenshots or example reports should be reviewed and explicitly approved before
being committed.

## Tests

Run the following selected synthetic/offline tests from this folder:

```powershell
.\.venv\Scripts\python.exe -m unittest test_market_structure test_market_transition test_qqq_options_advisor test_gc_audit_conclusions test_gc_audit_resolution test_gc_v19_gap_targets -v
```

This is a selected subset, not a claim that the entire historical suite is
self-contained. Keep older versioned modules that these tests import.
Some macro regressions depend on local JSON snapshots and reduced source
fixtures. Those files are intentionally withheld from the initial public
baseline until provenance, necessity and redistribution have been reviewed or
they have been replaced with clearly synthetic fixtures.

Full `unittest` discovery can therefore report missing-fixture errors in the
initial source-only publication. There is no CI badge or all-tests-passing claim
until a clean clone has been validated.

## Offline gold audit

Supply your own suitably licensed one-minute OHLCV history. Raw histories are
not distributed with this project.

```powershell
.\.venv\Scripts\python.exe gc_audit_backtest_v1_3_12.py --input-csv gc_1m_history.csv --profile quick --dry-run
.\.venv\Scripts\python.exe gc_audit_backtest_v1_3_12.py --input-csv gc_1m_history.csv --profile quick --workers 4 --output-dir audit_quick
```

| Profile | Evaluation interval |
|---|---|
| `quick` | Every 60 input bars |
| `standard` | Every 15 input bars |
| `detailed` | Every 5 input bars |
| `full` | Every eligible input bar |

`--workers` controls parallel processes; it does not control sampling density.
`--step-bars N` overrides the profile interval. A finer grid evaluates more
observations but does not automatically establish statistical independence.

Entry points pin specific engine versions by filename. At the time of this
publication the current checklist runner `run_gc_checklist_v1_7.py` imports
`gc_regime_1_0_19.py`, while the current audit `gc_audit_backtest_v1_3_12.py`
still pins `gc_regime_1_0_18.py`; rewiring the audit to the newer engine is a
separate reviewed change, not part of this baseline. The equity-index side
(`es_regime.py`, `nq_regime.py`, `index_regime.py`) uses unversioned filenames
instead of per-release copies. Older versioned `gc_regime_*`,
`gc_audit_backtest_v1_3_*` and `run_gc_checklist_*` files are retained because
tests and runners import them, and a few stale version references remain in
comments and console labels. Check the actual imports in the local files before
attributing a result to a particular version.

## Live-source modules

Broker-connected runners require an appropriately configured TWS/IB Gateway
session and the relevant data access. No account credentials, session files,
account identifiers or personal run configurations should be committed.

Macro collection may require locally supplied public-source contact information
and research inputs. Keep values such as `SEC_USER_AGENT` in the local
environment, not in source, screenshots or shared IDE configuration.
The macro workflow is not an out-of-the-box public demo in this first
publication because its local input snapshots are intentionally excluded.

## Data and publication policy

- **Included:** reviewed top-level Python modules and tests, this README,
  dependency metadata and individually approved implementation guides.
- **Excluded:** virtual environments, IDE state, credentials, personal
  configuration, raw histories, provider snapshots, caches, generated reports,
  source archives and unreviewed nested fixtures.
- **Review required:** `.py` files can contain secrets too. An allowed extension
  is not evidence that a file is safe to publish.
- **No full backup promise:** GitHub contains the reviewed source baseline,
  not a complete copy of the local research workstation.

The project-level `.gitignore` uses a default-deny policy for the initial
publication. Add precise exceptions only after reviewing file contents and
rights to redistribute data. Already tracked files are not protected by new
ignore patterns ([GitHub ignore documentation](https://docs.github.com/en/get-started/git-basics/ignoring-files)).

## Interpretation and limitations

- **No validated alpha claim:** diagnostic agreement or attractive forward
  outcomes do not demonstrate a profitable deployable strategy.
- **Audit scope:** forward price/ATR statistics are not an execution-aware
  net-PnL simulation with commissions, spreads and slippage.
- **Time semantics:** forward horizons follow input-row steps; missing bars can
  make them differ from elapsed wall-clock minutes.
- **Statistical caution:** filter comparisons are exploratory, not a causal
  ablation study or automatic permission to tune thresholds. Out-of-sample
  testing and multiple-comparison effects remain important.
- **Data limitations:** delayed, stale, partial and missing data can constrain
  conclusions. A missing field should not be interpreted as a measured zero.
- **Safety boundary:** research output and paper plans are not an instruction
  to place a live trade.

## Development approach and roadmap

Development is AI-assisted, including implementation, debugging, documentation
and test design. Generated suggestions are reviewed through iterative runs and
regression tests; AI assistance does not certify financial conclusions.

The planned sequence is to establish a reviewed source baseline, validate a
clean-clone demo and test subset, then refactor imports and stable module names
without changing behavior. A later standalone `market-regime-toolkit`
repository will become the canonical project; `python_pub` can remain the
portfolio index. No standalone repository is claimed to exist yet.

No license grant is implied by this README. Choose an explicit code license
separately and review third-party material before redistribution.
