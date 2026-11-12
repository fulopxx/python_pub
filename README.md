# Python Projects Portfolio

A collection of Python projects exploring analytical tooling, data integration,
visualization and desktop applications. Each project has its own scope and setup;
the repository is a portfolio collection, not a single installable application.

## Featured project

### [Market Regime Research Toolkit](crash_regime_rules/README.md)

A rules-based market-research and paper decision-support toolkit for gold and
equity-index futures, with complementary volatility, options and macro/credit
analysis modules. It emphasizes explainable diagnostics, data-quality checks and
repeatable offline evaluation rather than claims of predictive performance.

- **Engineering focus:** modular analytical pipelines, API integration,
  timezone-aware time series, multiprocessing and automated regression tests.
- **Technologies:** Python, NumPy, pandas, `ib_insync`, command-line interfaces
  and `unittest`.
- **Try it:** the project README includes a synthetic options demo that does not
  require a brokerage connection.
- **Status:** active research code; original module filenames are preserved
  during the initial publication phase.

See the [project overview, setup and limitations](crash_regime_rules/README.md).

## Earlier projects

These existing projects are retained in their original folders. Their runtime
requirements and maintenance status should be checked individually; the market
toolkit's dependency file does not apply to them.

| Project | Focus | Location |
|---|---|---|
| Fractal Video Generator | Fractal/video-generation project | [Browse files](Fractal_video_generator/) |
| SkyRadar FlightInfo GUI | Flight-information GUI project | [Browse files](SkyRadar_FlightInfo_GUI/) |
| SkyRadar FlightInfo GUI 2 | Second flight-information GUI project folder | [Browse files](SkyRadar_FlightInfo_GUI2/) |

The existing project names and locations are visible in the
[public repository](https://github.com/fulopxx/python_pub).

## Repository practices

- **Separate environments:** create a virtual environment for the project being
  used instead of assuming a shared interpreter or dependency set.
- **Source-first publication:** local credentials, private configuration, raw
  market histories and generated reports are not part of the toolkit's public
  source baseline.
- **Incremental improvement:** documentation, clean-clone reproducibility and
  automated checks are developed before larger structural refactoring.

## Development approach

The Market Regime Research Toolkit is developed iteratively with AI assistance
for implementation, debugging, documentation and test design. AI-generated work
requires review and testing; it is not evidence of trading validity or production
readiness.

The code and examples are provided for technical exploration. Consult each
project's documentation and any applicable license before reuse.
