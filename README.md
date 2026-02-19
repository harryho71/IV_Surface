# IV Surface: High-Performance Volatility Surface Library

A **production-ready implied volatility surface library** combining a C++ computational engine with a Python/Streamlit dashboard for options pricing and risk management.

**Status:** All features complete · 327 tests passing

---

## Quick Start

### Prerequisites

- Python 3.10+
- C++ compiler: MinGW-w64 (Windows) / GCC (Linux/Mac)
- CMake 3.15+

### Setup

```bash
# 1. Create virtual environment and install Python dependencies
python -m venv .venv
source .venv/Scripts/activate   # Windows
# source .venv/bin/activate     # Linux / Mac
pip install -r requirements.txt

# 2. Build C++ engine
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cp third_party/nlopt/bin/libnlopt.dll build/    # Windows only

# 3. Verify the engine
./build/sabr_cli.exe eval 100 105 0.25 0.25 0.7 -0.4 0.3
# Expected output: 0.074940

# 4. Launch the dashboard
```

The dashboard opens at `http://localhost:8501`.

> See [DEVELOPMENT.md](DEVELOPMENT.md) for the full build guide, C++ CLI reference, Python API, and testing details.

---

## Dashboard (`app.py`)

The main application is a Streamlit dashboard that exposes all features in a tabbed UI:

| Tab                          | Contents                                                             |
| ---------------------------- | -------------------------------------------------------------------- |
| **Phase 1 – Arbitrage**      | Butterfly & calendar violation checks, quote adjustment              |
| **Phase 2 – SSVI**           | SSVI surface fitting with Gatheral-Jacquier constraints              |
| **Phase 3 – Calibration**    | Vega-weighted, Tikhonov-regularized, warm-start SABR calibration     |
| **Phase 4 – Total Variance** | σ²T framework, Lee moment bounds, C++-accelerated grid operations    |
| **Phase 5 – Validation**     | Greeks smoothness, benchmark structures, PCA monitoring, backtesting |
| **Phase 6 – Operations**     | Bid-ask surface, audit trail, alerting, trader overrides, pipeline   |

**Data source:** choose _Synthetic_ (instant) or _Real Market Data_ (yfinance, cached under `data/raw/`).

---

## Features

| Category          | Capability                                                                                        |
| ----------------- | ------------------------------------------------------------------------------------------------- |
| **Models**        | SABR, SVI, SSVI, eSSVI (extended SSVI with ρ(θ))                                                  |
| **Arbitrage**     | Butterfly, calendar, total variance — zero violations by construction                             |
| **Calibration**   | Vega-weighted, bid-ask weighted, Tikhonov regularization, warm-start, multi-objective             |
| **Interpolation** | Total variance framework (w = σ²T), Lee moment bounds, monotonicity-preserving                    |
| **Validation**    | Dense-grid checker, Greeks smoothness, benchmark structures, PCA monitoring, backtesting          |
| **Operations**    | Calibration audit trail, parameter versioning, alerting, trader override workflow, daily pipeline |
| **Interface**     | C++ CLI via subprocess — language-agnostic, no Python ABI dependency                              |
| **Data**          | yfinance integration, smart JSON caching, forward curve inference, expiry classification          |

---

## Architecture

```
Python (orchestration, data, visualization)
    │  subprocess IPC (CSV args / newline output)
    ▼
build/sabr_cli.exe  (C++ engine)
    ├── SABR / SVI / SSVI / eSSVI evaluation
    ├── NLopt model calibration
    └── Total variance operations (8–10× faster than Python)
```

**Why subprocess instead of pybind11?** No Python ABI dependency — the same `.exe` works with Python 3.10–3.14+ and can be called from R, Julia, or any shell.

---

## Project Structure

```
IV_Surface/
├── app.py                          # Streamlit dashboard (main entry point)
├── src/
│   ├── cpp/
│   │   ├── include/                # C++ headers: pricing, volatility, interpolation, …
│   │   └── src/                    # Implementations + sabr_cli.cpp (CLI entry point)
│   └── python/
│       ├── cpp_unified_engine.py   # Subprocess bridge (SABREngine, CppTotalVarianceEngine)
│       ├── data/                   # Fetching, cleaning, forwards, pipeline, caching
│       └── surface/                # arbitrage, ssvi, essvi, total_variance, calibration,
│                                   # validation, greeks, benchmark, backtesting,
│                                   # audit, alerting, overrides, bid_ask, pipeline
├── tests/
│   ├── cpp/                        # C++ unit tests (pricing, interpolation, calibration)
│   └── python/                     # Python tests (327 passing)
├── examples/                       # Demo scripts for each phase
├── data/
│   ├── raw/                        # Cached market data (JSON per ticker)
│   └── processed/                  # Cleaned option chains (CSV)
├── output/                         # Generated plots and reports
├── third_party/
│   ├── nlopt/                      # Bundled NLopt optimizer (MIT/LGPL)
│   └── Eigen/                      # Header-only linear algebra (MPL2)
└── CMakeLists.txt                  # C++ build configuration
```

---

## Testing

```bash
# Python (327 tests)
pytest tests/python/ -v

# C++ unit tests
./build/test_cpp_pricing.exe
./build/test_cpp_interpolation.exe
./build/test_cpp_calibration.exe
```

---

## Performance

| Operation                        | Time    | Notes                 |
| -------------------------------- | ------- | --------------------- |
| Single SABR IV evaluation        | ~2 ms   | subprocess round-trip |
| 50-strike smile                  | ~95 ms  | batch CSV call        |
| SABR calibration                 | ~45 ms  | NLopt via C++         |
| σ ↔ w conversion (100 strikes)   | ~0.3 ms | C++ total variance    |
| Arbitrage validation (full grid) | ~2 ms   | vectorized C++        |
| Full surface (4T × 50K)          | ~850 ms | from real market data |

---

## Feature Status (February 2026)

| Phase     | Description                                                                    | Tests      |
| --------- | ------------------------------------------------------------------------------ | ---------- |
| 1         | Arbitrage enforcement (butterfly, calendar, total variance)                    | 18 ✅      |
| 2         | SSVI surface model with Gatheral-Jacquier constraints                          | 17 ✅      |
| 3         | Vega/bid-ask weighting, Tikhonov regularization, warm-start                    | 20 ✅      |
| 4         | Total variance framework, Lee bounds, C++ acceleration (8–10×)                 | 20 ✅      |
| 5         | Validation: Greeks smoothness, benchmarks, PCA monitoring, backtesting         | 35 ✅      |
| 6         | Operational: audit trail, versioning, alerting, overrides, daily pipeline      | 52 ✅      |
| Audit     | Standard compliance: forward space, eSSVI, expiry classification, data quality | 77 ✅      |
| **Total** |                                                                                | **327 ✅** |

---

## Requirements

Python packages: `numpy`, `pandas`, `scipy`, `matplotlib`, `plotly`, `streamlit`, `yfinance`
(full list in `requirements.txt`)

C++ dependencies: NLopt 2.7.1, Eigen 3.x (both bundled in `third_party/`)

---

## References

- Hagan et al. (2002) — "Managing Smile Risk" (SABR model)
- Gatheral & Jacquier (2014) — "Arbitrage-free SVI volatility surfaces" (SSVI/eSSVI)
- Gatheral (2006) — _The Volatility Surface: A Practitioner's Guide_
- NLopt: https://nlopt.readthedocs.io/

---

## License

MIT — see [LICENSE](LICENSE)
