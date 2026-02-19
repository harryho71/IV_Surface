# Development Guide

Technical reference for building, running, and extending the IV Surface library.

---

## Table of Contents

1. [Building the C++ Engine](#building-the-c-engine)
2. [C++ CLI Reference](#c-cli-reference)
3. [Python API](#python-api)
4. [Data Layer & Caching](#data-layer--caching)
5. [SABR Parameter Guide](#sabr-parameter-guide)
6. [Testing](#testing)
7. [Adding New Features](#adding-new-features)
8. [Troubleshooting](#troubleshooting)

---

## Building the C++ Engine

### Windows (MinGW-w64)

```bash
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sabr_cli
cp third_party/nlopt/bin/libnlopt.dll build/
```

### Linux / Mac (GCC / Clang)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sabr_cli
```

### Build targets

| Target | Output | Description |
|--------|--------|-------------|
| `sabr_cli` | `build/sabr_cli.exe` | Main computational engine (required) |
| `test_cpp_pricing` | `build/test_cpp_pricing.exe` | Black-Scholes & SABR unit tests |
| `test_cpp_interpolation` | `build/test_cpp_interpolation.exe` | Spline interpolation tests |
| `test_cpp_calibration` | `build/test_cpp_calibration.exe` | NLopt calibration tests |

Build all:

```bash
cmake --build build
```

### Verify the build

```bash
./build/sabr_cli.exe eval 100 105 0.25 0.25 0.7 -0.4 0.3
# Expected: 0.074940
```

---

## C++ CLI Reference

All C++ functionality is exposed via `build/sabr_cli.exe <command> <args…>`.  
Arrays are passed as comma-separated values (no spaces).

### `eval` — single SABR evaluation

```bash
./build/sabr_cli.exe eval <Forward> <Strike> <Maturity> <Alpha> <Beta> <Rho> <Nu>
```

Example:

```bash
./build/sabr_cli.exe eval 100 105 0.25 0.25 0.7 -0.4 0.3
# 0.074940
```

### `calibrate` — fit SABR to market quotes

```bash
./build/sabr_cli.exe calibrate <Maturity> "<Strikes>" "<IVs>" "<InitParams>"
```

Output: `alpha,beta,rho,nu,rmse,iterations,converged`

Example:

```bash
./build/sabr_cli.exe calibrate 0.25 "95,100,105,110" "0.25,0.23,0.25,0.27" "0.2,0.7,-0.3,0.3"
# 0.234567,0.700000,-0.287654,0.321098,0.002134,45,1
```

### `bs_prices` — Black-Scholes batch pricing

```bash
./build/sabr_cli.exe bs_prices <Forward> "<Strikes>" <Maturity> <Rate> "<IVs>"
```

Output: comma-separated call prices.

### `check_butterfly` — butterfly arbitrage check

```bash
./build/sabr_cli.exe check_butterfly "<Strikes>" "<CallPrices>" <Tolerance>
```

Output: `violations` (integer count).

### Total variance commands

```bash
# Convert implied vols to total variance (w = σ²T)
./build/sabr_cli.exe tv_sigma_to_w "<sigma_csv>" "<maturities_csv>"

# Convert total variance back to implied vols
./build/sabr_cli.exe tv_w_to_sigma "<w_csv>" "<maturities_csv>"

# Enforce monotonicity on flattened w grid
./build/sabr_cli.exe tv_enforce_monotonic "<w_csv>" <n_strikes> <n_maturities>

# Validate arbitrage-free property
./build/sabr_cli.exe tv_validate "<w_csv>" <n_strikes> <n_maturities> [tolerance]

# Compute Lee wing bounds
./build/sabr_cli.exe tv_lee_bounds <w_atm>
```

---

## Python API

### SABREngine (`src/python/cpp_unified_engine.py`)

```python
from cpp_unified_engine import SABREngine

engine = SABREngine()   # auto-detects build/sabr_cli.exe

# Single implied vol
iv = engine.evaluate(forward=100, strike=105, maturity=0.25,
                     alpha=0.25, beta=0.7, rho=-0.4, nu=0.3)

# Full smile
import numpy as np
strikes = np.linspace(90, 110, 50)
ivs = engine.evaluate_smile(100, strikes, 0.25, 0.25, 0.7, -0.4, 0.3)

# Calibration
params, rmse, iters, converged = engine.calibrate(
    maturity=0.25,
    strikes=[95, 98, 100, 102, 105],
    market_ivs=[0.26, 0.24, 0.23, 0.24, 0.25],
    initial_params=[0.2, 0.7, -0.3, 0.3],   # [alpha, beta, rho, nu]
)
print(f"α={params[0]:.4f} β={params[1]:.4f} ρ={params[2]:.4f} ν={params[3]:.4f}")
print(f"RMSE={rmse*100:.2f}vp  converged={converged}")
```

### CppTotalVarianceEngine

```python
from cpp_unified_engine import CppTotalVarianceEngine

tv = CppTotalVarianceEngine()

# Convert a vol grid to total variance
w_grid = tv.sigma_to_total_variance(sigma_grid, maturities)

# Enforce monotonicity
w_mono = tv.enforce_monotonicity(w_grid)

# Validate
bfly_viols, cal_viols = tv.validate_arbitrage_free(w_grid, tolerance=1e-6)
```

### Surface modules (`src/python/surface/`)

| Module | Key class | Purpose |
|--------|-----------|---------|
| `arbitrage.py` | `ArbitrageChecker`, `ArbitrageReport` | Butterfly / calendar checks |
| `quote_adjustment.py` | `QuoteAdjuster` | Constrained quote correction |
| `ssvi.py` | `SSVISurface`, `SSVIParameters` | SSVI surface model |
| `essvi.py` | `ESSVISurface`, `ESSVIParameters` | Extended SSVI (ρ(θ) parameterisation) |
| `calibration.py` | `SurfaceCalibrator` | Python-level calibration wrapper |
| `advanced_calibration.py` | `AdvancedCalibrator` | Vega-weighting, Tikhonov, warm-start |
| `total_variance.py` | `TotalVarianceInterpolator` | Total variance interpolation |
| `validation.py` | `SurfaceValidator` | Full surface validation suite |
| `greeks.py` | `GreeksCalculator`, `GreeksValidator` | Delta, Gamma, Vega, smoothness |
| `benchmark.py` | `BenchmarkStructurePricer` | Spread / straddle / butterfly pricing |
| `backtesting.py` | `Backtester`, `StressScenario` | Historical & stress backtesting |
| `parameter_monitoring.py` | `ParameterDynamicsMonitor` | PCA-based drift detection |
| `bid_ask.py` | `BidAskSurfaceGenerator` | Bid-ask spread surface |
| `audit.py` | — | Calibration audit trail |
| `alerting.py` | — | Threshold-based alert engine |
| `overrides.py` | — | Trader override workflow |
| `pipeline.py` | — | Daily end-of-day pipeline |

### Data modules (`src/python/data/`)

```python
from data.pipeline import fetch_and_save_market_data, load_or_fetch, snapshot_to_grid

# Fetch fresh data (or load from cache)
snapshot = load_or_fetch("SPY")

# Convert to calibration grid
grid = snapshot_to_grid(snapshot)

from data.forwards import ForwardCurveBuilder
fwd_builder = ForwardCurveBuilder()
forwards = fwd_builder.build(spot=580.0, rate=0.05, div_yield=0.01, maturities=[0.1, 0.25, 0.5])
```

---

## Data Layer & Caching

Market data is automatically cached as JSON under `data/raw/<TICKER>_market_data.json`.

### Cache file structure

```json
{
  "ticker": "SPY",
  "spot": 580.25,
  "forward": 580.25,
  "timestamp": "2026-02-18T13:45:00.123456",
  "data": {
    "0.068": {
      "strikes": [464.0, 469.0, 474.0],
      "ivs": [0.142, 0.138, 0.135],
      "expiry": "2026-02-27",
      "days": 9
    }
  }
}
```

### Cache priority (in `load_or_fetch`)

1. If the user requests a refresh → fetch from yfinance and overwrite cache.
2. Cached file exists → load and return.
3. No cache → fetch from yfinance and save.

Pre-cached tickers: `AAPL`, `GOOG`, `MSFT`, `PLTR`, `QQQ`, `SPY` (`data/raw/`).

---

## SABR Parameter Guide

| Parameter | Symbol | Typical range | Meaning |
|-----------|--------|---------------|---------|
| Alpha | α | 0.05 – 0.50 | ATM volatility level |
| Beta | β | 0.50 – 1.00 | CEV exponent (0.7 typical for equities) |
| Rho | ρ | −0.80 – 0.00 | Spot-vol correlation (negative → left skew) |
| Nu | ν | 0.10 – 0.80 | Volatility of volatility |

**Rules of thumb:**
- β = 0 → normal SABR (additive vol). β = 1 → log-normal SABR.
- More negative ρ → steeper put skew.
- Higher ν → more pronounced smile convexity.

---

## Testing

### Python tests

```bash
pytest tests/python/ -v              # all 327 tests
pytest tests/python/ -k "arbitrage"  # filter by name
pytest tests/python/ --tb=short      # compact traceback
```

### C++ unit tests

```bash
./build/test_cpp_pricing.exe
./build/test_cpp_interpolation.exe
./build/test_cpp_calibration.exe
```

### Full test suite

```bash
pytest tests/python/ -v && \
  ./build/test_cpp_pricing.exe && \
  ./build/test_cpp_interpolation.exe && \
  ./build/test_cpp_calibration.exe
```

### Test structure

| Test file | Coverage |
|-----------|---------|
| `tests/python/test_arbitrage.py` | Butterfly, calendar, total variance checks |
| `tests/python/test_calibration.py` | SABR / advanced calibration |
| `tests/python/test_data.py` | Market data fetching and caching |
| `tests/python/test_data_quality.py` | Data cleaning and validation |
| `tests/cpp/test_pricing.cpp` | Black-Scholes formulas and Greeks |
| `tests/cpp/test_interpolation.cpp` | Cubic spline, RBF |
| `tests/cpp/test_calibration.cpp` | NLopt integration |

---

## Adding New Features

### Adding a new C++ CLI command

1. Add the command to the if-else chain in `src/cpp/src/sabr_cli.cpp`:

```cpp
else if (command == "my_command")
{
    // parse CSV: auto vals = parse_csv_doubles(argv[2]);
    // compute
    std::cout << result << "\n";
    return 0;
}
```

2. Rebuild:

```bash
cmake --build build --target sabr_cli
```

3. Add a Python wrapper in `src/python/cpp_unified_engine.py`:

```python
def my_method(self, arg1, arg2):
    cmd = [self.exe_path, "my_command", str(arg1), str(arg2)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())
```

4. Write tests in `tests/python/` and `tests/cpp/`.

### Debugging C++ errors from Python

```python
result = subprocess.run(cmd, capture_output=True, text=True, check=False)
if result.returncode != 0:
    print("stderr:", result.stderr)   # C++ exception / assertion message
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `sabr_cli.exe not found` | C++ engine not built or not in `build/` | Run `cmake --build build --target sabr_cli` |
| `libnlopt.dll not found` | DLL missing from `build/` | `cp third_party/nlopt/bin/libnlopt.dll build/` |
| `ModuleNotFoundError` in app.py | Python path not set | Launch via `streamlit run app.py` from project root |
| Very slow calibration | subprocess overhead | Expected; single-call batch reduces round-trips |
| Arbitrage violations after calibration | Noisy market quotes | Use `QuoteAdjuster` to correct before re-calibrating |
| yfinance returns empty chain | Market closed or ticker invalid | Use cached data in `data/raw/` |
