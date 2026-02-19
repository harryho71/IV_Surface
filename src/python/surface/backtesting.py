"""
Backtesting Framework

Reconstructs and evaluates historical IV surface calibrations.

Provides:
  - Backtester.run()            — replay calibration over a date range
  - Backtester.predict()        — naive next-day parameter prediction
  - Backtester.measure_accuracy()— RMSE / MAE of prediction vs realised
  - Backtester.stress_test()    — evaluate surface under standard shock scenarios

Input history format:
  A list of dicts  [{"date": ..., "strikes": ..., "maturities": ..., "ivs": ...}, ...]
  or any sequence of IV surface snapshots.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BacktestSnapshot:
    """Single date snapshot used in backtesting."""
    date: str                    # ISO date string or label
    strikes: np.ndarray
    maturities: np.ndarray
    ivs: np.ndarray              # (n_strikes, n_maturities)
    predicted_ivs: Optional[np.ndarray] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None


@dataclass
class BacktestReport:
    """Aggregate backtesting results."""
    snapshots: List[BacktestSnapshot]
    mean_rmse: float
    mean_mae: float
    max_rmse: float
    n_dates: int
    rmse_threshold: float
    passed: bool                 # mean_rmse < rmse_threshold

    def summary(self) -> str:
        return (
            f"Backtest Report | Dates={self.n_dates} | "
            f"Mean RMSE={self.mean_rmse * 100:.3f} vol pts | "
            f"Max RMSE={self.max_rmse * 100:.3f} vol pts | "
            f"{'PASS' if self.passed else 'FAIL'} (thr={self.rmse_threshold * 100:.1f} vol pts)"
        )


@dataclass
class StressScenario:
    """Definition of a stress shock."""
    name: str
    spot_shock: float        # relative spot shock  e.g. -0.10 = −10 %
    vol_shock: float         # additive IV shock    e.g.  0.05 = +5 vol pts
    rate_shock: float = 0.0  # additive rate shock


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """
    Backtests IV surface models over historical data.

    Prediction method is pluggable via `predictor` callable:
        predicted_ivs = predictor(prev_ivs, prev_strikes, prev_maturities)

    Default predictor: naive carry-forward (today = yesterday).
    """

    # Default stress scenarios mirroring standard bank stress tests
    DEFAULT_SCENARIOS: List[StressScenario] = [
        StressScenario("down_10pct",   spot_shock=-0.10, vol_shock=+0.08),
        StressScenario("up_10pct",     spot_shock=+0.10, vol_shock=-0.04),
        StressScenario("vol_spike",    spot_shock=+0.00, vol_shock=+0.10),
        StressScenario("vol_crush",    spot_shock=+0.00, vol_shock=-0.05),
        StressScenario("crash_20pct",  spot_shock=-0.20, vol_shock=+0.15),
    ]

    def __init__(
        self,
        rmse_threshold: float = 0.01,
        predictor: Optional[Callable] = None,
    ):
        """
        Args:
            rmse_threshold: Acceptable mean RMSE (0.01 = 1 vol pt).
            predictor:      Callable(prev_ivs, prev_strikes, prev_maturities) → next_ivs.
                            Defaults to carry-forward.
        """
        self.rmse_threshold = rmse_threshold
        self.predictor = predictor or self._carry_forward

    # ------------------------------------------------------------------
    # Static predictor helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _carry_forward(
        prev_ivs: np.ndarray,
        prev_strikes: np.ndarray,
        prev_maturities: np.ndarray,
    ) -> np.ndarray:
        """Naive carry-forward: next-day IVs = today's IVs."""
        return prev_ivs.copy()

    @staticmethod
    def _mean_revert(
        prev_ivs: np.ndarray,
        prev_strikes: np.ndarray,
        prev_maturities: np.ndarray,
        long_run_vol: float = 0.20,
        speed: float = 0.05,
    ) -> np.ndarray:
        """Simple mean-reversion predictor."""
        return prev_ivs + speed * (long_run_vol - prev_ivs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        history: List[Dict],
    ) -> BacktestReport:
        """
        Replay calibration over a sequence of historical snapshots.

        Args:
            history: List of dicts, each with keys:
                       "date", "strikes", "maturities", "ivs"

        Returns:
            BacktestReport
        """
        if len(history) < 2:
            raise ValueError("Need at least 2 snapshots to backtest.")

        snapshots: List[BacktestSnapshot] = []

        for i in range(1, len(history)):
            prev = history[i - 1]
            curr = history[i]

            # Predict using previous snapshot
            pred_ivs = self.predictor(
                np.asarray(prev["ivs"]),
                np.asarray(prev["strikes"]),
                np.asarray(prev["maturities"]),
            )

            actual_ivs = np.asarray(curr["ivs"])

            # Align shapes (prediction on prev grid vs actual on curr grid)
            min_shape = (
                min(pred_ivs.shape[0], actual_ivs.shape[0]),
                min(pred_ivs.shape[1], actual_ivs.shape[1]),
            )
            p = pred_ivs[: min_shape[0], : min_shape[1]]
            a = actual_ivs[: min_shape[0], : min_shape[1]]

            diff = p - a
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            mae  = float(np.mean(np.abs(diff)))

            snapshots.append(BacktestSnapshot(
                date=str(curr.get("date", i)),
                strikes=np.asarray(curr["strikes"]),
                maturities=np.asarray(curr["maturities"]),
                ivs=actual_ivs,
                predicted_ivs=pred_ivs,
                rmse=rmse,
                mae=mae,
            ))

        rmse_vals = [s.rmse for s in snapshots]
        mae_vals  = [s.mae  for s in snapshots]
        mean_rmse = float(np.mean(rmse_vals))
        mean_mae  = float(np.mean(mae_vals))
        max_rmse  = float(np.max(rmse_vals))

        return BacktestReport(
            snapshots=snapshots,
            mean_rmse=mean_rmse,
            mean_mae=mean_mae,
            max_rmse=max_rmse,
            n_dates=len(snapshots),
            rmse_threshold=self.rmse_threshold,
            passed=(mean_rmse < self.rmse_threshold),
        )

    def predict(
        self,
        current_ivs: np.ndarray,
        strikes: np.ndarray,
        maturities: np.ndarray,
    ) -> np.ndarray:
        """
        Predict next-day IV surface.

        Args:
            current_ivs: (n_K, n_T) today's IV surface.
            strikes:     (n_K,)
            maturities:  (n_T,)

        Returns:
            Predicted (n_K, n_T) IV surface for next day.
        """
        return self.predictor(current_ivs, strikes, maturities)

    def measure_accuracy(
        self,
        predicted: np.ndarray,
        realised: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute point-in-time accuracy metrics.

        Args:
            predicted: (n_K, n_T)
            realised:  (n_K, n_T)

        Returns:
            Dict with rmse, mae, max_error (all in vol units).
        """
        diff = predicted - realised
        return {
            "rmse":      float(np.sqrt(np.mean(diff ** 2))),
            "mae":       float(np.mean(np.abs(diff))),
            "max_error": float(np.max(np.abs(diff))),
            "bias":      float(np.mean(diff)),
        }

    def stress_test(
        self,
        base_ivs: np.ndarray,
        strikes: np.ndarray,
        maturities: np.ndarray,
        spot: float,
        scenarios: Optional[List[StressScenario]] = None,
    ) -> Dict[str, Dict]:
        """
        Evaluate the surface under a set of stress scenarios.

        Each scenario applies:
          - A relative spot shock (shifts ATM up/down)
          - An additive vol shock (parallel shift of the IV surface)

        Args:
            base_ivs:   (n_K, n_T) base IV surface.
            strikes:    (n_K,)
            maturities: (n_T,)
            spot:       Current spot price.
            scenarios:  List of StressScenario; defaults to DEFAULT_SCENARIOS.

        Returns:
            Dict[scenario_name, {"shocked_ivs", "shocked_spot", "iv_change_mean"}]
        """
        if scenarios is None:
            scenarios = self.DEFAULT_SCENARIOS

        results: Dict[str, Dict] = {}
        for sc in scenarios:
            shocked_spot = spot * (1.0 + sc.spot_shock)
            shocked_ivs  = np.maximum(base_ivs + sc.vol_shock, 1e-4)

            # Scale strike dimension: implied by spot move (sticky-delta approximation)
            shocked_strikes = strikes * (1.0 + sc.spot_shock)

            iv_change = shocked_ivs - base_ivs
            results[sc.name] = {
                "shocked_ivs":    shocked_ivs,
                "shocked_strikes": shocked_strikes,
                "shocked_spot":   shocked_spot,
                "iv_change_mean": float(np.mean(iv_change)),
                "iv_change_max":  float(np.max(np.abs(iv_change))),
                "scenario":       sc,
            }

        return results

    def plot_backtest_rmse(
        self,
        report: BacktestReport,
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Plot RMSE time-series with threshold line.

        Args:
            report:    BacktestReport from run()
            save_path: PNG save path; None to close without saving.

        Returns:
            save_path if saved, else None.
        """
        import matplotlib.pyplot as plt

        dates = [s.date for s in report.snapshots]
        rmses = [s.rmse * 100 for s in report.snapshots]      # convert to vol pts %

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rmses, color="steelblue", lw=1.5, label="RMSE (vol pts)")
        ax.axhline(
            report.rmse_threshold * 100,
            color="red", lw=1.2, linestyle="--",
            label=f"Threshold ({report.rmse_threshold * 100:.1f} vol pts)",
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("RMSE (vol pts)")
        ax.set_title(f"Backtest RMSE  —  Mean={report.mean_rmse * 100:.3f} vol pts")
        ax.legend()
        ax.grid(True, alpha=0.3)

        step = max(1, len(dates) // 10)
        ax.set_xticks(range(0, len(dates), step))
        ax.set_xticklabels(dates[::step], rotation=30, ha="right", fontsize=8)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return save_path

        plt.close(fig)
        return None
