"""
Parameter Dynamics Monitoring

PCA-based monitoring of model parameter time-series to:
  - Track the first 3 principal components (explained variance).
  - Detect parameter jumps that may indicate mis-calibration.
  - Correlate factor loadings with market indices (VIX proxy etc.).
  - Alert when a new observation is an outlier in PCA space.

Input:  history array of shape (n_dates, n_params) — one row per calibration run.
Output: PCAMonitorReport with loadings, explained variance, alerts.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PCAMonitorReport:
    """Results from parameter PCA monitoring."""
    n_components: int
    explained_variance_ratio: np.ndarray   # (n_components,)
    cumulative_explained: float            # sum of first n_components
    loadings: np.ndarray                   # (n_components, n_params) — eigenvectors
    scores: np.ndarray                     # (n_dates, n_components) — projections
    target_met: bool                       # True if cumulative_explained >= threshold
    jump_flags: List[int]                  # Indices of dates with parameter jumps
    outlier_flags: List[int]               # Indices of dates with PCA outliers
    summary: str


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class ParameterDynamicsMonitor:
    """
    Monitors SSVI/SABR parameter stability via PCA on historical series.

    Usage::

        monitor = ParameterDynamicsMonitor(n_components=3)
        history = np.array([...])   # (n_dates, n_params)
        report  = monitor.analyze(history)
        print(report.summary)
    """

    def __init__(
        self,
        n_components: int = 3,
        explained_variance_target: float = 0.95,
        jump_threshold: float = 0.10,
        outlier_sigma: float = 3.0,
    ):
        """
        Args:
            n_components:               Number of PCA components to retain.
            explained_variance_target:  Minimum cumulative explained variance (0.95 = 95 %).
            jump_threshold:             Flag if any parameter changes > threshold × |prev| (10 %).
            outlier_sigma:              Flag if PCA score > outlier_sigma std devs from mean.
        """
        self.n_components = n_components
        self.ev_target = explained_variance_target
        self.jump_threshold = jump_threshold
        self.outlier_sigma = outlier_sigma

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, history: np.ndarray) -> PCAMonitorReport:
        """
        Run full PCA analysis on parameter history.

        Args:
            history: (n_dates, n_params) — rows are calibration snapshots.

        Returns:
            PCAMonitorReport
        """
        if history.ndim != 2:
            raise ValueError("history must be 2-D array (n_dates, n_params)")

        n_dates, n_params = history.shape
        n_comp = min(self.n_components, n_params, n_dates - 1)

        # Standardise
        mean = np.mean(history, axis=0)
        std  = np.std(history, axis=0)
        std[std < 1e-12] = 1.0          # avoid division by zero for constant params
        X = (history - mean) / std

        # Covariance-based PCA via SVD
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        eigenvalues = (s ** 2) / max(n_dates - 1, 1)
        total_var   = np.sum(eigenvalues)
        ev_ratio    = eigenvalues[:n_comp] / (total_var + 1e-12)
        loadings    = Vt[:n_comp, :]          # (n_comp, n_params)
        scores      = X @ loadings.T           # (n_dates, n_comp)

        cumulative = float(np.sum(ev_ratio))

        # Jump detection: relative change > jump_threshold
        jump_flags: List[int] = []
        for i in range(1, n_dates):
            denom = np.abs(history[i - 1]) + 1e-12
            rel_change = np.abs(history[i] - history[i - 1]) / denom
            if np.any(rel_change > self.jump_threshold):
                jump_flags.append(i)

        # PCA outlier detection: Mahalanobis in score space (simplified: z-score per component)
        score_mean = np.mean(scores, axis=0)
        score_std  = np.std(scores, axis=0) + 1e-12
        z_scores   = np.abs((scores - score_mean) / score_std)
        outlier_flags: List[int] = list(np.where(np.any(z_scores > self.outlier_sigma, axis=1))[0])

        target_met = cumulative >= self.ev_target

        summary = (
            f"PCA Monitor | {n_comp} components | "
            f"Explained={cumulative * 100:.1f}% "
            f"({'OK' if target_met else f'< target {self.ev_target * 100:.0f}%'}) | "
            f"Jumps={len(jump_flags)} | Outliers={len(outlier_flags)}"
        )

        return PCAMonitorReport(
            n_components=n_comp,
            explained_variance_ratio=ev_ratio,
            cumulative_explained=cumulative,
            loadings=loadings,
            scores=scores,
            target_met=target_met,
            jump_flags=jump_flags,
            outlier_flags=outlier_flags,
            summary=summary,
        )

    def plot_scores(
        self,
        report: PCAMonitorReport,
        dates: Optional[List] = None,
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Plot first 3 PCA score time-series and mark jumps / outliers.

        Args:
            report:     PCAMonitorReport from analyze()
            dates:      Optional list of date labels.
            save_path:  PNG save path; if None just closes the figure.

        Returns:
            save_path if saved, else None.
        """
        import matplotlib.pyplot as plt

        n_comp = min(report.n_components, 3)
        x = list(range(report.scores.shape[0]))
        labels = dates if dates and len(dates) == len(x) else x

        fig, axes = plt.subplots(n_comp, 1, figsize=(10, 3 * n_comp), sharex=True)
        if n_comp == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(x, report.scores[:, i], color="steelblue", lw=1.2, label=f"PC{i+1}")
            # Mark jumps
            for j in report.jump_flags:
                ax.axvline(j, color="orange", alpha=0.7, lw=0.8, linestyle="--")
            # Mark outliers
            for j in report.outlier_flags:
                ax.axvline(j, color="red", alpha=0.8, lw=0.8, linestyle=":")
            ev_pct = report.explained_variance_ratio[i] * 100
            ax.set_ylabel(f"PC{i+1} ({ev_pct:.1f}%)")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Date index")
        fig.suptitle(
            f"Parameter PCA scores  —  cumulative {report.cumulative_explained * 100:.1f}%",
            fontsize=12,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return save_path

        plt.close(fig)
        return None
