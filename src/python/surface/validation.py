"""
Validation Suite
Automated surface validation and reporting utilities.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from .arbitrage import ArbitrageChecker, ArbitrageReport, ArbitrageViolation


@dataclass
class ValidationSummary:
    """High-level validation summary."""
    is_valid: bool
    total_violations: int
    severe_violations: int
    report: ArbitrageReport


class SurfaceValidator:
    """Validation suite for IV surfaces."""

    def __init__(self, tolerance: float = 1e-6, bid_ask_buffer: float = 0.0):
        self.tolerance = tolerance
        self.bid_ask_buffer = bid_ask_buffer
        self.arb_checker = ArbitrageChecker(
            tolerance=tolerance,
            bid_ask_buffer=bid_ask_buffer
        )

    def dense_grid_check(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        ivs: np.ndarray,
        spot: float,
        rate: float = 0.0,
        dividend_yield: float = 0.0,
        bid_ask_spreads: Optional[np.ndarray] = None
    ) -> ValidationSummary:
        """Run arbitrage checks on a dense grid.

        Args:
            strikes: (n_strikes,)
            maturities: (n_maturities,)
            ivs: (n_strikes, n_maturities)
            spot: spot price
            rate: risk-free rate
            dividend_yield: dividend yield
            bid_ask_spreads: optional spreads grid (n_strikes, n_maturities)
        """
        violations: List[ArbitrageViolation] = []

        # Butterfly checks per maturity
        for t_idx, maturity in enumerate(maturities):
            iv_slice = ivs[:, t_idx]
            spreads_slice = None
            if bid_ask_spreads is not None:
                spreads_slice = bid_ask_spreads[:, t_idx]
            _, vios = self.arb_checker.check_butterfly_arbitrage(
                strikes=strikes,
                implied_vols=iv_slice,
                maturity=maturity,
                spot=spot,
                rate=rate,
                dividend_yield=dividend_yield,
                bid_ask_spreads=spreads_slice
            )
            violations.extend(vios)

        # Calendar checks per strike
        for k_idx, strike in enumerate(strikes):
            ivs_strike = ivs[k_idx, :]
            spreads_strike = None
            if bid_ask_spreads is not None:
                spreads_strike = bid_ask_spreads[k_idx, :]
            _, vios = self.arb_checker.check_calendar_arbitrage(
                strike=strike,
                maturities=maturities,
                ivs=ivs_strike,
                spot=spot,
                rate=rate,
                dividend_yield=dividend_yield,
                bid_ask_spreads=spreads_strike
            )
            violations.extend(vios)

        # Total variance monotonicity (w = sigma^2 * T)
        w_grid = (ivs ** 2) * maturities[np.newaxis, :]
        for k_idx, strike in enumerate(strikes):
            w_strike = w_grid[k_idx, :]
            dw = np.diff(w_strike)
            if np.any(dw < -self.tolerance):
                min_violation = np.min(dw)
                severity = self.arb_checker._classify_severity(abs(min_violation), self.tolerance)
                violations.append(ArbitrageViolation(
                    type='total_variance',
                    severity=severity,
                    value=min_violation,
                    tolerance=self.tolerance,
                    location=(strike,),
                    message=(
                        f"Total variance monotonicity violation at K={strike:.2f}: "
                        f"min Δw = {min_violation:.6f}"
                    )
                ))

        report = self._build_report(violations)
        severe = sum(1 for v in violations if v.severity == 'severe')

        return ValidationSummary(
            is_valid=report.is_arbitrage_free,
            total_violations=len(violations),
            severe_violations=severe,
            report=report
        )

    @staticmethod
    def report_violations(report: ArbitrageReport) -> Dict[str, int]:
        """Return counts of violations by type."""
        return {
            "butterfly": report.butterfly_violations,
            "calendar": report.calendar_violations,
            "total_variance": report.total_variance_violations
        }

    @staticmethod
    def visualize_violations(report: ArbitrageReport) -> Optional[str]:
        """Placeholder for visualization integration.

        Returns a simple text summary for now.
        """
        return report.summary

    @staticmethod
    def visualize_violations_heatmap(
        strikes: np.ndarray,
        maturities: np.ndarray,
        violations: List[ArbitrageViolation],
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """Render a simple heatmap of violation counts.

        Args:
            strikes: (n_strikes,)
            maturities: (n_maturities,)
            violations: list of ArbitrageViolation
            save_path: optional file path to save PNG
        """
        import matplotlib.pyplot as plt

        heat = np.zeros((len(strikes), len(maturities)))

        for violation in violations:
            if violation.type == "butterfly":
                _, k_mid, _, maturity = violation.location
                k_idx = int(np.argmin(np.abs(strikes - k_mid)))
                t_idx = int(np.argmin(np.abs(maturities - maturity)))
                heat[k_idx, t_idx] += 1.0
            elif violation.type == "calendar":
                strike, _, t2 = violation.location
                k_idx = int(np.argmin(np.abs(strikes - strike)))
                t_idx = int(np.argmin(np.abs(maturities - t2)))
                heat[k_idx, t_idx] += 1.0
            elif violation.type == "total_variance":
                strike = violation.location[0]
                k_idx = int(np.argmin(np.abs(strikes - strike)))
                heat[k_idx, :] += 1.0

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(heat, origin="lower", aspect="auto")
        ax.set_xlabel("Maturity")
        ax.set_ylabel("Strike")
        ax.set_xticks(range(len(maturities)))
        ax.set_yticks(range(len(strikes)))
        ax.set_xticklabels([f"{t:.2f}" for t in maturities])
        ax.set_yticklabels([f"{k:.2f}" for k in strikes])
        fig.colorbar(im, ax=ax, label="Violation Count")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            return save_path

        plt.close(fig)
        return None

    def _build_report(self, violations: List[ArbitrageViolation]) -> ArbitrageReport:
        butterfly = sum(1 for v in violations if v.type == 'butterfly')
        calendar = sum(1 for v in violations if v.type == 'calendar')
        total_variance = sum(1 for v in violations if v.type == 'total_variance')

        is_arbitrage_free = len(violations) == 0
        summary = (
            f"Arbitrage-free={is_arbitrage_free} | "
            f"Butterfly={butterfly}, Calendar={calendar}, TotalVariance={total_variance}"
        )

        return ArbitrageReport(
            is_arbitrage_free=is_arbitrage_free,
            violations=violations,
            butterfly_violations=butterfly,
            calendar_violations=calendar,
            total_variance_violations=total_variance,
            summary=summary
        )
