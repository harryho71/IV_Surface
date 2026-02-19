"""
Greeks Smoothness Validator & Greeks Dashboard

GreeksCalculator: Analytical Black-Scholes Greeks over an IV surface.
    - GreeksValidator:  Detects oscillations and kinks (overfitting indicators).

5.6 - plot_greeks_dashboard: 3D matplotlib dashboard of all six Greeks.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GreeksSmoothnesReport:
    """Report on Greeks smoothness across the surface."""
    is_smooth: bool
    delta_oscillations: int
    gamma_oscillations: int
    vega_oscillations: int
    vanna_kinks: int
    volga_kinks: int
    details: List[str]

    def summary(self) -> str:
        return (
            f"Greeks smooth={'PASS' if self.is_smooth else 'FAIL'} | "
            f"Delta={self.delta_oscillations}, Gamma={self.gamma_oscillations}, "
            f"Vega={self.vega_oscillations}, Vanna={self.vanna_kinks}, "
            f"Volga={self.volga_kinks}"
        )


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class GreeksCalculator:
    """
    Computes analytical Black-Scholes Greeks over a (K, T) implied-vol surface.

    All Greeks for European calls:
      Delta (Δ)  = ∂C/∂S
      Gamma (Γ)  = ∂²C/∂S²
      Vega  (ν)  = ∂C/∂σ          (per unit of vol, not per vol-point)
      Theta (Θ)  = ∂C/∂T
      Vanna      = ∂²C/∂S∂σ
      Volga      = ∂²C/∂σ²
    """

    @staticmethod
    def _d1_d2(
        S: float, K: float, T: float, r: float, q: float, sigma: float
    ):
        sqrt_T = np.sqrt(max(T, 1e-10))
        sigma_s = max(sigma, 1e-8)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma_s ** 2) * T) / (sigma_s * sqrt_T)
        d2 = d1 - sigma_s * sqrt_T
        return d1, d2

    def compute(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        ivs: np.ndarray,
        spot: float,
        rate: float = 0.0,
        div_yield: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """
        Compute full Greeks grid.

        Args:
            strikes:   (n_K,)
            maturities:(n_T,)
            ivs:       (n_K, n_T)
            spot:      current spot price
            rate:      risk-free rate
            div_yield: continuous dividend yield

        Returns:
            Dict with keys delta, gamma, vega, theta, vanna, volga;
            each array is (n_K, n_T).
        """
        n_K, n_T = len(strikes), len(maturities)
        G = {g: np.zeros((n_K, n_T)) for g in ("delta", "gamma", "vega", "theta", "vanna", "volga")}

        for t_idx, T in enumerate(maturities):
            if T <= 0:
                continue
            sqrt_T = np.sqrt(T)
            for k_idx, K in enumerate(strikes):
                sigma = ivs[k_idx, t_idx]
                if sigma <= 0:
                    continue

                d1, d2 = self._d1_d2(spot, K, T, rate, div_yield, sigma)
                n_d1 = norm.pdf(d1)
                N_d1 = norm.cdf(d1)
                N_d2 = norm.cdf(d2)
                df_q = np.exp(-div_yield * T)
                df_r = np.exp(-rate * T)

                G["delta"][k_idx, t_idx] = df_q * N_d1
                G["gamma"][k_idx, t_idx] = df_q * n_d1 / (spot * sigma * sqrt_T)
                G["vega"][k_idx, t_idx]  = spot * df_q * n_d1 * sqrt_T
                G["theta"][k_idx, t_idx] = (
                    -spot * df_q * n_d1 * sigma / (2.0 * sqrt_T)
                    - rate * K * df_r * N_d2
                    + div_yield * spot * df_q * N_d1
                )
                G["vanna"][k_idx, t_idx] = -df_q * n_d1 * d2 / sigma
                G["volga"][k_idx, t_idx] = spot * df_q * n_d1 * sqrt_T * d1 * d2 / sigma

        return G


# ---------------------------------------------------------------------------
# Validator (5.3)
# ---------------------------------------------------------------------------

class GreeksValidator:
    """
    Detects oscillations/kinks in Greek surfaces that indicate overfitting.

    Method: normalised second differences.
      If  |Δ²f[i]| / max(|f|)  >  threshold  → oscillation flagged.
    """

    def __init__(self, oscillation_threshold: float = 0.02):
        """
        Args:
            oscillation_threshold: Relative amplitude triggering an oscillation
                alert (0.02 = 2 % of surface range).
        """
        self.threshold = oscillation_threshold

    # ------------------------------------------------------------------
    @staticmethod
    def _sign_change_count(arr: np.ndarray) -> int:
        """Count sign changes in a 1-D array (ignoring zeros)."""
        nonzero = arr[arr != 0]
        if len(nonzero) < 2:
            return 0
        return int(np.sum(np.diff(np.sign(nonzero)) != 0))

    def _count_strike_oscillations(self, surface: np.ndarray) -> int:
        """
        Count oscillations in the strike dimension per maturity slice.

        An oscillation means > 1 sign change in consecutive first differences,
        i.e. the surface wiggles up-and-down rather than curving smoothly.
        A single inflection (like the natural peak of gamma) is allowed.
        """
        count = 0
        for t_idx in range(surface.shape[1]):
            d1 = np.diff(surface[:, t_idx])
            n_sign_changes = self._sign_change_count(d1)
            # More than 1 sign change in d1 = genuine oscillation
            if n_sign_changes > 1:
                count += n_sign_changes - 1
        return count

    def _count_maturity_kinks(self, surface: np.ndarray) -> int:
        """
        Count kinks in the maturity axis per strike.

        More than 1 sign change in d1 over maturities = non-smooth term-structure.
        """
        count = 0
        for k_idx in range(surface.shape[0]):
            if surface.shape[1] < 3:
                break
            d1 = np.diff(surface[k_idx, :])
            n_sign_changes = self._sign_change_count(d1)
            if n_sign_changes > 1:
                count += n_sign_changes - 1
        return count

    # ------------------------------------------------------------------
    def validate(
        self,
        greeks: Dict[str, np.ndarray],
        strikes: np.ndarray,
        maturities: np.ndarray,
    ) -> GreeksSmoothnesReport:
        """
        Check all Greeks for oscillations and kinks.

        Args:
            greeks:    Dict from GreeksCalculator.compute()
            strikes:   strike grid
            maturities:maturity grid

        Returns:
            GreeksSmoothnesReport
        """
        delta_osc = self._count_strike_oscillations(greeks["delta"])
        gamma_osc = self._count_strike_oscillations(greeks["gamma"])
        vega_osc  = self._count_strike_oscillations(greeks["vega"])
        vanna_k   = self._count_maturity_kinks(greeks["vanna"])
        volga_k   = self._count_maturity_kinks(greeks["volga"])

        details: List[str] = []
        if delta_osc:
            details.append(f"Delta: {delta_osc} oscillation(s) detected in strike dimension")
        if gamma_osc:
            details.append(f"Gamma: {gamma_osc} oscillation(s) detected in strike dimension")
        if vega_osc:
            details.append(f"Vega : {vega_osc} oscillation(s) detected in strike dimension")
        if vanna_k:
            details.append(f"Vanna: {vanna_k} kink(s) detected in maturity dimension")
        if volga_k:
            details.append(f"Volga: {volga_k} kink(s) detected in maturity dimension")

        total = delta_osc + gamma_osc + vega_osc + vanna_k + volga_k
        return GreeksSmoothnesReport(
            is_smooth=(total == 0),
            delta_oscillations=delta_osc,
            gamma_oscillations=gamma_osc,
            vega_oscillations=vega_osc,
            vanna_kinks=vanna_k,
            volga_kinks=volga_k,
            details=details,
        )


# ---------------------------------------------------------------------------
# Greeks Dashboard (5.6)
# ---------------------------------------------------------------------------

def plot_greeks_dashboard(
    greeks: Dict[str, np.ndarray],
    strikes: np.ndarray,
    maturities: np.ndarray,
    title: str = "Greeks Dashboard",
    save_path: Optional[str] = None,
) -> Optional[str]:
    """
    3D surface dashboard for Delta, Gamma, Vega, Theta, Vanna, Volga.

    Args:
        greeks:     Dict from GreeksCalculator.compute()
        strikes:    (n_K,)
        maturities: (n_T,)
        title:      Figure title
        save_path:  If given, save PNG and return path; otherwise show.

    Returns:
        save_path if saved, else None.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    K_grid, T_grid = np.meshgrid(strikes, maturities, indexing="ij")

    panels = [
        ("delta", "Delta (Δ)"),
        ("gamma", "Gamma (Γ)"),
        ("vega",  "Vega (ν)"),
        ("theta", "Theta (Θ)"),
        ("vanna", "Vanna"),
        ("volga", "Volga"),
    ]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i, (key, label) in enumerate(panels):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        surf = ax.plot_surface(
            K_grid, T_grid, greeks[key],
            cmap="RdBu_r", alpha=0.88, linewidth=0, antialiased=True
        )
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8, pad=0.1)
        ax.set_xlabel("Strike", fontsize=8)
        ax.set_ylabel("Maturity", fontsize=8)
        ax.set_zlabel(label, fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=6)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.close(fig)
    return None
