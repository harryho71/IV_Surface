#pragma once
#include <vector>
#include <functional>

namespace iv_surface
{

    /**
     * Wrapper around NLopt for model calibration
     * Fit SABR, SVI, or polynomial models to observed IV surface
     *
     * SABR Model (Stochastic Alpha, Beta, Rho):
     * σ(K) = σ₀ * F^(1-β) / ((F*K)^((1-β)/2) * (1 + ((1-β)²/24) * ln²(F/K)))
     * Parameters: {α, β, ρ, ν}
     * Best for: FX, interest rates
     *
     * SVI Model (Stochastic Volatility Inspired):
     * σ(m) = √[a + b(ρ(m - m*) + √((m - m*)² + σ²))]
     * Parameters: {a, b, ρ, σ, m*}
     * Best for: Equities, simple and interpretable
     *
     * Polynomial Model:
     * σ(m,τ) = (b₀ + b₁*m + b₂*m²) + (b₃ + b₄*m)*τ
     * Parameters: regression coefficients
     * Best for: Quick fits, computational efficiency
     */
    class ModelCalibrator
    {
    public:
        enum ModelType
        {
            SABR = 0,
            SVI = 1,
            POLYNOMIAL = 2
        };

        struct CalibrationResult
        {
            std::vector<double> parameters; // Calibrated parameter vector
            double rmse;                    // Root mean squared error
            double max_error;               // Maximum absolute error
            int iterations;                 // Number of iterations
            bool converged;                 // Convergence flag
        };

        /**
         * Calibrate model to observed IV surface
         * Minimize: Σ (observed_IV_i - model_IV_i)²
         *
         * @param model_type     Type of model: SABR, SVI, or POLYNOMIAL
         * @param observed_ivs   Observed market IVs for fitting
         * @param strikes        Strike prices corresponding to IVs
         * @param maturity       Fixed maturity (single-tenor fit)
         * @param initial_params Starting parameter guess (dimension depends on model)
         * @return CalibrationResult with fitted parameters and diagnostics
         *
         * @note Optimization via NLopt (multiple algorithms tried)
         * @note Local optimization (not global search)
         * @note Parameter bounds enforced:
         *   - SABR: 0 < α < 1, 0 ≤ β ≤ 1, -1 < ρ < 1, ν > 0
         *   - SVI: a ≥ 0, b ≥ 0, |ρ| < 1, σ ≥ 0
         *   - Polynomial: unconstrained (least-squares)
         *
         * @note Convergence criteria:
         *   - Parameter tolerance: 1e-6
         *   - Objective tolerance: 1e-8
         *   - Max iterations: 1000
         */
        static CalibrationResult calibrate(
            ModelType model_type,
            const std::vector<double> &observed_ivs,
            const std::vector<double> &strikes,
            double maturity,
            const std::vector<double> &initial_params,
            double forward = 0.0);

        /**
         * SABR Model Evaluation
         * σ_SABR(K) = α * F^(1-β) / ((F*K)^((1-β)/2) * D(β, ν, ρ, ln(F/K)))
         *
         * where D is the scaling function accounting for stochastic vol
         *
         * @param forward      Forward price (F = S * exp((r-q)*T))
         * @param strike       Strike price
         * @param maturity     Time to maturity (years)
         * @param params       {α, β, ρ, ν}
         *   - α (alpha):     ATM volatility
         *   - β (beta):      CEV exponent (0=normal, 1=lognormal)
         *   - ρ (rho):       Spot-vol correlation
         *   - ν (nu):        Volatility of volatility
         * @return Implied volatility at (K, T)
         */
        static double sabr_volatility(
            double forward, double strike, double maturity,
            const std::vector<double> &params);

        /**
         * SVI Model Evaluation
         * σ_SVI(m) = √[a + b(ρ(m - m*) + √((m - m*)² + σ²))]
         *
         * where m = ln(K/F) is log-moneyness
         *
         * @param moneyness    Log-moneyness m = ln(K/F)
         * @param params       {a, b, ρ, σ, m*}
         *   - a (a):         Variance floor (ATM curvature)
         *   - b (b):         Volatility slope (skew magnitude)
         *   - ρ (rho):       Skew direction (-1 to 1)
         *   - σ (sigma):     Smoothness parameter (smile width)
         *   - m* (m_star):   Strike offset (center of smile)
         * @return Implied volatility at moneyness m
         *
         * @note SVI satisfies arbitrage-free conditions by construction
         */
        static double svi_volatility(
            double moneyness,
            const std::vector<double> &params);

    private:
        // Objective function for NLopt (wrapper for user-defined model)
        static double objective_function(
            const std::vector<double> &params,
            void *user_data);

        // Helper: compute forward price from spot, rates, yields, maturity
        static double compute_forward(double spot, double rate, double maturity);
    };

} // namespace iv_surface
