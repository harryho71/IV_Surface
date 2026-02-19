#pragma once
#include <vector>
#include <Eigen/Dense>

namespace iv_surface
{

    /**
     * Multi-method interpolation for sparse IV grids
     * Transforms scattered (K, T, σ) data into smooth 2D surface
     *
     * Three methods provided:
     *
     * 1. CUBIC_SPLINE (Default):
     *    - Piecewise cubic polynomials with C² continuity
     *    - Smooth and locally controlled
     *    - Best for regular grids
     *
     * 2. RBF_THINPLATE:
     *    - Radial Basis Function (thin-plate spline)
     *    - φ(r) = r² ln(r) + polynomial
     *    - Better for scattered/irregular data
     *    - More expensive but more flexible
     *
     * 3. POLYNOMIAL:
     *    - Parametric model: σ(m,τ) = f(moneyness, maturity)
     *    - Fast evaluation, interpretable
     *    - Less flexible but more stable
     */
    class Interpolator
    {
    public:
        enum Method
        {
            CUBIC_SPLINE = 0,
            RBF_THINPLATE = 1,
            POLYNOMIAL = 2
        };

        /**
         * Build interpolator from scattered IV data
         *
         * @param strikes    Strike prices (x-axis of grid)
         * @param maturities Time to maturity in years (y-axis)
         * @param ivs        Implied volatility values (z-axis)
         * @param method     Interpolation method (CUBIC_SPLINE default)
         *
         * @note Data points need not form regular grid
         * @note Missing data points (NaN) handled gracefully
         * @note Automatically normalizes strikes to moneyness m = ln(K/F)
         * @note Constructor performs expensive precomputation (fitting)
         */
        Interpolator(
            const std::vector<double> &strikes,
            const std::vector<double> &maturities,
            const std::vector<double> &ivs,
            Method method = CUBIC_SPLINE);

        /**
         * Evaluate interpolated IV at arbitrary (K, T) point
         * Single point evaluation (fast, O(1) after fitting)
         *
         * @param strike     Strike price (actual price, not moneyness)
         * @param maturity   Time to maturity in years
         * @return Interpolated IV value (as decimal, e.g., 0.20 = 20%)
         *
         * @note Extrapolation behavior:
         *   - Cubic spline: constant value beyond boundary
         *   - RBF: extrapolates based on basis function
         *   - Polynomial: linear extrapolation
         *
         * @note Returns NaN if point outside reasonable bounds
         */
        double evaluate(double strike, double maturity) const;

        /**
         * Batch evaluation at grid of points
         * Efficient vectorized evaluation
         *
         * @param strikes     Vector of strike prices (query points)
         * @param maturities  Vector of maturities (query points)
         * @return Vector of interpolated IV values
         *
         * @note Size: output vector = min(strikes.size(), maturities.size())
         * @note For grid evaluation, use evaluate_grid separately for each row
         */
        std::vector<double> evaluate_grid(
            const std::vector<double> &strikes,
            const std::vector<double> &maturities) const;

        /**
         * Compute local IV gradients (numerical derivatives)
         * Essential for Greeks computation and arbitrage checks
         *
         * @param strike    Strike price
         * @param maturity  Time to maturity
         * @return IVGradient struct with partial derivatives
         *
         * Structure:
         *   dSigma_dK    : Volatility smile slope (∂σ/∂K)
         *   dSigma_dT    : Term structure slope (∂σ/∂T)
         *   d2Sigma_dK2  : Smile convexity (∂²σ/∂K²)
         *
         * @note Computed via finite differences (central difference method)
         * @note Critical for Greeks skew computation (Vega×dσ/dS)
         * @note Used in arbitrage detection (butterfly spreads)
         */
        struct IVGradient
        {
            double dSigma_dK;   // Smile slope
            double dSigma_dT;   // Term structure slope
            double d2Sigma_dK2; // Smile convexity
        };

        IVGradient compute_gradient(double strike, double maturity) const;

    private:
        Method method_;
        std::vector<double> strikes_;
        std::vector<double> maturities_;
        std::vector<double> ivs_;
        Eigen::MatrixXd fit_matrix_;
        Eigen::VectorXd coefficients_;

        // Fitting methods (called in constructor)
        void fit_cubic_spline(); // 2D tensor-product cubic spline
        void fit_rbf();          // Thin-plate spline basis
        void fit_polynomial();   // Least-squares polynomial fit
    };

} // namespace iv_surface
