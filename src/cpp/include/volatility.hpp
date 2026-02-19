#pragma once
#include "pricing.hpp"
#include <optional>

namespace iv_surface
{

    /**
     * Implied Volatility Solver using multiple algorithms
     *
     * Core Problem: Given option market price P_market, solve for σ such that:
     *   BS_Price(σ) = P_market
     *
     * Non-linear root-finding problem with known derivative (Vega).
     * Three algorithms for robustness:
     *
     * 1. Newton-Raphson (Algorithm 0): Fast, needs Vega
     *    σ_{n+1} = σ_n - (BS_Price(σ_n) - P_market) / Vega(σ_n)
     *    Converges in 2-5 iterations typically
     *
     * 2. Brent's Method (Algorithm 1): Robust, no derivative
     *    Combines bisection, secant, inverse quadratic interpolation
     *    Guaranteed convergence, ~10-15 iterations
     *
     * 3. Bisection (Algorithm 2): Fallback for pathological cases
     *    Linear convergence, guaranteed but slow
     */
    class IVSolver
    {
    public:
        struct SolverConfig
        {
            double tolerance = 1e-6;  // Target accuracy: 0.0001%
            int max_iterations = 100; // Upper bound on iterations
            double min_iv = 0.001;    // Lower bound: 0.1% (1 basis point)
            double max_iv = 3.0;      // Upper bound: 300% (extreme volatility)
            int algorithm = 0;        // 0=Newton-Raphson, 1=Brent, 2=Bisection
        };

        // Get default configuration
        static SolverConfig default_config();

        /**
         * Solve for implied volatility (single option)
         *
         * @param option_type 'C' for call option, 'P' for put option
         * @param S           Spot price of underlying
         * @param K           Strike price
         * @param T           Time to expiration (years, must be > 0)
         * @param r           Risk-free rate (annualized)
         * @param market_price Observed market price of option
         * @param q           Dividend yield (annualized, default 0)
         * @param config      Solver configuration (algorithm, tolerance, bounds)
         * @return std::optional<double> IV value if converged, nullopt otherwise
         *
         * @note Returns nullopt if:
         *   - No solution exists in [min_iv, max_iv]
         *   - Market price violates bounds (< intrinsic, > spot for call)
         *   - Convergence failed after max_iterations
         *
         * @note Put-Call Parity: If call IV fails, use put via synthetic call
         *   C - P = S*exp(-q*T) - K*exp(-r*T)
         *
         * @note Algorithm selection:
         *   - Newton-Raphson: Fast, preferred for liquid options
         *   - Brent: Use if Newton fails, more robust
         *   - Bisection: Last resort (slow but always converges)
         */
        static std::optional<double> solve(
            char option_type,
            double S, double K, double T, double r,
            double market_price, double q = 0.0,
            const SolverConfig *config = nullptr);

        /**
         * Batch IV calculation with parallel execution
         * Process 1000+ options simultaneously at C++ level
         *
         * @param option_types Vector of 'C' or 'P' flags
         * @param spots        Vector of spot prices
         * @param strikes      Vector of strike prices
         * @param times        Vector of times to expiration
         * @param r            Risk-free rate (scalar)
         * @param market_prices Vector of observed option prices
         * @param q            Dividend yield (scalar)
         * @param config       Solver configuration
         * @return Vector of std::optional<double> IVs (nullopt for failures)
         *
         * @note OpenMP parallelization: each option solved independently
         * @note #pragma omp parallel for distribution strategy
         * @note 10-50x speedup vs sequential for large batches
         *
         * @note Error Handling:
         *   - Failed conversions return nullopt (not exception)
         *   - Continue processing despite failures
         *   - Log failures to stderr with option details
         */
        static std::vector<std::optional<double>> solve_batch(
            const std::vector<char> &option_types,
            const std::vector<double> &spots,
            const std::vector<double> &strikes,
            const std::vector<double> &times,
            double r,
            const std::vector<double> &market_prices,
            double q = 0.0,
            const SolverConfig *config = nullptr);

    private:
        // Newton-Raphson implementation
        static std::optional<double> newton_raphson(
            char option_type, double S, double K, double T,
            double r, double market_price, double q,
            const SolverConfig &config);

        // Brent's method implementation
        static std::optional<double> brent_method(
            char option_type, double S, double K, double T,
            double r, double market_price, double q,
            const SolverConfig &config);

        // Objective function: BS_Price(σ) - P_market
        static double objective_function(
            char option_type, double S, double K, double T,
            double r, double sigma, double market_price, double q);
    };

} // namespace iv_surface
