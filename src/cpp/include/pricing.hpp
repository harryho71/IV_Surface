#pragma once
#include <cmath>
#include <vector>
#include <array>

namespace iv_surface
{

    /**
     * Black-Scholes Pricing and Greeks Calculation
     * Optimized for batch processing with SIMD vectorization
     *
     * All Greeks computed analytically:
     * - Delta: ∂C/∂S (directional exposure)
     * - Gamma: ∂²C/∂S² (convexity risk)
     * - Vega: ∂C/∂σ (volatility exposure, critical for IV surface)
     * - Theta: -∂C/∂T (time decay)
     * - Rho: ∂C/∂r (interest rate exposure)
     */
    class BlackScholesEngine
    {
    public:
        struct Greeks
        {
            double price;
            double delta;
            double gamma;
            double vega;
            double theta;
            double rho;
        };

        /**
         * Calculate option price and Greeks (single option)
         *
         * @param S     Spot price (underlying asset price)
         * @param K     Strike price (exercise price)
         * @param T     Time to expiration in years (days/365)
         * @param r     Risk-free interest rate (annualized)
         * @param sigma Volatility (annualized, as decimal: 0.2 = 20%)
         * @param q     Dividend yield (annualized)
         * @param is_call True for call option, false for put
         * @return Greeks struct with all computed values
         *
         * @note Handles edge cases: T→0, very deep ITM/OTM
         * @note Use scipy.stats.norm for numerical stability
         */
        static Greeks calculate_greeks(
            double S, double K, double T, double r,
            double sigma, double q, bool is_call);

        /**
         * Batch Greeks calculation (vectorized, parallel execution)
         * Used for fast Greeks computation across portfolios
         *
         * @param spots      Vector of spot prices
         * @param strikes    Vector of strike prices
         * @param times      Vector of times to expiration (years)
         * @param r          Risk-free rate (scalar, same for all)
         * @param sigmas     Vector of implied volatilities
         * @param q          Dividend yield (scalar)
         * @param is_call    True for all calls, false for all puts
         * @return Vector of Greeks structs
         *
         * @note Vectorization at C++ level eliminates Python loop overhead
         * @note OpenMP parallelization across options
         * @note ~10-100x faster than pure Python
         */
        static std::vector<Greeks> calculate_batch(
            const std::vector<double> &spots,
            const std::vector<double> &strikes,
            const std::vector<double> &times,
            double r,
            const std::vector<double> &sigmas,
            double q,
            bool is_call);

        /**
         * Simple call price calculation (no Greeks)
         * Convenience function for arbitrage checking
         *
         * @param S     Spot price
         * @param K     Strike price
         * @param T     Time to expiration (years)
         * @param r     Risk-free rate
         * @param sigma Implied volatility
         * @param q     Dividend yield (default 0.0)
         * @return Call option price
         */
        static double call_price(
            double S, double K, double T, double r,
            double sigma, double q = 0.0);

    private:
        // Standard normal probability density function
        static double norm_pdf(double x);

        // Standard normal cumulative distribution function
        static double norm_cdf(double x);

        // Helper: compute d1 and d2 from Black-Scholes formula
        static std::array<double, 2> calculate_d1_d2(
            double S, double K, double T, double r,
            double sigma, double q);
    };

} // namespace iv_surface
