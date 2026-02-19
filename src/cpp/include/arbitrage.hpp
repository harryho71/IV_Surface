#pragma once
#include <vector>
#include <string>
#include <optional>

namespace iv_surface
{

    /**
     * Arbitrage-free surface validation
     *
     * Implements fundamental no-arbitrage checks required for
     * production-grade IV surfaces in investment banks:
     *
     * 1. Butterfly arbitrage: Call prices convex in strike
     * 2. Calendar arbitrage: Call prices non-decreasing in time
     * 3. Total variance monotonicity: w = σ²T non-decreasing in T
     */
    class ArbitrageChecker
    {
    public:
        /**
         * Violation record
         */
        struct Violation
        {
            std::string type;             // "butterfly", "calendar", "total_variance"
            double value;                 // Magnitude of violation
            double tolerance;             // Tolerance threshold
            std::vector<double> location; // Strike(s) and maturity
            std::string severity;         // "minor", "moderate", "severe"
        };

        /**
         * Validation report
         */
        struct Report
        {
            bool passed;
            int total_checks;
            std::vector<Violation> violations;
            bool butterfly_pass;
            bool calendar_pass;
            bool total_variance_pass;
        };

        /**
         * Check butterfly arbitrage: call prices must be convex in strike
         *
         * Condition: C(K-h) - 2*C(K) + C(K+h) >= -tolerance
         *
         * @param strikes Sorted array of strike prices
         * @param call_prices Corresponding call prices
         * @param maturity Time to maturity
         * @param tolerance Acceptable tolerance (default: 1e-6)
         * @param bid_ask_spreads Optional spreads for adaptive tolerance
         * @return Vector of violations (empty if passed)
         */
        static std::vector<Violation> check_butterfly(
            const std::vector<double> &strikes,
            const std::vector<double> &call_prices,
            double maturity,
            double tolerance = 1e-6,
            const std::vector<double> *bid_ask_spreads = nullptr);

        /**
         * Check calendar arbitrage: call prices must be non-decreasing in time
         *
         * Condition: C(K, T₂) >= C(K, T₁) - tolerance for T₂ > T₁
         *
         * @param strike Fixed strike price
         * @param maturities Sorted array of maturities
         * @param call_prices Corresponding call prices at fixed strike
         * @param tolerance Acceptable tolerance (default: 1e-6)
         * @param bid_ask_spreads Optional spreads for adaptive tolerance
         * @return Vector of violations (empty if passed)
         */
        static std::vector<Violation> check_calendar(
            double strike,
            const std::vector<double> &maturities,
            const std::vector<double> &call_prices,
            double tolerance = 1e-6,
            const std::vector<double> *bid_ask_spreads = nullptr);

        /**
         * Check total variance monotonicity: w(k,T) = σ²T must increase with T
         *
         * Condition: w(k, T₂) >= w(k, T₁) - tolerance for T₂ > T₁
         *
         * @param strikes Array of strike prices
         * @param maturities Array of maturities
         * @param implied_vols Flattened array of IVs (row-major: strikes × maturities)
         * @param tolerance Acceptable tolerance (default: 1e-6)
         * @return Vector of violations (empty if passed)
         */
        static std::vector<Violation> check_total_variance(
            const std::vector<double> &strikes,
            const std::vector<double> &maturities,
            const std::vector<double> &implied_vols,
            double tolerance = 1e-6);

        /**
         * Complete surface validation
         *
         * @param strikes Array of strike prices
         * @param maturities Array of maturities
         * @param implied_vols Flattened IVs (row-major: strikes × maturities)
         * @param spot Current spot price
         * @param rate Risk-free rate
         * @param dividend_yield Dividend yield (default: 0)
         * @param tolerance Absolute tolerance (default: 1e-6)
         * @param bid_ask_spreads Optional spreads (default: nullptr)
         * @return Complete validation report
         */
        static Report validate_surface(
            const std::vector<double> &strikes,
            const std::vector<double> &maturities,
            const std::vector<double> &implied_vols,
            double spot,
            double rate,
            double dividend_yield = 0.0,
            double tolerance = 1e-6,
            const std::vector<double> *bid_ask_spreads = nullptr);

    private:
        static std::string classify_severity(double violation_magnitude, double tolerance);
    };

} // namespace iv_surface
