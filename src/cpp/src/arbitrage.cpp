#include "arbitrage.hpp"
#include "pricing.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace iv_surface
{

    std::string ArbitrageChecker::classify_severity(double violation_magnitude, double tolerance)
    {
        if (tolerance <= 0.0)
        {
            return "severe";
        }

        double ratio = violation_magnitude / tolerance;

        if (ratio < 2.0)
        {
            return "minor";
        }
        else if (ratio < 5.0)
        {
            return "moderate";
        }
        else
        {
            return "severe";
        }
    }

    std::vector<ArbitrageChecker::Violation> ArbitrageChecker::check_butterfly(
        const std::vector<double> &strikes,
        const std::vector<double> &call_prices,
        double maturity,
        double tolerance,
        const std::vector<double> *bid_ask_spreads)
    {
        std::vector<Violation> violations;

        if (strikes.size() < 3 || call_prices.size() < 3)
        {
            return violations;
        }

        const std::size_t n = std::min(strikes.size(), call_prices.size());

        // Check each consecutive triplet
        for (std::size_t i = 1; i + 1 < n; ++i)
        {
            const double K_left = strikes[i - 1];
            const double K_mid = strikes[i];
            const double K_right = strikes[i + 1];

            const double C_left = call_prices[i - 1];
            const double C_mid = call_prices[i];
            const double C_right = call_prices[i + 1];

            const double h_left = K_mid - K_left;
            const double h_right = K_right - K_mid;

            if (h_left <= 0.0 || h_right <= 0.0)
            {
                continue;
            }

            // Butterfly value
            double butterfly;
            if (std::abs(h_left - h_right) < 1e-6)
            {
                // Equal spacing: standard butterfly
                butterfly = C_left - 2.0 * C_mid + C_right;
            }
            else
            {
                // Unequal spacing: linear interpolation check
                const double alpha = h_right / (h_left + h_right);
                const double C_mid_linear = alpha * C_left + (1.0 - alpha) * C_right;
                butterfly = C_mid_linear - C_mid;
            }

            // Determine tolerance
            double tol = tolerance;
            if (bid_ask_spreads != nullptr && bid_ask_spreads->size() > i)
            {
                tol = 0.5 * (*bid_ask_spreads)[i];
            }

            // Check violation
            if (butterfly < -tol)
            {
                Violation v;
                v.type = "butterfly";
                v.value = butterfly;
                v.tolerance = tol;
                v.location = {K_left, K_mid, K_right, maturity};
                v.severity = classify_severity(std::abs(butterfly), tol);
                violations.push_back(v);
            }
        }

        return violations;
    }

    std::vector<ArbitrageChecker::Violation> ArbitrageChecker::check_calendar(
        double strike,
        const std::vector<double> &maturities,
        const std::vector<double> &call_prices,
        double tolerance,
        const std::vector<double> *bid_ask_spreads)
    {
        std::vector<Violation> violations;

        if (maturities.size() < 2 || call_prices.size() < 2)
        {
            return violations;
        }

        const std::size_t n = std::min(maturities.size(), call_prices.size());

        // Check each consecutive pair
        for (std::size_t i = 0; i + 1 < n; ++i)
        {
            const double T1 = maturities[i];
            const double T2 = maturities[i + 1];
            const double C1 = call_prices[i];
            const double C2 = call_prices[i + 1];

            // Calendar spread value (should be non-negative)
            const double calendar = C2 - C1;

            // Determine tolerance
            double tol = tolerance;
            if (bid_ask_spreads != nullptr && bid_ask_spreads->size() > i + 1)
            {
                tol = 0.5 * ((*bid_ask_spreads)[i] + (*bid_ask_spreads)[i + 1]);
            }

            // Check violation
            if (calendar < -tol)
            {
                Violation v;
                v.type = "calendar";
                v.value = calendar;
                v.tolerance = tol;
                v.location = {strike, T1, T2};
                v.severity = classify_severity(std::abs(calendar), tol);
                violations.push_back(v);
            }
        }

        return violations;
    }

    std::vector<ArbitrageChecker::Violation> ArbitrageChecker::check_total_variance(
        const std::vector<double> &strikes,
        const std::vector<double> &maturities,
        const std::vector<double> &implied_vols,
        double tolerance)
    {
        std::vector<Violation> violations;

        if (strikes.empty() || maturities.empty())
        {
            return violations;
        }

        const std::size_t n_strikes = strikes.size();
        const std::size_t n_maturities = maturities.size();

        if (implied_vols.size() != n_strikes * n_maturities)
        {
            return violations;
        }

        // Check monotonicity for each strike
        for (std::size_t i = 0; i < n_strikes; ++i)
        {
            const double K = strikes[i];

            // Check each consecutive maturity pair
            for (std::size_t j = 0; j + 1 < n_maturities; ++j)
            {
                const double T1 = maturities[j];
                const double T2 = maturities[j + 1];

                // Access IVs (row-major indexing)
                const double sigma1 = implied_vols[i * n_maturities + j];
                const double sigma2 = implied_vols[i * n_maturities + j + 1];

                if (!std::isfinite(sigma1) || !std::isfinite(sigma2))
                {
                    continue;
                }

                // Compute total variance w = σ²T
                const double w1 = (T1 > 0.0) ? sigma1 * sigma1 * T1 : 0.0;
                const double w2 = (T2 > 0.0) ? sigma2 * sigma2 * T2 : 0.0;

                // Should have w2 >= w1
                const double diff = w2 - w1;

                // Tolerance scales with magnitude
                const double tol = std::max(tolerance, tolerance * std::abs(w1));

                if (diff < -tol)
                {
                    Violation v;
                    v.type = "total_variance";
                    v.value = diff;
                    v.tolerance = tol;
                    v.location = {K, T1, T2};
                    v.severity = classify_severity(std::abs(diff), tol);
                    violations.push_back(v);
                }
            }
        }

        return violations;
    }

    ArbitrageChecker::Report ArbitrageChecker::validate_surface(
        const std::vector<double> &strikes,
        const std::vector<double> &maturities,
        const std::vector<double> &implied_vols,
        double spot,
        double rate,
        double dividend_yield,
        double tolerance,
        const std::vector<double> *bid_ask_spreads)
    {
        Report report;
        report.passed = true;
        report.total_checks = 0;
        report.butterfly_pass = true;
        report.calendar_pass = true;
        report.total_variance_pass = true;

        const std::size_t n_strikes = strikes.size();
        const std::size_t n_maturities = maturities.size();

        if (implied_vols.size() != n_strikes * n_maturities)
        {
            return report;
        }

        // 1. Butterfly checks (for each maturity)
        for (std::size_t j = 0; j < n_maturities; ++j)
        {
            const double T = maturities[j];
            if (T <= 0.0)
            {
                continue;
            }

            // Convert IVs to call prices
            std::vector<double> call_prices(n_strikes);
            for (std::size_t i = 0; i < n_strikes; ++i)
            {
                const double K = strikes[i];
                const double sigma = implied_vols[i * n_maturities + j];

                if (sigma > 0.0 && std::isfinite(sigma))
                {
                    call_prices[i] = BlackScholesEngine::call_price(
                        spot, K, T, rate, sigma, dividend_yield);
                }
                else
                {
                    call_prices[i] = 0.0;
                }
            }

            // Bid-ask spreads for this maturity (if provided)
            const std::vector<double> *spreads_at_T = nullptr;
            std::vector<double> spreads_slice;
            if (bid_ask_spreads != nullptr)
            {
                spreads_slice.resize(n_strikes);
                for (std::size_t i = 0; i < n_strikes; ++i)
                {
                    spreads_slice[i] = (*bid_ask_spreads)[i * n_maturities + j];
                }
                spreads_at_T = &spreads_slice;
            }

            auto viols = check_butterfly(strikes, call_prices, T, tolerance, spreads_at_T);
            report.total_checks += static_cast<int>(std::max<std::size_t>(1, n_strikes - 2));
            if (!viols.empty())
            {
                report.butterfly_pass = false;
                report.violations.insert(report.violations.end(), viols.begin(), viols.end());
            }
        }

        // 2. Calendar checks (for each strike)
        for (std::size_t i = 0; i < n_strikes; ++i)
        {
            const double K = strikes[i];

            // Extract call prices at this strike
            std::vector<double> call_prices(n_maturities);
            for (std::size_t j = 0; j < n_maturities; ++j)
            {
                const double T = maturities[j];
                const double sigma = implied_vols[i * n_maturities + j];

                if (T > 0.0 && sigma > 0.0 && std::isfinite(sigma))
                {
                    call_prices[j] = BlackScholesEngine::call_price(
                        spot, K, T, rate, sigma, dividend_yield);
                }
                else
                {
                    call_prices[j] = 0.0;
                }
            }

            // Bid-ask spreads for this strike (if provided)
            const std::vector<double> *spreads_at_K = nullptr;
            std::vector<double> spreads_slice;
            if (bid_ask_spreads != nullptr)
            {
                spreads_slice.resize(n_maturities);
                for (std::size_t j = 0; j < n_maturities; ++j)
                {
                    spreads_slice[j] = (*bid_ask_spreads)[i * n_maturities + j];
                }
                spreads_at_K = &spreads_slice;
            }

            auto viols = check_calendar(K, maturities, call_prices, tolerance, spreads_at_K);
            report.total_checks += static_cast<int>(std::max<std::size_t>(1, n_maturities - 1));
            if (!viols.empty())
            {
                report.calendar_pass = false;
                report.violations.insert(report.violations.end(), viols.begin(), viols.end());
            }
        }

        // 3. Total variance monotonicity
        auto viols = check_total_variance(strikes, maturities, implied_vols, tolerance);
        report.total_checks += static_cast<int>(n_strikes * std::max<std::size_t>(1, n_maturities - 1));
        if (!viols.empty())
        {
            report.total_variance_pass = false;
            report.violations.insert(report.violations.end(), viols.begin(), viols.end());
        }

        // Final verdict
        report.passed = report.butterfly_pass && report.calendar_pass && report.total_variance_pass;

        return report;
    }

} // namespace iv_surface
