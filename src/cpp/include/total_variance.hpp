/**
 * Total Variance Framework - C++ Implementation
 * High-performance computation for total variance interpolation
 *
 * This module provides:
 * - Total variance space transformations (σ ↔ w)
 * - Monotonicity enforcement (∂w/∂T ≥ 0)
 * - Convexity enforcement (∂²C/∂K² ≥ 0)
 * - Lee moment bounds computation
 * - Surface interpolation and extrapolation
 * - Arbitrage validation
 */

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>

namespace iv_surface
{

    /**
     * Total Variance Utilities
     * High-speed batch conversion and computation
     */
    class TotalVariance
    {
    public:
        /**
         * Convert implied volatility grid to total variance grid
         * w(K,T) = σ²(K,T) * T
         *
         * Args:
         *   sigma_grid: IV grid (row-major: n_strikes x n_maturities)
         *   maturities: Time to maturity for each column (n_maturities)
         *   n_strikes: Number of strikes
         *   n_maturities: Number of maturities
         *
         * Returns:
         *   Total variance grid (row-major: n_strikes x n_maturities)
         */
        static std::vector<double> sigma_to_total_variance(
            const std::vector<double> &sigma_grid,
            const std::vector<double> &maturities,
            size_t n_strikes,
            size_t n_maturities)
        {
            if (sigma_grid.size() != n_strikes * n_maturities)
                throw std::runtime_error("sigma_grid size mismatch");
            if (maturities.size() != n_maturities)
                throw std::runtime_error("maturities size mismatch");

            std::vector<double> w_grid(n_strikes * n_maturities);

            for (size_t t = 0; t < n_maturities; ++t)
            {
                double T = maturities[t];
                for (size_t k = 0; k < n_strikes; ++k)
                {
                    size_t idx = k * n_maturities + t;
                    double sigma = sigma_grid[idx];
                    w_grid[idx] = sigma * sigma * T;
                }
            }

            return w_grid;
        }

        /**
         * Convert total variance grid to implied volatility grid
         * σ(K,T) = √(w(K,T) / T)
         *
         * Args:
         *   w_grid: Total variance grid (row-major: n_strikes x n_maturities)
         *   maturities: Time to maturity for each column (n_maturities)
         *   n_strikes: Number of strikes
         *   n_maturities: Number of maturities
         *
         * Returns:
         *   IV grid (row-major: n_strikes x n_maturities)
         */
        static std::vector<double> total_variance_to_sigma(
            const std::vector<double> &w_grid,
            const std::vector<double> &maturities,
            size_t n_strikes,
            size_t n_maturities)
        {
            if (w_grid.size() != n_strikes * n_maturities)
                throw std::runtime_error("w_grid size mismatch");
            if (maturities.size() != n_maturities)
                throw std::runtime_error("maturities size mismatch");

            std::vector<double> sigma_grid(n_strikes * n_maturities);

            for (size_t t = 0; t < n_maturities; ++t)
            {
                double T = maturities[t];
                if (T <= 0)
                    throw std::runtime_error("Invalid maturity");

                for (size_t k = 0; k < n_strikes; ++k)
                {
                    size_t idx = k * n_maturities + t;
                    double w = w_grid[idx];
                    // Clamp to avoid numerical issues
                    w = std::max(w, 1e-8);
                    sigma_grid[idx] = std::sqrt(w / T);
                }
            }

            return sigma_grid;
        }

        /**
         * Enforce monotonicity constraint: ∂w/∂T ≥ 0 for each strike
         *
         * Args:
         *   w_grid: Total variance grid (row-major: n_strikes x n_maturities)
         *   n_strikes: Number of strikes
         *   n_maturities: Number of maturities
         *
         * Returns:
         *   Corrected grid with monotonicity enforced
         */
        static std::vector<double> enforce_monotonicity(
            const std::vector<double> &w_grid,
            size_t n_strikes,
            size_t n_maturities)
        {
            if (w_grid.size() != n_strikes * n_maturities)
                throw std::runtime_error("w_grid size mismatch");

            std::vector<double> w_corrected = w_grid;

            // For each strike (row), enforce monotonicity in maturity (columns)
            for (size_t k = 0; k < n_strikes; ++k)
            {
                double prev_w = w_corrected[k * n_maturities];

                for (size_t t = 1; t < n_maturities; ++t)
                {
                    size_t idx = k * n_maturities + t;
                    if (w_corrected[idx] < prev_w)
                    {
                        // Enforce: w(T) >= w(T-1)
                        w_corrected[idx] = prev_w;
                    }
                    prev_w = w_corrected[idx];
                }
            }

            return w_corrected;
        }

        /**
         * Compute Lee moment bounds for wing extrapolation
         *
         * Args:
         *   w_atm: ATM total variance
         *
         * Returns:
         *   (left_bound, right_bound) - slopes for wing extrapolation
         */
        static std::pair<double, double> compute_lee_bounds(
            double w_atm)
        {
            if (w_atm <= 0)
                throw std::runtime_error("Invalid ATM variance");

            double sqrt_2pi = std::sqrt(2.0 * M_PI);
            double sigma_atm = std::sqrt(w_atm);

            // Lee bounds: ± √(2π) / σ_atm
            double bound_magnitude = sqrt_2pi / sigma_atm;

            return {-bound_magnitude, bound_magnitude};
        }

        /**
         * Detect calendar arbitrage violations
         * Returns count of strikes where ∂w/∂T < 0
         *
         * Args:
         *   w_grid: Total variance grid (row-major)
         *   n_strikes: Number of strikes
         *   n_maturities: Number of maturities
         *   tolerance: Violation threshold (default 1e-6)
         *
         * Returns:
         *   Number of calendar arbitrage violations
         */
        static int count_calendar_arb_violations(
            const std::vector<double> &w_grid,
            size_t n_strikes,
            size_t n_maturities,
            double tolerance = 1e-6)
        {
            int violations = 0;

            for (size_t k = 0; k < n_strikes; ++k)
            {
                for (size_t t = 1; t < n_maturities; ++t)
                {
                    size_t idx_prev = k * n_maturities + (t - 1);
                    size_t idx_curr = k * n_maturities + t;

                    double dw = w_grid[idx_curr] - w_grid[idx_prev];
                    if (dw < -tolerance)
                    {
                        violations++;
                    }
                }
            }

            return violations;
        }

        /**
         * Detect butterfly arbitrage violations (convexity)
         * Returns count of maturities where ∂²w/∂K² < 0
         *
         * Args:
         *   w_grid: Total variance grid (row-major)
         *   n_strikes: Number of strikes
         *   n_maturities: Number of maturities
         *   tolerance: Violation threshold (default 1e-6)
         *
         * Returns:
         *   Number of butterfly arbitrage violations
         */
        static int count_butterfly_arb_violations(
            const std::vector<double> &w_grid,
            size_t n_strikes,
            size_t n_maturities,
            double tolerance = 1e-6)
        {
            int violations = 0;

            // Check convexity in strike dimension for each maturity
            for (size_t t = 0; t < n_maturities; ++t)
            {
                for (size_t k = 1; k < n_strikes - 1; ++k)
                {
                    double w_left = w_grid[(k - 1) * n_maturities + t];
                    double w_mid = w_grid[k * n_maturities + t];
                    double w_right = w_grid[(k + 1) * n_maturities + t];

                    // Second derivative: w_right - 2*w_mid + w_left
                    double d2w = w_right - 2.0 * w_mid + w_left;

                    if (d2w < -tolerance)
                    {
                        violations++;
                    }
                }
            }

            return violations;
        }

        /**
         * Batch arbitrage validation
         *
         * Returns: (calendar_violations, butterfly_violations)
         */
        static std::pair<int, int> validate_arbitrage_free(
            const std::vector<double> &w_grid,
            size_t n_strikes,
            size_t n_maturities,
            double tolerance = 1e-6)
        {
            return {
                count_calendar_arb_violations(w_grid, n_strikes, n_maturities, tolerance),
                count_butterfly_arb_violations(w_grid, n_strikes, n_maturities, tolerance)};
        }
    };

    /**
     * Interpolation utilities for total variance
     */
    class TotalVarianceInterpolation
    {
    public:
        /**
         * Linear interpolation for a single strike across maturities
         * Preserves monotonicity
         *
         * Args:
         *   t_input: Input maturities (n_input)
         *   w_input: Total variance at input maturities (n_input)
         *   t_output: Output maturities (n_output)
         *
         * Returns:
         *   Interpolated total variance (n_output)
         */
        static std::vector<double> interpolate_maturities_linear(
            const std::vector<double> &t_input,
            const std::vector<double> &w_input,
            const std::vector<double> &t_output)
        {
            if (t_input.size() != w_input.size())
                throw std::runtime_error("Input size mismatch");
            if (t_input.size() < 2)
                throw std::runtime_error("Need at least 2 input points");

            std::vector<double> w_output;
            w_output.reserve(t_output.size());

            for (double t : t_output)
            {
                // Find bracketing interval
                auto it = std::lower_bound(t_input.begin(), t_input.end(), t);

                if (it == t_input.end())
                {
                    // Extrapolate beyond last point
                    size_t n = t_input.size();
                    double t1 = t_input[n - 2];
                    double t2 = t_input[n - 1];
                    double w1 = w_input[n - 2];
                    double w2 = w_input[n - 1];

                    double w = w2 + (w2 - w1) / (t2 - t1) * (t - t2);
                    w_output.push_back(std::max(w, w2)); // Don't decrease
                }
                else if (it == t_input.begin())
                {
                    // Extrapolate before first point
                    w_output.push_back(w_input[0]);
                }
                else
                {
                    // Interpolate
                    size_t idx2 = std::distance(t_input.begin(), it);
                    size_t idx1 = idx2 - 1;

                    double t1 = t_input[idx1];
                    double t2 = t_input[idx2];
                    double w1 = w_input[idx1];
                    double w2 = w_input[idx2];

                    double alpha = (t - t1) / (t2 - t1);
                    double w = w1 + alpha * (w2 - w1);
                    w_output.push_back(w);
                }
            }

            return w_output;
        }

        /**
         * Extract statistics from total variance grid
         */
        struct GridStatistics
        {
            double mean_variance;
            double min_variance;
            double max_variance;
            double std_dev_variance;
        };

        static GridStatistics compute_statistics(
            const std::vector<double> &w_grid,
            size_t n_strikes,
            size_t n_maturities)
        {
            if (w_grid.empty())
                throw std::runtime_error("Empty grid");

            double sum = std::accumulate(w_grid.begin(), w_grid.end(), 0.0);
            double mean = sum / w_grid.size();

            double min_val = *std::min_element(w_grid.begin(), w_grid.end());
            double max_val = *std::max_element(w_grid.begin(), w_grid.end());

            double sum_sq_dev = 0.0;
            for (double w : w_grid)
            {
                double dev = w - mean;
                sum_sq_dev += dev * dev;
            }
            double std_dev = std::sqrt(sum_sq_dev / w_grid.size());

            return {mean, min_val, max_val, std_dev};
        }
    };

} // namespace iv_surface
