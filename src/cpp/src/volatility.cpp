#include "volatility.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <mutex>
#include <vector>

namespace iv_surface
{

    namespace
    {
        constexpr double kEps = 1e-12;
        constexpr double kTwoPi = 6.2831853071795864769;

        std::mutex g_log_mutex;

        bool is_call_option(char option_type)
        {
            return option_type == 'C' || option_type == 'c';
        }

        void log_failure(const char *reason, char option_type, double S, double K, double T, double P)
        {
            std::lock_guard<std::mutex> lock(g_log_mutex);
            std::cerr << "IVSolver failure (" << reason << ") "
                      << "type=" << option_type << " S=" << S << " K=" << K
                      << " T=" << T << " P=" << P << '\n';
        }

        void log_batch_failure(const char *reason)
        {
            std::lock_guard<std::mutex> lock(g_log_mutex);
            std::cerr << "IVSolver batch failure (" << reason << ")\n";
        }
    } // namespace

    IVSolver::SolverConfig IVSolver::default_config()
    {
        return SolverConfig{
            .tolerance = 1e-6,
            .max_iterations = 100,
            .min_iv = 0.001,
            .max_iv = 3.0,
            .algorithm = 0};
    }

    double IVSolver::objective_function(
        char option_type, double S, double K, double T,
        double r, double sigma, double market_price, double q)
    {
        const auto greeks = BlackScholesEngine::calculate_greeks(
            S, K, T, r, sigma, q, is_call_option(option_type));
        return greeks.price - market_price;
    }

    std::optional<double> IVSolver::brent_method(
        char option_type, double S, double K, double T,
        double r, double market_price, double q,
        const SolverConfig &config)
    {
        double a = config.min_iv;
        double b = config.max_iv;

        double fa = objective_function(option_type, S, K, T, r, a, market_price, q);
        double fb = objective_function(option_type, S, K, T, r, b, market_price, q);

        if (!std::isfinite(fa) || !std::isfinite(fb))
        {
            log_failure("brent_nan", option_type, S, K, T, market_price);
            return std::nullopt;
        }

        if (fa * fb > 0.0)
        {
            const double expand = 0.25 * (config.max_iv - config.min_iv);
            a = std::max(config.min_iv - expand, kEps);
            b = config.max_iv + expand;
            fa = objective_function(option_type, S, K, T, r, a, market_price, q);
            fb = objective_function(option_type, S, K, T, r, b, market_price, q);
            if (!std::isfinite(fa) || !std::isfinite(fb) || fa * fb > 0.0)
            {
                log_failure("brent_no_bracket", option_type, S, K, T, market_price);
                return std::nullopt;
            }
        }

        if (std::abs(fa) < std::abs(fb))
        {
            std::swap(a, b);
            std::swap(fa, fb);
        }

        double c = a;
        double fc = fa;
        double d = 0.0;
        double s = 0.0;
        bool mflag = true;

        for (int iter = 0; iter < config.max_iterations; ++iter)
        {
            if (std::abs(fb) < config.tolerance)
            {
                return b;
            }

            if (fa != fc && fb != fc)
            {
                s = a * fb * fc / ((fa - fb) * (fa - fc)) + b * fa * fc / ((fb - fa) * (fb - fc)) + c * fa * fb / ((fc - fa) * (fc - fb));
            }
            else
            {
                s = b - fb * (b - a) / (fb - fa);
            }

            const double condition1 = (s < (3.0 * a + b) * 0.25) || (s > b);
            const double condition2 = mflag && (std::abs(s - b) >= std::abs(b - c) * 0.5);
            const double condition3 = !mflag && (std::abs(s - b) >= std::abs(c - d) * 0.5);
            const double condition4 = mflag && (std::abs(b - c) < config.tolerance);
            const double condition5 = !mflag && (std::abs(c - d) < config.tolerance);

            if (condition1 || condition2 || condition3 || condition4 || condition5)
            {
                s = 0.5 * (a + b);
                mflag = true;
            }
            else
            {
                mflag = false;
            }

            const double fs = objective_function(option_type, S, K, T, r, s, market_price, q);
            if (!std::isfinite(fs))
            {
                log_failure("brent_nan", option_type, S, K, T, market_price);
                return std::nullopt;
            }

            d = c;
            c = b;
            fc = fb;

            if (fa * fs < 0.0)
            {
                b = s;
                fb = fs;
            }
            else
            {
                a = s;
                fa = fs;
            }

            if (std::abs(fa) < std::abs(fb))
            {
                std::swap(a, b);
                std::swap(fa, fb);
            }

            if (std::abs(b - a) < config.tolerance * std::max(std::abs(b), 1.0))
            {
                return 0.5 * (a + b);
            }
        }

        log_failure("brent_no_convergence", option_type, S, K, T, market_price);
        return std::nullopt;
    }

    std::optional<double> IVSolver::newton_raphson(
        char option_type, double S, double K, double T,
        double r, double market_price, double q,
        const SolverConfig &config)
    {
        if (S <= 0.0 || K <= 0.0 || T <= 0.0 || market_price <= 0.0)
        {
            log_failure("invalid_inputs", option_type, S, K, T, market_price);
            return std::nullopt;
        }

        const double sigma0 = std::sqrt(kTwoPi / T) * (market_price / S);
        if (sigma0 < config.min_iv || sigma0 > config.max_iv)
        {
            return brent_method(option_type, S, K, T, r, market_price, q, config);
        }

        double sigma = std::clamp(sigma0, config.min_iv, config.max_iv);

        for (int iter = 0; iter < config.max_iterations; ++iter)
        {
            const auto greeks = BlackScholesEngine::calculate_greeks(
                S, K, T, r, sigma, q, is_call_option(option_type));

            if (!std::isfinite(greeks.price) || !std::isfinite(greeks.vega))
            {
                return brent_method(option_type, S, K, T, r, market_price, q, config);
            }

            const double price_diff = greeks.price - market_price;
            if (std::abs(price_diff) < config.tolerance)
            {
                return sigma;
            }

            const double vega = greeks.vega * 100.0;
            if (!std::isfinite(vega) || std::abs(vega) < kEps)
            {
                return brent_method(option_type, S, K, T, r, market_price, q, config);
            }

            const double sigma_next = sigma - price_diff / vega;
            if (!std::isfinite(sigma_next))
            {
                return brent_method(option_type, S, K, T, r, market_price, q, config);
            }

            if (sigma_next < config.min_iv || sigma_next > config.max_iv)
            {
                return brent_method(option_type, S, K, T, r, market_price, q, config);
            }

            if (std::abs(sigma_next - sigma) / std::max(sigma, kEps) < config.tolerance)
            {
                return sigma_next;
            }

            sigma = sigma_next;
        }

        log_failure("newton_no_convergence", option_type, S, K, T, market_price);
        return std::nullopt;
    }

    std::optional<double> IVSolver::solve(
        char option_type,
        double S, double K, double T, double r,
        double market_price, double q,
        const SolverConfig *config)
    {
        // Use default config if nullptr
        SolverConfig default_cfg = default_config();
        const SolverConfig &cfg = (config != nullptr) ? *config : default_cfg;

        if (!is_call_option(option_type) && option_type != 'P' && option_type != 'p')
        {
            log_failure("invalid_option_type", option_type, S, K, T, market_price);
            return std::nullopt;
        }

        switch (cfg.algorithm)
        {
        case 1:
            return brent_method(option_type, S, K, T, r, market_price, q, cfg);
        case 2:
        {
            double a = cfg.min_iv;
            double b = cfg.max_iv;
            double fa = objective_function(option_type, S, K, T, r, a, market_price, q);
            double fb = objective_function(option_type, S, K, T, r, b, market_price, q);

            if (!std::isfinite(fa) || !std::isfinite(fb) || fa * fb > 0.0)
            {
                log_failure("bisection_no_bracket", option_type, S, K, T, market_price);
                return std::nullopt;
            }

            for (int iter = 0; iter < cfg.max_iterations; ++iter)
            {
                const double mid = 0.5 * (a + b);
                const double fm = objective_function(option_type, S, K, T, r, mid, market_price, q);
                if (!std::isfinite(fm))
                {
                    log_failure("bisection_nan", option_type, S, K, T, market_price);
                    return std::nullopt;
                }
                if (std::abs(fm) < cfg.tolerance || (b - a) / std::max(mid, 1.0) < cfg.tolerance)
                {
                    return mid;
                }
                if (fa * fm <= 0.0)
                {
                    b = mid;
                    fb = fm;
                }
                else
                {
                    a = mid;
                    fa = fm;
                }
            }

            log_failure("bisection_no_convergence", option_type, S, K, T, market_price);
            return std::nullopt;
        }
        default:
            return newton_raphson(option_type, S, K, T, r, market_price, q, cfg);
        }
    }

    std::vector<std::optional<double>> IVSolver::solve_batch(
        const std::vector<char> &option_types,
        const std::vector<double> &spots,
        const std::vector<double> &strikes,
        const std::vector<double> &times,
        double r,
        const std::vector<double> &market_prices,
        double q,
        const SolverConfig *config)
    {
        const std::vector<char> option_types_local = option_types;
        const std::vector<double> spots_local = spots;
        const std::vector<double> strikes_local = strikes;
        const std::vector<double> times_local = times;
        const std::vector<double> market_prices_local = market_prices;

        const std::size_t n = option_types_local.size();
        const std::size_t min_n = std::min(
            {option_types_local.size(), spots_local.size(), strikes_local.size(), times_local.size(), market_prices_local.size()});

        std::vector<std::optional<double>> results(n, std::nullopt);

        if (min_n != n)
        {
            log_batch_failure("input_size_mismatch");
        }

#pragma omp parallel for schedule(dynamic, 1000)
        for (int i = 0; i < static_cast<int>(min_n); ++i)
        {
            results[i] = solve(
                option_types_local[static_cast<std::size_t>(i)],
                spots_local[static_cast<std::size_t>(i)],
                strikes_local[static_cast<std::size_t>(i)],
                times_local[static_cast<std::size_t>(i)],
                r,
                market_prices_local[static_cast<std::size_t>(i)],
                q,
                config);
        }

        return results;
    }

} // namespace iv_surface
