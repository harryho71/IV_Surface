#include "volatility.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace
{
    bool approx_equal(double a, double b, double tol)
    {
        return std::abs(a - b) <= tol;
    }

    double bs_price(char option_type, double S, double K, double T, double r, double sigma, double q)
    {
        const bool is_call = (option_type == 'C' || option_type == 'c');
        const auto greeks = iv_surface::BlackScholesEngine::calculate_greeks(S, K, T, r, sigma, q, is_call);
        return greeks.price;
    }
}

int main()
{
    using iv_surface::IVSolver;

    // 1. Known IV recovery
    {
        const double sigma = 0.20;
        const double q = 0.0;
        struct Case
        {
            double S;
            double K;
            double T;
            double r;
        };
        const std::vector<Case> cases = {
            {100.0, 100.0, 1.0, 0.05},
            {120.0, 110.0, 0.5, 0.02},
            {80.0, 100.0, 2.0, 0.01},
            {150.0, 140.0, 0.25, 0.03}};

        for (const auto &c : cases)
        {
            const double price = bs_price('C', c.S, c.K, c.T, c.r, sigma, q);
            IVSolver::SolverConfig cfg;
            cfg.algorithm = 0;
            cfg.tolerance = 1e-8;
            const auto iv = IVSolver::solve('C', c.S, c.K, c.T, c.r, price, q, cfg);
            assert(iv.has_value());
            assert(std::abs(*iv - sigma) < 1e-6);
        }
    }

    // 2. Convergence verification (estimate iterations via max_iterations sweep)
    {
        const double S = 100.0;
        const double K = 100.0;
        const double T = 1.0;
        const double r = 0.05;
        const double q = 0.0;
        const double sigma = 0.2;
        const double price = bs_price('C', S, K, T, r, sigma, q);

        int worst_required = 0;
        for (int max_iter = 1; max_iter <= 10; ++max_iter)
        {
            IVSolver::SolverConfig cfg;
            cfg.algorithm = 0;
            cfg.tolerance = 1e-8;
            cfg.max_iterations = max_iter;
            const auto iv = IVSolver::solve('C', S, K, T, r, price, q, cfg);
            if (iv.has_value())
            {
                worst_required = max_iter;
                break;
            }
        }
        std::cout << "Estimated iterations to converge: " << worst_required << "\n";
        assert(worst_required > 0);
        assert(worst_required <= 5);
    }

    // 3. Algorithm comparison
    {
        const double S = 100.0;
        const double K = 100.0;
        const double T = 1.0;
        const double r = 0.05;
        const double q = 0.0;
        const double sigma = 0.2;
        const double price = bs_price('C', S, K, T, r, sigma, q);

        IVSolver::SolverConfig cfg;
        cfg.tolerance = 1e-8;

        cfg.algorithm = 0;
        const auto iv_newton = IVSolver::solve('C', S, K, T, r, price, q, cfg);
        cfg.algorithm = 1;
        const auto iv_brent = IVSolver::solve('C', S, K, T, r, price, q, cfg);
        cfg.algorithm = 2;
        const auto iv_bisect = IVSolver::solve('C', S, K, T, r, price, q, cfg);

        assert(iv_newton.has_value());
        assert(iv_brent.has_value());
        assert(iv_bisect.has_value());

        assert(approx_equal(*iv_newton, *iv_brent, 1e-6));
        assert(approx_equal(*iv_newton, *iv_bisect, 1e-6));

        const auto t0 = std::chrono::high_resolution_clock::now();
        cfg.algorithm = 0;
        IVSolver::solve('C', S, K, T, r, price, q, cfg);
        const auto t1 = std::chrono::high_resolution_clock::now();
        cfg.algorithm = 1;
        IVSolver::solve('C', S, K, T, r, price, q, cfg);
        const auto t2 = std::chrono::high_resolution_clock::now();
        cfg.algorithm = 2;
        IVSolver::solve('C', S, K, T, r, price, q, cfg);
        const auto t3 = std::chrono::high_resolution_clock::now();

        const auto dt_newton = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        const auto dt_brent = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        const auto dt_bisect = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        std::cout << "Timing (us) Newton/Brent/Bisection: " << dt_newton << "/" << dt_brent << "/" << dt_bisect << "\n";
    }

    // 4. Edge cases
    {
        struct Case
        {
            double S;
            double K;
            double T;
            double r;
            double sigma;
        };
        const std::vector<Case> cases = {
            {200.0, 100.0, 1.0, 0.05, 0.2},         // deep ITM
            {50.0, 100.0, 1.0, 0.05, 0.2},          // deep OTM
            {100.0, 100.0, 1.0 / 365.0, 0.01, 0.2}, // short expiry
            {100.0, 100.0, 6.0, 0.03, 0.2},         // long expiry
            {100.0, 100.0, 1.0, 0.02, 0.01},        // low vol
            {100.0, 100.0, 1.0, 0.02, 2.0}          // high vol
        };

        IVSolver::SolverConfig cfg;
        cfg.tolerance = 1e-8;
        cfg.algorithm = 0;

        for (const auto &c : cases)
        {
            const double price = bs_price('C', c.S, c.K, c.T, c.r, c.sigma, 0.0);
            const auto iv = IVSolver::solve('C', c.S, c.K, c.T, c.r, price, 0.0, cfg);
            assert(iv.has_value());
            assert(approx_equal(*iv, c.sigma, 1e-4));
        }
    }

    // 5. Error conditions
    {
        IVSolver::SolverConfig cfg;
        cfg.algorithm = 1; // brent for bracket checks
        cfg.tolerance = 1e-8;

        const double S = 100.0;
        const double K = 90.0;
        const double T = 1.0;
        const double r = 0.02;

        const double intrinsic_call = std::max(S - K, 0.0);
        const double below_intrinsic = intrinsic_call * 0.5;
        const auto iv_fail_intrinsic = IVSolver::solve('C', S, K, T, r, below_intrinsic, 0.0, cfg);
        assert(!iv_fail_intrinsic.has_value());

        const double above_spot = S + 1.0;
        const auto iv_fail_spot = IVSolver::solve('C', S, K, T, r, above_spot, 0.0, cfg);
        assert(!iv_fail_spot.has_value());

        const auto iv_fail_zeroT = IVSolver::solve('C', S, K, 0.0, r, 1.0, 0.0, cfg);
        assert(!iv_fail_zeroT.has_value());
    }

    // 6. Batch processing
    {
        const std::size_t n = 1000;
        std::vector<char> types(n, 'C');
        std::vector<double> spots(n, 100.0);
        std::vector<double> strikes(n);
        std::vector<double> times(n, 1.0);
        std::vector<double> prices(n);
        std::vector<double> sigmas(n);
        const double r = 0.01;

        for (std::size_t i = 0; i < n; ++i)
        {
            strikes[i] = 80.0 + 0.05 * static_cast<double>(i);
            sigmas[i] = 0.1 + 0.001 * static_cast<double>(i % 50);
            prices[i] = bs_price('C', spots[i], strikes[i], times[i], r, sigmas[i], 0.0);
        }

        IVSolver::SolverConfig cfg;
        cfg.tolerance = 1e-8;
        cfg.algorithm = 0;

        const auto t0 = std::chrono::high_resolution_clock::now();
        const auto batch = IVSolver::solve_batch(types, spots, strikes, times, r, prices, 0.0, cfg);
        const auto t1 = std::chrono::high_resolution_clock::now();

        for (std::size_t i = 0; i < n; ++i)
        {
            const auto single = IVSolver::solve(types[i], spots[i], strikes[i], times[i], r, prices[i], 0.0, cfg);
            assert(batch[i].has_value());
            assert(single.has_value());
            assert(approx_equal(*batch[i], *single, 1e-6));
        }

        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "Batch solve (" << n << ") time: " << ms << " ms\n";
    }

    // 7. Put-Call parity recovery
    {
        const double S = 100.0;
        const double K = 100.0;
        const double T = 1.0;
        const double r = 0.05;
        const double q = 0.0;
        const double sigma = 0.2;

        const double call_price = bs_price('C', S, K, T, r, sigma, q);
        const double put_price = bs_price('P', S, K, T, r, sigma, q);

        IVSolver::SolverConfig cfg;
        cfg.tolerance = 1e-8;
        cfg.algorithm = 0;

        const auto call_iv = IVSolver::solve('C', S, K, T, r, call_price, q, cfg);
        const auto put_iv = IVSolver::solve('P', S, K, T, r, put_price, q, cfg);
        assert(call_iv.has_value());
        assert(put_iv.has_value());

        const auto call_reprice = bs_price('C', S, K, T, r, *call_iv, q);
        const auto put_reprice = bs_price('P', S, K, T, r, *put_iv, q);
        const double parity_lhs = call_reprice - put_reprice;
        const double parity_rhs = S * std::exp(-q * T) - K * std::exp(-r * T);
        assert(approx_equal(parity_lhs, parity_rhs, 1e-6));

        const double invalid_call_price = S + 1.0;
        const auto call_fail = IVSolver::solve('C', S, K, T, r, invalid_call_price, q, cfg);
        const auto put_ok = IVSolver::solve('P', S, K, T, r, put_price, q, cfg);
        assert(!call_fail.has_value());
        assert(put_ok.has_value());
    }

    return 0;
}
