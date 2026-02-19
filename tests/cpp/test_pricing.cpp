#define private public
#include "pricing.hpp"
#undef private

#include <cassert>
#include <cmath>

namespace
{
    bool approx_equal(double a, double b, double tol)
    {
        return std::abs(a - b) <= tol;
    }
}

int main()
{
    using iv_surface::BlackScholesEngine;

    // 1. Test norm_cdf against known values
    assert(approx_equal(BlackScholesEngine::norm_cdf(0.0), 0.5, 1e-6));
    assert(approx_equal(BlackScholesEngine::norm_cdf(1.96), 0.975, 5e-3));
    assert(approx_equal(BlackScholesEngine::norm_cdf(-1.96), 0.025, 5e-3));

    // 2. Test Black-Scholes price against reference value
    {
        const double S = 100.0;
        const double K = 100.0;
        const double T = 1.0;
        const double r = 0.05;
        const double sigma = 0.2;
        const double q = 0.0;
        const auto greeks = BlackScholesEngine::calculate_greeks(S, K, T, r, sigma, q, true);
        assert(approx_equal(greeks.price, 10.45, 0.1));
    }

    // 3. Test Greeks validation
    {
        const auto call = BlackScholesEngine::calculate_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, true);
        const auto put = BlackScholesEngine::calculate_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, false);
        assert(call.delta > 0.0 && call.delta < 1.0);
        assert(put.delta < 0.0);
        assert(call.gamma > 0.0);
        assert(call.vega > 0.0);
    }

    // 4. Finite difference checks
    {
        const double S = 100.0;
        const double K = 100.0;
        const double T = 1.0;
        const double r = 0.05;
        const double sigma = 0.2;
        const double q = 0.0;

        const double dS = 0.01;
        const double dSigma = 1e-3;
        const double dT = 1.0 / 365.0;

        const auto base = BlackScholesEngine::calculate_greeks(S, K, T, r, sigma, q, true);
        const auto up = BlackScholesEngine::calculate_greeks(S + dS, K, T, r, sigma, q, true);
        const auto down = BlackScholesEngine::calculate_greeks(S - dS, K, T, r, sigma, q, true);

        const double fd_gamma = (up.delta - down.delta) / (2.0 * dS);
        assert(approx_equal(fd_gamma, base.gamma, 5e-3));

        const auto up_vol = BlackScholesEngine::calculate_greeks(S, K, T, r, sigma + dSigma, q, true);
        const auto down_vol = BlackScholesEngine::calculate_greeks(S, K, T, r, sigma - dSigma, q, true);
        const double fd_vega = ((up_vol.price - down_vol.price) / (2.0 * dSigma)) * 0.01;
        assert(approx_equal(fd_vega, base.vega, 5e-3));

        const auto tomorrow = BlackScholesEngine::calculate_greeks(S, K, T - dT, r, sigma, q, true);
        const double fd_theta = (tomorrow.price - base.price) / dT;
        assert(approx_equal(fd_theta, base.theta, 5e-2));
    }

    // 5. Edge cases
    {
        // T -> 0, price approaches intrinsic
        const double S = 105.0;
        const double K = 100.0;
        const double T = 1e-6;
        const double r = 0.05;
        const double sigma = 0.2;
        const double q = 0.0;
        const auto greeks = BlackScholesEngine::calculate_greeks(S, K, T, r, sigma, q, true);
        const double intrinsic = std::max(S - K, 0.0);
        assert(approx_equal(greeks.price, intrinsic, 1e-2));

        // Deep ITM and deep OTM deltas
        const auto deep_itm = BlackScholesEngine::calculate_greeks(200.0, 100.0, 1.0, 0.05, 0.2, 0.0, true);
        const auto deep_otm = BlackScholesEngine::calculate_greeks(50.0, 100.0, 1.0, 0.05, 0.2, 0.0, true);
        assert(deep_itm.delta > 0.98);
        assert(deep_otm.delta < 0.02);

        // sigma -> 0, vega -> 0
        const auto tiny_vol = BlackScholesEngine::calculate_greeks(100.0, 100.0, 1.0, 0.05, 1e-6, 0.0, true);
        assert(std::abs(tiny_vol.vega) < 1e-3);
    }

    return 0;
}
