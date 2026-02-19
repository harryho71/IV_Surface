#include "pricing.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace iv_surface
{

    namespace
    {
        constexpr double kSqrt2 = 1.4142135623730950488;
        constexpr double kSqrt2Pi = 2.5066282746310005024;
        constexpr double kEps = 1e-12;

#ifndef M_PI
        constexpr double M_PI = 3.14159265358979323846;
#endif

        BlackScholesEngine::Greeks nan_greeks()
        {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            return {nan, nan, nan, nan, nan, nan};
        }
    } // namespace

    double BlackScholesEngine::norm_pdf(double x)
    {
        if (!std::isfinite(x))
        {
            return 0.0;
        }
        if (std::abs(x) > 40.0)
        {
            return 0.0;
        }
        const double exponent = -0.5 * x * x;
        const double denom = std::sqrt(2.0 * M_PI);
        return std::exp(exponent) / denom;
    }

    double BlackScholesEngine::norm_cdf(double x)
    {
        if (!std::isfinite(x))
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (x < -10.0)
        {
            return 0.0;
        }
        if (x > 10.0)
        {
            return 1.0;
        }
        return 0.5 * (1.0 + std::erf(x / kSqrt2));
    }

    std::array<double, 2> BlackScholesEngine::calculate_d1_d2(
        double S, double K, double T, double r,
        double sigma, double q)
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (S <= 0.0 || K <= 0.0 || T < 0.0 || sigma < 0.0)
        {
            return {nan, nan};
        }
        if (T <= kEps || sigma <= kEps)
        {
            return {nan, nan};
        }

        const double sqrt_t = std::sqrt(T);
        const double vol_sqrt_t = sigma * sqrt_t;
        const double log_moneyness = std::log(S / K);
        const double drift = (r - q + 0.5 * sigma * sigma) * T;
        const double d1 = (log_moneyness + drift) / vol_sqrt_t;
        const double d2 = d1 - vol_sqrt_t;
        return {d1, d2};
    }

    BlackScholesEngine::Greeks BlackScholesEngine::calculate_greeks(
        double S, double K, double T, double r,
        double sigma, double q, bool is_call)
    {
        if (S <= 0.0 || K <= 0.0 || T < 0.0 || sigma < 0.0)
        {
            return nan_greeks();
        }

        if (T <= kEps)
        {
            const double discount_r = std::exp(-r * T);
            const double discount_q = std::exp(-q * T);
            const double intrinsic_call = std::max(S * discount_q - K * discount_r, 0.0);
            const double intrinsic_put = std::max(K * discount_r - S * discount_q, 0.0);
            if (is_call)
            {
                const double delta = (S > K) ? 1.0 : 0.0;
                return {intrinsic_call, delta, 0.0, 0.0, 0.0, 0.0};
            }
            const double delta = (S < K) ? -1.0 : 0.0;
            return {intrinsic_put, delta, 0.0, 0.0, 0.0, 0.0};
        }

        const auto d1d2 = calculate_d1_d2(S, K, T, r, sigma, q);
        if (!std::isfinite(d1d2[0]) || !std::isfinite(d1d2[1]))
        {
            return nan_greeks();
        }

        const double d1 = d1d2[0];
        const double d2 = d1d2[1];
        const double sqrt_t = std::sqrt(T);
        const double discount_r = std::exp(-r * T);
        const double discount_q = std::exp(-q * T);
        const double pdf_d1 = norm_pdf(d1);
        const double cdf_d1 = norm_cdf(d1);
        const double cdf_d2 = norm_cdf(d2);

        const double gamma = pdf_d1 / (S * sigma * sqrt_t);
        const double vega = S * pdf_d1 * sqrt_t / 100.0;

        if (is_call)
        {
            const double price = S * discount_q * cdf_d1 - K * discount_r * cdf_d2;
            const double delta = cdf_d1;
            const double theta = -(S * pdf_d1 * sigma) / (2.0 * sqrt_t) - r * K * discount_r * cdf_d2;
            const double rho = K * T * discount_r * cdf_d2;
            return {price, delta, gamma, vega, theta, rho};
        }

        const double price = K * discount_r * norm_cdf(-d2) - S * discount_q * norm_cdf(-d1);
        const double delta = cdf_d1 - 1.0;
        const double theta = -(S * pdf_d1 * sigma) / (2.0 * sqrt_t) + r * K * discount_r * norm_cdf(-d2);
        const double rho = -K * T * discount_r * norm_cdf(-d2);
        return {price, delta, gamma, vega, theta, rho};
    }

    std::vector<BlackScholesEngine::Greeks> BlackScholesEngine::calculate_batch(
        const std::vector<double> &spots,
        const std::vector<double> &strikes,
        const std::vector<double> &times,
        double r,
        const std::vector<double> &sigmas,
        double q,
        bool is_call)
    {
        const std::vector<double> spots_local = spots;
        const std::vector<double> strikes_local = strikes;
        const std::vector<double> times_local = times;
        const std::vector<double> sigmas_local = sigmas;

        const std::size_t n = std::min(
            {spots_local.size(), strikes_local.size(), times_local.size(), sigmas_local.size()});

        std::vector<Greeks> results(n);

#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(n); ++i)
        {
            results[i] = calculate_greeks(
                spots_local[i],
                strikes_local[i],
                times_local[i],
                r,
                sigmas_local[i],
                q,
                is_call);
        }

        return results;
    }

    double BlackScholesEngine::call_price(
        double S, double K, double T, double r,
        double sigma, double q)
    {
        Greeks greeks = calculate_greeks(S, K, T, r, sigma, q, true);
        return greeks.price;
    }

} // namespace iv_surface
