#include "interpolation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace iv_surface
{

    namespace
    {
        constexpr double kEps = 1e-12;

        bool has_nan(const std::vector<double> &values)
        {
            for (double v : values)
            {
                if (!std::isfinite(v))
                {
                    return true;
                }
            }
            return false;
        }

        std::vector<double> spline_second_derivatives(
            const std::vector<double> &x,
            const std::vector<double> &y)
        {
            const std::size_t n = x.size();
            std::vector<double> y2(n, 0.0);
            if (n < 2 || y.size() != n)
            {
                std::fill(y2.begin(), y2.end(), std::numeric_limits<double>::quiet_NaN());
                return y2;
            }

            for (std::size_t i = 1; i < n; ++i)
            {
                if (x[i] <= x[i - 1])
                {
                    std::fill(y2.begin(), y2.end(), std::numeric_limits<double>::quiet_NaN());
                    return y2;
                }
            }

            std::vector<double> u(n - 1, 0.0);
            y2[0] = 0.0;
            u[0] = 0.0;

            for (std::size_t i = 1; i + 1 < n; ++i)
            {
                const double sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
                const double p = sig * y2[i - 1] + 2.0;
                y2[i] = (sig - 1.0) / p;
                const double dd = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
                u[i] = (6.0 * dd / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
            }

            y2[n - 1] = 0.0;
            for (std::size_t k = n - 1; k-- > 0;)
            {
                y2[k] = y2[k] * y2[k + 1] + u[k];
            }

            return y2;
        }

        double spline_interpolate(
            const std::vector<double> &x,
            const std::vector<double> &y,
            const std::vector<double> &y2,
            double xq)
        {
            const std::size_t n = x.size();
            if (n < 2 || y.size() != n || y2.size() != n)
            {
                return std::numeric_limits<double>::quiet_NaN();
            }
            if (xq < x.front() || xq > x.back())
            {
                return std::numeric_limits<double>::quiet_NaN();
            }

            auto it = std::upper_bound(x.begin(), x.end(), xq);
            std::size_t khi = std::clamp<std::size_t>(static_cast<std::size_t>(it - x.begin()), 1, n - 1);
            std::size_t klo = khi - 1;

            const double h = x[khi] - x[klo];
            if (h <= kEps)
            {
                return std::numeric_limits<double>::quiet_NaN();
            }

            const double a = (x[khi] - xq) / h;
            const double b = (xq - x[klo]) / h;
            return a * y[klo] + b * y[khi] + ((a * a * a - a) * y2[klo] + (b * b * b - b) * y2[khi]) * (h * h) / 6.0;
        }
    } // namespace

    Interpolator::Interpolator(
        const std::vector<double> &strikes,
        const std::vector<double> &maturities,
        const std::vector<double> &ivs,
        Method method) : method_(method), strikes_(strikes), maturities_(maturities), ivs_(ivs)
    {
        switch (method_)
        {
        case CUBIC_SPLINE:
            fit_cubic_spline();
            break;
        case RBF_THINPLATE:
            fit_rbf();
            break;
        case POLYNOMIAL:
            fit_polynomial();
            break;
        default:
            fit_cubic_spline();
            break;
        }
    }

    void Interpolator::fit_cubic_spline()
    {
        const std::size_t n_strikes = strikes_.size();
        const std::size_t n_maturities = maturities_.size();
        if (n_strikes == 0 || n_maturities == 0)
        {
            fit_matrix_.resize(0, 0);
            coefficients_.resize(0);
            return;
        }
        if (ivs_.size() != n_strikes * n_maturities)
        {
            fit_matrix_.resize(0, 0);
            coefficients_.resize(0);
            return;
        }

        fit_matrix_.resize(static_cast<Eigen::Index>(n_strikes), static_cast<Eigen::Index>(n_maturities));
        coefficients_.resize(static_cast<Eigen::Index>(n_strikes * n_maturities));

        for (std::size_t j = 0; j < n_maturities; ++j)
        {
            std::vector<double> y(n_strikes);
            for (std::size_t i = 0; i < n_strikes; ++i)
            {
                y[i] = ivs_[i * n_maturities + j];
            }
            const auto y2 = has_nan(y) ? std::vector<double>(n_strikes, std::numeric_limits<double>::quiet_NaN())
                                       : spline_second_derivatives(strikes_, y);
            for (std::size_t i = 0; i < n_strikes; ++i)
            {
                fit_matrix_(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = y2[i];
            }
        }

        for (std::size_t i = 0; i < n_strikes; ++i)
        {
            std::vector<double> y(n_maturities);
            for (std::size_t j = 0; j < n_maturities; ++j)
            {
                y[j] = ivs_[i * n_maturities + j];
            }
            const auto y2 = has_nan(y) ? std::vector<double>(n_maturities, std::numeric_limits<double>::quiet_NaN())
                                       : spline_second_derivatives(maturities_, y);
            for (std::size_t j = 0; j < n_maturities; ++j)
            {
                coefficients_(static_cast<Eigen::Index>(i * n_maturities + j)) = y2[j];
            }
        }
    }

    void Interpolator::fit_rbf()
    {
        const std::size_t n_strikes = strikes_.size();
        const std::size_t n_maturities = maturities_.size();
        if (n_strikes == 0 || n_maturities == 0 || ivs_.size() != n_strikes * n_maturities)
        {
            fit_matrix_.resize(0, 0);
            coefficients_.resize(0);
            return;
        }

        std::vector<double> xs;
        std::vector<double> ys;
        std::vector<double> zs;
        xs.reserve(ivs_.size());
        ys.reserve(ivs_.size());
        zs.reserve(ivs_.size());

        for (std::size_t i = 0; i < n_strikes; ++i)
        {
            for (std::size_t j = 0; j < n_maturities; ++j)
            {
                const double z = ivs_[i * n_maturities + j];
                if (std::isfinite(z))
                {
                    xs.push_back(strikes_[i]);
                    ys.push_back(maturities_[j]);
                    zs.push_back(z);
                }
            }
        }

        const std::size_t n = xs.size();
        if (n < 3)
        {
            fit_matrix_.resize(0, 0);
            coefficients_.resize(0);
            return;
        }

        Eigen::MatrixXd system = Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(n + 3), static_cast<Eigen::Index>(n + 3));
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n + 3));

        for (std::size_t i = 0; i < n; ++i)
        {
            rhs(static_cast<Eigen::Index>(i)) = zs[i];
            for (std::size_t j = 0; j < n; ++j)
            {
                const double dx = xs[i] - xs[j];
                const double dy = ys[i] - ys[j];
                const double r = std::sqrt(dx * dx + dy * dy);
                double phi = 0.0;
                if (r > kEps)
                {
                    const double r2 = r * r;
                    phi = r2 * std::log(r);
                }
                system(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = phi;
            }

            system(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(n)) = 1.0;
            system(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(n + 1)) = xs[i];
            system(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(n + 2)) = ys[i];

            system(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(i)) = 1.0;
            system(static_cast<Eigen::Index>(n + 1), static_cast<Eigen::Index>(i)) = xs[i];
            system(static_cast<Eigen::Index>(n + 2), static_cast<Eigen::Index>(i)) = ys[i];
        }

        Eigen::VectorXd solution = system.colPivHouseholderQr().solve(rhs);
        if (solution.size() != static_cast<Eigen::Index>(n + 3))
        {
            fit_matrix_.resize(0, 0);
            coefficients_.resize(0);
            return;
        }

        fit_matrix_.resize(static_cast<Eigen::Index>(n), 2);
        for (std::size_t i = 0; i < n; ++i)
        {
            fit_matrix_(static_cast<Eigen::Index>(i), 0) = xs[i];
            fit_matrix_(static_cast<Eigen::Index>(i), 1) = ys[i];
        }

        coefficients_ = solution;
    }

    void Interpolator::fit_polynomial()
    {
        fit_matrix_.resize(0, 0);
        coefficients_.resize(0);
    }

    double Interpolator::evaluate(double strike, double maturity) const
    {
        if (method_ == RBF_THINPLATE)
        {
            const Eigen::Index n = fit_matrix_.rows();
            if (n <= 0 || coefficients_.size() != n + 3)
            {
                return std::numeric_limits<double>::quiet_NaN();
            }

            double value = 0.0;
            for (Eigen::Index i = 0; i < n; ++i)
            {
                const double dx = strike - fit_matrix_(i, 0);
                const double dy = maturity - fit_matrix_(i, 1);
                const double r = std::sqrt(dx * dx + dy * dy);
                double phi = 0.0;
                if (r > kEps)
                {
                    const double r2 = r * r;
                    phi = r2 * std::log(r);
                }
                value += coefficients_(i) * phi;
            }

            const double c0 = coefficients_(n);
            const double c1 = coefficients_(n + 1);
            const double c2 = coefficients_(n + 2);
            return value + c0 + c1 * strike + c2 * maturity;
        }

        if (method_ == POLYNOMIAL)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const std::size_t n_strikes = strikes_.size();
        const std::size_t n_maturities = maturities_.size();
        if (n_strikes < 2 || n_maturities < 2)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (ivs_.size() != n_strikes * n_maturities)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (strike < strikes_.front() || strike > strikes_.back())
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (maturity < maturities_.front() || maturity > maturities_.back())
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        std::vector<double> temp_values(n_maturities);
        for (std::size_t j = 0; j < n_maturities; ++j)
        {
            std::vector<double> y(n_strikes);
            std::vector<double> y2(n_strikes);
            for (std::size_t i = 0; i < n_strikes; ++i)
            {
                y[i] = ivs_[i * n_maturities + j];
                y2[i] = fit_matrix_(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
            }
            if (has_nan(y) || has_nan(y2))
            {
                return std::numeric_limits<double>::quiet_NaN();
            }
            temp_values[j] = spline_interpolate(strikes_, y, y2, strike);
            if (!std::isfinite(temp_values[j]))
            {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }

        if (has_nan(temp_values))
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const auto y2T = spline_second_derivatives(maturities_, temp_values);
        if (has_nan(y2T))
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return spline_interpolate(maturities_, temp_values, y2T, maturity);
    }

    std::vector<double> Interpolator::evaluate_grid(
        const std::vector<double> &strikes,
        const std::vector<double> &maturities) const
    {
        const std::size_t n = std::min(strikes.size(), maturities.size());
        std::vector<double> results(n, std::numeric_limits<double>::quiet_NaN());
        for (std::size_t i = 0; i < n; ++i)
        {
            results[i] = evaluate(strikes[i], maturities[i]);
        }
        return results;
    }

    Interpolator::IVGradient Interpolator::compute_gradient(double strike, double maturity) const
    {
        const double hK = 0.001 * strike;
        const double hT = 0.001 * maturity;

        const double center = evaluate(strike, maturity);
        IVGradient grad{};

        if (!std::isfinite(center) || hK <= kEps || hT <= kEps)
        {
            grad.dSigma_dK = std::numeric_limits<double>::quiet_NaN();
            grad.dSigma_dT = std::numeric_limits<double>::quiet_NaN();
            grad.d2Sigma_dK2 = std::numeric_limits<double>::quiet_NaN();
            return grad;
        }

        const double upK = evaluate(strike + hK, maturity);
        const double downK = evaluate(strike - hK, maturity);
        if (std::isfinite(upK) && std::isfinite(downK))
        {
            grad.dSigma_dK = (upK - downK) / (2.0 * hK);
            grad.d2Sigma_dK2 = (upK - 2.0 * center + downK) / (hK * hK);
        }
        else
        {
            const double forwardK = evaluate(strike + hK, maturity);
            const double backwardK = evaluate(strike - hK, maturity);
            if (std::isfinite(forwardK))
            {
                grad.dSigma_dK = (forwardK - center) / hK;
                grad.d2Sigma_dK2 = std::numeric_limits<double>::quiet_NaN();
            }
            else if (std::isfinite(backwardK))
            {
                grad.dSigma_dK = (center - backwardK) / hK;
                grad.d2Sigma_dK2 = std::numeric_limits<double>::quiet_NaN();
            }
            else
            {
                grad.dSigma_dK = std::numeric_limits<double>::quiet_NaN();
                grad.d2Sigma_dK2 = std::numeric_limits<double>::quiet_NaN();
            }
        }

        const double upT = evaluate(strike, maturity + hT);
        const double downT = evaluate(strike, maturity - hT);
        if (std::isfinite(upT) && std::isfinite(downT))
        {
            grad.dSigma_dT = (upT - downT) / (2.0 * hT);
        }
        else
        {
            const double forwardT = evaluate(strike, maturity + hT);
            const double backwardT = evaluate(strike, maturity - hT);
            if (std::isfinite(forwardT))
            {
                grad.dSigma_dT = (forwardT - center) / hT;
            }
            else if (std::isfinite(backwardT))
            {
                grad.dSigma_dT = (center - backwardT) / hT;
            }
            else
            {
                grad.dSigma_dT = std::numeric_limits<double>::quiet_NaN();
            }
        }

        return grad;
    }

} // namespace iv_surface
