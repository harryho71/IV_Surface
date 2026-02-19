#include "optimization.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>

#include <nlopt.hpp>

namespace iv_surface
{

    namespace
    {
        struct CalibrationData
        {
            ModelCalibrator::ModelType model_type;
            const std::vector<double> *observed_ivs;
            const std::vector<double> *strikes;
            double maturity;
            double forward;
            mutable int eval_count = 0; // Track function evaluations
        };

        double evaluate_model(
            ModelCalibrator::ModelType model_type,
            double forward,
            double strike,
            double maturity,
            const std::vector<double> &params)
        {
            switch (model_type)
            {
            case ModelCalibrator::SABR:
                return ModelCalibrator::sabr_volatility(forward, strike, maturity, params);
            case ModelCalibrator::SVI:
            {
                const double m = std::log(strike / forward);
                return ModelCalibrator::svi_volatility(m, params);
            }
            case ModelCalibrator::POLYNOMIAL:
            {
                if (params.size() < 5)
                {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                const double m = std::log(strike / forward);
                const double b0 = params[0];
                const double b1 = params[1];
                const double b2 = params[2];
                const double b3 = params[3];
                const double b4 = params[4];
                return (b0 + b1 * m + b2 * m * m) + (b3 + b4 * m) * maturity;
            }
            default:
                return std::numeric_limits<double>::quiet_NaN();
            }
        }
    } // namespace

    double ModelCalibrator::sabr_volatility(
        double forward, double strike, double maturity,
        const std::vector<double> &params)
    {
        if (params.size() < 4 || forward <= 0.0 || strike <= 0.0 || maturity < 0.0)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double alpha = params[0];
        const double beta = params[1];
        const double rho = params[2];
        const double nu = params[3];

        if (!(alpha > 0.0) || !(nu > 0.0) || !(beta >= 0.0 && beta <= 1.0) || !(rho > -1.0 && rho < 1.0))
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double F = forward;
        const double K = strike;
        const double one_minus_beta = 1.0 - beta;
        const double log_fk = std::log(F / K);
        const double log_fk2 = log_fk * log_fk;
        const double log_fk4 = log_fk2 * log_fk2;

        const double fk_pow = std::pow(F * K, 0.5 * one_minus_beta);
        const double f_pow = std::pow(F, one_minus_beta);

        const double term1 = 1.0 + (one_minus_beta * one_minus_beta / 24.0) * log_fk2 + (std::pow(one_minus_beta, 4) / 1920.0) * log_fk4;

        const double term2 = 1.0 + ((one_minus_beta * one_minus_beta) / 24.0) * (alpha * alpha) / (fk_pow * fk_pow) + (rho * beta * nu * alpha) / (4.0 * fk_pow) + (2.0 - 3.0 * rho * rho) * (nu * nu) / 24.0;

        const double t_adj = term2 * maturity;

        const double atm_cutoff = 1e-7;
        double sigma = 0.0;

        if (std::abs(log_fk) < atm_cutoff)
        {
            sigma = (alpha / f_pow) * (1.0 + t_adj);
        }
        else
        {
            const double z = (nu / alpha) * fk_pow * log_fk;
            const double sqrt_term = std::sqrt(1.0 - 2.0 * rho * z + z * z);
            const double x_z = std::log((sqrt_term + z - rho) / (1.0 - rho));
            const double z_over_x = (std::abs(x_z) > 0.0) ? (z / x_z) : 1.0;

            sigma = (alpha / (fk_pow * term1)) * z_over_x * (1.0 + t_adj);
        }

        if (!std::isfinite(sigma))
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return std::clamp(sigma, 0.001, 3.0);
    }

    double ModelCalibrator::svi_volatility(
        double moneyness,
        const std::vector<double> &params)
    {
        if (params.size() < 5)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double a = params[0];
        const double b = params[1];
        const double rho = params[2];
        const double sigma = params[3];
        const double m_star = params[4];

        if (a < 0.0 || b < 0.0 || sigma < 0.0 || !(rho > -1.0 && rho < 1.0))
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double x = moneyness - m_star;
        const double term = rho * x + std::sqrt(x * x + sigma * sigma);
        const double variance = a + b * term;
        if (!std::isfinite(variance) || variance < 0.0)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double vol = std::sqrt(variance);
        return std::max(vol, 0.001);
    }

    double ModelCalibrator::objective_function(
        const std::vector<double> &params,
        void *user_data)
    {
        auto *data = static_cast<CalibrationData *>(user_data);
        if (!data || !data->observed_ivs || !data->strikes)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Increment evaluation counter
        ++(data->eval_count);

        const auto &observed = *data->observed_ivs;
        const auto &strikes = *data->strikes;
        const std::size_t n = std::min(observed.size(), strikes.size());
        if (n == 0)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        double sum_sq = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const double model_iv = evaluate_model(data->model_type, data->forward, strikes[i], data->maturity, params);
            if (!std::isfinite(model_iv))
            {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double diff = observed[i] - model_iv;
            sum_sq += diff * diff;
        }

        return sum_sq;
    }

    double ModelCalibrator::compute_forward(double spot, double rate, double maturity)
    {
        return spot * std::exp(rate * maturity);
    }

    ModelCalibrator::CalibrationResult ModelCalibrator::calibrate(
        ModelType model_type,
        const std::vector<double> &observed_ivs,
        const std::vector<double> &strikes,
        double maturity,
        const std::vector<double> &initial_params,
        double forward)
    {
        CalibrationResult result{};
        result.parameters = initial_params;
        result.rmse = std::numeric_limits<double>::infinity();
        result.max_error = std::numeric_limits<double>::infinity();
        result.iterations = 0;
        result.converged = false;

        if (observed_ivs.empty() || strikes.empty() || initial_params.empty())
        {
            return result;
        }

        const std::size_t n = std::min(observed_ivs.size(), strikes.size());

        // Use provided forward price, or compute from spot if forward <= 0
        double F = forward;
        if (F <= 0.0)
        {
            F = strikes[n / 2]; // Fallback: use middle strike
        }

        CalibrationData data{model_type, &observed_ivs, &strikes, maturity, F};

        auto run_optimizer = [&](nlopt::algorithm algo) -> bool
        {
            std::vector<double> params = initial_params;
            nlopt::opt opt(algo, params.size());
            opt.set_min_objective(
                [](const std::vector<double> &x, std::vector<double> &, void *ud)
                {
                    return ModelCalibrator::objective_function(x, ud);
                },
                &data);
            // Set reasonable convergence criteria
            opt.set_ftol_rel(1e-6);  // Relative function value tolerance
            opt.set_ftol_abs(1e-10); // Absolute function value tolerance
            opt.set_xtol_rel(1e-8);  // Relative parameter tolerance
            opt.set_xtol_abs(1e-10); // Absolute parameter tolerance
            opt.set_maxeval(2000);   // Maximum function evaluations
            opt.set_maxtime(30.0);   // Maximum time (seconds)

            if (model_type == SABR)
            {
                std::vector<double> lb{1e-6, 0.0, -0.999, 1e-6};
                std::vector<double> ub{1.0, 1.0, 0.999, 5.0};
                opt.set_lower_bounds(lb);
                opt.set_upper_bounds(ub);
            }
            else if (model_type == SVI)
            {
                std::vector<double> lb{0.0, 0.0, -0.999, 0.0, -10.0};
                std::vector<double> ub{10.0, 10.0, 0.999, 10.0, 10.0};
                opt.set_lower_bounds(lb);
                opt.set_upper_bounds(ub);
            }

            // Set initial step size for derivative-free methods
            opt.set_initial_step(std::vector<double>(params.size(), 0.1));

            double minf = 0.0;
            nlopt::result opt_result = nlopt::FAILURE;

            try
            {
                opt_result = opt.optimize(params, minf);
            }
            catch (const std::exception &e)
            {
                std::cerr << "NLopt exception: " << e.what() << std::endl;
                return false;
            }
            catch (...)
            {
                std::cerr << "Unknown exception during optimization" << std::endl;
                return false;
            }

            // Debug output (disabled for production)
            // std::cerr << "[DEBUG] opt_result=" << static_cast<int>(opt_result)
            //           << " minf=" << minf
            //           << " evals=" << data.eval_count << std::endl;

            double max_error = 0.0;
            double sum_sq = 0.0;
            for (std::size_t i = 0; i < n; ++i)
            {
                const double model_iv = evaluate_model(model_type, F, strikes[i], maturity, params);
                const double diff = observed_ivs[i] - model_iv;
                sum_sq += diff * diff;
                max_error = std::max(max_error, std::abs(diff));
            }

            const double rmse = std::sqrt(sum_sq / static_cast<double>(n));

            if (rmse < result.rmse)
            {
                result.parameters = params;
                result.rmse = rmse;
                result.max_error = max_error;
                // Use the actual evaluation counter from the objective function
                result.iterations = data.eval_count;
                result.converged = (opt_result > 0);
            }

            return opt_result > 0;
        };

        const bool local_ok = run_optimizer(nlopt::LN_NELDERMEAD); // Simplex method (always iterates)
        if (!local_ok)
        {
            run_optimizer(nlopt::LD_SLSQP); // SLSQP
        }
        if (!local_ok)
        {
            run_optimizer(nlopt::LD_LBFGS); // LBFGS
        }
        if (!local_ok)
        {
            run_optimizer(nlopt::GN_CRS2_LM); // Global method
        }

        return result;
    }

} // namespace iv_surface
