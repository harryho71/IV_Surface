/**
 * SABR CLI Tool - Standalone executable for Python subprocess calls
 * Provides SABR model evaluation and calibration via command line
 */

#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cctype>
#include <fstream>
#include "optimization.hpp"
#include "volatility.hpp"
#include "interpolation.hpp"
#include "total_variance.hpp"

using namespace iv_surface;

void print_usage()
{
    std::cout << "Usage: sabr_cli <command> <args>\n\n";
    std::cout << "Commands:\n";
    std::cout << "  eval <F> <K> <T> <alpha> <beta> <rho> <nu>\n";
    std::cout << "    - Evaluate SABR volatility at strike K\n";
    std::cout << "    - F: forward price, K: strike, T: maturity\n";
    std::cout << "    - alpha, beta, rho, nu: SABR parameters\n\n";
    std::cout << "  calibrate <maturity> <strike1,strike2,...> <iv1,iv2,...> <alpha0,beta0,rho0,nu0>\n";
    std::cout << "    - Calibrate SABR model to observed IVs\n";
    std::cout << "    - Returns: alpha,beta,rho,nu,rmse,iterations,converged\n";
    std::cout << "  iv <C|P> <S> <K> <T> <r> <price> [q]\n";
    std::cout << "    - Solve implied volatility from option price\n";
    std::cout << "    - Returns: IV (decimal) or nan if failed\n";
    std::cout << "  interp_grid <method> <strikes> <maturities> <ivs> <grid_strikes> <grid_maturities>\n";
    std::cout << "    - Interpolate IVs on a grid using C++ interpolator\n";
    std::cout << "    - method: cubic_spline|rbf|polynomial (or 0/1/2)\n";
    std::cout << "    - Returns: comma-separated IVs (row-major: strikes x maturities)\n";
    std::cout << "  interp_grid_file <method> <data_file> <grid_strikes> <grid_maturities>\n";
    std::cout << "    - data_file: 3 lines (strikes CSV, maturities CSV, ivs CSV)\n\n";
    std::cout << "Total Variance Commands:\n";
    std::cout << "  tv_sigma_to_w <sigma_csv> <maturities_csv>\n";
    std::cout << "    - Convert IV to total variance: w = σ²T\n";
    std::cout << "    - sigma_csv: flattened row-major grid\n";
    std::cout << "    - Returns: comma-separated total variance\n";
    std::cout << "  tv_w_to_sigma <w_csv> <maturities_csv>\n";
    std::cout << "    - Convert total variance to IV: σ = √(w/T)\n";
    std::cout << "  tv_enforce_monotonic <w_csv> <n_strikes> <n_maturities>\n";
    std::cout << "    - Enforce ∂w/∂T ≥ 0 (calendar arbitrage prevention)\n";
    std::cout << "  tv_validate <w_csv> <n_strikes> <n_maturities>\n";
    std::cout << "    - Validate arbitrage-free: returns calendar_violations,butterfly_violations\n";
    std::cout << "  tv_lee_bounds <w_atm>\n";
    std::cout << "    - Compute Lee moment bounds: left_bound,right_bound\n";
}

static std::vector<double> parse_csv_doubles(const std::string &csv)
{
    std::vector<double> values;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        if (!token.empty())
        {
            values.push_back(std::stod(token));
        }
    }
    return values;
}

static Interpolator::Method parse_method(const std::string &method)
{
    std::string m;
    m.reserve(method.size());
    for (char c : method)
    {
        m.push_back(static_cast<char>(std::tolower(c)));
    }

    if (m == "0" || m == "cubic_spline" || m == "cubic" || m == "spline")
    {
        return Interpolator::CUBIC_SPLINE;
    }
    if (m == "1" || m == "rbf" || m == "rbf_thinplate" || m == "thinplate")
    {
        return Interpolator::RBF_THINPLATE;
    }
    if (m == "2" || m == "polynomial" || m == "poly")
    {
        return Interpolator::POLYNOMIAL;
    }

    throw std::runtime_error("Unknown interpolation method: " + method);
}

static bool read_interpolator_data(
    const std::string &file_path,
    std::vector<double> &strikes,
    std::vector<double> &maturities,
    std::vector<double> &ivs)
{
    std::ifstream in(file_path);
    if (!in.is_open())
    {
        return false;
    }

    std::string line;
    if (!std::getline(in, line))
        return false;
    strikes = parse_csv_doubles(line);

    if (!std::getline(in, line))
        return false;
    maturities = parse_csv_doubles(line);

    if (!std::getline(in, line))
        return false;
    ivs = parse_csv_doubles(line);

    return true;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        print_usage();
        return 1;
    }

    std::string command = argv[1];

    try
    {
        if (command == "eval")
        {
            // SABR evaluation: sabr_cli eval F K T alpha beta rho nu
            if (argc != 9)
            {
                std::cerr << "Error: eval requires 7 arguments\n";
                print_usage();
                return 1;
            }

            double F = std::stod(argv[2]);
            double K = std::stod(argv[3]);
            double T = std::stod(argv[4]);
            double alpha = std::stod(argv[5]);
            double beta = std::stod(argv[6]);
            double rho = std::stod(argv[7]);
            double nu = std::stod(argv[8]);

            std::vector<double> params = {alpha, beta, rho, nu};
            double iv = ModelCalibrator::sabr_volatility(F, K, T, params);

            // Output just the IV value (easy to parse from Python)
            std::cout << std::fixed << std::setprecision(6) << iv << std::endl;
            return 0;
        }
        else if (command == "calibrate")
        {
            // SABR calibration: sabr_cli calibrate T "K1,K2,K3" "IV1,IV2,IV3" "a,b,r,n" [F]
            if (argc < 6 || argc > 7)
            {
                std::cerr << "Error: calibrate requires 4-5 arguments\n";
                print_usage();
                return 1;
            }

            double maturity = std::stod(argv[2]);

            // Parse comma-separated strikes
            std::vector<double> strikes;
            std::stringstream ss_strikes(argv[3]);
            std::string token;
            while (std::getline(ss_strikes, token, ','))
            {
                strikes.push_back(std::stod(token));
            }

            // Parse comma-separated IVs
            std::vector<double> ivs;
            std::stringstream ss_ivs(argv[4]);
            while (std::getline(ss_ivs, token, ','))
            {
                ivs.push_back(std::stod(token));
            }

            // Parse initial parameters
            std::vector<double> initial_params;
            std::stringstream ss_params(argv[5]);
            while (std::getline(ss_params, token, ','))
            {
                initial_params.push_back(std::stod(token));
            }

            // Optional forward price (defaults to 0, which means auto-compute)
            double forward = 0.0;
            if (argc == 7)
            {
                forward = std::stod(argv[6]);
            }

            if (strikes.size() != ivs.size())
            {
                std::cerr << "Error: strikes and IVs must have same length\n";
                return 1;
            }

            if (initial_params.size() != 4)
            {
                std::cerr << "Error: initial params must be [alpha,beta,rho,nu]\n";
                return 1;
            }

            // Calibrate
            auto result = ModelCalibrator::calibrate(
                ModelCalibrator::SABR,
                ivs,
                strikes,
                maturity,
                initial_params,
                forward);

            // Output: alpha,beta,rho,nu,rmse,iterations,converged
            std::cout << std::fixed << std::setprecision(6);
            std::cout << result.parameters[0] << ","
                      << result.parameters[1] << ","
                      << result.parameters[2] << ","
                      << result.parameters[3] << ","
                      << result.rmse << ","
                      << result.iterations << ","
                      << (result.converged ? "1" : "0") << std::endl;
            return 0;
        }
        else if (command == "iv")
        {
            // Implied vol: sabr_cli iv C S K T r price [q]
            if (argc != 8 && argc != 9)
            {
                std::cerr << "Error: iv requires 6 or 7 arguments\n";
                print_usage();
                return 1;
            }

            std::string type_str = argv[2];
            if (type_str.empty())
            {
                std::cerr << "Error: option type must be C or P\n";
                return 1;
            }

            char option_type = static_cast<char>(std::toupper(type_str[0]));
            double S = std::stod(argv[3]);
            double K = std::stod(argv[4]);
            double T = std::stod(argv[5]);
            double r = std::stod(argv[6]);
            double price = std::stod(argv[7]);
            double q = (argc == 9) ? std::stod(argv[8]) : 0.0;

            auto cfg = IVSolver::default_config();
            auto iv = IVSolver::solve(option_type, S, K, T, r, price, q, &cfg);

            if (iv.has_value())
            {
                std::cout << std::fixed << std::setprecision(6) << iv.value() << std::endl;
            }
            else
            {
                std::cout << "nan" << std::endl;
            }
            return 0;
        }
        else if (command == "interp_grid")
        {
            // Interpolator: sabr_cli interp_grid method "K1,K2" "T1,T2" "IV1,IV2" "Kq1,Kq2" "Tq1,Tq2"
            if (argc != 8)
            {
                std::cerr << "Error: interp_grid requires 6 arguments\n";
                print_usage();
                return 1;
            }

            auto method = parse_method(argv[2]);
            auto strikes = parse_csv_doubles(argv[3]);
            auto maturities = parse_csv_doubles(argv[4]);
            auto ivs = parse_csv_doubles(argv[5]);
            auto grid_strikes = parse_csv_doubles(argv[6]);
            auto grid_maturities = parse_csv_doubles(argv[7]);

            if (strikes.size() != maturities.size() || strikes.size() != ivs.size())
            {
                std::cerr << "Error: strikes, maturities, and ivs must have same length\n";
                return 1;
            }

            if (grid_strikes.empty() || grid_maturities.empty())
            {
                std::cerr << "Error: grid_strikes and grid_maturities must be non-empty\n";
                return 1;
            }

            Interpolator interpolator(strikes, maturities, ivs, method);

            std::cout << std::fixed << std::setprecision(6);
            bool first = true;
            for (double K : grid_strikes)
            {
                for (double T : grid_maturities)
                {
                    double val = interpolator.evaluate(K, T);
                    if (!first)
                    {
                        std::cout << ",";
                    }
                    std::cout << val;
                    first = false;
                }
            }
            std::cout << std::endl;
            return 0;
        }
        else if (command == "interp_grid_file")
        {
            // Interpolator with file input: sabr_cli interp_grid_file method data_file "Kq1,Kq2" "Tq1,Tq2"
            if (argc != 6)
            {
                std::cerr << "Error: interp_grid_file requires 4 arguments\n";
                print_usage();
                return 1;
            }

            auto method = parse_method(argv[2]);
            std::string file_path = argv[3];
            auto grid_strikes = parse_csv_doubles(argv[4]);
            auto grid_maturities = parse_csv_doubles(argv[5]);

            std::vector<double> strikes;
            std::vector<double> maturities;
            std::vector<double> ivs;
            if (!read_interpolator_data(file_path, strikes, maturities, ivs))
            {
                std::cerr << "Error: failed to read interpolator data file\n";
                return 1;
            }

            if (strikes.size() != maturities.size() || strikes.size() != ivs.size())
            {
                std::cerr << "Error: strikes, maturities, and ivs must have same length\n";
                return 1;
            }

            if (grid_strikes.empty() || grid_maturities.empty())
            {
                std::cerr << "Error: grid_strikes and grid_maturities must be non-empty\n";
                return 1;
            }

            Interpolator interpolator(strikes, maturities, ivs, method);

            std::cout << std::fixed << std::setprecision(6);
            bool first = true;
            for (double K : grid_strikes)
            {
                for (double T : grid_maturities)
                {
                    double val = interpolator.evaluate(K, T);
                    if (!first)
                    {
                        std::cout << ",";
                    }
                    std::cout << val;
                    first = false;
                }
            }
            std::cout << std::endl;
            return 0;
        }
        else if (command == "bs_prices")
        {
            // Black-Scholes batch pricing: sabr_cli bs_prices S "K1,K2,..." T r "IV1,IV2,..." [q]
            if (argc < 7 || argc > 8)
            {
                std::cerr << "Error: bs_prices requires 5-6 arguments\n";
                return 1;
            }

            double S = std::stod(argv[2]);
            std::vector<double> strikes = parse_csv_doubles(argv[3]);
            double T = std::stod(argv[4]);
            double r = std::stod(argv[5]);
            std::vector<double> ivs = parse_csv_doubles(argv[6]);
            double q = (argc == 8) ? std::stod(argv[7]) : 0.0;

            if (strikes.size() != ivs.size())
            {
                std::cerr << "Error: strikes and IVs must have same length\n";
                return 1;
            }

            // Compute call prices using Black-Scholes
            bool first = true;
            for (size_t i = 0; i < strikes.size(); ++i)
            {
                double K = strikes[i];
                double sigma = ivs[i];

                // Black-Scholes call price
                double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
                double d2 = d1 - sigma * std::sqrt(T);

                // Standard normal CDF approximation
                auto norm_cdf = [](double x)
                {
                    return 0.5 * std::erfc(-x * M_SQRT1_2);
                };

                double call_price = S * std::exp(-q * T) * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);

                if (!first)
                    std::cout << ",";
                std::cout << std::fixed << std::setprecision(8) << call_price;
                first = false;
            }
            std::cout << std::endl;
            return 0;
        }
        else if (command == "check_butterfly")
        {
            // Butterfly arbitrage check: sabr_cli check_butterfly "K1,K2,..." "C1,C2,..." tolerance
            if (argc != 5)
            {
                std::cerr << "Error: check_butterfly requires 3 arguments\n";
                return 1;
            }

            std::vector<double> strikes = parse_csv_doubles(argv[2]);
            std::vector<double> call_prices = parse_csv_doubles(argv[3]);
            double tolerance = std::stod(argv[4]);

            if (strikes.size() != call_prices.size() || strikes.size() < 3)
            {
                std::cerr << "Error: need at least 3 strikes/prices\n";
                return 1;
            }

            // Check each consecutive triplet
            int violations = 0;
            for (size_t i = 1; i + 1 < strikes.size(); ++i)
            {
                double K_left = strikes[i - 1];
                double K_mid = strikes[i];
                double K_right = strikes[i + 1];
                double C_left = call_prices[i - 1];
                double C_mid = call_prices[i];
                double C_right = call_prices[i + 1];

                double h_left = K_mid - K_left;
                double h_right = K_right - K_mid;

                if (h_left <= 0 || h_right <= 0)
                    continue;

                double butterfly;
                if (std::abs(h_left - h_right) < 1e-6)
                {
                    // Equal spacing: standard butterfly
                    butterfly = C_left - 2.0 * C_mid + C_right;
                }
                else
                {
                    // Unequal spacing: linear interpolation check
                    double alpha = h_right / (h_left + h_right);
                    double C_mid_linear = alpha * C_left + (1.0 - alpha) * C_right;
                    butterfly = C_mid_linear - C_mid;
                }

                if (butterfly < -tolerance)
                {
                    violations++;
                }
            }

            // Output: number of violations
            std::cout << violations << std::endl;
            return 0;
        }
        else if (command == "check_calendar")
        {
            // Calendar arbitrage check: sabr_cli check_calendar K "T1,T2,..." "C1,C2,..." tolerance
            if (argc != 5)
            {
                std::cerr << "Error: check_calendar requires 3 arguments\n";
                return 1;
            }

            std::vector<double> maturities = parse_csv_doubles(argv[2]);
            std::vector<double> call_prices = parse_csv_doubles(argv[3]);
            double tolerance = std::stod(argv[4]);

            if (maturities.size() != call_prices.size() || maturities.size() < 2)
            {
                std::cerr << "Error: need at least 2 maturities/prices\n";
                return 1;
            }

            // Check each consecutive pair
            int violations = 0;
            for (size_t i = 0; i + 1 < maturities.size(); ++i)
            {
                double calendar = call_prices[i + 1] - call_prices[i];
                if (calendar < -tolerance)
                {
                    violations++;
                }
            }

            // Output: number of violations
            std::cout << violations << std::endl;
            return 0;
        }
        else if (command == "tv_sigma_to_w")
        {
            // Convert IV to total variance: w = σ²T
            if (argc < 4)
            {
                std::cerr << "Usage: sabr_cli tv_sigma_to_w <sigma_csv> <maturities_csv>\n";
                return 1;
            }

            try
            {
                std::vector<double> sigma_grid = parse_csv_doubles(argv[2]);
                std::vector<double> maturities = parse_csv_doubles(argv[3]);

                if (sigma_grid.empty() || maturities.empty())
                    throw std::runtime_error("Empty input");

                size_t n_maturities = maturities.size();
                size_t n_strikes = sigma_grid.size() / n_maturities;

                if (n_strikes * n_maturities != sigma_grid.size())
                    throw std::runtime_error("Size mismatch");

                auto w_grid = TotalVariance::sigma_to_total_variance(
                    sigma_grid, maturities, n_strikes, n_maturities);

                // Output comma-separated
                for (size_t i = 0; i < w_grid.size(); ++i)
                {
                    if (i > 0)
                        std::cout << ",";
                    std::cout << std::fixed << std::setprecision(8) << w_grid[i];
                }
                std::cout << std::endl;
                return 0;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error: " << e.what() << std::endl;
                return 1;
            }
        }
        else if (command == "tv_w_to_sigma")
        {
            // Convert total variance to IV: σ = √(w/T)
            if (argc < 4)
            {
                std::cerr << "Usage: sabr_cli tv_w_to_sigma <w_csv> <maturities_csv>\n";
                return 1;
            }

            try
            {
                std::vector<double> w_grid = parse_csv_doubles(argv[2]);
                std::vector<double> maturities = parse_csv_doubles(argv[3]);

                if (w_grid.empty() || maturities.empty())
                    throw std::runtime_error("Empty input");

                size_t n_maturities = maturities.size();
                size_t n_strikes = w_grid.size() / n_maturities;

                if (n_strikes * n_maturities != w_grid.size())
                    throw std::runtime_error("Size mismatch");

                auto sigma_grid = TotalVariance::total_variance_to_sigma(
                    w_grid, maturities, n_strikes, n_maturities);

                // Output comma-separated
                for (size_t i = 0; i < sigma_grid.size(); ++i)
                {
                    if (i > 0)
                        std::cout << ",";
                    std::cout << std::fixed << std::setprecision(8) << sigma_grid[i];
                }
                std::cout << std::endl;
                return 0;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error: " << e.what() << std::endl;
                return 1;
            }
        }
        else if (command == "tv_enforce_monotonic")
        {
            // Enforce monotonicity: ∂w/∂T ≥ 0
            if (argc < 5)
            {
                std::cerr << "Usage: sabr_cli tv_enforce_monotonic <w_csv> <n_strikes> <n_maturities>\n";
                return 1;
            }

            try
            {
                std::vector<double> w_grid = parse_csv_doubles(argv[2]);
                size_t n_strikes = std::stoull(argv[3]);
                size_t n_maturities = std::stoull(argv[4]);

                if (n_strikes * n_maturities != w_grid.size())
                    throw std::runtime_error("Size mismatch");

                auto w_corrected = TotalVariance::enforce_monotonicity(
                    w_grid, n_strikes, n_maturities);

                // Output comma-separated
                for (size_t i = 0; i < w_corrected.size(); ++i)
                {
                    if (i > 0)
                        std::cout << ",";
                    std::cout << std::fixed << std::setprecision(8) << w_corrected[i];
                }
                std::cout << std::endl;
                return 0;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error: " << e.what() << std::endl;
                return 1;
            }
        }
        else if (command == "tv_validate")
        {
            // Validate arbitrage-free
            if (argc < 5)
            {
                std::cerr << "Usage: sabr_cli tv_validate <w_csv> <n_strikes> <n_maturities>\n";
                return 1;
            }

            try
            {
                std::vector<double> w_grid = parse_csv_doubles(argv[2]);
                size_t n_strikes = std::stoull(argv[3]);
                size_t n_maturities = std::stoull(argv[4]);
                double tolerance = argc > 5 ? std::stod(argv[5]) : 1e-6;

                if (n_strikes * n_maturities != w_grid.size())
                    throw std::runtime_error("Size mismatch");

                auto [calendar_viols, butterfly_viols] = TotalVariance::validate_arbitrage_free(
                    w_grid, n_strikes, n_maturities, tolerance);

                // Output: calendar_violations,butterfly_violations
                std::cout << calendar_viols << "," << butterfly_viols << std::endl;
                return 0;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error: " << e.what() << std::endl;
                return 1;
            }
        }
        else if (command == "tv_lee_bounds")
        {
            // Compute Lee moment bounds
            if (argc < 3)
            {
                std::cerr << "Usage: sabr_cli tv_lee_bounds <w_atm>\n";
                return 1;
            }

            try
            {
                double w_atm = std::stod(argv[2]);
                auto [left_bound, right_bound] = TotalVariance::compute_lee_bounds(w_atm);

                // Output: left_bound,right_bound
                std::cout << std::fixed << std::setprecision(8)
                          << left_bound << "," << right_bound << std::endl;
                return 0;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error: " << e.what() << std::endl;
                return 1;
            }
        }
        else
        {
            std::cerr << "Error: Unknown command '" << command << "'\n";
            print_usage();
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
