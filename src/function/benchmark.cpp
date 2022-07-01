#include <mutex>
#include <nano/function/benchmark.h>

#include <nano/function/axis_ellipsoid.h>
#include <nano/function/cauchy.h>
#include <nano/function/chung_reynolds.h>
#include <nano/function/dixon_price.h>
#include <nano/function/elastic_net.h>
#include <nano/function/exponential.h>
#include <nano/function/geometric.h>
#include <nano/function/kinks.h>
#include <nano/function/powell.h>
#include <nano/function/qing.h>
#include <nano/function/quadratic.h>
#include <nano/function/rosenbrock.h>
#include <nano/function/rotated_ellipsoid.h>
#include <nano/function/sargan.h>
#include <nano/function/schumer_steiglitz.h>
#include <nano/function/sphere.h>
#include <nano/function/styblinski_tang.h>
#include <nano/function/trid.h>
#include <nano/function/zakharov.h>

using namespace nano;

function_factory_t& benchmark_function_t::all()
{
    static function_factory_t manager;

    static std::once_flag flag;
    std::call_once(
        flag,
        []()
        {
            manager.add<function_trid_t>("trid", "Trid function: https://www.sfu.ca/~ssurjano/trid.html");
            manager.add<function_qing_t>("qing", "Qing function: http://benchmarkfcns.xyz/benchmarkfcns/qingfcn.html");
            manager.add<function_kinks_t>("kinks", "random kinks: f(x_ = sum(|x - K_i|, i)");
            manager.add<function_cauchy_t>("cauchy", "Cauchy function: f(x) = log(1 + x.dot(x))");
            manager.add<function_sargan_t>(
                "sargan", "Sargan function: http://infinity77.net/global_optimization/test_functions_nd_S.html");
            manager.add<function_powell_t>("powell", "Powell function: https://www.sfu.ca/~ssurjano/powell.html");
            manager.add<function_sphere_t>("sphere", "sphere function: f(x) = x.dot(x)");
            manager.add<function_zakharov_t>("zakharov",
                                             "Zakharov function: https://www.sfu.ca/~ssurjano/zakharov.html");
            manager.add<function_quadratic_t>("quadratic",
                                              "random quadratic function: f(x) = x.dot(a) + x * A * x, where A is PD");
            manager.add<function_rosenbrock_t>(
                "rosenbrock", "Rosenbrock function: https://en.wikipedia.org/wiki/Test_functions_for_optimization");
            manager.add<function_exponential_t>("exponential", "exponential function: f(x) = exp(1 + x.dot(x) / D)");
            manager.add<function_dixon_price_t>("dixon-price",
                                                "Dixon-Price function: https://www.sfu.ca/~ssurjano/dixonpr.html");
            manager.add<function_chung_reynolds_t>("chung-reynolds", "Chung-Reynolds function: f(x) = (x.dot(x))^2");
            manager.add<function_axis_ellipsoid_t>(
                "axis-ellipsoid", "axis-parallel hyper-ellipsoid function: f(x) = sum(i*x+i^2, i=1,D)");
            manager.add<function_styblinski_tang_t>(
                "styblinski-tang", "Styblinski-Tang function: https://www.sfu.ca/~ssurjano/stybtang.html");
            manager.add<function_schumer_steiglitz_t>("schumer-steiglitz",
                                                      "Schumer-Steiglitz No. 02 function: f(x) = sum(x_i^4, i=1,D)");
            manager.add<function_rotated_ellipsoid_t>(
                "rotated-ellipsoid", "rotated hyper-ellipsoid function: https://www.sfu.ca/~ssurjano/rothyp.html");
            manager.add<function_geometric_optimization_t>(
                "geometric", "generic geometric optimization function: f(x) = sum(i, exp(alpha_i + a_i.dot(x)))");

            manager.add<function_enet_mse_t>("mse+ridge", "mean squared error with ridge-like regularization", 10, 0.0,
                                             1.0);
            manager.add<function_enet_mse_t>("mse+lasso", "mean squared error with lasso-like regularization", 10, 1.0,
                                             0.0);
            manager.add<function_enet_mse_t>("mse+elasticnet",
                                             "mean squared error with elastic net-like regularization", 10, 1.0, 1.0);

            manager.add<function_enet_mae_t>("mae+ridge", "mean squared error with ridge-like regularization", 10, 0.0,
                                             1.0);
            manager.add<function_enet_mae_t>("mae+lasso", "mean squared error with lasso-like regularization", 10, 1.0,
                                             0.0);
            manager.add<function_enet_mae_t>("mae+elasticnet",
                                             "mean squared error with elastic net-like regularization", 10, 1.0, 1.0);

            manager.add<function_enet_hinge_t>("hinge+ridge", "hinge loss (linear SVM) with ridge-like regularization",
                                               10, 0.0, 1.0);
            manager.add<function_enet_hinge_t>("hinge+lasso", "hinge loss (linear SVM) with lasso-like regularization",
                                               10, 1.0, 0.0);
            manager.add<function_enet_hinge_t>(
                "hinge+elasticnet", "hinge loss (linear SVM) with elastic net-like regularization", 10, 1.0, 1.0);

            manager.add<function_enet_cauchy_t>(
                "cauchy+ridge", "cauchy loss (robust regression) with ridge-like regularization", 10, 0.0, 1.0);
            manager.add<function_enet_cauchy_t>(
                "cauchy+lasso", "cauchy loss (robust regression) with lasso-like regularization", 10, 1.0, 0.0);
            manager.add<function_enet_cauchy_t>("cauchy+elasticnet",
                                                "cauchy loss (robust regression) with elastic net-like regularization",
                                                10, 1.0, 1.0);

            manager.add<function_enet_logistic_t>("logistic+ridge",
                                                  "logistic regression with ridge-like regularization", 10, 0.0, 1.0);
            manager.add<function_enet_logistic_t>("logistic+lasso",
                                                  "logistic regression with lasso-like regularization", 10, 1.0, 0.0);
            manager.add<function_enet_logistic_t>(
                "logistic+elasticnet", "logistic regression with elastic net-like regularization", 10, 1.0, 1.0);
        });

    return manager;
}

rfunctions_t benchmark_function_t::make(benchmark_function_t::config_t config, const std::regex& id_regex)
{
    const auto convexity  = config.m_convexity;
    const auto smoothness = config.m_smoothness;

    const auto min_dims = std::min(config.m_min_dims, config.m_max_dims);
    const auto max_dims = std::max(config.m_min_dims, config.m_max_dims);
    assert(min_dims >= 1);

    const auto& factory = benchmark_function_t::all();
    const auto  ids     = factory.ids(id_regex);

    rfunctions_t functions;
    for (tensor_size_t dims = min_dims; dims <= max_dims;)
    {
        for (const auto& id : ids)
        {
            auto function = factory.get(id);

            if ((convexity == convexity::ignore || (function->convex() == (convexity == convexity::yes))) &&
                (smoothness == smoothness::ignore || (function->smooth() == (smoothness == smoothness::yes))))
            {
                functions.push_back(function->make(dims, config.m_summands));
            }
        }

        if (dims < 4)
        {
            ++dims;
        }
        else
        {
            dims *= 2;
        }
    }

    return functions;
} // LCOV_EXCL_LINE
