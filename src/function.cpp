#include <mutex>

#include <nano/core/strutil.h>
#include <nano/function/benchmark/axis_ellipsoid.h>
#include <nano/function/benchmark/cauchy.h>
#include <nano/function/benchmark/chung_reynolds.h>
#include <nano/function/benchmark/dixon_price.h>
#include <nano/function/benchmark/elastic_net.h>
#include <nano/function/benchmark/exponential.h>
#include <nano/function/benchmark/geometric.h>
#include <nano/function/benchmark/kinks.h>
#include <nano/function/benchmark/powell.h>
#include <nano/function/benchmark/qing.h>
#include <nano/function/benchmark/quadratic.h>
#include <nano/function/benchmark/rosenbrock.h>
#include <nano/function/benchmark/rotated_ellipsoid.h>
#include <nano/function/benchmark/sargan.h>
#include <nano/function/benchmark/schumer_steiglitz.h>
#include <nano/function/benchmark/sphere.h>
#include <nano/function/benchmark/styblinski_tang.h>
#include <nano/function/benchmark/trid.h>
#include <nano/function/benchmark/zakharov.h>

using namespace nano;

function_t::function_t(string_t id, tensor_size_t size)
    : clonable_t(std::move(id))
    , m_size(size)
{
}

function_t::function_t(const function_t&) = default;

function_t& function_t::operator=(const function_t&) = default;

void function_t::convex(bool convex)
{
    m_convex = convex;
}

void function_t::smooth(bool smooth)
{
    m_smooth = smooth;
}

void function_t::strong_convexity(scalar_t sconvexity)
{
    m_sconvexity = sconvexity;
}

string_t function_t::name(bool with_size) const
{
    return with_size ? scat(type_id(), "[", size(), "D]") : type_id();
}

bool function_t::constrain(constraint_t&& constraint)
{
    if (compatible(constraint, *this))
    {
        m_constraints.emplace_back(std::move(constraint));
        return true;
    }
    return false;
}

bool function_t::constrain(scalar_t min, scalar_t max, tensor_size_t dimension)
{
    if (min < max && dimension >= 0 && dimension < size())
    {
        m_constraints.emplace_back(constraint::minimum_t{min, dimension});
        m_constraints.emplace_back(constraint::maximum_t{max, dimension});
        return true;
    }
    return false;
}

bool function_t::constrain(scalar_t min, scalar_t max)
{
    if (min < max)
    {
        for (tensor_size_t i = 0, size = this->size(); i < size; ++i)
        {
            m_constraints.emplace_back(constraint::minimum_t{min, i});
            m_constraints.emplace_back(constraint::maximum_t{max, i});
        }
        return true;
    }
    return false;
}

bool function_t::constrain(const vector_t& min, const vector_t& max)
{
    if (min.size() == size() && max.size() == size() && (max - min).minCoeff() > 0.0)
    {
        for (tensor_size_t i = 0, size = this->size(); i < size; ++i)
        {
            m_constraints.emplace_back(constraint::minimum_t{min(i), i});
            m_constraints.emplace_back(constraint::maximum_t{max(i), i});
        }
        return true;
    }
    return false;
}

bool function_t::valid(const vector_t& x) const
{
    assert(x.size() == size());

    const auto op = [&](const constraint_t& constraint)
    { return ::nano::valid(constraint, x) < std::numeric_limits<scalar_t>::epsilon(); };

    return std::all_of(m_constraints.begin(), m_constraints.end(), op);
}

const constraints_t& function_t::constraints() const
{
    return m_constraints;
}

scalar_t function_t::vgrad(const vector_t& x, vector_t* gx) const
{
    m_fcalls += 1;
    m_gcalls += (gx != nullptr) ? 1 : 0;
    return do_vgrad(x, gx);
}

tensor_size_t function_t::fcalls() const
{
    return m_fcalls;
}

tensor_size_t function_t::gcalls() const
{
    return m_gcalls;
}

void function_t::clear_statistics() const
{
    m_fcalls = 0;
    m_gcalls = 0;
}

rfunction_t function_t::make(tensor_size_t, tensor_size_t) const
{
    assert(false);
    return rfunction_t{};
}

factory_t<function_t>& function_t::all()
{
    static auto manager = factory_t<function_t>{};
    const auto  op      = []()
    {
        manager.add<function_trid_t>("Trid function: https://www.sfu.ca/~ssurjano/trid.html");
        manager.add<function_qing_t>("Qing function: http://benchmarkfcns.xyz/benchmarkfcns/qingfcn.html");
        manager.add<function_kinks_t>("random kinks: f(x_ = sum(|x - K_i|, i)");
        manager.add<function_cauchy_t>("Cauchy function: f(x) = log(1 + x.dot(x))");
        manager.add<function_sargan_t>(
            "Sargan function: http://infinity77.net/global_optimization/test_functions_nd_S.html");
        manager.add<function_powell_t>("Powell function: https://www.sfu.ca/~ssurjano/powell.html");
        manager.add<function_sphere_t>("sphere function: f(x) = x.dot(x)");
        manager.add<function_zakharov_t>("Zakharov function: https://www.sfu.ca/~ssurjano/zakharov.html");
        manager.add<function_quadratic_t>("random quadratic function: f(x) = x.dot(a) + x * A * x, where A is PD");
        manager.add<function_rosenbrock_t>(
            "Rosenbrock function: https://en.wikipedia.org/wiki/Test_functions_for_optimization");
        manager.add<function_exponential_t>("exponential function: f(x) = exp(1 + x.dot(x) / D)");
        manager.add<function_dixon_price_t>("Dixon-Price function: https://www.sfu.ca/~ssurjano/dixonpr.html");
        manager.add<function_chung_reynolds_t>("Chung-Reynolds function: f(x) = (x.dot(x))^2");
        manager.add<function_axis_ellipsoid_t>("axis-parallel hyper-ellipsoid function: f(x) = sum(i*x+i^2, i=1,D)");
        manager.add<function_styblinski_tang_t>("Styblinski-Tang function: https://www.sfu.ca/~ssurjano/stybtang.html");
        manager.add<function_schumer_steiglitz_t>("Schumer-Steiglitz No. 02 function: f(x) = sum(x_i^4, i=1,D)");
        manager.add<function_rotated_ellipsoid_t>(
            "rotated hyper-ellipsoid function: https://www.sfu.ca/~ssurjano/rothyp.html");
        manager.add<function_geometric_optimization_t>(
            "generic geometric optimization function: f(x) = sum(i, exp(alpha_i + a_i.dot(x)))");

        manager.add<function_enet_mse_t>("mean squared error with ridge-like regularization", 10, 0.0, 1.0);
        manager.add<function_enet_mse_t>("mean squared error with lasso-like regularization", 10, 1.0, 0.0);
        manager.add<function_enet_mse_t>("mean squared error with elastic net-like regularization", 10, 1.0, 1.0);

        manager.add<function_enet_mae_t>("mean squared error with ridge-like regularization", 10, 0.0, 1.0);
        manager.add<function_enet_mae_t>("mean squared error with lasso-like regularization", 10, 1.0, 0.0);
        manager.add<function_enet_mae_t>("mean squared error with elastic net-like regularization", 10, 1.0, 1.0);

        manager.add<function_enet_hinge_t>("hinge loss (linear SVM) with ridge-like regularization", 10, 0.0, 1.0);
        manager.add<function_enet_hinge_t>("hinge loss (linear SVM) with lasso-like regularization", 10, 1.0, 0.0);
        manager.add<function_enet_hinge_t>("hinge loss (linear SVM) with elastic net-like regularization", 10, 1.0,
                                           1.0);

        manager.add<function_enet_cauchy_t>("cauchy loss (robust regression) with ridge-like regularization", 10, 0.0,
                                            1.0);
        manager.add<function_enet_cauchy_t>("cauchy loss (robust regression) with lasso-like regularization", 10, 1.0,
                                            0.0);
        manager.add<function_enet_cauchy_t>("cauchy loss (robust regression) with elastic net-like regularization", 10,
                                            1.0, 1.0);

        manager.add<function_enet_logistic_t>("logistic regression with ridge-like regularization", 10, 0.0, 1.0);
        manager.add<function_enet_logistic_t>("logistic regression with lasso-like regularization", 10, 1.0, 0.0);
        manager.add<function_enet_logistic_t>("logistic regression with elastic net-like regularization", 10, 1.0, 1.0);
    };

    static std::once_flag flag;
    std::call_once(flag, op);

    return manager;
}

rfunctions_t function_t::make(function_t::config_t config, const std::regex& id_regex)
{
    const auto convexity  = config.m_convexity;
    const auto smoothness = config.m_smoothness;

    const auto min_dims = std::min(config.m_min_dims, config.m_max_dims);
    const auto max_dims = std::max(config.m_min_dims, config.m_max_dims);
    assert(min_dims >= 1);

    const auto& factory = function_t::all();
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
