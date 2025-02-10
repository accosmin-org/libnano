#include <mutex>

#include <function/benchmark/axis_ellipsoid.h>
#include <function/benchmark/cauchy.h>
#include <function/benchmark/chained_cb3I.h>
#include <function/benchmark/chained_cb3II.h>
#include <function/benchmark/chained_lq.h>
#include <function/benchmark/chung_reynolds.h>
#include <function/benchmark/dixon_price.h>
#include <function/benchmark/exponential.h>
#include <function/benchmark/geometric.h>
#include <function/benchmark/kinks.h>
#include <function/benchmark/maxhilb.h>
#include <function/benchmark/maxq.h>
#include <function/benchmark/maxquad.h>
#include <function/benchmark/powell.h>
#include <function/benchmark/qing.h>
#include <function/benchmark/quadratic.h>
#include <function/benchmark/rosenbrock.h>
#include <function/benchmark/rotated_ellipsoid.h>
#include <function/benchmark/sargan.h>
#include <function/benchmark/schumer_steiglitz.h>
#include <function/benchmark/sphere.h>
#include <function/benchmark/styblinski_tang.h>
#include <function/benchmark/trid.h>
#include <function/benchmark/zakharov.h>

#include <function/program/cvx410.h>
#include <function/program/cvx48b.h>
#include <function/program/cvx48c.h>
#include <function/program/cvx48d.h>
#include <function/program/cvx48e.h>
#include <function/program/cvx48f.h>
#include <function/program/cvx49.h>
#include <function/program/numopt162.h>
#include <function/program/numopt1625.h>

#include <function/mlearn/elasticnet.h>
#include <function/mlearn/lasso.h>
#include <function/mlearn/ridge.h>

#include <nano/core/strutil.h>

using namespace nano;

function_t::function_t(string_t id, tensor_size_t size)
    : typed_t(std::move(id))
    , m_size(size)
{
}

void function_t::convex(const convexity c)
{
    m_convexity = c;
}

void function_t::smooth(const smoothness s)
{
    m_smoothness = s;
}

void function_t::strong_convexity(const scalar_t strong_convexity)
{
    m_strong_convexity = strong_convexity;
}

string_t function_t::name(const bool with_size) const
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

const constraints_t& function_t::constraints() const
{
    return m_constraints;
}

bool function_t::valid(const vector_t& x) const
{
    const auto op = [&](const constraint_t& constraint)
    { return ::nano::valid(constraint, x) < std::numeric_limits<scalar_t>::epsilon(); };

    return x.size() == size() && std::all_of(m_constraints.begin(), m_constraints.end(), op);
}

tensor_size_t function_t::n_equalities() const
{
    return ::nano::n_equalities(m_constraints);
}

tensor_size_t function_t::n_inequalities() const
{
    return ::nano::n_inequalities(m_constraints);
}

scalar_t function_t::operator()(vector_cmap_t x, vector_map_t gx) const
{
    assert(x.size() == size());
    assert(gx.size() == 0 || gx.size() == size());

    m_fcalls += 1;
    m_gcalls += (gx.size() == size()) ? 1 : 0;
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

rfunction_t function_t::make(tensor_size_t) const
{
    return rfunction_t{};
}

bool function_t::optimum(vector_t xbest)
{
    if (xbest.size() != size())
    {
        return false;
    }
    else
    {
        m_optimum.m_xbest = std::move(xbest);
        m_optimum.m_fbest = do_vgrad(m_optimum.m_xbest, vector_map_t{});
        return true;
    }
}

bool function_t::optimum(const scalar_t fbest)
{
    if (!std::isfinite(fbest))
    {
        return false;
    }
    else
    {
        m_optimum.m_fbest = fbest;
        return true;
    }
}

const optimum_t& function_t::optimum() const
{
    return m_optimum;
}

factory_t<function_t>& function_t::all()
{
    static auto manager = factory_t<function_t>{};
    const auto  op      = []()
    {
        manager.add<function_maxq_t>("MAXQ function: f(x) = max(i, x_i^2)");
        manager.add<function_maxquad_t>("MAXQUAD function: f(x) = max(k, x.dot(A_k*x) - b_k.dot(x))");
        manager.add<function_maxhilb_t>("MAXHILB function: f(x) = max(i, sum(j, xj / (i + j = 1))");
        manager.add<function_chained_lq_t>("chained LQ function (see documentation)");
        manager.add<function_chained_cb3I_t>("chained CB3 I function (see documentation)");
        manager.add<function_chained_cb3II_t>("chained CB3 II function (see documentation)");

        manager.add<function_trid_t>("Trid function: https://www.sfu.ca/~ssurjano/trid.html");
        manager.add<function_qing_t>("Qing function: http://benchmarkfcns.xyz/benchmarkfcns/qingfcn.html");
        manager.add<function_kinks_t>("random kinks: f(x) = sum(|x - K_i|, i)");
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

        manager.add<function_lasso_mse_t>("mean squared error (MSE) with lasso regularization");
        manager.add<function_lasso_mae_t>("mean absolute error (MAE) with lasso regularization");
        manager.add<function_lasso_hinge_t>("hinge loss (linear SVM) with lasso regularization");
        manager.add<function_lasso_cauchy_t>("cauchy loss (robust regression) with lasso regularization");
        manager.add<function_lasso_logistic_t>("logistic loss (logistic regression) with lasso regularization");

        manager.add<function_ridge_mse_t>("mean squared error (MSE) with ridge regularization");
        manager.add<function_ridge_mae_t>("mean absolute error (MAE) with ridge regularization");
        manager.add<function_ridge_hinge_t>("hinge loss (linear SVM) with ridge regularization");
        manager.add<function_ridge_cauchy_t>("cauchy loss (robust regression) with ridge regularization");
        manager.add<function_ridge_logistic_t>("logistic loss (logistic regression) with ridge regularization");

        manager.add<function_elasticnet_mse_t>("mean squared error (MSE) with elastic net regularization");
        manager.add<function_elasticnet_mae_t>("mean absolute error (MAE) with elastic net regularization");
        manager.add<function_elasticnet_hinge_t>("hinge loss (linear SVM) with elastic net regularization");
        manager.add<function_elasticnet_cauchy_t>("cauchy loss (robust regression) with elastic net regularization");
        manager.add<function_elasticnet_logistic_t>(
            "logistic loss (logistic regression) with elastic net regularization");

        // manager.add<linear_program_cvx48b_t>("linear program, exercise 4.8 (b), 'Convex Optimization'");
        // manager.add<linear_program_cvx48b_t>("linear program, exercise 4.8 (b), 'Convex Optimization'");

        // TODO: fix type_id for configuration benchmark functions
        // TODO: change fixture to update alpha1, alpha2 ... for testing solvers...
    };

    static std::once_flag flag;
    std::call_once(flag, op);

    return manager;
}

rfunctions_t function_t::make(const function_t::config_t& config, const std::regex& id_regex)
{
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
            assert(function != nullptr);

            const auto is_convex = function->convex();
            const auto is_smooth = function->smooth();

            const auto has_constraints = !(function->constraints().empty());

            if ((config.m_convexity == convexity::ignore || is_convex == (config.m_convexity == convexity::yes)) &&
                (config.m_smoothness == smoothness::ignore || is_smooth == (config.m_smoothness == smoothness::yes)) &&
                (config.m_constrained == constrained::ignore ||
                 has_constraints == (config.m_constrained == constrained::yes)))
            {
                functions.emplace_back(function->make(dims));
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
