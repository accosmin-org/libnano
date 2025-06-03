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
#include <function/program/eqcqp.h>
#include <function/program/numopt162.h>
#include <function/program/numopt1625.h>
#include <function/program/portfolio.h>
#include <function/program/randomqp.h>

#include <function/mlearn/elasticnet.h>
#include <function/mlearn/lasso.h>
#include <function/mlearn/ridge.h>

#include <nano/core/strutil.h>

using namespace nano;

namespace
{
void make_function(rfunction_t& function, const tensor_size_t dims, rfunctions_t& functions)
{
    // NB: generate different regularization parameters for various linear ML models
    // to assess the difficulty of the resulting numerical optimization problems.
    if (function->parameter_if("lasso::alpha1") != nullptr)
    {
        for (const auto alpha1 : {1e+0, 1e+2, 1e+4, 1e+6})
        {
            function->config("lasso::alpha1", alpha1);
            functions.emplace_back(function->make(dims));
        }
    }

    else if (function->parameter_if("ridge::alpha2") != nullptr)
    {
        for (const auto alpha2 : {1e+0, 1e+2, 1e+4, 1e+6})
        {
            function->config("ridge::alpha2", alpha2);
            functions.emplace_back(function->make(dims));
        }
    }

    else if (function->parameter_if("elasticnet::alpha1") != nullptr)
    {
        for (const auto alpha12 : {1e+0, 1e+2, 1e+4, 1e+6})
        {
            function->config("elasticnet::alpha1", alpha12);
            function->config("elasticnet::alpha2", alpha12);
            functions.emplace_back(function->make(dims));
        }
    }

    else if (function->parameter_if("cvx48b::lambda") != nullptr)
    {
        for (const auto lambda : {-1e-6, -1e+0, -1e+1, -1e+2})
        {
            function->config("cvx48b::lambda", lambda);
            functions.emplace_back(function->make(dims));
        }
    }

    else if (function->parameter_if("cvx48e-eq::alpha") != nullptr)
    {
        for (const auto alpha : {0.0, 0.5, 1.0})
        {
            function->config("cvx48e-eq::alpha", alpha);
            functions.emplace_back(function->make(dims));
        }
    }

    else if (function->parameter_if("cvx48e-ineq::alpha") != nullptr)
    {
        for (const auto alpha : {1e-6, 0.5, 1.0})
        {
            function->config("cvx48e-ineq::alpha", alpha);
            functions.emplace_back(function->make(dims));
        }
    }

    else if (function->parameter_if("cvx48f::alpha") != nullptr)
    {
        for (const auto alpha : {0.0, 0.3, 0.7, 1.0})
        {
            function->config("cvx48f::alpha", alpha);
            functions.emplace_back(function->make(dims));
        }
    }

    else if (function->parameter_if("numopt162::neqs") != nullptr)
    {
        for (const auto neqs : {1e-6, 0.1, 0.2, 0.5, 0.8, 1.0})
        {
            function->config("numopt162::neqs", neqs);
            functions.emplace_back(function->make(dims));
        }
    }

    else if (function->parameter_if("randomqp::nineqs") != nullptr)
    {
        for (const auto nineqs : {5.0, 10.0, 20.0})
        {
            function->config("randomqp::nineqs", nineqs);
            functions.emplace_back(function->make(dims));
        }
    }

    else if (function->parameter_if("eqcqp::neqs") != nullptr)
    {
        for (const auto neqs : {0.1, 0.5, 0.9})
        {
            function->config("eqcqp::neqs", neqs);
            functions.emplace_back(function->make(dims));
        }
    }

    else if (function->parameter_if("portfolio::factors") != nullptr)
    {
        for (const auto factors : {0.1, 0.5, 0.9})
        {
            function->config("portfolio::factors", factors);
            functions.emplace_back(function->make(dims));
        }
    }

    else
    {
        functions.emplace_back(function->make(dims));
    }
}
} // namespace

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
    return with_size ? scat(do_name(), "[", size(), "D]") : do_name();
}

string_t function_t::do_name() const
{
    return type_id();
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

void function_t::clear_constraints()
{
    m_constraints.clear();
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

        manager.add<linear_program_cvx48b_t>("linear program: ex. 4.8(b), 'Convex Optimization', 2nd edition");
        manager.add<linear_program_cvx48c_t>("linear program: ex. 4.8(c), 'Convex Optimization', 2nd edition");
        manager.add<linear_program_cvx48d_eq_t>(
            "linear program: ex. 4.8(d) - equality case, 'Convex Optimization', 2nd edition");
        manager.add<linear_program_cvx48d_ineq_t>(
            "linear program: ex. 4.8(d) - inequality case, 'Convex Optimization', 2nd edition");
        manager.add<linear_program_cvx48e_eq_t>(
            "linear program: ex. 4.8(e) - equality case, 'Convex Optimization', 2nd edition");
        manager.add<linear_program_cvx48e_ineq_t>(
            "linear program: ex. 4.8(e) - inequality case, 'Convex Optimization', 2nd edition");
        manager.add<linear_program_cvx48f_t>("linear program: ex. 4.8(f), 'Convex Optimization', 2nd edition");
        manager.add<linear_program_cvx49_t>("linear program: ex. 4.9, 'Convex Optimization', 2nd edition");
        manager.add<linear_program_cvx410_t>("linear program: ex. 4.10, 'Convex Optimization', 2nd edition");

        manager.add<quadratic_program_numopt162_t>(
            "quadratic program: ex. 16.2, 'Numerical optimization', 2nd edition");
        manager.add<quadratic_program_numopt1625_t>(
            "quadratic program: ex. 16.25, 'Numerical optimization', 2nd edition");
        manager.add<quadratic_program_randomqp_t>(
            "random quadratic program: A.1, 'OSQP: an operator splitting solver for quadratic programs'");
        manager.add<quadratic_program_eqcqp_t>(
            "equality constrained quadratic program: A.2, 'OSQP: an operator splitting solver for quadratic programs'");
        manager.add<quadratic_program_portfolio_t>(
            "portfolio optimization: A.4, 'OSQP: an operator splitting solver for quadratic programs'");
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

            switch (config.m_function_type)
            {
            case function_type::convex:
                if (function->convex() && function->constraints().empty())
                {
                    make_function(function, dims, functions);
                }
                break;

            case function_type::smooth:
                if (function->smooth() && function->constraints().empty())
                {
                    make_function(function, dims, functions);
                }
                break;

            case function_type::convex_smooth:
                if (function->convex() && function->smooth() && function->constraints().empty())
                {
                    make_function(function, dims, functions);
                }
                break;

            case function_type::convex_nonsmooth:
                if (function->convex() && !function->smooth() && function->constraints().empty())
                {
                    make_function(function, dims, functions);
                }
                break;

            case function_type::linear_program:
                if (dynamic_cast<const linear_program_t*>(function.get()) != nullptr)
                {
                    make_function(function, dims, functions);
                }
                break;

            case function_type::quadratic_program:
                if (dynamic_cast<const quadratic_program_t*>(function.get()) != nullptr)
                {
                    make_function(function, dims, functions);
                }
                break;

            case function_type::any:
                functions.emplace_back(function->make(dims));
                break;
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
