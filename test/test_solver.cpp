#include "fixture/function.h"
#include "fixture/lsearch0.h"
#include "fixture/lsearchk.h"
#include "fixture/solver.h"
#include <iomanip>
#include <nano/core/logger.h>
#include <nano/function/benchmark/sphere.h>
#include <nano/solver/quasi.h>

using namespace nano;

struct solver_description_t
{
    solver_description_t() = default;

    explicit solver_description_t(const solver_type type)
        : m_type(type)
    {
    }

    auto& smooth_config(const minimize_config_t& config)
    {
        m_smooth_config = config;
        return *this;
    }

    auto& nonsmooth_config(const minimize_config_t& config)
    {
        m_nonsmooth_config = config;
        return *this;
    }

    solver_type       m_type{solver_type::line_search};
    minimize_config_t m_smooth_config{};
    minimize_config_t m_nonsmooth_config{};
};

static auto make_description(const string_t& solver_id)
{
    if (solver_id == "cgd-n" || solver_id == "cgd-hs" || solver_id == "cgd-fr" || solver_id == "cgd-pr" ||
        solver_id == "cgd-cd" || solver_id == "cgd-ls" || solver_id == "cgd-dy" || solver_id == "cgd-dycd" ||
        solver_id == "cgd-dyhs" || solver_id == "cgd-frpr" || solver_id == "lbfgs" || solver_id == "sr1" ||
        solver_id == "bfgs" || solver_id == "hoshino" || solver_id == "fletcher")
    {
        return solver_description_t{solver_type::line_search}.smooth_config(
            minimize_config_t{}.epsilon(5e-8).max_evals(1000).expected_maximum_deviation(1e-6));
    }
    else if (solver_id == "dfp")
    {
        // NB: DFP needs many more iterations to reach the solution!
        return solver_description_t{solver_type::line_search}.smooth_config(
            minimize_config_t{}.epsilon(5e-8).max_evals(10000).expected_maximum_deviation(1e-6));
    }
    else if (solver_id == "gd")
    {
        // NB: gradient descent (GD) needs many more iterations to minimize badly conditioned problems!
        return solver_description_t{solver_type::line_search}.smooth_config(
            minimize_config_t{}.epsilon(5e-7).max_evals(10000).expected_maximum_deviation(1e-5));
    }
    else if (solver_id == "sgm" || solver_id == "cocob")
    {
        return solver_description_t{solver_type::non_monotonic}
            .smooth_config(minimize_config_t{}.epsilon(1e-5).max_evals(20000).expected_maximum_deviation(1e-3))
            .nonsmooth_config(minimize_config_t{}.epsilon(1e-5).max_evals(20000).expected_maximum_deviation(1e-2));
    }
    else if (solver_id == "sda" || solver_id == "wda")
    {
        // NB: SDA/WDA can take way too many iterations to converge reliably to the solution!
        // NB: also the distance to the optimum `D` is not usually known and it impacts the convergence and its speed!
        return solver_description_t{solver_type::non_monotonic}.smooth_config(
            minimize_config_t{}.epsilon(1e-3).max_evals(1000).expected_convergence(false).expected_maximum_deviation(
                1e+1));
    }
    else if (solver_id == "ellipsoid" || solver_id == "osga")
    {
        return solver_description_t{solver_type::non_monotonic}
            .smooth_config(minimize_config_t{}.epsilon(5e-8).expected_maximum_deviation(1e-6))
            .nonsmooth_config(minimize_config_t{}.epsilon(5e-8).expected_maximum_deviation(1e-5));
    }
    else
    {
        assert(false);
        return solver_description_t{};
    }
}

static auto make_lsearch0_ids()
{
    return lsearch0_t::all().ids();
}

static auto make_lsearchk_ids()
{
    return lsearchk_t::all().ids();
}

static auto make_solver_ids()
{
    return solver_t::all().ids(std::regex(".+"));
}

static auto make_smooth_solver_ids()
{
    return solver_t::all().ids(std::regex(".+"));
}

static auto make_nonsmooth_solver_ids()
{
    return strings_t{"ellipsoid", "osga", "sgm"}; // FIXME: have all methods converge!!!, "cocob"};
}

static auto make_best_smooth_solver_ids()
{
    return strings_t{"cgd-pr", "lbfgs", "bfgs"};
}

UTEST_BEGIN_MODULE(test_solver_lsearch)

UTEST_CASE(solver_type_str)
{
    UTEST_CHECK_EQUAL(scat(solver_type::line_search), "unconstrained with line-search");
    UTEST_CHECK_EQUAL(scat(solver_type::non_monotonic), "unconstrained non-monotonic");
    UTEST_CHECK_EQUAL(scat(solver_type::constrained), "constrained");
}

UTEST_CASE(state_str)
{
    for (const auto status : enum_values<solver_status>())
    {
        std::stringstream stream;
        stream << status;
        UTEST_CHECK_EQUAL(stream.str(), scat(status));
    }
}

UTEST_CASE(state_valid)
{
    const auto function = function_sphere_t{7};
    const auto state    = solver_state_t{function, make_random_x0(function)};
    UTEST_CHECK(state.valid());
}

UTEST_CASE(state_invalid_fNAN)
{
    const auto function = function_sphere_t{3};
    const auto state    = solver_state_t{function, make_vector<scalar_t>(NAN, 1.0, 0.0)};
    UTEST_CHECK(!state.valid());
}

UTEST_CASE(state_has_descent)
{
    const auto function = function_sphere_t{5};
    const auto state    = solver_state_t{function, make_random_x0(function)};
    UTEST_CHECK(state.has_descent(-state.gx()));
}

UTEST_CASE(state_has_no_descent0)
{
    const auto function = function_sphere_t{2};
    const auto state    = solver_state_t{function, make_random_x0(function)};
    UTEST_CHECK(!state.has_descent(make_vector<scalar_t>(.0, 0.0)));
}

UTEST_CASE(state_has_no_descent1)
{
    const auto function = function_sphere_t{7};
    const auto state    = solver_state_t{function, make_random_x0(function)};
    UTEST_CHECK(!state.has_descent(state.gx()));
}

UTEST_CASE(state_update_if_better)
{
    const auto function = function_sphere_t{2};
    const auto x0       = make_vector<scalar_t>(0.0, 0.0);
    const auto x1       = make_vector<scalar_t>(1.0, 1.0);
    const auto x2       = make_vector<scalar_t>(2.0, 2.0);
    const auto x09      = make_vector<scalar_t>(0.9, 0.9);

    // update path: x1, x2 (up), NaN (div), x1 (=), x09 (down), x0 (down)

    auto state = solver_state_t{function, x1};
    UTEST_CHECK_CLOSE(state.x(), x1, 1e-12);
    UTEST_CHECK_CLOSE(state.fx(), function.vgrad(x1), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(0), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(1), std::numeric_limits<scalar_t>::max(), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(2), std::numeric_limits<scalar_t>::max(), 1e-12);

    UTEST_CHECK(!state.update_if_better(x2, function.vgrad(x2)));

    UTEST_CHECK_CLOSE(state.x(), x1, 1e-12);
    UTEST_CHECK_CLOSE(state.fx(), function.vgrad(x1), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(0), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(1), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(2), std::numeric_limits<scalar_t>::max(), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(3), std::numeric_limits<scalar_t>::max(), 1e-12);

    UTEST_CHECK(!state.update_if_better(x2, std::numeric_limits<scalar_t>::quiet_NaN()));

    UTEST_CHECK_CLOSE(state.x(), x1, 1e-12);
    UTEST_CHECK_CLOSE(state.fx(), function.vgrad(x1), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(0), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(1), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(2), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(3), std::numeric_limits<scalar_t>::max(), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(4), std::numeric_limits<scalar_t>::max(), 1e-12);

    UTEST_CHECK(!state.update_if_better(x1, function.vgrad(x1)));

    UTEST_CHECK_CLOSE(state.x(), x1, 1e-12);
    UTEST_CHECK_CLOSE(state.fx(), function.vgrad(x1), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(0), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(1), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(2), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(3), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(4), std::numeric_limits<scalar_t>::max(), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(5), std::numeric_limits<scalar_t>::max(), 1e-12);

    UTEST_CHECK(state.update_if_better(x09, function.vgrad(x09)));

    UTEST_CHECK_CLOSE(state.x(), x09, 1e-12);
    UTEST_CHECK_CLOSE(state.fx(), function.vgrad(x09), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(0), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(1), 0.38, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(2), 0.38, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(3), 0.38, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(4), 0.38, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(5), 0.38, 1e-12);

    UTEST_CHECK(state.update_if_better(x0, function.vgrad(x0)));

    UTEST_CHECK_CLOSE(state.x(), x0, 1e-12);
    UTEST_CHECK_CLOSE(state.fx(), function.vgrad(x0), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(0), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(1), 1.62, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(2), 1.62, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(3), 1.62, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(4), 1.62, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(5), 1.62, 1e-12);

    UTEST_CHECK(!state.update_if_better(x0, function.vgrad(x0)));

    UTEST_CHECK_CLOSE(state.x(), x0, 1e-12);
    UTEST_CHECK_CLOSE(state.fx(), function.vgrad(x0), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(0), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(1), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(2), 1.62, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(3), 1.62, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(4), 1.62, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(5), 1.62, 1e-12);
}

UTEST_CASE(state_update_if_better_constrained)
{
    auto function = function_sphere_t{2};
    function.constrain(constraint::euclidean_ball_inequality_t{make_vector<scalar_t>(0, 0), 1.0});

    auto state = solver_state_t{function, make_vector<scalar_t>(1.0, 1.0)};
    {
        auto cstate = solver_state_t{function, make_vector<scalar_t>(NAN, NAN)};
        UTEST_CHECK(!cstate.valid());
        UTEST_CHECK(!state.update_if_better_constrained(cstate, 1e-6));
    }
    {
        auto cstate = solver_state_t{function, make_vector<scalar_t>(0.0, 0.0)};
        UTEST_CHECK(cstate.valid());
        UTEST_CHECK(state.update_if_better_constrained(cstate, 1e-6));
        UTEST_CHECK_CLOSE(state.fx(), 0.0, 1e-12);
        UTEST_CHECK_CLOSE(cstate.fx(), 0.0, 1e-12);
    }
    {
        auto cstate = solver_state_t{function, make_vector<scalar_t>(2.0, 2.0)};
        UTEST_CHECK(cstate.valid());
        UTEST_CHECK(!state.update_if_better_constrained(cstate, 1e-6));
        UTEST_CHECK_CLOSE(state.fx(), 0.0, 1e-12);
        UTEST_CHECK_CLOSE(cstate.fx(), 8.0, 1e-12);
    }
}

UTEST_CASE(state_convergence0)
{
    const auto function = function_sphere_t{7};
    const auto state    = solver_state_t{function, make_vector<scalar_t>(0, 0, 0, 0, 0, 0, 0)};
    UTEST_CHECK_GREATER_EQUAL(state.gradient_test(), 0);
    UTEST_CHECK_LESS(state.gradient_test(), epsilon0<scalar_t>());
}

UTEST_CASE(state_convergence1)
{
    const auto function = function_sphere_t{7};
    const auto state    = solver_state_t{function, make_random_x0(function, epsilon1<scalar_t>())};
    UTEST_CHECK_GREATER_EQUAL(state.gradient_test(), 0);
    UTEST_CHECK_LESS(state.gradient_test(), epsilon2<scalar_t>());
}

UTEST_CASE(state_convergence2)
{
    const auto function = function_sphere_t{3};
    const auto bstate   = solver_state_t{function, make_vector<scalar_t>(1e-3, 1e-3, 1e-3)};
    const auto cstate   = solver_state_t{function, make_vector<scalar_t>(1e-4, 1e-4, 1e-4)};
    UTEST_CHECK(::nano::converged(bstate, cstate, 1e-3));
    UTEST_CHECK(!::nano::converged(bstate, cstate, 1e-4));
}

UTEST_CASE(factory)
{
    for (const auto& solver_id : solver_t::all().ids())
    {
        const auto solver = make_solver(solver_id);

        const auto desc = make_description(solver_id);
        UTEST_CHECK_EQUAL(solver->type(), desc.m_type);
    }
}

UTEST_CASE(config_solvers)
{
    for (const auto& solver_id : make_solver_ids())
    {
        const auto solver = make_solver(solver_id);

        // NB: 0 < c1 < c2 < 1
        UTEST_CHECK_NOTHROW(solver->parameter("solver::tolerance") = std::make_tuple(1e-4, 1e-1));
        UTEST_CHECK_EQUAL(solver->parameter("solver::tolerance").value_pair<scalar_t>(), std::make_tuple(1e-4, 1e-1));

        UTEST_CHECK_THROW(solver->parameter("solver::tolerance") = std::make_tuple(2e-1, 1e-1), std::runtime_error);
        UTEST_CHECK_THROW(solver->parameter("solver::tolerance") = std::make_tuple(1e-1, 1e-4), std::runtime_error);
        UTEST_CHECK_THROW(solver->parameter("solver::tolerance") = std::make_tuple(1e-1, +1.1), std::runtime_error);
        UTEST_CHECK_THROW(solver->parameter("solver::tolerance") = std::make_tuple(1e-1, -0.1), std::runtime_error);
        UTEST_CHECK_THROW(solver->parameter("solver::tolerance") = std::make_tuple(-0.1, +1.1), std::runtime_error);
        UTEST_CHECK_EQUAL(solver->parameter("solver::tolerance").value_pair<scalar_t>(), std::make_tuple(1e-4, 1e-1));

        UTEST_CHECK_NOTHROW(solver->parameter("solver::tolerance") = std::make_tuple(1e-1, 9e-1));
        UTEST_CHECK_EQUAL(solver->parameter("solver::tolerance").value_pair<scalar_t>(), std::make_tuple(1e-1, 9e-1));

        UTEST_CHECK_THROW(solver->lsearch0("invalid-lsearch0-id"), std::runtime_error);
        UTEST_CHECK_THROW(solver->lsearchk("invalid-lsearchk-id"), std::runtime_error);

        UTEST_CHECK_NOTHROW(solver->lsearch0("constant"));
        UTEST_CHECK_NOTHROW(solver->lsearch0(*make_lsearch0("constant")));

        UTEST_CHECK_NOTHROW(solver->lsearchk("backtrack"));
        UTEST_CHECK_NOTHROW(solver->lsearchk(*make_lsearchk("backtrack")));
    }
}

UTEST_CASE(default_solvers_on_smooth_convex)
{
    for (const auto& function : function_t::make({1, 4, convexity::yes, smoothness::yes, 100}))
    {
        UTEST_REQUIRE(function);

        for (const auto& x0 : make_random_x0s(*function))
        {
            auto config = minimize_config_t{};
            for (const auto& solver_id : make_smooth_solver_ids())
            {
                UTEST_NAMED_CASE(scat(function->name(), "/", solver_id));

                const auto descr = make_description(solver_id);
                config.config(descr.m_smooth_config);

                const auto solver = make_solver(solver_id);
                const auto state  = check_minimize(*solver, *function, x0, config);
                config.expected_minimum(state.fx());

                log_info() << function->name() << ": solver=" << solver_id << ", f=" << state.fx() << ".";
            }
        }
    }
}

UTEST_CASE(default_solvers_on_nonsmooth_convex)
{
    for (const auto& function : function_t::make({4, 4, convexity::yes, smoothness::no, 100}))
    {
        UTEST_REQUIRE(function);

        for (const auto& x0 : make_random_x0s(*function))
        {
            auto config = minimize_config_t{};
            for (const auto& solver_id : make_nonsmooth_solver_ids())
            {
                UTEST_NAMED_CASE(scat(function->name(), "/", solver_id));

                const auto descr = make_description(solver_id);
                config.config(descr.m_nonsmooth_config);

                const auto solver = make_solver(solver_id);
                const auto state  = check_minimize(*solver, *function, x0, config);
                config.expected_minimum(state.fx());

                log_info() << function->name() << ": solver=" << solver_id << ", f=" << state.fx() << ".";
            }
        }
    }
}

UTEST_CASE(best_solvers_with_lsearches_on_smooth)
{
    for (const auto& function : function_t::make({4, 4, convexity::ignore, smoothness::yes, 100}))
    {
        UTEST_REQUIRE(function);

        for (const auto& x0 : make_random_x0s(*function))
        {
            auto config = minimize_config_t{};
            for (const auto& solver_id : make_best_smooth_solver_ids())
            {
                const auto solver = make_solver(solver_id);
                for (const auto& lsearch0_id : make_lsearch0_ids())
                {
                    for (const auto& lsearchk_id : make_lsearchk_ids())
                    {
                        // NB: CGD, LBFGS and quasi-Newton methods cannot work with Armijo-based line-search!
                        if (lsearchk_id == "backtrack")
                        {
                            continue;
                        }

                        // NB: CGD cannot work with non-strong Wolfe-based line-search!
                        if (solver_id == "cgd-pr" && lsearchk_id == "lemarechal")
                        {
                            continue;
                        }

                        UTEST_NAMED_CASE(scat(function->name(), "/", solver_id, "/", lsearch0_id, "/", lsearchk_id));
                        UTEST_REQUIRE_NOTHROW(solver->lsearch0(lsearch0_id));
                        UTEST_REQUIRE_NOTHROW(solver->lsearchk(lsearchk_id));

                        const auto state = check_minimize(*solver, *function, x0, config);
                        config.expected_minimum(state.fx());

                        log_info() << function->name() << ": solver=" << solver_id << ", lsearch0=" << lsearch0_id
                                   << ", lsearchk=" << lsearchk_id << ", f=" << state.fx() << ".";
                    }
                }
            }
        }
    }
}

UTEST_CASE(best_solvers_with_cgdescent_very_accurate_on_smooth)
{
    for (const auto& function : function_t::make({4, 4, convexity::ignore, smoothness::yes, 100}))
    {
        UTEST_REQUIRE(function);

        for (const auto& x0 : make_random_x0s(*function))
        {
            auto config = minimize_config_t{}.max_evals(10000).epsilon(1e-10).expected_maximum_deviation(1e-9);
            for (const auto& solver_id : make_best_smooth_solver_ids())
            {
                UTEST_NAMED_CASE(scat(function->name(), "/", solver_id));

                const auto solver = make_solver(solver_id);
                UTEST_REQUIRE_NOTHROW(solver->lsearch0("cgdescent"));
                UTEST_REQUIRE_NOTHROW(solver->lsearchk("cgdescent"));

                const auto state = check_minimize(*solver, *function, x0, config);
                config.expected_minimum(state.fx());

                log_info() << function->name() << ": solver=" << solver_id << ", lsearch0=cgdescent"
                           << ", lsearchk=cgdescent, f=" << state.fx() << ".";
            }
        }
    }
}

UTEST_CASE(best_solvers_with_tolerances_on_smooth)
{
    for (const auto& function : function_t::make({4, 4, convexity::ignore, smoothness::yes, 100}))
    {
        UTEST_REQUIRE(function);

        for (const auto& x0 : make_random_x0s(*function))
        {
            for (const auto& solver_id : make_best_smooth_solver_ids())
            {
                UTEST_NAMED_CASE(scat(function->name(), "/", solver_id));

                const auto solver = make_solver(solver_id);

                UTEST_REQUIRE_NOTHROW(solver->parameter("solver::tolerance") = std::make_tuple(1e-4, 1e-1));
                check_minimize(*solver, *function, x0);

                UTEST_REQUIRE_NOTHROW(solver->parameter("solver::tolerance") = std::make_tuple(1e-4, 9e-1));
                check_minimize(*solver, *function, x0);

                UTEST_REQUIRE_NOTHROW(solver->parameter("solver::tolerance") = std::make_tuple(1e-1, 9e-1));
                check_minimize(*solver, *function, x0);
            }
        }
    }
}

UTEST_CASE(quasi_bfgs_with_initializations)
{
    for (const auto& function : function_t::make({4, 4, convexity::yes, smoothness::yes, 100}))
    {
        UTEST_REQUIRE(function);

        for (const auto& x0 : make_random_x0s(*function))
        {
            {
                const auto* const pname  = "solver::quasi::initialization";
                auto              solver = solver_quasi_bfgs_t{};

                UTEST_REQUIRE_NOTHROW(solver.parameter(pname) = solver_quasi_t::initialization::identity);
                check_minimize(solver, *function, x0);

                UTEST_REQUIRE_NOTHROW(solver.parameter(pname) = solver_quasi_t::initialization::scaled);
                check_minimize(solver, *function, x0);
            }
            {
                const auto* const pname  = "solver::quasi::initialization";
                auto              solver = solver_quasi_fletcher_t{};

                UTEST_REQUIRE_NOTHROW(solver.parameter(pname) = solver_quasi_t::initialization::identity);
                check_minimize(solver, *function, x0);

                UTEST_REQUIRE_NOTHROW(solver.parameter(pname) = solver_quasi_t::initialization::scaled);
                check_minimize(solver, *function, x0);
            }
        }
    }
}

UTEST_END_MODULE()
