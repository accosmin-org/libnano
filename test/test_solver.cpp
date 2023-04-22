#include "fixture/function.h"
#include "fixture/lsearch0.h"
#include "fixture/lsearchk.h"
#include "fixture/solver.h"
#include <nano/function/benchmark/sphere.h>
#include <nano/solver.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_solver)

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
    for (const auto& solver_id : solver_t::all().ids())
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

UTEST_END_MODULE()
