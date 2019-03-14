#include <utest/utest.h>
#include <nano/solver.h>
#include <nano/numeric.h>
#include <nano/function/sphere.h>

using namespace nano;

static void test(const solver_t& solver, const string_t& solver_id, const function_t& function, const vector_t& x0)
{
    const auto state0 = solver_state_t{function, x0};
    const auto f0 = state0.f;
    const auto g0 = state0.convergence_criterion();

    // minimize
    const auto state = solver.minimize(function, x0);
    const auto x = state.x;
    const auto f = state.f;
    const auto g = state.convergence_criterion();

    if (state.m_status != solver_state_t::status::converged)
    {
        std::cout << function.name() << " " << solver_id
            << ": x = [" << x0.transpose() << "]/[" << x.transpose() << "]"
            << ",f=" << f0 << "/" << f
            << ",g=" << g0 << "/" << g
            << "[" << to_string(state.m_status) << "]"
            << ",calls=" << state.m_fcalls << "/" << state.m_gcalls << ".\n";

        json_t json;
        solver.to_json(json);
        std::cout << " ... using " << json.dump() << "\n";
    }

    // check function value decrease
    UTEST_CHECK_LESS_EQUAL(f, f0 + epsilon1<scalar_t>());

    // check convergence
    UTEST_CHECK_LESS(g, solver.epsilon());
    UTEST_CHECK_EQUAL(state.m_status, solver_state_t::status::converged);
}

UTEST_BEGIN_MODULE(test_solvers)

UTEST_CASE(state_valid)
{
    const function_sphere_t function(7);
    solver_state_t state(function, vector_t::Random(function.size()));
    UTEST_CHECK(state);
}

UTEST_CASE(state_invalid_tINF)
{
    const function_sphere_t function(7);
    solver_state_t state(function, vector_t::Random(function.size()));
    state.t = INFINITY;
    UTEST_CHECK(!state);
}

UTEST_CASE(state_invalid_fNAN)
{
    const function_sphere_t function(7);
    solver_state_t state(function, vector_t::Random(function.size()));
    state.f = NAN;
    UTEST_CHECK(!state);
}

UTEST_CASE(state_has_descent)
{
    const function_sphere_t function(7);
    solver_state_t state(function, vector_t::Random(function.size()));
    state.d = -state.g;
    UTEST_CHECK(state.has_descent());
}

UTEST_CASE(state_has_no_descent0)
{
    const function_sphere_t function(7);
    solver_state_t state(function, vector_t::Random(function.size()));
    state.d.setZero();
    UTEST_CHECK(!state.has_descent());
}

UTEST_CASE(state_has_no_descent1)
{
    const function_sphere_t function(7);
    solver_state_t state(function, vector_t::Random(function.size()));
    state.d = state.g;
    UTEST_CHECK(!state.has_descent());
}

UTEST_CASE(state_convergence0)
{
    const function_sphere_t function(7);
    solver_state_t state(function, vector_t::Zero(function.size()));
    UTEST_CHECK(state.converged(epsilon2<scalar_t>()));
    UTEST_CHECK_GREATER_EQUAL(state.convergence_criterion(), 0);
    UTEST_CHECK_LESS(state.convergence_criterion(), epsilon0<scalar_t>());
}

UTEST_CASE(state_convergence1)
{
    const function_sphere_t function(7);
    solver_state_t state(function, vector_t::Random(function.size()) * epsilon1<scalar_t>());
    UTEST_CHECK(state.converged(epsilon2<scalar_t>()));
    UTEST_CHECK_GREATER_EQUAL(state.convergence_criterion(), 0);
    UTEST_CHECK_LESS(state.convergence_criterion(), epsilon2<scalar_t>());
}

UTEST_CASE(default_solvers)
{
    for (const auto& function : get_convex_functions(1, 4))
    {
        UTEST_REQUIRE(function);

        for (const auto& solver_id : get_solvers().ids())
        {
            const auto solver = get_solvers().get(solver_id);
            UTEST_REQUIRE(solver);

            for (auto t = 0; t < 10; ++ t)
            {
                test(*solver, solver_id, *function, vector_t::Random(function->size()));
            }
        }
    }
}

UTEST_CASE(lsearch_strategies)
{
    for (const auto& function : get_convex_functions(1, 4))
    {
        UTEST_REQUIRE(function);

        for (const auto& solver_id : {"gd", "cgd", "lbfgs", "bfgs"})
        {
            const auto solver = get_solvers().get(solver_id);
            UTEST_REQUIRE(solver);

            for (const auto& lsearch_id : get_lsearch_strategies().ids())
            {
                solver->lsearch(get_lsearch_strategies().get(lsearch_id));

                for (auto t = 0; t < 10; ++ t)
                {
                    test(*solver, solver_id, *function, vector_t::Random(function->size()));
                }
            }
        }
    }
}

UTEST_END_MODULE()
