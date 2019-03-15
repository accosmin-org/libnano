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

        const auto json = solver.config();
        std::cout << " ... using " << json.dump() << "\n";
    }

    // check function value decrease
    UTEST_CHECK_LESS_EQUAL(f, f0 + epsilon1<scalar_t>());

    // check convergence
    UTEST_CHECK_LESS(g, solver.epsilon());
    UTEST_CHECK_EQUAL(state.m_status, solver_state_t::status::converged);
}

// todo: verbose only when a failure is detected - add support for this in utest

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

UTEST_CASE(config_solvers)
{
    const auto valid0 = to_json("c1", 1e-4, "c2", 1e-1);
    const auto valid1 = to_json("c1", 1e-4, "c2", 9e-1);
    const auto valid2 = to_json("c1", 1e-1, "c2", 9e-1);

    const auto invalid_c10 = to_json("c1", "not-a-scalar");
    const auto invalid_c11 = to_json("c1", -1);
    const auto invalid_c12 = to_json("c1", +0);
    const auto invalid_c13 = to_json("c1", +1);
    const auto invalid_c14 = to_json("c1", +2);

    const auto invalid_c20 = to_json("c2", "not-a-scalar");
    const auto invalid_c21 = to_json("c2", -1);
    const auto invalid_c22 = to_json("c2", +0);
    const auto invalid_c23 = to_json("c2", +1);
    const auto invalid_c24 = to_json("c2", +2);
    const auto invalid_c25 = to_json("c1", 1e-1, "c2", 1e-4, "why", "c1 > c2");

    for (const auto& solver_id : solver_t::all().ids())
    {
        const auto solver = solver_t::all().get(solver_id);
        UTEST_REQUIRE(solver);

        UTEST_CHECK_NOTHROW(solver->config(valid0));
        UTEST_CHECK_NOTHROW(solver->config(valid1));
        UTEST_CHECK_NOTHROW(solver->config(valid2));

        UTEST_CHECK_THROW(solver->config(invalid_c10), std::invalid_argument);
        UTEST_CHECK_THROW(solver->config(invalid_c11), std::invalid_argument);
        UTEST_CHECK_THROW(solver->config(invalid_c12), std::invalid_argument);
        UTEST_CHECK_THROW(solver->config(invalid_c13), std::invalid_argument);
        UTEST_CHECK_THROW(solver->config(invalid_c14), std::invalid_argument);

        UTEST_CHECK_THROW(solver->config(invalid_c20), std::invalid_argument);
        UTEST_CHECK_THROW(solver->config(invalid_c21), std::invalid_argument);
        UTEST_CHECK_THROW(solver->config(invalid_c22), std::invalid_argument);
        UTEST_CHECK_THROW(solver->config(invalid_c23), std::invalid_argument);
        UTEST_CHECK_THROW(solver->config(invalid_c24), std::invalid_argument);
        UTEST_CHECK_THROW(solver->config(invalid_c25), std::invalid_argument);
    }
}

UTEST_CASE(default_solvers)
{
    for (const auto& function : get_convex_functions(1, 4))
    {
        UTEST_REQUIRE(function);

        for (const auto& solver_id : solver_t::all().ids())
        {
            const auto solver = solver_t::all().get(solver_id);
            UTEST_REQUIRE(solver);

            for (auto t = 0; t < 10; ++ t)
            {
                test(*solver, solver_id, *function, vector_t::Random(function->size()));
            }
        }
    }
}

UTEST_CASE(various_lsearches)
{
    for (const auto& function : get_convex_functions(1, 4))
    {
        UTEST_REQUIRE(function);

        for (const auto& solver_id : {"gd", "cgd", "lbfgs", "bfgs"})
        {
            const auto solver = solver_t::all().get(solver_id);
            UTEST_REQUIRE(solver);

            for (const auto& lsearch_id : lsearch_strategy_t::all().ids())
            {
                solver->lsearch(lsearch_strategy_t::all().get(lsearch_id));

                for (auto t = 0; t < 10; ++ t)
                {
                    test(*solver, solver_id, *function, vector_t::Random(function->size()));
                }
            }
        }
    }
}

UTEST_END_MODULE()
