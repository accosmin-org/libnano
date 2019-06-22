#include <iomanip>
#include <utest/utest.h>
#include <nano/solver.h>
#include <nano/numeric.h>
#include <nano/function/sphere.h>

using namespace nano;

static void setup_logger(const rsolver_t& solver, std::stringstream& stream, size_t& iterations)
{
    // log the optimization steps
    solver->logger([&] (const solver_state_t& state)
    {
        ++ iterations;
        stream
            << "\tdescent: i=" << state.m_iterations << ",f=" << state.f << ",g=" << state.convergence_criterion()
            << "[" << to_string(state.m_status) << "]" << ",calls=" << state.m_fcalls << "/" << state.m_gcalls << ".\n";
        return true;
    });

    // log the line-search steps
    solver->lsearch0_logger([&] (const solver_state_t& state0, const scalar_t t)
    {
        stream
            << "\t\tlsearch(0): t=" << state0.t << ",f=" << state0.f << ",g=" << state0.convergence_criterion()
            << ",t=" << t << ".\n";
    });

    solver->lsearchk_logger([&] (const solver_state_t& state0, const solver_state_t& state)
    {
        stream
            << "\t\tlsearch(t):t=" << state.t << ",f=" << state.f << ",g=" << state.convergence_criterion()
            << ",armijo=" << state.has_armijo(state0, solver->c1())
            << ",wolfe=" << state.has_wolfe(state0, solver->c2())
            << ",swolfe=" << state.has_strong_wolfe(state0, solver->c2()) << ".\n";
    });
}

static void test(const rsolver_t& solver, const string_t& solver_id, const function_t& function, const vector_t& x0)
{
    const auto old_n_failures = utest_n_failures.load();
    const auto state0 = solver_state_t{function, x0};

    std::stringstream stream;
    stream
        << std::fixed << std::setprecision(16)
        << function.name() << " " << solver_id << "[" << solver->config().dump() << "]\n"
        << ":x0=[" << state0.x.transpose() << "],f0=" << state0.f<< ",g0=" << state0.convergence_criterion() << "\n";

    size_t iterations = 0;
    setup_logger(solver, stream, iterations);

    // minimize
    solver->epsilon(1e-6);
    solver->max_iterations(5000);
    const auto state = solver->minimize(function, x0);
    UTEST_CHECK(state);

    // check function value decrease
    UTEST_CHECK_LESS_EQUAL(state.f, state0.f + epsilon1<scalar_t>());

    // check convergence
    UTEST_CHECK_LESS(state.convergence_criterion(), solver->epsilon());
    UTEST_CHECK_EQUAL(state.m_status, solver_state_t::status::converged);
    UTEST_CHECK_EQUAL(iterations, state.m_iterations);

    if (old_n_failures != utest_n_failures.load())
    {
        std::cout << stream.str();
    }
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

        UTEST_CHECK_THROW(solver->lsearch0("constant", rlsearch0_t()), std::invalid_argument);
        UTEST_CHECK_NOTHROW(solver->lsearch0(lsearch0_t::all().get("constant")->config()));
        UTEST_CHECK_NOTHROW(solver->lsearch0(lsearch0_t::all().get("constant")->config_with_id("constant")));

        UTEST_CHECK_THROW(solver->lsearchk("backtrack", rlsearchk_t()), std::invalid_argument);
        UTEST_CHECK_NOTHROW(solver->lsearchk(lsearchk_t::all().get("backtrack")->config()));
        UTEST_CHECK_NOTHROW(solver->lsearchk(lsearchk_t::all().get("backtrack")->config_with_id("backtrack")));
    }
}

const auto all_functions = get_functions(4, 4);
const auto convex_functions = get_convex_functions(4, 4);

const auto all_solver_ids = solver_t::all().ids();
const auto best_solver_ids = solver_t::all().ids(std::regex("cgd|lbfgs|bfgs"));
const auto all_lsearch0_ids = lsearch0_t::all().ids();
const auto all_lsearchk_ids = lsearchk_t::all().ids();

UTEST_CASE(all_default_solvers)
{
    for (const auto& function : convex_functions)
    {
        UTEST_REQUIRE(function);

        for (const auto& solver_id : all_solver_ids)
        {
            const auto solver = solver_t::all().get(solver_id);
            UTEST_REQUIRE(solver);

            test(solver, solver_id, *function, vector_t::Random(function->size()));
        }
    }
}

UTEST_CASE(best_solvers_with_lsearches)
{
    for (const auto& function : all_functions)
    {
        UTEST_REQUIRE(function);

        for (const auto& solver_id : best_solver_ids)
        {
            const auto solver = solver_t::all().get(solver_id);
            UTEST_REQUIRE(solver);

            for (const auto& lsearch0_id : all_lsearch0_ids)
            {
                for (const auto& lsearchk_id : all_lsearchk_ids)
                {
                    UTEST_REQUIRE_NOTHROW(solver->lsearch0(lsearch0_id));
                    UTEST_REQUIRE_NOTHROW(solver->lsearchk(lsearchk_id));

                    test(solver, solver_id, *function, vector_t::Random(function->size()));
                }
            }
        }
    }
}

UTEST_CASE(best_solvers_with_tolerances)
{
    for (const auto& function : all_functions)
    {
        UTEST_REQUIRE(function);

        for (const auto& solver_id : best_solver_ids)
        {
            const auto solver = solver_t::all().get(solver_id);
            UTEST_REQUIRE(solver);

            UTEST_REQUIRE_NOTHROW(solver->config(to_json("c1", 1e-4, "c2", 1e-1)));
            test(solver, solver_id, *function, vector_t::Random(function->size()));

            UTEST_REQUIRE_NOTHROW(solver->config(to_json("c1", 1e-4, "c2", 9e-1)));
            test(solver, solver_id, *function, vector_t::Random(function->size()));

            UTEST_REQUIRE_NOTHROW(solver->config(to_json("c1", 1e-1, "c2", 9e-1)));
            test(solver, solver_id, *function, vector_t::Random(function->size()));
        }
    }
}

UTEST_CASE(quasi_with_initializations)
{
    for (const auto& function : convex_functions)
    {
        UTEST_REQUIRE(function);

        const auto solver_id = "bfgs";
        const auto solver = solver_t::all().get(solver_id);
        UTEST_REQUIRE(solver);

        UTEST_REQUIRE_NOTHROW(solver->config(to_json("H0", "identity")));
        test(solver, solver_id, *function, vector_t::Random(function->size()));

        UTEST_REQUIRE_NOTHROW(solver->config(to_json("H0", "scaled")));
        test(solver, solver_id, *function, vector_t::Random(function->size()));
    }
}

UTEST_END_MODULE()
