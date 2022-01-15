#include <iomanip>
#include <utest/utest.h>
#include <nano/core/numeric.h>
#include <nano/solver/quasi.h>
#include <nano/function/sphere.h>
#include <nano/function/benchmark.h>

using namespace nano;

static void setup_logger(solver_t& solver, std::stringstream& stream, tensor_size_t& iterations)
{
    // log the optimization steps
    solver.logger([&] (const solver_state_t& state)
    {
        ++ iterations;
        stream << "\tdescent: " << state << ".\n";
        return true;
    });

    // log the line-search steps
    solver.lsearch0_logger([&] (const solver_state_t& state0, const scalar_t t)
    {
        stream
            << "\t\tlsearch(0): t=" << state0.t << ",f=" << state0.f << ",g=" << state0.convergence_criterion()
            << ",t=" << t << ".\n";
    });

    solver.lsearchk_logger([&] (const solver_state_t& state0, const solver_state_t& state)
    {
        stream
            << "\t\tlsearch(t):t=" << state.t << ",f=" << state.f << ",g=" << state.convergence_criterion()
            << ",armijo=" << state.has_armijo(state0, solver.c1())
            << ",wolfe=" << state.has_wolfe(state0, solver.c2())
            << ",swolfe=" << state.has_strong_wolfe(state0, solver.c2()) << ".\n";
    });
}

static auto test(solver_t& solver, const string_t& solver_id, const function_t& function, const vector_t& x0)
{
    const auto old_n_failures = utest_n_failures.load();
    const auto state0 = solver_state_t{function, x0};

    const auto lsearch0_id = solver.monotonic() ? solver.lsearch0_id() : "N/A";
    const auto lsearchk_id = solver.monotonic() ? solver.lsearchk_id() : "N/A";

    std::stringstream stream;
    stream
        << std::fixed << std::setprecision(16)
        << function.name() << " " << solver_id << "[" << lsearch0_id << "," << lsearchk_id << "]\n"
        << ":x0=[" << state0.x.transpose() << "],f0=" << state0.f<< ",g0=" << state0.convergence_criterion() << "\n";

    tensor_size_t iterations = 0;
    setup_logger(solver, stream, iterations);

    const auto epsilon = 1e-6;

    // minimize
    solver.epsilon(epsilon);
    solver.max_evals(10000);
    auto state = solver.minimize(function, x0);
    UTEST_CHECK(state);

    // check function value decrease
    UTEST_CHECK_LESS_EQUAL(state.f, state0.f + epsilon1<scalar_t>());

    // check convergence
    if (function.smooth() && solver.monotonic())
    {
        UTEST_CHECK_LESS(state.convergence_criterion(), epsilon);
    }
    UTEST_CHECK_EQUAL(state.m_status, solver_state_t::status::converged);
    UTEST_CHECK_EQUAL(iterations, state.m_iterations);

    if (old_n_failures != utest_n_failures.load())
    {
        std::cout << stream.str();
    }

    return state;
}

static void check_consistency(const function_t& function, const std::vector<scalar_t>& fvalues, scalar_t epsilon = 1e-6)
{
    if (function.convex())
    {
        const auto min = *std::min_element(fvalues.begin(), fvalues.end());
        const auto max = *std::max_element(fvalues.begin(), fvalues.end());
        UTEST_CHECK_CLOSE(min, max, epsilon);
    }
}

static auto make_lsearch0_ids() { return lsearch0_t::all().ids(); }
static auto make_lsearchk_ids() { return lsearchk_t::all().ids(); }

static auto make_solver_ids() { return solver_t::all().ids(std::regex(".+")); }
static auto make_smooth_solver_ids() { return solver_t::all().ids(std::regex(".+")); }
static auto make_nonsmooth_solver_ids() { return solver_t::all().ids(std::regex("osga|sgm")); }
static auto make_best_smooth_solver_ids() { return solver_t::all().ids(std::regex("cgd|lbfgs|bfgs"));}

UTEST_BEGIN_MODULE(test_solver_lsearch)

UTEST_CASE(state_str)
{
    for (const auto status : enum_values<solver_state_t::status>())
    {
        std::stringstream stream;
        stream << status;
        UTEST_CHECK_EQUAL(stream.str(), scat(status));
    }
}

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

UTEST_CASE(state_update_if_better)
{
    const function_sphere_t function(2);
    const auto x0 = vector_t::Constant(function.size(), 0.0);
    const auto x1 = vector_t::Constant(function.size(), 1.0);
    const auto x2 = vector_t::Constant(function.size(), 2.0);

    solver_state_t state(function, x1);
    UTEST_CHECK_CLOSE(state.f, 2.0, 1e-12);
    UTEST_CHECK(!state.update_if_better(x2, 8.0));
    UTEST_CHECK_CLOSE(state.f, 2.0, 1e-12);
    UTEST_CHECK(!state.update_if_better(x2, std::numeric_limits<scalar_t>::quiet_NaN()));
    UTEST_CHECK_CLOSE(state.f, 2.0, 1e-12);
    UTEST_CHECK(!state.update_if_better(x1, 2.0));
    UTEST_CHECK_CLOSE(state.f, 2.0, 1e-12);
    UTEST_CHECK(state.update_if_better(x0, 0.0));
    UTEST_CHECK_CLOSE(state.f, 0.0, 1e-12);
    UTEST_CHECK(!state.update_if_better(x2, 8.0));
    UTEST_CHECK_CLOSE(state.f, 0.0, 1e-12);
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

UTEST_CASE(factory)
{
    const auto description = std::map<string_t, std::tuple<bool>>
    {
        {"gd", std::make_tuple(true)},
        {"cgd", std::make_tuple(true)},
        {"cgd-n", std::make_tuple(true)},
        {"cgd-hs", std::make_tuple(true)},
        {"cgd-fr", std::make_tuple(true)},
        {"cgd-pr", std::make_tuple(true)},
        {"cgd-cd", std::make_tuple(true)},
        {"cgd-ls", std::make_tuple(true)},
        {"cgd-dy", std::make_tuple(true)},
        {"cgd-dycd", std::make_tuple(true)},
        {"cgd-dyhs", std::make_tuple(true)},
        {"cgd-prfr", std::make_tuple(true)},
        {"lbfgs", std::make_tuple(true)},
        {"dfp", std::make_tuple(true)},
        {"sr1", std::make_tuple(true)},
        {"bfgs", std::make_tuple(true)},
        {"hoshino", std::make_tuple(true)},
        {"fletcher", std::make_tuple(true)},
        {"osga", std::make_tuple(false)},
        {"sgm", std::make_tuple(false)},
    };

    const auto ids = solver_t::all().ids();

    UTEST_REQUIRE_EQUAL(ids.size(), description.size());
    for (const auto& id : ids)
    {
        const auto it = description.find(id);
        UTEST_REQUIRE(it != description.end());

        const auto solver = solver_t::all().get(id);
        UTEST_REQUIRE(solver);

        UTEST_CHECK_EQUAL(solver->monotonic(), std::get<0>(it->second));
    }
}

UTEST_CASE(config_solvers)
{
    for (const auto& solver_id : make_solver_ids())
    {
        const auto solver = solver_t::all().get(solver_id);
        UTEST_REQUIRE(solver);

        // NB: 0 < c1 < c2 < 1
        UTEST_CHECK_NOTHROW(solver->tolerance(1e-4, 1e-1));
        UTEST_CHECK_EQUAL(solver->c1(), 1e-4);
        UTEST_CHECK_EQUAL(solver->c2(), 1e-1);

        UTEST_CHECK_THROW(solver->tolerance(2e-1, 1e-1), std::runtime_error);
        UTEST_CHECK_THROW(solver->tolerance(1e-1, 1e-4), std::runtime_error);
        UTEST_CHECK_THROW(solver->tolerance(1e-1, +1.1), std::runtime_error);
        UTEST_CHECK_THROW(solver->tolerance(1e-1, -0.1), std::runtime_error);
        UTEST_CHECK_THROW(solver->tolerance(-0.1, +1.1), std::runtime_error);
        UTEST_CHECK_EQUAL(solver->c1(), 1e-4);
        UTEST_CHECK_EQUAL(solver->c2(), 1e-1);

        UTEST_CHECK_NOTHROW(solver->tolerance(1e-1, 9e-1));
        UTEST_CHECK_EQUAL(solver->c1(), 1e-1);
        UTEST_CHECK_EQUAL(solver->c2(), 9e-1);

        UTEST_CHECK_THROW(solver->lsearch0("invalid-lsearch0-id"), std::runtime_error);
        UTEST_CHECK_THROW(solver->lsearch0("constant", rlsearch0_t()), std::runtime_error);

        UTEST_CHECK_NOTHROW(solver->lsearch0("constant"));
        UTEST_CHECK_NOTHROW(solver->lsearch0("constant", lsearch0_t::all().get("constant")));

        UTEST_CHECK_THROW(solver->lsearchk("invalid-lsearchk-id"), std::runtime_error);
        UTEST_CHECK_THROW(solver->lsearchk("backtrack", rlsearchk_t()), std::runtime_error);

        UTEST_CHECK_NOTHROW(solver->lsearchk("backtrack"));
        UTEST_CHECK_NOTHROW(solver->lsearchk("backtrack", lsearchk_t::all().get("backtrack")));
    }
}

UTEST_CASE(solver_function)
{
    for (const auto& function : benchmark_function_t::make({2, 4, convexity::ignore, smoothness::ignore, 10}))
    {
        const auto sfunction = solver_function_t{*function};

        UTEST_CHECK_EQUAL(sfunction.size(), function->size());
        UTEST_CHECK_EQUAL(sfunction.name(), function->name());
        UTEST_CHECK_EQUAL(sfunction.convex(), function->convex());
        UTEST_CHECK_EQUAL(sfunction.smooth(), function->smooth());
        UTEST_CHECK_CLOSE(sfunction.strong_convexity(), function->strong_convexity(), epsilon0<scalar_t>());

        UTEST_CHECK_EQUAL(sfunction.fcalls(), 0);
        UTEST_CHECK_EQUAL(sfunction.gcalls(), 0);

        vector_t x = vector_t::Random(sfunction.size());
        sfunction.vgrad(x);

        UTEST_CHECK_EQUAL(sfunction.fcalls(), 1);
        UTEST_CHECK_EQUAL(sfunction.gcalls(), 0);

        vector_t gx(x.size());
        sfunction.vgrad(x, &gx);

        UTEST_CHECK_EQUAL(sfunction.fcalls(), 2);
        UTEST_CHECK_EQUAL(sfunction.gcalls(), 1);

        if (sfunction.summands() > 1)
        {
            for (tensor_size_t begin = 0; begin < sfunction.summands(); begin += 5)
            {
                const auto end = std::min(begin + 5, sfunction.summands());
                sfunction.vgrad(x, &gx, vgrad_config_t{make_range(begin, end)});
            }

            UTEST_CHECK_EQUAL(sfunction.fcalls(), 3);
            UTEST_CHECK_EQUAL(sfunction.gcalls(), 2);

            sfunction.vgrad(x);

            UTEST_CHECK_EQUAL(sfunction.fcalls(), 4);
            UTEST_CHECK_EQUAL(sfunction.gcalls(), 2);
        }
    }
}

UTEST_CASE(default_monotonic_solvers)
{
    for (const auto& function : benchmark_function_t::make({4, 4, convexity::yes, smoothness::yes, 100}))
    {
        UTEST_REQUIRE(function);

        const vector_t x0 = vector_t::Random(function->size());

        std::vector<scalar_t> fvalues;
        for (const auto& solver_id : make_smooth_solver_ids())
        {
            const auto solver = solver_t::all().get(solver_id);
            UTEST_REQUIRE(solver);

            const auto state = test(*solver, solver_id, *function, x0);
            fvalues.push_back(state.f);
            log_info() << function->name() << ": solver=" << solver_id << ", f=" << state.f << ".";
        }

        check_consistency(*function, fvalues, 1e-6);
    }
}

UTEST_CASE(default_nonmonotonic_solvers)
{
    for (const auto& function : benchmark_function_t::make({4, 4, convexity::yes, smoothness::no, 100}))
    {
        UTEST_REQUIRE(function);

        const vector_t x0 = vector_t::Random(function->size());

        std::vector<scalar_t> fvalues;
        for (const auto& solver_id : make_nonsmooth_solver_ids())
        {
            const auto solver = solver_t::all().get(solver_id);
            UTEST_REQUIRE(solver);

            const auto state = test(*solver, solver_id, *function, x0);
            fvalues.push_back(state.f);
            log_info() << function->name() << ": solver=" << solver_id << ", f=" << state.f << ".";
        }

        check_consistency(*function, fvalues, 1e-3);
    }
}

UTEST_CASE(best_solvers_with_lsearches)
{
    for (const auto& function : benchmark_function_t::make({4, 4, convexity::ignore, smoothness::yes, 100}))
    {
        UTEST_REQUIRE(function);

        const vector_t x0 = vector_t::Random(function->size());

        std::vector<scalar_t> fvalues;
        for (const auto& solver_id : make_best_smooth_solver_ids())
        {
            const auto solver = solver_t::all().get(solver_id);
            UTEST_REQUIRE(solver);

            for (const auto& lsearch0_id : make_lsearch0_ids())
            {
                for (const auto& lsearchk_id : make_lsearchk_ids())
                {
                    UTEST_REQUIRE_NOTHROW(solver->lsearch0(lsearch0_id));
                    UTEST_REQUIRE_NOTHROW(solver->lsearchk(lsearchk_id));

                    const auto state = test(*solver, solver_id, *function, x0);
                    fvalues.push_back(state.f);
                }
            }
        }

        check_consistency(*function, fvalues);
    }
}

UTEST_CASE(best_solvers_with_tolerances)
{
    for (const auto& function : benchmark_function_t::make({4, 4, convexity::ignore, smoothness::yes, 100}))
    {
        UTEST_REQUIRE(function);

        for (const auto& solver_id : make_best_smooth_solver_ids())
        {
            const auto solver = solver_t::all().get(solver_id);
            UTEST_REQUIRE(solver);

            UTEST_REQUIRE_NOTHROW(solver->tolerance(1e-4, 1e-1));
            test(*solver, solver_id, *function, vector_t::Random(function->size()));

            UTEST_REQUIRE_NOTHROW(solver->tolerance(1e-4, 9e-1));
            test(*solver, solver_id, *function, vector_t::Random(function->size()));

            UTEST_REQUIRE_NOTHROW(solver->tolerance(1e-1, 9e-1));
            test(*solver, solver_id, *function, vector_t::Random(function->size()));
        }
    }
}

UTEST_CASE(quasi_with_initializations)
{
    for (const auto& function : benchmark_function_t::make({4, 4, convexity::yes, smoothness::yes, 10}))
    {
        UTEST_REQUIRE(function);
        {
            const auto *const solver_id = "bfgs";
            auto solver = solver_quasi_bfgs_t{};

            UTEST_REQUIRE_NOTHROW(solver.init(solver_quasi_t::initialization::identity));
            test(solver, solver_id, *function, vector_t::Random(function->size()));

            UTEST_REQUIRE_NOTHROW(solver.init(solver_quasi_t::initialization::scaled));
            test(solver, solver_id, *function, vector_t::Random(function->size()));
        }
        {
            const auto *const solver_id = "fletcher";
            auto solver = solver_quasi_fletcher_t{};

            UTEST_REQUIRE_NOTHROW(solver.init(solver_quasi_t::initialization::identity));
            test(solver, solver_id, *function, vector_t::Random(function->size()));

            UTEST_REQUIRE_NOTHROW(solver.init(solver_quasi_t::initialization::scaled));
            test(solver, solver_id, *function, vector_t::Random(function->size()));
        }
    }
}

UTEST_END_MODULE()
