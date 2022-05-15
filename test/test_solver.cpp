#include <iomanip>
#include <utest/utest.h>
#include "fixture/solver.h"
#include <nano/core/logger.h>
#include <nano/core/numeric.h>
#include <nano/solver/quasi.h>
#include <nano/function/sphere.h>
#include <nano/function/benchmark.h>

using namespace nano;

template <typename tscalar>
static auto& operator<<(std::ostream& stream, const std::tuple<tscalar, tscalar>& values)
{
    return stream << std::get<0>(values) << "," << std::get<1>(values);
}

static void check_consistency(
    const function_t& function, const std::vector<scalar_t>& fvalues, const std::vector<scalar_t>& epsilons,
    size_t reference = 0U)
{
    if (function.convex())
    {
        for (size_t i = 0U; i < fvalues.size(); ++ i)
        {
            UTEST_CHECK_CLOSE(fvalues[reference], fvalues[i], epsilons[i]);
        }
    }
}

struct solver_description_t
{
    bool            m_monotonic{true};
    bool            m_converges{true};
    tensor_size_t   m_max_evals{1000};
    scalar_t        m_epsilon{1e-6};
};

static solver_description_t make_description(const string_t& solver_id)
{
    if (solver_id == "gd") { return {true, true, 10000, 1e-6}; }
    else if (solver_id == "cgd") { return {true, true, 1000, 1e-6}; }
    else if (solver_id == "cgd-n") { return {true, true, 1001, 1e-6}; }
    else if (solver_id == "cgd-hs") { return {true, true, 1002, 1e-6}; }
    else if (solver_id == "cgd-fr") { return {true, true, 1003, 1e-6}; }
    else if (solver_id == "cgd-pr") { return {true, true, 1004, 1e-6}; }
    else if (solver_id == "cgd-cd") { return {true, true, 1005, 1e-6}; }
    else if (solver_id == "cgd-ls") { return {true, true, 1006, 1e-6}; }
    else if (solver_id == "cgd-dy") { return {true, true, 1007, 1e-6}; }
    else if (solver_id == "cgd-dycd") { return {true, true, 1008, 1e-6}; }
    else if (solver_id == "cgd-dyhs") { return {true, true, 1009, 1e-6}; }
    else if (solver_id == "cgd-prfr") { return {true, true, 1010, 1e-6}; }
    else if (solver_id == "lbfgs") { return {true, true, 1011, 1e-6}; }
    else if (solver_id == "dfp") { return {true, true, 1012, 1e-6}; }
    else if (solver_id == "sr1") { return {true, true, 1013, 1e-6}; }
    else if (solver_id == "bfgs") { return {true, true, 1014, 1e-6}; }
    else if (solver_id == "hoshino") { return {true, true, 1015, 1e-6}; }
    else if (solver_id == "fletcher") { return {true, true, 1016, 1e-6}; }
    else if (solver_id == "osga") { return {false, true, 10000, 1e-6}; }
    else if (solver_id == "pgm") { return {false, false, 301, 3e-2}; }
    else if (solver_id == "dgm") { return {false, false, 302, 3e-2}; }
    else if (solver_id == "fgm") { return {false, false, 303, 3e-2}; }
    else { assert(false); return {}; }
};

static auto make_lsearch0_ids() { return lsearch0_t::all().ids(); }
static auto make_lsearchk_ids() { return lsearchk_t::all().ids(); }

static auto make_solver_ids() { return solver_t::all().ids(std::regex(".+")); }
static auto make_smooth_solver_ids() { return solver_t::all().ids(std::regex(".+")); }
static auto make_nonsmooth_solver_ids() { return solver_t::all().ids(std::regex("osga|pgm|dgm|fgm")); }
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
    for (const auto& solver_id : solver_t::all().ids())
    {
        const auto solver = solver_t::all().get(solver_id);
        UTEST_REQUIRE(solver);

        const auto desc = make_description(solver_id);
        UTEST_CHECK_EQUAL(solver->monotonic(), desc.m_monotonic);
    }
}

UTEST_CASE(config_solvers)
{
    for (const auto& solver_id : make_solver_ids())
    {
        const auto solver = solver_t::all().get(solver_id);
        UTEST_REQUIRE(solver);

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

UTEST_CASE(default_solvers_on_smooth_convex)
{
    for (const auto& function : benchmark_function_t::make({4, 4, convexity::yes, smoothness::yes, 100}))
    {
        UTEST_REQUIRE(function);

        const vector_t x0 = vector_t::Random(function->size());

        std::vector<scalar_t> fvalues, epsilons;
        for (const auto& solver_id : make_smooth_solver_ids())
        {
            const auto solver = solver_t::all().get(solver_id);
            UTEST_REQUIRE(solver);

            const auto state = check_minimize(*solver, solver_id, *function, x0);
            fvalues.push_back(state.f);
            epsilons.push_back(1e-6);
            log_info() << function->name() << ": solver=" << solver_id << ", f=" << state.f << ".";
        }

        check_consistency(*function, fvalues, epsilons);
    }
}

UTEST_CASE(default_solvers_on_nonsmooth_conex)
{
    for (const auto& function : benchmark_function_t::make({4, 4, convexity::yes, smoothness::no, 100}))
    {
        UTEST_REQUIRE(function);

        const vector_t x0 = vector_t::Ones(function->size());

        size_t reference = 0U;
        std::vector<scalar_t> fvalues, epsilons;
        for (const auto& solver_id : make_nonsmooth_solver_ids())
        {
            const auto solver = solver_t::all().get(solver_id);
            UTEST_REQUIRE(solver);

            const auto dd = make_description(solver_id);
            if (solver_id == string_t{"osga"})
            {
                reference = fvalues.size();
            }

            const auto state = check_minimize(*solver, solver_id, *function, x0, dd.m_max_evals, dd.m_epsilon, dd.m_converges);
            fvalues.push_back(state.f);
            epsilons.push_back(dd.m_epsilon);
            log_info() << function->name() << ": solver=" << solver_id << ", f=" << state.f << ".";
        }

        check_consistency(*function, fvalues, epsilons, reference);
    }
}

UTEST_CASE(best_solvers_with_lsearches_on_smooth)
{
    for (const auto& function : benchmark_function_t::make({4, 4, convexity::ignore, smoothness::yes, 100}))
    {
        UTEST_REQUIRE(function);

        const vector_t x0 = vector_t::Random(function->size());

        std::vector<scalar_t> fvalues, epsilons;
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

                    const auto state = check_minimize(*solver, solver_id, *function, x0);
                    fvalues.push_back(state.f);
                    epsilons.push_back(1e-6);
                }
            }
        }

        check_consistency(*function, fvalues, epsilons);
    }
}

UTEST_CASE(best_solvers_with_tolerances_on_smooth)
{
    for (const auto& function : benchmark_function_t::make({4, 4, convexity::ignore, smoothness::yes, 100}))
    {
        UTEST_REQUIRE(function);

        for (const auto& solver_id : make_best_smooth_solver_ids())
        {
            const auto solver = solver_t::all().get(solver_id);
            UTEST_REQUIRE(solver);

            UTEST_REQUIRE_NOTHROW(solver->parameter("solver::tolerance") = std::make_tuple(1e-4, 1e-1));
            check_minimize(*solver, solver_id, *function, vector_t::Random(function->size()));

            UTEST_REQUIRE_NOTHROW(solver->parameter("solver::tolerance") = std::make_tuple(1e-4, 9e-1));
            check_minimize(*solver, solver_id, *function, vector_t::Random(function->size()));

            UTEST_REQUIRE_NOTHROW(solver->parameter("solver::tolerance") = std::make_tuple(1e-1, 9e-1));
            check_minimize(*solver, solver_id, *function, vector_t::Random(function->size()));
        }
    }
}

UTEST_CASE(quasi_bfgs_with_initializations)
{
    for (const auto& function : benchmark_function_t::make({4, 4, convexity::yes, smoothness::yes, 100}))
    {
        UTEST_REQUIRE(function);
        {
            const auto *const solver_id = "bfgs";
            const auto *const pname = "solver::quasi::initialization";
            auto solver = solver_quasi_bfgs_t{};

            UTEST_REQUIRE_NOTHROW(solver.parameter(pname) = solver_quasi_t::initialization::identity);
            check_minimize(solver, solver_id, *function, vector_t::Random(function->size()));

            UTEST_REQUIRE_NOTHROW(solver.parameter(pname) = solver_quasi_t::initialization::scaled);
            check_minimize(solver, solver_id, *function, vector_t::Random(function->size()));
        }
        {
            const auto *const solver_id = "fletcher";
            const auto *const pname = "solver::quasi::initialization";
            auto solver = solver_quasi_fletcher_t{};

            UTEST_REQUIRE_NOTHROW(solver.parameter(pname) = solver_quasi_t::initialization::identity);
            check_minimize(solver, solver_id, *function, vector_t::Random(function->size()));

            UTEST_REQUIRE_NOTHROW(solver.parameter(pname) = solver_quasi_t::initialization::scaled);
            check_minimize(solver, solver_id, *function, vector_t::Random(function->size()));
        }
    }
}

UTEST_END_MODULE()
