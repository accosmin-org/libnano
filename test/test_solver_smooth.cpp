#include <fixture/lsearch0.h>
#include <fixture/lsearchk.h>
#include <fixture/solver.h>
#include <iomanip>
#include <solver/quasi.h>

using namespace nano;

namespace
{
auto make_lsearch0_ids()
{
    return lsearch0_t::all().ids();
}

auto make_lsearchk_ids()
{
    return lsearchk_t::all().ids();
}

auto make_best_smooth_solver_ids()
{
    return strings_t{"cgd-pr", "lbfgs", "bfgs"};
}

auto make_solver_ids()
{
    return strings_t{
        // clang-format off
        "ellipsoid",
        "gd",
        "cgd-n", "cgd-hs", "cgd-fr", "cgd-pr", "cgd-cd", "cgd-ls", "cgd-dy", "cgd-dycd", "cgd-dyhs", "cgd-frpr",
        "lbfgs",
        "sr1", "dfp", "bfgs", "hoshino", "fletcher"
        // clang-format on
    };
}
} // namespace

UTEST_BEGIN_MODULE(test_solver_smooth)

UTEST_CASE(default_solvers)
{
    check_minimize(make_solver_ids(), function_t::make({1, 4, function_type::convex_smooth}));
}

UTEST_CASE(best_solvers_with_lsearches_on_smooth)
{
    for (const auto& function : function_t::make({4, 4, function_type::smooth}))
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

                        // NB: these two line-search algorithms are not very robust in general!
                        config.m_expected_failure = lsearchk_id == "fletcher" || lsearchk_id == "lemarechal";

                        UTEST_NAMED_CASE(scat(function->name(), "/", solver_id, "/", lsearch0_id, "/", lsearchk_id));
                        UTEST_REQUIRE_NOTHROW(solver->lsearch0(lsearch0_id));
                        UTEST_REQUIRE_NOTHROW(solver->lsearchk(lsearchk_id));

                        const auto state = check_minimize(*solver, *function, x0, config);
                        config.expected_minimum(state.fx());

                        log_info(std::setprecision(10), function->name(), ": solver=", solver_id,
                                 ",lsearch0=", lsearch0_id, ",lsearchk=", lsearchk_id, ",fx=", state.fx(),
                                 ",calls=", state.fcalls(), "|", state.gcalls(), ".\n");
                    }
                }
            }
        }
    }
}

UTEST_CASE(best_solvers_with_cgdescent_very_accurate_on_smooth)
{
    for (const auto& function : function_t::make({4, 4, function_type::smooth}))
    {
        UTEST_REQUIRE(function);

        for (const auto& x0 : make_random_x0s(*function))
        {
            auto config = minimize_config_t{}.expected_maximum_deviation(1e-9);
            for (const auto& solver_id : make_best_smooth_solver_ids())
            {
                UTEST_NAMED_CASE(scat(function->name(), "/", solver_id));

                const auto solver = make_solver(solver_id);
                UTEST_REQUIRE_NOTHROW(solver->lsearch0("cgdescent"));
                UTEST_REQUIRE_NOTHROW(solver->lsearchk("cgdescent"));
                solver->parameter("solver::max_evals") = 10000;
                solver->parameter("solver::epsilon")   = 1e-10;

                const auto state = check_minimize(*solver, *function, x0, config);
                config.expected_minimum(state.fx());

                log_info(std::setprecision(10), function->name(), ": solver=", solver_id,
                         ",lsearch0=cgdescent,lsearchk=cgdescent,fx=", state.fx(), ",calls=", state.fcalls(), "|",
                         state.gcalls(), ".\n");
            }
        }
    }
}

UTEST_CASE(best_solvers_with_tolerances_on_smooth)
{
    for (const auto& function : function_t::make({4, 4, function_type::smooth}))
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
    for (const auto& function : function_t::make({4, 4, function_type::convex_smooth}))
    {
        UTEST_REQUIRE(function);

        for (const auto& x0 : make_random_x0s(*function))
        {
            {
                const auto* const pname  = "solver::quasi::initialization";
                auto              solver = solver_quasi_bfgs_t{};

                UTEST_REQUIRE_NOTHROW(solver.parameter(pname) = quasi_initialization::identity);
                check_minimize(solver, *function, x0);

                UTEST_REQUIRE_NOTHROW(solver.parameter(pname) = quasi_initialization::scaled);
                check_minimize(solver, *function, x0);
            }
            {
                const auto* const pname  = "solver::quasi::initialization";
                auto              solver = solver_quasi_fletcher_t{};

                UTEST_REQUIRE_NOTHROW(solver.parameter(pname) = quasi_initialization::identity);
                check_minimize(solver, *function, x0);

                UTEST_REQUIRE_NOTHROW(solver.parameter(pname) = quasi_initialization::scaled);
                check_minimize(solver, *function, x0);
            }
        }
    }
}

UTEST_END_MODULE()
