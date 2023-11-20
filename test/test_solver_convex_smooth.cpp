#include "fixture/function.h"
#include "fixture/solver.h"
#include <nano/core/logger.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_solver_on_convex_smooth)

UTEST_CASE(default_solvers)
{
    for (const auto& function : function_t::make({1, 4, convexity::yes, smoothness::yes, 100}))
    {
        UTEST_REQUIRE(function);

        for (const auto& x0 : make_random_x0s(*function))
        {
            auto config = minimize_config_t{};
            for (const auto& solver : make_smooth_solvers())
            {
                const auto solver_id = solver->type_id();
                UTEST_NAMED_CASE(scat(function->name(), "/", solver_id));

                const auto descr = make_description(solver_id);
                config.config(descr.m_smooth_config);

                const auto state = check_minimize(*solver, *function, x0, config);
                config.expected_minimum(state.fx());

                log_info() << function->name() << ": solver=" << solver_id << ", f=" << state.fx() << ".";
            }
        }
    }
}

UTEST_END_MODULE()
