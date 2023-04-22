#include "fixture/function.h"
#include "fixture/solver.h"
#include <iomanip>
#include <nano/core/logger.h>

using namespace nano;

static auto make_nonsmooth_solver_ids()
{
    return strings_t{"ellipsoid", "osga", "sgm", "cocob"}; // FIXME: have all methods converge!!!, "sda", "wda"};
}

UTEST_BEGIN_MODULE(test_solver_nonsmooth)

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

UTEST_END_MODULE()
