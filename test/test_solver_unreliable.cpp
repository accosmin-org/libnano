#include "fixture/function.h"
#include "fixture/solver.h"
#include <nano/core/logger.h>

using namespace nano;

inline auto make_solver_ids()
{
    return strings_t{"ellipsoid", "sgm", "cocob", "sda", "wda", "pgm", "dgm", "fgm", "asga2", "asga4"};
}

UTEST_BEGIN_MODULE(test_solver_unreliable)

UTEST_CASE(smooth)
{
    for (const auto& function : function_t::make({1, 4, convexity::yes, smoothness::yes, 100}))
    {
        UTEST_REQUIRE(function);

        for (const auto& x0 : make_random_x0s(*function))
        {
            auto config = minimize_config_t{};
            for (const auto& solver_id : make_solver_ids())
            {
                const auto solver = make_solver(solver_id);
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

UTEST_CASE(nonsmooth)
{
    for (const auto& function : function_t::make({4, 4, convexity::yes, smoothness::no, 100}))
    {
        UTEST_REQUIRE(function);

        for (const auto& x0 : make_random_x0s(*function))
        {
            auto config = minimize_config_t{};
            for (const auto& solver_id : make_solver_ids())
            {
                const auto solver = make_solver(solver_id);
                UTEST_NAMED_CASE(scat(function->name(), "/", solver_id));

                const auto descr = make_description(solver_id);
                config.config(descr.m_nonsmooth_config);

                const auto state = check_minimize(*solver, *function, x0, config);
                config.expected_minimum(state.fx());

                log_info() << function->name() << ": solver=" << solver_id << ", f=" << state.fx() << ".";
            }
        }
    }
}

UTEST_END_MODULE()
