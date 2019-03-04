#include "solver_gd.h"

using namespace nano;

void solver_gd_t::from_json(const json_t& json)
{
    nano::from_json(json, "init", m_init, "strat", m_strat, "c1", m_c1, "c2", m_c2);
}

void solver_gd_t::to_json(json_t& json) const
{
    nano::to_json(json,
        "init", to_string(m_init) + join(enum_values<lsearch_t::initializer>()),
        "strat", to_string(m_strat) + join(enum_values<lsearch_t::strategy>()),
        "c1", m_c1, "c2", m_c2);
}

solver_state_t solver_gd_t::minimize(const size_t max_iterations, const scalar_t epsilon,
    const solver_function_t& function, const vector_t& x0, const logger_t& logger) const
{
    lsearch_t lsearch(m_init, m_strat, m_c1, m_c2);

    auto cstate = solver_state_t{function, x0};
    for (size_t i = 0; i < max_iterations; ++ i, ++ cstate.m_iterations)
    {
        // descent direction
        cstate.d = -cstate.g;

        // line-search
        const auto iter_ok = lsearch(cstate);
        if (solver_t::done(logger, function, cstate, epsilon, iter_ok))
        {
            break;
        }
    }

    return cstate;
}
