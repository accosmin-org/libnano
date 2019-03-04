#include "solver_cgd.h"

using namespace nano;

template <typename tcgd_update>
void solver_cgd_base_t<tcgd_update>::from_json(const json_t& json)
{
    nano::from_json(json,
        "init", m_init, "strat", m_strat, "c1", m_c1, "c2", m_c2, "orthotest", m_orthotest);
}

template <typename tcgd_update>
void solver_cgd_base_t<tcgd_update>::to_json(json_t& json) const
{
    nano::to_json(json,
        "init", to_string(m_init) + join(enum_values<lsearch_t::initializer>()),
        "strat", to_string(m_strat) + join(enum_values<lsearch_t::strategy>()),
        "c1", m_c1, "c2", m_c2, "orthotest", m_orthotest);
}

template <typename tcgd_update>
solver_state_t solver_cgd_base_t<tcgd_update>::minimize(const size_t max_iterations, const scalar_t epsilon,
    const solver_function_t& function, const vector_t& x0, const logger_t& logger) const
{
    lsearch_t lsearch(m_init, m_strat, m_c1, m_c2);

    auto cstate = solver_state_t{function, x0};
    auto pstate = cstate;

    for (size_t i = 0; i < max_iterations; ++ i, ++ cstate.m_iterations)
    {
        // descent direction
        if (i == 0)
        {
            cstate.d = -cstate.g;
        }
        else
        {
            const scalar_t beta = tcgd_update::get(pstate, cstate);
            cstate.d = -cstate.g + beta * pstate.d;

            // restart:
            //  - if not a descent direction
            //  - or two consecutive gradients far from being orthogonal
            //      (see "Numerical optimization", Nocedal & Wright, 2nd edition, p.124-125)
            if (!cstate.has_descent())
            {
                cstate.d = -cstate.g;
            }
            else if (std::fabs(cstate.g.dot(pstate.g)) >= m_orthotest * cstate.g.dot(cstate.g))
            {
                cstate.d = -cstate.g;
            }
        }

        // line-search
        pstate = cstate;
        const auto iter_ok = lsearch(cstate);
        if (solver_t::done(logger, function, cstate, epsilon, iter_ok))
        {
            break;
        }
    }

    return cstate;
}

template class nano::solver_cgd_base_t<cgd_step_HS>;
template class nano::solver_cgd_base_t<cgd_step_FR>;
template class nano::solver_cgd_base_t<cgd_step_PRP>;
template class nano::solver_cgd_base_t<cgd_step_CD>;
template class nano::solver_cgd_base_t<cgd_step_LS>;
template class nano::solver_cgd_base_t<cgd_step_DY>;
template class nano::solver_cgd_base_t<cgd_step_N>;
template class nano::solver_cgd_base_t<cgd_step_DYCD>;
template class nano::solver_cgd_base_t<cgd_step_DYHS>;
