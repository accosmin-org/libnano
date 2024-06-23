#include <nano/solver/lsearch.h>

using namespace nano;

lsearch_t::lsearch_t(rlsearch0_t&& lsearch0, rlsearchk_t&& lsearchk)
    : m_lsearch0(std::move(lsearch0))
    , m_lsearchk(std::move(lsearchk))
{
}

bool lsearch_t::get(solver_state_t& state, const vector_t& descent) const
{
    assert(m_lsearch0);
    assert(m_lsearchk);

    const auto init_step_size = m_lsearch0->get(state, descent, m_last_step_size);
    m_lsearch0->log("[", m_lsearch0->type_id(), "]: t=", init_step_size, ",f=", state.fx(),
                    ",g=", state.gradient_test(), ".\n");

    const auto [ok, step_size] = m_lsearchk->get(state, descent, init_step_size);
    m_last_step_size           = step_size;
    return ok;
}
