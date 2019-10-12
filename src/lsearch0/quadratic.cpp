#include <nano/lsearch0/quadratic.h>

using namespace nano;

rlsearch0_t lsearch0_quadratic_t::clone() const
{
    return std::make_unique<lsearch0_quadratic_t>(*this);
}

scalar_t lsearch0_quadratic_t::get(const solver_state_t& state)
{
    scalar_t t0;

    if (state.m_iterations <= 1)
    {
        t0 = 1;
    }
    else
    {
        t0 = std::min(scalar_t(1), -alpha() * 2 * std::max(m_prevf - state.f, beta() * epsilon()) / m_prevdg);
    }

    m_prevf = state.f;
    m_prevdg = state.dg();

    log(state, t0);
    return t0;
}
