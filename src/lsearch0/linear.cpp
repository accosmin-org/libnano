#include <nano/lsearch0/linear.h>

using namespace nano;

rlsearch0_t lsearch0_linear_t::clone() const
{
    return std::make_unique<lsearch0_linear_t>(*this);
}

scalar_t lsearch0_linear_t::get(const solver_state_t& state)
{
    scalar_t t0 = 0;

    const auto dg = state.dg();
    if (state.m_iterations <= 1)
    {
        t0 = 1;
    }
    else
    {
        t0 = std::min(scalar_t(1), -alpha() * std::max(-state.t * m_prevdg, beta() * epsilon()) / dg);
    }

    m_prevdg = dg;

    log(state, t0);
    return t0;
}
