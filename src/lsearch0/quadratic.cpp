#include <nano/lsearch0/quadratic.h>

using namespace nano;

lsearch0_quadratic_t::lsearch0_quadratic_t()
    : lsearch0_t("quadratic")
{
    register_parameter(parameter_t::make_scalar("lsearch0::quadratic::beta", 1, LT, 10.0, LT, 1e+6));
    register_parameter(parameter_t::make_scalar("lsearch0::quadratic::alpha", 1, LT, 1.01, LT, 1e+6));
}

rlsearch0_t lsearch0_quadratic_t::clone() const
{
    return std::make_unique<lsearch0_quadratic_t>(*this);
}

scalar_t lsearch0_quadratic_t::get(const solver_state_t& state, const vector_t& descent, const scalar_t last_step_size)
{
    const auto beta    = parameter("lsearch0::quadratic::beta").value<scalar_t>();
    const auto alpha   = parameter("lsearch0::quadratic::alpha").value<scalar_t>();
    const auto epsilon = parameter("lsearch0::epsilon").value<scalar_t>();

    scalar_t t0 = 0;
    if (last_step_size < 0)
    {
        t0 = 1;
    }
    else
    {
        t0 = std::min(scalar_t(1), -alpha * 2 * std::max(m_prevf - state.fx(), beta * epsilon) / m_prevdg);
    }

    m_prevf  = state.fx();
    m_prevdg = state.dg(descent);
    log(state, t0);
    return t0;
}
