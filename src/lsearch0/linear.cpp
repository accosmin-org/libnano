#include <lsearch0/linear.h>

using namespace nano;

lsearch0_linear_t::lsearch0_linear_t()
    : lsearch0_t("linear")
{
    register_parameter(parameter_t::make_scalar("lsearch0::linear::beta", 1, LT, 10.0, LT, 1e+6));
    register_parameter(parameter_t::make_scalar("lsearch0::linear::alpha", 1, LT, 1.01, LT, 1e+6));
}

rlsearch0_t lsearch0_linear_t::clone() const
{
    return std::make_unique<lsearch0_linear_t>(*this);
}

scalar_t lsearch0_linear_t::get(const solver_state_t& state, const vector_t& descent, const scalar_t last_step_size)
{
    const auto beta    = parameter("lsearch0::linear::beta").value<scalar_t>();
    const auto alpha   = parameter("lsearch0::linear::alpha").value<scalar_t>();
    const auto epsilon = parameter("lsearch0::epsilon").value<scalar_t>();

    const auto dg = state.dg(descent);

    scalar_t t0 = 0;
    if (last_step_size < 0.0)
    {
        t0 = 1.0;
    }
    else
    {
        t0 = std::min(1.0, -alpha * std::max(-last_step_size * m_prevdg, beta * epsilon) / dg);
    }

    m_prevdg = dg;
    return t0;
}
