#include <nano/lsearch0/constant.h>

using namespace nano;

lsearch0_constant_t::lsearch0_constant_t()
    : lsearch0_t("constant")
{
    register_parameter(parameter_t::make_scalar("lsearch0::constant::t0", 0, LT, 1, LT, 1e+6));
}

rlsearch0_t lsearch0_constant_t::clone() const
{
    return std::make_unique<lsearch0_constant_t>(*this);
}

scalar_t lsearch0_constant_t::get(const solver_state_t& state, const vector_t&, const scalar_t)
{
    const auto t0 = parameter("lsearch0::constant::t0").value<scalar_t>();

    log(state, t0);
    return t0;
}
