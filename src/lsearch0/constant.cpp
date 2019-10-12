#include <nano/lsearch0/constant.h>

using namespace nano;

rlsearch0_t lsearch0_constant_t::clone() const
{
    return std::make_unique<lsearch0_constant_t>(*this);
}

scalar_t lsearch0_constant_t::get(const solver_state_t& state)
{
    log(state, t0());
    return t0();
}
