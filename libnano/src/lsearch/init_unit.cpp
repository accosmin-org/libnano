#include "init_unit.h"

using namespace nano;

json_t lsearch_unit_init_t::config() const
{
    json_t json;
    // todo: parametrizable constant line-search step length (default 1)
    return json;
}

void lsearch_unit_init_t::config(const json_t&)
{
}

scalar_t lsearch_unit_init_t::get(const solver_state_t&)
{
    return 1;
}
