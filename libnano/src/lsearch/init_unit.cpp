#include "init_unit.h"

using namespace nano;

json_t lsearch_unit_init_t::config() const
{
    json_t json;
    return json;
}

void lsearch_unit_init_t::config(const json_t&)
{
}

scalar_t lsearch_unit_init_t::get(const solver_state_t&, const int)
{
    return 1;
}
