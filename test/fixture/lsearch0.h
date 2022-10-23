#include <nano/lsearch0.h>
#include <utest/utest.h>

using namespace nano;

[[maybe_unused]] inline auto make_lsearch0(const string_t& name = "cgdescent")
{
    auto lsearch0 = lsearch0_t::all().get(name);
    UTEST_REQUIRE(lsearch0);
    return lsearch0;
}
