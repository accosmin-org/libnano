#include <nano/lsearchk.h>
#include <utest/utest.h>

using namespace nano;

[[maybe_unused]] inline auto make_lsearchk(const string_t& name = "cgdescent")
{
    auto lsearchk = lsearchk_t::all().get(name);
    UTEST_REQUIRE(lsearchk);
    return lsearchk;
}
