#include <nano/tuner.h>
#include <utest/utest.h>

using namespace nano;

[[maybe_unused]] inline auto make_tuner(const string_t& name = "surrogate")
{
    auto tuner = tuner_t::all().get(name);
    UTEST_REQUIRE(tuner);
    return tuner;
}
