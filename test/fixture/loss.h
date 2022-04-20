#include <nano/loss.h>
#include <utest/utest.h>

using namespace nano;

[[maybe_unused]] inline auto make_loss(const char* name = "squared")
{
    auto loss = loss_t::all().get(name);
    UTEST_REQUIRE(loss);
    return loss;
}
