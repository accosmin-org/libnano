#include <nano/wlearner.h>
#include <utest/utest.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_wlearner_factory)

UTEST_CASE(factory)
{
    const auto& wlearners = wlearner_t::all();
    UTEST_CHECK_EQUAL(wlearners.ids().size(), 6U);
    UTEST_CHECK(wlearners.get("affine") != nullptr);
    UTEST_CHECK(wlearners.get("hinge") != nullptr);
    UTEST_CHECK(wlearners.get("dstep") != nullptr);
    UTEST_CHECK(wlearners.get("dtree") != nullptr);
    UTEST_CHECK(wlearners.get("table") != nullptr);
    UTEST_CHECK(wlearners.get("stump") != nullptr);
}

UTEST_END_MODULE()
