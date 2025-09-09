#include <nano/wlearner.h>
#include <utest/utest.h>

using namespace nano;

UTEST_BEGIN_MODULE()

UTEST_CASE(factory)
{
    const auto& wlearners = wlearner_t::all();
    UTEST_CHECK_EQUAL(wlearners.ids().size(), 8U);
    UTEST_CHECK(wlearners.get("affine") != nullptr);
    UTEST_CHECK(wlearners.get("hinge") != nullptr);
    UTEST_CHECK(wlearners.get("dtree") != nullptr);
    UTEST_CHECK(wlearners.get("stump") != nullptr);
    UTEST_CHECK(wlearners.get("dense-table") != nullptr);
    UTEST_CHECK(wlearners.get("dstep-table") != nullptr);
    UTEST_CHECK(wlearners.get("kbest-table") != nullptr);
    UTEST_CHECK(wlearners.get("ksplit-table") != nullptr);
}

UTEST_END_MODULE()
