#include <nano/generator/generator.h>
#include <utest/utest.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_generator_factory)

UTEST_CASE(factory)
{
    const auto& generators = generator_t::all();
    UTEST_CHECK_EQUAL(generators.ids().size(), 6U);
    UTEST_CHECK(generators.get("gradient") != nullptr);
    UTEST_CHECK(generators.get("product") != nullptr);
    UTEST_CHECK(generators.get("identity-sclass") != nullptr);
    UTEST_CHECK(generators.get("identity-mclass") != nullptr);
    UTEST_CHECK(generators.get("identity-scalar") != nullptr);
    UTEST_CHECK(generators.get("identity-struct") != nullptr);
}

UTEST_END_MODULE()
