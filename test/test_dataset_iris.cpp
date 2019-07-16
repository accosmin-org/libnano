#include <utest/utest.h>
#include <nano/dataset.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_dataset_iris)

UTEST_CASE(load)
{
    auto dataset = dataset_t::all().get("iris");
    UTEST_REQUIRE(dataset);

    UTEST_REQUIRE(dataset->load());
}

UTEST_END_MODULE()
