#include <nano/datasource.h>
#include <utest/utest.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_datasource_factory)

UTEST_CASE(factory)
{
    const auto& datasets = datasource_t::all();
    UTEST_CHECK_EQUAL(datasets.ids().size(), 12U);
    UTEST_CHECK(datasets.get("abalone") != nullptr);
    UTEST_CHECK(datasets.get("adult") != nullptr);
    UTEST_CHECK(datasets.get("bank-marketing") != nullptr);
    UTEST_CHECK(datasets.get("breast-cancer") != nullptr);
    UTEST_CHECK(datasets.get("forest-fires") != nullptr);
    UTEST_CHECK(datasets.get("iris") != nullptr);
    UTEST_CHECK(datasets.get("wine") != nullptr);
    UTEST_CHECK(datasets.get("cifar100c") != nullptr);
    UTEST_CHECK(datasets.get("cifar100f") != nullptr);
    UTEST_CHECK(datasets.get("cifar10") != nullptr);
    UTEST_CHECK(datasets.get("fashion-mnist") != nullptr);
    UTEST_CHECK(datasets.get("mnist") != nullptr);
}

UTEST_END_MODULE()
