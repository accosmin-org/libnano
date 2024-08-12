#include <fixture/configurable.h>
#include <nano/datasource/linear.h>
#include <utest/utest.h>

using namespace nano;

template <class... targs>
[[maybe_unused]] static auto make_linear_datasource(const tensor_size_t samples, const tensor_size_t targets,
                                                    const tensor_size_t features, const targs... args)
{
    auto datasource                                      = linear_datasource_t{};
    datasource.parameter("datasource::linear::samples")  = samples;
    datasource.parameter("datasource::linear::targets")  = targets;
    datasource.parameter("datasource::linear::features") = features;
    ::config(datasource, args...);
    UTEST_REQUIRE_NOTHROW(datasource.load());
    return datasource;
}
