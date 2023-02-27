#include "fixture/dataset.h"
#include "fixture/datasource/linear.h"
#include "fixture/model.h"
#include <nano/dataset/iterator.h>
#include <nano/linear/model.h>
#include <nano/linear/result.h>

using namespace nano;

[[maybe_unused]] static auto make_linear_datasource(const tensor_size_t samples, const tensor_size_t targets,
                                                    const tensor_size_t features, const tensor_size_t modulo = 31,
                                                    const scalar_t noise = 0.0)
{
    auto datasource = linear_datasource_t{samples, features, targets};
    datasource.noise(noise);
    datasource.modulo(modulo);
    UTEST_REQUIRE_NOTHROW(datasource.load());
    return datasource;
}

template <typename tweights, typename tbias>
[[maybe_unused]] static void check_linear(const dataset_t& dataset, tweights weights, tbias bias, scalar_t epsilon)
{
    const auto samples = dataset.samples();

    auto called = make_full_tensor<tensor_size_t>(make_dims(samples), 0);

    auto iterator = flatten_iterator_t{dataset, arange(0, samples)};
    iterator.batch(11);
    iterator.scaling(scaling_type::none);
    iterator.loop(
        [&](tensor_range_t range, size_t, tensor2d_cmap_t inputs, tensor4d_cmap_t targets)
        {
            for (tensor_size_t i = 0, size = range.size(); i < size; ++i)
            {
                UTEST_CHECK_CLOSE(targets.vector(i), weights * inputs.vector(i) + bias, epsilon);
                called(range.begin() + i) = 1;
            }
        });

    UTEST_CHECK_EQUAL(called, make_full_tensor<tensor_size_t>(make_dims(samples), 1));
}

[[maybe_unused]] static void check_result(const fit_result_t& result, const strings_t& expected_param_names,
                                          const tensor_size_t expected_folds, const scalar_t epsilon)
{
    ::check_result(result, expected_param_names,
                   expected_param_names.empty()        ? 1U
                   : expected_param_names.size() == 1U ? 6U
                                                       : 15U,
                   expected_folds, epsilon);

    for (const auto& param_result : result.param_results())
    {
        for (tensor_size_t fold = 0; fold < expected_folds; ++fold)
        {
            const auto& result = std::any_cast<linear::fit_result_t>(param_result.extra(fold));

            UTEST_REQUIRE_EQUAL(result.m_statistics.size(), 3);

            const auto fcalls = result.m_statistics(0);
            const auto gcalls = result.m_statistics(1);
            const auto status = result.m_statistics(2);

            UTEST_CHECK_NOT_EQUAL(static_cast<solver_status>(static_cast<int>(status)), solver_status::failed);

            UTEST_CHECK_GREATER_EQUAL(fcalls, 1);
            UTEST_CHECK_GREATER_EQUAL(gcalls, 1);
        }
    }
}
