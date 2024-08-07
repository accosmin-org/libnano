#include <fixture/configurable.h>
#include <fixture/dataset.h>
#include <fixture/learner.h>
#include <nano/dataset/iterator.h>
#include <nano/datasource/linear.h>
#include <nano/linear/model.h>
#include <nano/linear/result.h>

using namespace nano;
using namespace nano::ml;

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

template <class tweights, class tbias>
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

[[maybe_unused]] static void check_result(const result_t& result, const strings_t& expected_param_names,
                                          const tensor_size_t expected_folds, const scalar_t epsilon)
{
    ::check_result(result, expected_param_names,
                   expected_param_names.empty()        ? 1U
                   : expected_param_names.size() == 1U ? 6U
                                                       : 15U,
                   expected_folds, epsilon);

    for (tensor_size_t trial = 0; trial < result.trials(); ++trial)
    {
        for (tensor_size_t fold = 0; fold < expected_folds; ++fold)
        {
            const auto& pfresult = std::any_cast<linear::result_t>(result.extra(trial, fold));

            UTEST_REQUIRE_EQUAL(pfresult.m_statistics.size(), 3);

            const auto fcalls = pfresult.m_statistics(0);
            const auto gcalls = pfresult.m_statistics(1);
            const auto status = pfresult.m_statistics(2);

            UTEST_CHECK_NOT_EQUAL(static_cast<solver_status>(static_cast<int>(status)), solver_status::failed);

            UTEST_CHECK_GREATER_EQUAL(fcalls, 1);
            UTEST_CHECK_GREATER_EQUAL(gcalls, 1);
        }
    }
}
