#include <fixture/configurable.h>
#include <fixture/dataset.h>
#include <fixture/learner.h>
#include <nano/dataset/iterator.h>
#include <nano/linear.h>
#include <nano/linear/result.h>

using namespace nano;
using namespace nano::ml;

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

[[maybe_unused]] static void check_fitting(const std::any& extra, const string_t& log_path)
{
    const auto old_n_failures = utest_n_failures.load();

    const auto& pfresult = std::any_cast<linear::result_t>(extra);

    UTEST_REQUIRE_EQUAL(pfresult.m_statistics.size(), 3);

    const auto fcalls = pfresult.m_statistics(0);
    const auto gcalls = pfresult.m_statistics(1);
    const auto status = pfresult.m_statistics(2);

    UTEST_CHECK_EQUAL(static_cast<solver_status>(static_cast<int>(status)), solver_status::converged);

    UTEST_CHECK_GREATER_EQUAL(fcalls, 1);
    UTEST_CHECK_GREATER_EQUAL(gcalls, 1);

    if (old_n_failures != utest_n_failures.load())
    {
        std::ifstream in(log_path);
        const auto    stream = string_t{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
        std::cout << stream;
    }
}

[[maybe_unused]] static void check_result(const result_t& result, const strings_t& expected_param_names,
                                          const tensor_size_t expected_folds, const scalar_t epsilon)
{
    ::check_result(result, expected_param_names,
                   expected_param_names.empty()        ? 1U
                   : expected_param_names.size() == 1U ? 6U
                                                       : 15U,
                   expected_folds, epsilon);

    // the solver should converge for all hyper-parameter trials and all folds
    for (tensor_size_t trial = 0; trial < result.trials(); ++trial)
    {
        for (tensor_size_t fold = 0; fold < expected_folds; ++fold)
        {
            check_fitting(result.extra(trial, fold), result.log_path(trial, fold));
        }
    }

    // the solver should converge at the final refitting step as well
    check_fitting(result.extra(), result.refit_log_path());

    // TODO: the tuning strategy should not fail as well!!!
}
