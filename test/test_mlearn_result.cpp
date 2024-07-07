#include <nano/mlearn/result.h>
#include <utest/utest.h>

using namespace nano;
using namespace nano::ml;

namespace
{
void check_stats(const stats_t& stats, const scalar_t expected_mean, const scalar_t expected_stdev,
                 const scalar_t expected_count, const scalar_t expected_per01, const scalar_t expected_per05,
                 const scalar_t expected_per10, const scalar_t expected_per20, const scalar_t expected_per50,
                 const scalar_t expected_per80, const scalar_t expected_per90, const scalar_t expected_per95,
                 const scalar_t expected_per99, const scalar_t epsilon = 1e-12)
{
    UTEST_CHECK_CLOSE(stats.m_mean, expected_mean, epsilon);
    UTEST_CHECK_CLOSE(stats.m_stdev, expected_stdev, epsilon);
    UTEST_CHECK_CLOSE(stats.m_count, expected_count, epsilon);
    UTEST_CHECK_CLOSE(stats.m_per01, expected_per01, epsilon);
    UTEST_CHECK_CLOSE(stats.m_per05, expected_per05, epsilon);
    UTEST_CHECK_CLOSE(stats.m_per10, expected_per10, epsilon);
    UTEST_CHECK_CLOSE(stats.m_per20, expected_per20, epsilon);
    UTEST_CHECK_CLOSE(stats.m_per50, expected_per50, epsilon);
    UTEST_CHECK_CLOSE(stats.m_per80, expected_per80, epsilon);
    UTEST_CHECK_CLOSE(stats.m_per90, expected_per90, epsilon);
    UTEST_CHECK_CLOSE(stats.m_per95, expected_per95, epsilon);
    UTEST_CHECK_CLOSE(stats.m_per99, expected_per99, epsilon);
}
} // namespace

UTEST_BEGIN_MODULE(test_mlearn_result)

UTEST_CASE(result_optimum)
{
    const auto folds        = 3;
    const auto param_spaces = param_spaces_t{
        param_space_t{"l1reg", param_space_t::type::log10, 1e-3, 1e+0, 1e+3},
        param_space_t{"l2reg", param_space_t::type::log10, 1e-3, 1e+0, 1e+3}
    };

    auto result = result_t{param_spaces, folds};
    UTEST_CHECK_EQUAL(result.folds(), folds);
    UTEST_CHECK_EQUAL(result.trials(), 0);

    const auto make_errors_losses = [](const tensor_size_t min, const tensor_size_t max)
    {
        auto values = tensor2d_t{2, max - min + 1};
        for (tensor_size_t val = min; val <= max; ++val)
        {
            values(0, val - min) = 1e-3 * static_cast<scalar_t>(val - min);
            values(1, val - min) = 1e-4 * static_cast<scalar_t>(max - val);
        }
        return values;
    };

    {
        const auto closest_trial = result.closest_trial(make_tensor<scalar_t>(make_dims(2), 0.0, 0.99), 0);
        UTEST_CHECK_EQUAL(closest_trial, 0);
    }
    {
        result.add(make_tensor<scalar_t>(make_dims(1, 2), 0.0, 1.0));
        result.store(0, 0, make_errors_losses(0, 100), make_errors_losses(1000, 1200), 1);
        result.store(0, 1, make_errors_losses(1, 101), make_errors_losses(1001, 1301), "2");
        result.store(0, 2, make_errors_losses(2, 102), make_errors_losses(1003, 1403), 3.14);
        UTEST_CHECK_EQUAL(result.folds(), folds);
        UTEST_CHECK_EQUAL(result.trials(), 1);
        UTEST_CHECK_EQUAL(result.optimum_trial(), 0);

        UTEST_CHECK_EQUAL(std::any_cast<int>(result.extra(0, 0)), 1);
        UTEST_CHECK_EQUAL(std::any_cast<const char*>(result.extra(0, 1)), string_t{"2"});
        UTEST_CHECK_EQUAL(std::any_cast<double>(result.extra(0, 2)), 3.14);

        check_stats(result.stats(0, 0, split_type::train, value_type::errors), 1e-3 * 50, 0.002915475947, 101.0, 1e-3,
                    5e-3, 10e-3, 20e-3, 50e-3, 80e-3, 90e-3, 95e-3, 99e-3);
        check_stats(result.stats(0, 1, split_type::valid, value_type::losses), 1e-4 * 150, 0.000501663898, 301.0, 3e-4,
                    15e-4, 30e-4, 60e-4, 150e-4, 240e-4, 270e-4, 285e-4, 297e-4);

        result.store(make_errors_losses(0, 10));

        check_stats(result.stats(value_type::errors), 1e-3 * 5, 1e-3, 11.0, 5e-4, 5e-4, 10e-4, 20e-4, 50e-4, 80e-4,
                    90e-4, 95e-4, 95e-4);
        check_stats(result.stats(value_type::losses), 1e-4 * 5, 1e-4, 11.0, 5e-5, 5e-5, 10e-5, 20e-5, 50e-5, 80e-5,
                    90e-5, 95e-5, 95e-5);
    }
    {
        const auto closest_trial = result.closest_trial(make_tensor<scalar_t>(make_dims(2), 0.0, 0.99), 1);
        UTEST_CHECK_EQUAL(closest_trial, 0);
    }
    {
        result.add(make_tensor<scalar_t>(make_dims(1, 2), 1.0, 2.0));
        result.store(1, 0, make_errors_losses(10, 110), make_errors_losses(1000, 1100));
        result.store(1, 1, make_errors_losses(11, 111), make_errors_losses(1001, 1201));
        result.store(1, 2, make_errors_losses(12, 112), make_errors_losses(1003, 1303));
        UTEST_CHECK_EQUAL(result.folds(), folds);
        UTEST_CHECK_EQUAL(result.trials(), 2);
        UTEST_CHECK_EQUAL(result.optimum_trial(), 1);
    }
    {
        result.add(make_tensor<scalar_t>(make_dims(1, 2), 0.5, 1.2));
        result.store(2, 0, make_errors_losses(10, 110), make_errors_losses(1000, 1010));
        result.store(2, 1, make_errors_losses(11, 111), make_errors_losses(1001, 1021));
        result.store(2, 2, make_errors_losses(12, 112), make_errors_losses(1003, 1033));
        UTEST_CHECK_EQUAL(result.folds(), folds);
        UTEST_CHECK_EQUAL(result.trials(), 3);
        UTEST_CHECK_EQUAL(result.optimum_trial(), 2);
    }
    {
        const auto closest_trial = result.closest_trial(make_tensor<scalar_t>(make_dims(2), 0.5, 1.21), 3);
        UTEST_CHECK_EQUAL(closest_trial, 2);
    }
    {
        result.add(make_tensor<scalar_t>(make_dims(1, 2), 0.9, 1.1));
        result.store(3, 0, make_errors_losses(10, 110), make_errors_losses(1000, 1040));
        result.store(3, 1, make_errors_losses(11, 111), make_errors_losses(1001, 1061));
        result.store(3, 2, make_errors_losses(12, 112), make_errors_losses(1003, 1033));
        UTEST_CHECK_EQUAL(result.folds(), folds);
        UTEST_CHECK_EQUAL(result.trials(), 4);
        UTEST_CHECK_EQUAL(result.optimum_trial(), 2);
    }
}

UTEST_END_MODULE()
