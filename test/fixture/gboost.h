#include "fixture/model.h"
#include "fixture/solver.h"
#include "fixture/splitter.h"
#include "fixture/tuner.h"
#include "fixture/wlearner.h"
#include <nano/gboost/enums.h>
#include <nano/gboost/model.h>
#include <nano/gboost/result.h>

using namespace nano;

static auto make_gbooster()
{
    auto model = gboost_model_t{};
    model.logger(model_t::make_logger_stdio());
    model.parameter("gboost::max_rounds") = 100;
    model.parameter("gboost::epsilon")    = 1e-6;
    model.parameter("gboost::patience")   = 2;
    return model;
}

static void check_predict(const gboost_model_t& model, const dataset_t& dataset, const scalar_t epsilon = 1e-12)
{
    const auto samples  = arange(0, dataset.samples());
    const auto outputs  = model.predict(dataset, samples);
    const auto iterator = targets_iterator_t{dataset, samples};

    iterator.loop([&](const tensor_range_t& range, size_t, tensor4d_cmap_t targets)
                  { UTEST_CHECK_CLOSE(targets, outputs.slice(range), epsilon); });
}

static void check_predict_throws(const gboost_model_t& model)
{
    const auto datasource1 = make_random_datasource(make_features_too_few());
    const auto datasource2 = make_random_datasource(make_features_too_many());
    const auto datasource3 = make_random_datasource(make_features_invalid_target());

    const auto dataset1 = make_dataset(datasource1);
    const auto dataset2 = make_dataset(datasource2);
    const auto dataset3 = make_dataset(datasource3);

    UTEST_CHECK_THROW(model.predict(dataset1, arange(0, dataset1.samples())), std::runtime_error);
    UTEST_CHECK_THROW(model.predict(dataset2, arange(0, dataset2.samples())), std::runtime_error);
    UTEST_CHECK_THROW(model.predict(dataset3, arange(0, dataset3.samples())), std::runtime_error);
}

static void check_equal(const fit_result_t::stats_t& lhs, const fit_result_t::stats_t& rhs,
                        const scalar_t epsilon = 1e-15)
{
    UTEST_CHECK_CLOSE(lhs.m_mean, rhs.m_mean, epsilon);
    UTEST_CHECK_CLOSE(lhs.m_stdev, rhs.m_stdev, epsilon);
    UTEST_CHECK_CLOSE(lhs.m_count, rhs.m_count, epsilon);
    UTEST_CHECK_CLOSE(lhs.m_per01, rhs.m_per01, epsilon);
    UTEST_CHECK_CLOSE(lhs.m_per05, rhs.m_per05, epsilon);
    UTEST_CHECK_CLOSE(lhs.m_per10, rhs.m_per10, epsilon);
    UTEST_CHECK_CLOSE(lhs.m_per20, rhs.m_per20, epsilon);
    UTEST_CHECK_CLOSE(lhs.m_per50, rhs.m_per50, epsilon);
    UTEST_CHECK_CLOSE(lhs.m_per80, rhs.m_per80, epsilon);
    UTEST_CHECK_CLOSE(lhs.m_per90, rhs.m_per90, epsilon);
    UTEST_CHECK_CLOSE(lhs.m_per95, rhs.m_per95, epsilon);
    UTEST_CHECK_CLOSE(lhs.m_per99, rhs.m_per99, epsilon);
}

static void check_equal(const fit_result_t& lhs, const fit_result_t& rhs, const scalar_t epsilon = 1e-15)
{
    UTEST_CHECK_EQUAL(lhs.param_names(), rhs.param_names());

    UTEST_REQUIRE_EQUAL(lhs.param_results().size(), rhs.param_results().size());
    for (size_t i = 0U; i < lhs.param_results().size(); ++i)
    {
        const auto& ilhs = lhs.param_results()[i];
        const auto& irhs = rhs.param_results()[i];

        UTEST_CHECK_EQUAL(ilhs.folds(), irhs.folds());
        UTEST_CHECK_CLOSE(ilhs.params(), irhs.params(), epsilon);
        UTEST_CHECK_CLOSE(ilhs.values(), irhs.values(), epsilon);

        for (tensor_size_t fold = 0; fold < ilhs.folds(); ++fold)
        {
            const auto& xlhs = std::any_cast<gboost::fit_result_t>(ilhs.extra(fold));
            const auto& xrhs = std::any_cast<gboost::fit_result_t>(irhs.extra(fold));

            UTEST_CHECK_CLOSE(xlhs.m_bias, xrhs.m_bias, epsilon);
            UTEST_CHECK_CLOSE(xlhs.m_statistics, xrhs.m_statistics, epsilon);
        }
    }

    check_equal(lhs.stats(value_type::errors), rhs.stats(value_type::errors), epsilon);
    check_equal(lhs.stats(value_type::losses), rhs.stats(value_type::losses), epsilon);
}

static void check_result(const fit_result_t& result, const strings_t& expected_param_names,
                         const tensor_size_t expected_folds = 2, const scalar_t epsilon = 1e-5)
{
    ::check_result(result, expected_param_names, expected_param_names.empty() ? 1U : 4U, expected_folds, epsilon);

    if (expected_param_names.empty())
    {
        UTEST_CHECK_EQUAL(result.param_results().size(), 1U);
    }

    const auto delta_train_loss =
        (expected_param_names.size() == 1U && expected_param_names[0U] == "vAreg") ? 1e-3 : 0.0;

    for (const auto& param_result : result.param_results())
    {
        for (tensor_size_t fold = 0; fold < expected_folds; ++fold)
        {
            const auto& result          = std::any_cast<gboost::fit_result_t>(param_result.extra(fold));
            const auto [rounds, nstats] = result.m_statistics.dims();
            const auto optimum_round    = static_cast<tensor_size_t>(result.m_wlearners.size());

            auto last_train_loss = std::numeric_limits<scalar_t>::max();

            UTEST_CHECK_LESS(rounds, 200);
            UTEST_REQUIRE_EQUAL(nstats, 7);
            UTEST_CHECK_GREATER_EQUAL(rounds, optimum_round);
            for (tensor_size_t round = 0; round < optimum_round + 1; ++round)
            {
                const auto train_error = result.m_statistics(round, 0);
                const auto train_loss  = result.m_statistics(round, 1);
                const auto valid_error = result.m_statistics(round, 2);
                const auto valid_loss  = result.m_statistics(round, 3);
                const auto fcalls      = result.m_statistics(round, 4);
                const auto gcalls      = result.m_statistics(round, 5);
                const auto status      = result.m_statistics(round, 6);

                UTEST_CHECK(std::isfinite(train_error));
                UTEST_CHECK(std::isfinite(train_loss));
                UTEST_CHECK(std::isfinite(valid_error));
                UTEST_CHECK(std::isfinite(valid_loss));

                UTEST_CHECK_GREATER_EQUAL(train_error, 0.0);
                UTEST_CHECK_GREATER_EQUAL(train_loss, 0.0);
                UTEST_CHECK_GREATER_EQUAL(valid_error, 0.0);
                UTEST_CHECK_GREATER_EQUAL(valid_loss, 0.0);

                UTEST_CHECK_GREATER(last_train_loss + delta_train_loss, train_loss);
                last_train_loss = train_loss;

                UTEST_CHECK_GREATER_EQUAL(fcalls, 1);
                UTEST_CHECK_GREATER_EQUAL(gcalls, 1);

                UTEST_CHECK_EQUAL(static_cast<solver_status>(static_cast<int>(status)), solver_status::converged);
            }
        }
    }
}

template <typename tdatasource>
auto check_gbooster(gboost_model_t model, const tdatasource& datasource0, const tensor_size_t folds = 2)
{
    const auto loss     = make_loss("mse");
    const auto solver   = make_solver("cgd-n", 1e-3, 1000);
    const auto dataset  = make_dataset(datasource0);
    const auto splitter = make_splitter("k-fold", folds, 42U);
    const auto tuner    = make_tuner("surrogate");
    const auto samples  = arange(0, dataset.samples());

    // fitting should fail if no weak learner to chose from
    UTEST_REQUIRE_THROW(make_gbooster().fit(dataset, samples, *loss, *solver, *splitter, *tuner), std::runtime_error);

    // fitting should work when properly setup
    auto fit_result = fit_result_t{};
    UTEST_REQUIRE_NOTHROW(fit_result = model.fit(dataset, samples, *loss, *solver, *splitter, *tuner));

    // check model
    datasource0.check_gbooster(model);
    check_predict(model, dataset, 1e-5);
    check_predict_throws(model);

    // check model loading and saving from and to binary streams
    const auto imodel = check_stream(model);
    datasource0.check_gbooster(model);
    check_predict(imodel, dataset, 1e-5);
    check_predict_throws(imodel);

    return fit_result;
}
