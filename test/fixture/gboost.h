#include <fixture/learner.h>
#include <fixture/solver.h>
#include <fixture/splitter.h>
#include <fixture/tuner.h>
#include <fixture/wlearner.h>
#include <nano/gboost/enums.h>
#include <nano/gboost/model.h>
#include <nano/gboost/result.h>

using namespace nano;
using namespace nano::ml;

static auto make_gbooster()
{
    auto model                            = gboost_model_t{};
    model.parameter("gboost::max_rounds") = 100;
    model.parameter("gboost::epsilon")    = 1e-6;
    model.parameter("gboost::patience")   = 2;
    return model;
}

static auto make_wlearners()
{
    auto wlearners = rwlearners_t{};
    wlearners.emplace_back(wlearner_t::all().get("affine"));
    wlearners.emplace_back(wlearner_t::all().get("dense-table"));
    return wlearners;
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

static void check_equal(const stats_t& lhs, const stats_t& rhs, const scalar_t epsilon = 1e-12)
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

static void check_equal(const result_t& lhs, const result_t& rhs, const scalar_t epsilon = 1e-12)
{
    UTEST_REQUIRE_EQUAL(lhs.folds(), rhs.folds());
    UTEST_REQUIRE_EQUAL(lhs.trials(), rhs.trials());

    UTEST_REQUIRE_EQUAL(lhs.param_spaces().size(), rhs.param_spaces().size());
    for (size_t i = 0; i < lhs.param_spaces().size(); ++i)
    {
        UTEST_CHECK_EQUAL(lhs.param_spaces()[i].name(), rhs.param_spaces()[i].name());
    }

    for (tensor_size_t trial = 0; trial < lhs.trials(); ++trial)
    {
        UTEST_CHECK_CLOSE(lhs.params(trial), rhs.params(trial), epsilon);

        for (tensor_size_t fold = 0; fold < lhs.folds(); ++fold)
        {
            const auto& xlhs = std::any_cast<gboost::result_t>(lhs.extra(trial, fold));
            const auto& xrhs = std::any_cast<gboost::result_t>(rhs.extra(trial, fold));

            UTEST_CHECK_CLOSE(xlhs.m_bias, xrhs.m_bias, epsilon);
            UTEST_CHECK_CLOSE(xlhs.m_statistics, xrhs.m_statistics, epsilon);
        }
    }

    check_equal(lhs.stats(value_type::errors), rhs.stats(value_type::errors), epsilon);
    check_equal(lhs.stats(value_type::losses), rhs.stats(value_type::losses), epsilon);
}

static void check_result(const result_t& result, const strings_t& expected_param_names,
                         const tensor_size_t expected_folds = 2, const scalar_t epsilon = 1e-5)
{
    ::check_result(result, expected_param_names, expected_param_names.empty() ? 1U : 4U, expected_folds, epsilon);

    for (tensor_size_t trial = 0; trial < result.trials(); ++trial)
    {
        for (tensor_size_t fold = 0; fold < expected_folds; ++fold)
        {
            const auto& pfresult        = std::any_cast<gboost::result_t>(result.extra(trial, fold));
            const auto [rounds, nstats] = pfresult.m_statistics.dims();
            const auto optimum_round    = static_cast<tensor_size_t>(pfresult.m_wlearners.size());

            auto last_train_loss = std::numeric_limits<scalar_t>::max();

            UTEST_CHECK_LESS(rounds, 200);
            UTEST_REQUIRE_EQUAL(nstats, 8);
            UTEST_CHECK_GREATER_EQUAL(rounds, optimum_round);
            for (tensor_size_t round = 0; round < rounds; ++round)
            {
                UTEST_NAMED_CASE(scat("params=", result.params(trial).array().transpose(), ",fold=", fold,
                                      ",round=", round, ",optim_round=", optimum_round));

                const auto train_error = pfresult.m_statistics(round, 0);
                const auto train_loss  = pfresult.m_statistics(round, 1);
                const auto valid_error = pfresult.m_statistics(round, 2);
                const auto valid_loss  = pfresult.m_statistics(round, 3);
                const auto shrinkage   = pfresult.m_statistics(round, 4);
                const auto fcalls      = pfresult.m_statistics(round, 5);
                const auto gcalls      = pfresult.m_statistics(round, 6);
                const auto status      = pfresult.m_statistics(round, 7);

                UTEST_CHECK(std::isfinite(train_error));
                UTEST_CHECK(std::isfinite(train_loss));
                UTEST_CHECK(std::isfinite(valid_error));
                UTEST_CHECK(std::isfinite(valid_loss));

                UTEST_CHECK_GREATER_EQUAL(train_error, 0.0);
                UTEST_CHECK_GREATER_EQUAL(train_loss, 0.0);
                UTEST_CHECK_GREATER_EQUAL(valid_error, 0.0);
                UTEST_CHECK_GREATER_EQUAL(valid_loss, 0.0);
                UTEST_CHECK_GREATER_EQUAL(shrinkage, 0.1);
                UTEST_CHECK_LESS_EQUAL(shrinkage, 1.0);

                UTEST_CHECK_GREATER_EQUAL(last_train_loss + epsilon, train_loss);
                if (round <= optimum_round)
                {
                    UTEST_CHECK_NOT_EQUAL(static_cast<solver_status>(static_cast<int>(status)), solver_status::failed);
                }
                last_train_loss = train_loss;

                UTEST_CHECK_GREATER_EQUAL(fcalls, 1);
                UTEST_CHECK_GREATER_EQUAL(gcalls, 1);
            }
        }
    }
}

template <class tdatasource>
auto check_gbooster(gboost_model_t model, const tdatasource& datasource0, const tensor_size_t folds = 2)
{
    const auto loss       = make_loss("mse");
    const auto dataset    = make_dataset(datasource0);
    const auto samples    = arange(0, dataset.samples());
    const auto splitter   = make_splitter("k-fold", folds, 42U);
    const auto fit_params = params_t{}.splitter(splitter).logger(make_stdout_logger());
    const auto wlearners  = make_wlearners();

    // fitting should fail if no weak learner to chose from
    {
        auto model0 = make_gbooster();
        model0.prototypes(rwlearners_t{});
        UTEST_REQUIRE_THROW(make_gbooster().fit(dataset, samples, *loss, fit_params), std::runtime_error);
    }

    // fitting should work when properly setup
    auto fit_result = result_t{};
    model.prototypes(wlearners);
    UTEST_REQUIRE_NOTHROW(fit_result = model.fit(dataset, samples, *loss, fit_params));

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
