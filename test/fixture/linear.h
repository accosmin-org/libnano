#include <fixture/configurable.h>
#include <fixture/dataset.h>
#include <fixture/learner.h>
#include <fixture/splitter.h>
#include <nano/dataset/iterator.h>
#include <nano/linear.h>
#include <nano/linear/result.h>
#include <nano/linear/util.h>

using namespace nano;
using namespace nano::ml;

template <class tweights, class tbias>
[[maybe_unused]] static void check_linear(const dataset_t& dataset, tweights weights, tbias bias, scalar_t epsilon)
{
    const auto samples = dataset.samples();

    auto all_outputs = tensor4d_t{cat_dims(samples, dataset.target_dims())};
    auto all_targets = tensor4d_t{cat_dims(samples, dataset.target_dims())};
    auto all_called  = make_full_tensor<tensor_size_t>(make_dims(samples), 0);

    auto iterator = flatten_iterator_t{dataset, arange(0, samples)};
    iterator.batch(11);
    iterator.scaling(scaling_type::none);
    iterator.loop(
        [&](tensor_range_t range, size_t, tensor2d_cmap_t inputs, tensor4d_cmap_t targets)
        {
            for (tensor_size_t i = 0, size = range.size(); i < size; ++i)
            {
                all_called(range.begin() + i)         = 1;
                all_targets.vector(range.begin() + i) = targets.vector(i);
                all_outputs.vector(range.begin() + i) = weights * inputs.vector(i) + bias;
            }
        });

    UTEST_CHECK_EQUAL(all_called, make_full_tensor<tensor_size_t>(make_dims(samples), 1));
    UTEST_CHECK_CLOSE(all_targets, all_outputs, epsilon);
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

[[maybe_unused]] auto make_model(const string_t& id, const scaling_type scaling, const tensor_size_t batch = 100)
{
    auto model = linear_t::all().get(id);
    UTEST_REQUIRE(model);
    model->parameter("linear::batch")   = batch;
    model->parameter("linear::scaling") = scaling;
    return model;
}

[[maybe_unused]] auto make_fit_params(const rsolver_t& solver)
{
    return params_t{}.splitter(make_splitter("k-fold", 2)).solver(solver).logger(make_stdout_logger());
}

[[maybe_unused]] void check_outputs(const dataset_t& dataset, const indices_t& samples, const tensor4d_t& outputs,
                                    const scalar_t epsilon)
{
    auto all_targets = tensor4d_t{cat_dims(samples.size(), dataset.target_dims())};

    auto iterator = flatten_iterator_t{dataset, samples};
    iterator.batch(7);
    iterator.scaling(scaling_type::none);
    iterator.loop([&](tensor_range_t range, size_t, tensor4d_cmap_t targets) { all_targets.slice(range) = targets; });

    UTEST_CHECK_CLOSE(all_targets, outputs, epsilon);
}

[[maybe_unused]] void check_model(const linear_t& model, const dataset_t& dataset, const indices_t& samples,
                                  const scalar_t epsilon)
{
    const auto outputs = model.predict(dataset, samples);
    check_outputs(dataset, samples, outputs, epsilon);

    UTEST_CHECK_EQUAL(model.weights().dims(), make_dims(1, dataset.columns()));
    UTEST_CHECK_EQUAL(model.bias().dims(), make_dims(1));

    string_t str;
    {
        std::ostringstream stream;
        UTEST_REQUIRE_NOTHROW(model.write(stream));
        str = stream.str();
    }
    {
        auto               new_model = make_model("ordinary", scaling_type::none);
        std::istringstream stream(str);
        UTEST_REQUIRE_NOTHROW(new_model->read(stream));
        const auto new_outputs = new_model->predict(dataset, samples);
        UTEST_CHECK_CLOSE(outputs, new_outputs, epsilon0<scalar_t>());
    }
}

[[maybe_unused]] void check_importance(const linear_t& model, const dataset_t& dataset, const indices_t& relevancy)
{
    const auto importance         = linear::feature_importance(dataset, model.weights());
    const auto sparsity           = linear::sparsity_ratio(importance);
    const auto expected_revelancy = static_cast<scalar_t>(relevancy.sum()) / static_cast<scalar_t>(dataset.features());

    UTEST_REQUIRE_EQUAL(relevancy.size(), dataset.features());
    UTEST_REQUIRE_EQUAL(relevancy.size(), importance.size());

    for (tensor_size_t feature = 0, features = dataset.features(); feature < features; ++feature)
    {
        if (relevancy(feature) != 0)
        {
            UTEST_CHECK_GREATER(importance(feature), 1e-1);
        }
        else
        {
            UTEST_CHECK_LESS(importance(feature), 1e-6);
        }
    }

    UTEST_CHECK_CLOSE(sparsity, 1.0 - expected_revelancy, 1e-15);
}
