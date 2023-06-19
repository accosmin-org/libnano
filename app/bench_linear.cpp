#include "util.h"
#include <iomanip>
#include <nano/core/cmdline.h>
#include <nano/core/parameter_tracker.h>
#include <nano/core/table.h>
#include <nano/dataset.h>
#include <nano/dataset/iterator.h>
#include <nano/linear/model.h>
#include <nano/linear/util.h>

using namespace nano;

namespace
{
template <typename tfactory>
auto make_object(const cmdline_t::result_t& options, const tfactory& factory, const char* const option,
                 const char* const obj_name)
{
    const auto ids = factory.ids(std::regex(options.get<string_t>(option)));
    critical(ids.size() != 1U, "expecting a single ", obj_name, ", got (", ids.size(), ") instead!");

    auto object = factory.get(ids[0U]);
    assert(object != nullptr);
    return object;
}

auto print_scalar(const scalar_t value)
{
    return scat(std::setprecision(6), std::fixed, value);
}

auto print_params(const ml::result_t& result)
{
    if (result.param_names().empty())
    {
        return string_t{"N/A"};
    }
    else
    {
        const auto& param_names  = result.param_names();
        const auto& param_values = result.optimum().params();
        assert(static_cast<tensor_size_t>(param_names.size()) == param_values.size());

        string_t str;
        for (size_t i = 0U; i < param_names.size(); ++i)
        {
            str += scat(param_names[i], "=", std::fixed, std::setprecision(8),
                        param_values(static_cast<tensor_size_t>(i)), " ");
        }
        return str;
    }
}

int unsafe_main(int argc, const char* argv[])
{
    // parse the command line
    cmdline_t cmdline("benchmark linear machine learning models");
    cmdline.add("", "loss", "regex to select loss functions", "<select>");
    cmdline.add("", "solver", "regex to select solvers", "lbfgs");
    cmdline.add("", "tuner", "regex to select hyper-parameter tuning methods", "surrogate");
    cmdline.add("", "splitter", "regex to select train-validation splitting methods (evaluation aka outer splits)",
                "k-fold");
    cmdline.add("", "datasource", "regex to select machine learning datasets", "<select>");
    cmdline.add("", "generator", "regex to select feature generation methods", "identity.+");
    cmdline.add("", "list-linear-params", "list the parameters of the linear model");

    const auto options = cmdline.process(argc, argv);
    if (options.has("help"))
    {
        cmdline.usage();
        std::exit(EXIT_SUCCESS);
    }
    if (options.has("list-linear-params"))
    {
        table_t table;
        table.header() << "parameter"
                       << "value"
                       << "domain";
        table.delim();
        const auto configurable = linear_model_t{};
        for (const auto& param : configurable.parameters())
        {
            table.append() << param.name() << param.value() << param.domain();
        }
        std::cout << table;
        return EXIT_SUCCESS;
    }

    // check arguments and options
    const auto rloss       = make_object(options, loss_t::all(), "loss", "loss function");
    const auto rtuner      = make_object(options, tuner_t::all(), "tuner", "hyper-parameter tuning method");
    const auto rsolver     = make_object(options, solver_t::all(), "solver", "solver");
    const auto rsplitter   = make_object(options, splitter_t::all(), "splitter", "train-validation splitting method");
    const auto rdatasource = make_object(options, datasource_t::all(), "datasource", "machine learning dataset");
    const auto generator_ids = generator_t::all().ids(std::regex(options.get<string_t>("generator")));

    // TODO: option to save trained models
    // TODO: option to save training history to csv
    // TODO: wrapper script to generate plots?!
    // TODO: experiments to evaluate feature value scaling, regularization method, feature generation (products!)

    auto param_tracker = parameter_tracker_t{options};
    param_tracker.setup(*rloss);
    param_tracker.setup(*rtuner);
    param_tracker.setup(*rsolver);
    param_tracker.setup(*rsplitter);
    param_tracker.setup(*rdatasource);

    // load dataset
    rdatasource->load();
    const auto dataset = ::load_dataset(*rdatasource, generator_ids);

    // train the model using nested cross-validation with respecting the datasource's test samples (if given):
    //  for each outer fold...
    //      make (training, validation) split
    //      fit (and tune) on the training samples
    //      evaluate on the validation samples
    auto table = table_t{};
    table.header() << "fold"
                   << "optimum params"
                   << "train error"
                   << "valid error"
                   << "refit error"
                   << "test error";
    table.delim();

    const auto test_samples = rdatasource->test_samples();
    const auto eval_samples = rdatasource->train_samples();
    const auto tr_vd_splits = rsplitter->split(eval_samples);
    for (size_t outer_fold = 0U; outer_fold < tr_vd_splits.size(); ++outer_fold)
    {
        const auto& [train_samples, valid_samples] = tr_vd_splits[outer_fold];

        auto model = linear_model_t{};
        param_tracker.setup(model);

        const auto fit_logger = ml::params_t::make_stdio_logger();
        const auto fit_params = ml::params_t{}.solver(*rsolver).tuner(*rtuner).logger(fit_logger);
        const auto fit_result = model.fit(dataset, train_samples, *rloss, fit_params);

        const auto test_errors_values = model.evaluate(dataset, valid_samples, *rloss);

        table.append() << scat(outer_fold + 1, "/", tr_vd_splits.size()) << print_params(fit_result)
                       << print_scalar(fit_result.optimum().value(ml::split_type::train, ml::value_type::errors))
                       << print_scalar(fit_result.optimum().value(ml::split_type::valid, ml::value_type::errors))
                       << print_scalar(fit_result.stats(ml::value_type::errors).m_mean)
                       << print_scalar(test_errors_values.tensor(0).mean());
        std::cout << table;

        // TODO: export inner/outer splits' results!
        // TODO: check the selected features are the expected ones(lasso, elasticnet)
        // TODO: synthetic linear dataset (classification and regression) with known relevant feature sets
        const auto feature_importance = linear::feature_importance(dataset, model.weights());

        log_info() << std::fixed << std::setprecision(6) << "sparsity_ratio:"
                   << " @1e-2=" << linear::sparsity_ratio(feature_importance, 1e-2)
                   << ",@1e-3=" << linear::sparsity_ratio(feature_importance, 1e-3)
                   << ",@1e-4=" << linear::sparsity_ratio(feature_importance, 1e-4)
                   << ",@1e-5=" << linear::sparsity_ratio(feature_importance, 1e-5)
                   << ",@1e-6=" << linear::sparsity_ratio(feature_importance, 1e-6);

        for (tensor_size_t ifeature = 0, features = dataset.features(); ifeature < features; ++ifeature)
        {
            const auto& feature = dataset.feature(ifeature);
            log_info() << "feature=" << feature << ",importance=" << feature_importance(ifeature);
        }

        (void)test_samples;
        // const auto tr_samples = rdatasource->train_samples();
    }

    // OK
    return EXIT_SUCCESS;
}
} // namespace

int main(int argc, const char* argv[])
{
    return nano::safe_main(unsafe_main, argc, argv);
}
