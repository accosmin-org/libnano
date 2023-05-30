#include "util.h"
#include <iomanip>
#include <nano/core/parameter_tracker.h>
#include <nano/dataset.h>
#include <nano/dataset/iterator.h>
#include <nano/linear/model.h>

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

    cmdline.add("", "list-loss", "list the available loss functions");
    cmdline.add("", "list-tuner", "list the available hyper-parameter tuning methods");
    cmdline.add("", "list-solver", "list the available solvers");
    cmdline.add("", "list-splitter", "list the available train-validation splitting methods");
    cmdline.add("", "list-datasource", "list the available machine learning datasets");
    cmdline.add("", "list-generator", "list the available feature generation methods");

    cmdline.add("", "list-linear-params", "list the parameters of the linear model");
    cmdline.add("", "list-loss-params", "list the parameters of the selected loss functions");
    cmdline.add("", "list-tuner-params", "list the parameters of the selected hyper-parameter tuning methods");
    cmdline.add("", "list-solver-params", "list the parameters of the selected solvers");
    cmdline.add("", "list-splitter-params", "list the parameters of the selected train-validation splitting methods");
    cmdline.add("", "list-datasource-params", "list the parameters of the selected machine learning datasets");
    cmdline.add("", "list-generator-params", "list the parameters of the selected feature generation methods");

    const auto options = ::process(cmdline, argc, argv);

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
    // TODO: experiments to evaluate feature value scaling, regularization method

    auto param_tracker = parameter_tracker_t{options};
    param_tracker.setup(*rloss);
    param_tracker.setup(*rtuner);
    param_tracker.setup(*rsolver);
    param_tracker.setup(*rsplitter);

    // load dataset
    ::load_datasource(*rdatasource);
    const auto dataset = ::load_dataset(*rdatasource, generator_ids);

    // train the model using nested cross-validation with respecting the datasource's test samples (if given):
    //  for each outer fold...
    //      make (training, validation) split
    //      fit (and tune) on the training samples
    //      evaluate on the validation samples
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

        const auto errors_values = model.evaluate(dataset, valid_samples, *rloss);

        log_info() << std::setprecision(8) << std::fixed << "linear: tests=" << errors_values.tensor(1).mean() << "/"
                   << errors_values.tensor(0).mean() << ",outer_fold=" << (outer_fold + 1) << "/" << tr_vd_splits.size()
                   << ".";

        // TODO: export inner/outer splits' results!

        // TODO: compute and export feature weights
        // TODO: check the selected features are the expected ones(lasso, elasticnet)
        // TODO: compute some sparsity factor
        // TODO: synthetic linear dataset (classification and regression) with known relevant feature sets
        auto feature_weights = make_full_tensor<scalar_t>(make_dims(dataset.features()), 0.0);

        const auto& weights = model.weights();
        for (tensor_size_t column = 0, columns = dataset.columns(); column < columns; ++column)
        {
            const auto feature = dataset.column2feature(column);
            feature_weights(feature) += weights.matrix().col(column).array().abs().sum();
        }

        std::cout << "feature_weights=" << feature_weights << std::endl;

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
