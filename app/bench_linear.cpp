#include <iomanip>
#include <nano/core/cmdline.h>
#include <nano/core/table.h>
#include <nano/critical.h>
#include <nano/linear.h>
#include <nano/linear/util.h>
#include <nano/main.h>

using namespace nano;

namespace
{
template <class tfactory>
auto make_object(const cmdresult_t& options, const tfactory& factory, const std::string_view option_name,
                 const std::string_view obj_name)
{
    const auto ids = factory.ids(std::regex(options.get<string_t>(option_name)));
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
    if (result.param_spaces().empty())
    {
        return string_t{"N/A"};
    }
    else
    {
        const auto& param_spaces = result.param_spaces();
        const auto  param_values = result.params(result.optimum_trial());
        assert(static_cast<tensor_size_t>(param_spaces.size()) == param_values.size());

        string_t str;
        for (size_t i = 0U; i < param_spaces.size(); ++i)
        {
            str += scat(param_spaces[i].name(), "=", std::fixed, std::setprecision(8),
                        param_values(static_cast<tensor_size_t>(i)), " ");
        }
        return str;
    }
}

int unsafe_main(int argc, const char* argv[])
{
    // parse the command line
    cmdline_t cmdline("benchmark linear machine learning models");
    cmdline.add("--linear", "regex to select linear model type", "elastic_net");
    cmdline.add("--loss", "regex to select loss functions", "<mandatory>");
    cmdline.add("--solver", "regex to select solvers", "lbfgs");
    cmdline.add("--tuner", "regex to select hyper-parameter tuning methods", "surrogate");
    cmdline.add("--splitter", "regex to select train-validation splitting methods (evaluation aka outer splits)",
                "k-fold");
    cmdline.add("--datasource", "regex to select machine learning datasets", "<mandatory>");
    cmdline.add("--generator", "regex to select feature generation methods", "identity.+");
    cmdline.add("--list-linear-params", "list the parameters of the linear model");

    const auto options = cmdline.process(argc, argv);
    if (cmdline.handle(options))
    {
        return EXIT_SUCCESS;
    }

    // check arguments and options
    const auto rmodel      = make_object(options, linear_t::all(), "--linear", "linear model");
    const auto rloss       = make_object(options, loss_t::all(), "--loss", "loss function");
    const auto rtuner      = make_object(options, tuner_t::all(), "--tuner", "hyper-parameter tuning method");
    const auto rsolver     = make_object(options, solver_t::all(), "--solver", "solver");
    const auto rsplitter   = make_object(options, splitter_t::all(), "--splitter", "train-validation splitting method");
    const auto rdatasource = make_object(options, datasource_t::all(), "--datasource", "machine learning dataset");
    const auto generator_ids = generator_t::all().ids(std::regex(options.get<string_t>("--generator")));

    if (options.has("--list-linear-params"))
    {
        table_t table;
        table.header() << "parameter"
                       << "value"
                       << "domain";
        table.delim();
        for (const auto& param : rmodel->parameters())
        {
            table.append() << param.name() << param.value() << param.domain();
        }
        std::cout << table;
        return EXIT_SUCCESS;
    }

    // TODO: option to save trained models
    // TODO: option to save training history to csv
    // TODO: wrapper script to generate plots?!
    // TODO: experiments to evaluate feature value scaling, regularization method, feature generation (products!)

    auto rconfig = cmdconfig_t{options};
    rconfig.setup(*rloss);
    rconfig.setup(*rtuner);
    rconfig.setup(*rsolver);
    rconfig.setup(*rsplitter);
    rconfig.setup(*rdatasource);

    // load dataset
    rdatasource->load();
    auto dataset = dataset_t{*rdatasource};
    for (const auto& generator_id : generator_ids)
    {
        dataset.add(generator_t::all().get(generator_id));
    }

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

        rconfig.setup(*rmodel);

        const auto fit_params = ml::params_t{}.solver(*rsolver).tuner(*rtuner).logger(make_stdout_logger());
        const auto fit_result = rmodel->fit(dataset, train_samples, *rloss, fit_params);

        const auto test_errors_values = rmodel->evaluate(dataset, valid_samples, *rloss);
        const auto optimum_trial      = fit_result.optimum_trial();

        table.append() << scat(outer_fold + 1, "/", tr_vd_splits.size()) << print_params(fit_result)
                       << print_scalar(fit_result.value(optimum_trial, ml::split_type::train, ml::value_type::errors))
                       << print_scalar(fit_result.value(optimum_trial, ml::split_type::valid, ml::value_type::errors))
                       << print_scalar(fit_result.stats(ml::value_type::errors).m_mean)
                       << print_scalar(test_errors_values.tensor(0).mean());
        std::cout << table;

        // TODO: export inner/outer splits' results!
        // TODO: check the selected features are the expected ones(lasso, elasticnet)
        // TODO: synthetic linear dataset (classification and regression) with known relevant feature sets
        const auto feature_importance = linear::feature_importance(dataset, rmodel->weights());

        const auto logger = make_stdout_logger();
        logger.log(log_type::info, std::fixed, std::setprecision(6),
                   "sparsity_ratio:", " @1e-2=", linear::sparsity_ratio(feature_importance, 1e-2),
                   ",@1e-3=", linear::sparsity_ratio(feature_importance, 1e-3),
                   ",@1e-4=", linear::sparsity_ratio(feature_importance, 1e-4),
                   ",@1e-5=", linear::sparsity_ratio(feature_importance, 1e-5),
                   ",@1e-6=", linear::sparsity_ratio(feature_importance, 1e-6));

        for (tensor_size_t ifeature = 0, features = dataset.features(); ifeature < features; ++ifeature)
        {
            const auto& feature = dataset.feature(ifeature);
            logger.log(log_type::info, "feature=", feature, ",importance=", feature_importance(ifeature));
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
