#include <iomanip>
#include <nano/core/cmdline.h>
#include <nano/core/table.h>
#include <nano/critical.h>
#include <nano/gboost/model.h>
#include <nano/main.h>
#include <nano/wlearner.h>

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
    cmdline_t cmdline("benchmark gradient boosting machine learning models");
    cmdline.add("--loss", "regex to select loss functions", "<mandatory>");
    cmdline.add("--solver", "regex to select solvers", "lbfgs");
    cmdline.add("--tuner", "regex to select hyper-parameter tuning methods", "surrogate");
    cmdline.add("--splitter", "regex to select train-validation splitting methods (evaluation aka outer splits)",
                "k-fold");
    cmdline.add("--datasource", "regex to select machine learning datasets", "<mandatory>");
    cmdline.add("--generator", "regex to select feature generation methods", "identity.+");
    cmdline.add("--wlearner", "regex to select weak learners", "<mandatory>");
    cmdline.add("--list-gboost-params", "list the parameters of the gradient boosting model");

    const auto options = cmdline.process(argc, argv);
    if (cmdline.handle(options))
    {
        return EXIT_SUCCESS;
    }
    if (options.has("--list-gboost-params"))
    {
        table_t table;
        table.header() << "parameter"
                       << "value"
                       << "domain";
        table.delim();
        const auto configurable = gboost_model_t{};
        for (const auto& param : configurable.parameters())
        {
            table.append() << param.name() << param.value() << param.domain();
        }
        std::cout << table;
        return EXIT_SUCCESS;
    }

    // check arguments and options
    const auto rloss       = make_object(options, loss_t::all(), "--loss", "loss function");
    const auto rtuner      = make_object(options, tuner_t::all(), "--tuner", "hyper-parameter tuning method");
    const auto rsolver     = make_object(options, solver_t::all(), "--solver", "solver");
    const auto rsplitter   = make_object(options, splitter_t::all(), "--splitter", "train-validation splitting method");
    const auto rdatasource = make_object(options, datasource_t::all(), "--datasource", "machine learning dataset");
    const auto generator_ids = generator_t::all().ids(std::regex(options.get<string_t>("--generator")));
    const auto wlearner_ids  = wlearner_t::all().ids(std::regex(options.get<string_t>("--wlearner")));

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

    auto wlearners = rwlearners_t{};
    for (const auto& wlearner_id : wlearner_ids)
    {
        auto wlearner = wlearner_t::all().get(wlearner_id);
        assert(wlearner != nullptr);
        rconfig.setup(*wlearner);
        wlearners.emplace_back(std::move(wlearner));
    }

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

        auto model = gboost_model_t{};
        rconfig.setup(model);

        const auto fit_logger = ml::params_t::make_stdio_logger();
        const auto fit_params = ml::params_t{}.solver(*rsolver).tuner(*rtuner).logger(fit_logger);
        const auto fit_result = model.fit(dataset, train_samples, *rloss, wlearners, fit_params);

        const auto test_errors_values = model.evaluate(dataset, valid_samples, *rloss);
        const auto optimum_trial      = fit_result.optimum_trial();

        table.append() << scat(outer_fold + 1, "/", tr_vd_splits.size()) << print_params(fit_result)
                       << print_scalar(fit_result.value(optimum_trial, ml::split_type::train, ml::value_type::errors))
                       << print_scalar(fit_result.value(optimum_trial, ml::split_type::valid, ml::value_type::errors))
                       << print_scalar(fit_result.stats(ml::value_type::errors).m_mean)
                       << print_scalar(test_errors_values.tensor(0).mean());
        std::cout << table;

        // TODO: export inner/outer splits' results!
        // TODO: check the selected features are the expected ones(lasso, elasticnet)
        // TODO: feature importance analysis

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
