#include <nano/table.h>
#include <nano/chrono.h>
#include <nano/logger.h>
#include <nano/cmdline.h>
#include <nano/imclass.h>
#include <nano/linear/model.h>
#include <nano/iterator/memfixed.h>

using namespace nano;

static auto get_imclass(const string_t& id, const size_t folds, const int train_percentage)
{
    const auto start = nano::timer_t{};

    auto dataset = imclass_dataset_t::all().get(id);
    critical(!dataset, scat("invalid dataset '", id, "'"));
    dataset->folds(folds);
    dataset->train_percentage(train_percentage);
    critical(!dataset->load(), scat("failed to load dataset '", id, "'"));

    log_info() << ">>> loading done in " << start.elapsed() << ".";
    return dataset;
}

static auto get_loss(const string_t& id)
{
    auto loss = loss_t::all().get(id);
    critical(!loss, scat("invalid loss '", id, "'"));
    return loss;
}

static auto get_solver(const string_t& id, const scalar_t epsilon, const int max_iterations)
{
    auto solver = solver_t::all().get(id);
    critical(!solver, scat("invalid solver '", id, "'"));
    solver->epsilon(epsilon);
    solver->max_iterations(max_iterations);
    return solver;
}

static int unsafe_main(int argc, const char* argv[])
{
    // parse the command line
    cmdline_t cmdline("report statistics on datasets");
    cmdline.add("", "imclass",          "regex to select image classification datasets", ".+");
    cmdline.add("", "solver",           "regex to select the solvers to benchmark", "lbfgs");
    cmdline.add("", "loss",             "regex to select the loss functions to benchmark", "s-classnll");
    cmdline.add("", "regularization",   "regex to select the regularization methods to benchmark", ".+");
    cmdline.add("", "epsilon",          "convergence criterion (solver)", 1e-3);
    cmdline.add("", "max-iterations",   "maximum number of iterations (solver)", 1000);
    cmdline.add("", "max-trials-tune",  "maximum number of trials per tuning step", 7);
    cmdline.add("", "tune-steps",       "number of tuning steps", 2);
    cmdline.add("", "folds",            "number of folds [1, 100]", 10);
    cmdline.add("", "train-percentage", "percentage of training samples excluding the test samples [10, 90]", 80);
    cmdline.add("", "batch",            "batch size in number of samples [1, 4092]", 32);

    // todo: table with stats for various configurations
    // todo: option to save training history to csv
    // todo: option to save trained models
    // todo: wrapper bash script to generate plots with gnuplot?!

    cmdline.process(argc, argv);

    if (cmdline.has("help"))
    {
        cmdline.usage();
        return EXIT_SUCCESS;
    }

    const auto batch = cmdline.get<int>("batch");
    const auto folds = cmdline.get<size_t>("folds");
    const auto epsilon = cmdline.get<scalar_t>("epsilon");
    const auto tune_steps = cmdline.get<int>("tune-steps");
    const auto max_iterations = cmdline.get<int>("max-iterations");
    const auto train_percentage = cmdline.get<int>("train-percentage");
    const auto max_trials_per_tune_step = cmdline.get<int>("max-trials-tune");
    const auto regularizations = enum_values<linear_model_t::regularization>(std::regex(cmdline.get<string_t>("regularization")));

    // prepare table header
    table_t table;
    auto& header0 = table.header();
    header0 << colspan(3) << "";
    for (const auto regularization : regularizations)
    {
        header0 << alignment::center << colspan(3) << scat("regularization[", regularization, "]");
    }
    table.delim();
    auto& header1 = table.append();
    header1 << alignment::center << "imclass" << "solver" << "loss";
    for (const auto regularization : regularizations)
    {
        NANO_UNUSED1(regularization);
        header1 << alignment::center << "error" << alignment::center << "eval[ms]" << alignment::center << "train[ms]";
    }

    // for each image classification dataset...
    for (const auto& id_imclass : imclass_dataset_t::all().ids(std::regex(cmdline.get<string_t>("imclass"))))
    {
        const auto dataset = get_imclass(id_imclass, folds, train_percentage);
        const auto iterator = memfixed_iterator_t<uint8_t>{*dataset};

        // for each numerical optimization method...
        for (const auto& id_solver : solver_t::all().ids(std::regex(cmdline.get<string_t>("solver"))))
        {
            const auto solver = get_solver(id_solver, epsilon, max_iterations);

            // for each loss function
            for (const auto& id_loss : loss_t::all().ids(std::regex(cmdline.get<string_t>("loss"))))
            {
                const auto loss = get_loss(id_loss);

                table.delim();
                auto& row = table.append();
                row << id_imclass << id_solver << id_loss;

                // for each regularization method
                auto model = linear_model_t{};
                for (const auto regularization : regularizations)
                {
                    const auto training = model.train(*loss, iterator, *solver, regularization,
                        batch, max_trials_per_tune_step, tune_steps);

                    stats_t te_errors, eval_times, train_times;
                    for (const auto& tfold : training)
                    {
                        te_errors(tfold.m_te_error);
                        eval_times(tfold.m_eval_time.count());
                        train_times(tfold.m_train_time.count());
                    }

                    row << scat(std::setprecision(2), std::fixed, te_errors.median())
                        << std::lround(eval_times.median())
                        << std::lround(train_times.median());
                }
            }
        }
    }

    std::cout << table;

    // OK
    return EXIT_SUCCESS;
}

int main(int argc, const char* argv[])
{
    return nano::main(unsafe_main, argc, argv);
}
