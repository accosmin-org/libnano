#include <iomanip>
#include <nano/core/logger.h>
#include <nano/mlearn/params.h>
#include <nano/mlearn/result.h>

using namespace nano;
using namespace nano::ml;

params_t::logger_t params_t::make_logger_stdio(const int precision)
{
    return [=, last_trial = size_t{0U}](const result_t& result, const string_t& prefix) mutable
    {
        const auto& param_names        = result.param_names();
        const auto& param_results      = result.param_results();
        const auto  optim_errors_stats = result.stats(value_type::errors);
        const auto  optim_losses_stats = result.stats(value_type::losses);

        const auto print_params = [&](const tensor1d_t& param_values, const auto... tokens)
        {
            assert(param_names.size() == static_cast<size_t>(param_values.size()));

            auto logger = log_info();
            logger << std::fixed << std::setprecision(precision) << std::fixed << prefix << ": ";
            for (size_t i = 0U, size = param_names.size(); i < size; ++i)
            {
                logger << param_names[i] << "=" << param_values(static_cast<tensor_size_t>(i)) << ",";
            }
            (logger << ... << tokens) << ".";
        };

        for (size_t trial = last_trial; trial < param_results.size(); ++trial)
        {
            const auto& param_result = param_results[trial];
            const auto  folds        = param_result.folds();
            const auto  norm         = static_cast<scalar_t>(folds);

            auto sum_train_losses = 0.0;
            auto sum_train_errors = 0.0;
            auto sum_valid_losses = 0.0;
            auto sum_valid_errors = 0.0;
            for (tensor_size_t fold = 0; fold < folds; ++fold)
            {
                const auto fold_train_value = param_result.stats(fold, split_type::train, value_type::losses).m_mean;
                const auto fold_train_error = param_result.stats(fold, split_type::train, value_type::errors).m_mean;
                const auto fold_valid_value = param_result.stats(fold, split_type::valid, value_type::losses).m_mean;
                const auto fold_valid_error = param_result.stats(fold, split_type::valid, value_type::errors).m_mean;

                print_params(param_result.params(), "train=", fold_train_value, "/", fold_train_error, ",",
                             "valid=", fold_valid_value, "/", fold_valid_error, ",fold=", (fold + 1), "/", folds);

                sum_train_losses += fold_train_value;
                sum_train_errors += fold_train_error;
                sum_valid_losses += fold_valid_value;
                sum_valid_errors += fold_valid_error;
            }

            print_params(param_result.params(), "train=", sum_train_losses / norm, "/", sum_train_errors / norm, ",",
                         "valid=", sum_valid_losses / norm, "/", sum_valid_errors / norm, "(average)");
        }
        last_trial = param_results.size();

        if (std::isfinite(optim_errors_stats.m_mean))
        {
            const auto& optimum_params = result.optimum().params();
            print_params(optimum_params, "refit=", optim_losses_stats.m_mean, "/", optim_errors_stats.m_mean);
        }
    };
}

params_t::params_t()
{
    tuner("surrogate");
    solver("hoshino");
    splitter("k-fold");
}

params_t::params_t(params_t&&) noexcept = default;

params_t::params_t(const params_t& other)
    : m_logger(other.m_logger)
    , m_tuner(other.m_tuner->clone())
    , m_solver(other.m_solver->clone())
    , m_splitter(other.m_splitter->clone())
{
}

params_t& params_t::operator=(params_t&&) noexcept = default;

params_t& params_t::operator=(const params_t& other)
{
    if (this != &other)
    {
        m_logger   = other.m_logger;
        m_tuner    = other.m_tuner->clone();
        m_solver   = other.m_solver->clone();
        m_splitter = other.m_splitter->clone();
    }
    return *this;
}

params_t::~params_t() = default;

params_t& params_t::tuner(const tuner_t& tuner)
{
    m_tuner = tuner.clone();
    return *this;
}

params_t& params_t::tuner(rtuner_t&& tuner)
{
    critical(!tuner, "params: unitialized tuner!");
    m_tuner = std::move(tuner);
    return *this;
}

params_t& params_t::tuner(const rtuner_t& tuner)
{
    critical(!tuner, "params: unitialized tuner!");
    m_tuner = tuner->clone();
    return *this;
}

params_t& params_t::tuner(const string_t& id)
{
    return tuner(tuner_t::all().get(id));
}

params_t& params_t::solver(const solver_t& solver)
{
    m_solver = solver.clone();
    return *this;
}

params_t& params_t::solver(rsolver_t&& solver)
{
    critical(!solver, "params: unitialized solver!");
    m_solver = std::move(solver);
    return *this;
}

params_t& params_t::solver(const rsolver_t& solver)
{
    critical(!solver, "params: unitialized solver!");
    m_solver = solver->clone();
    return *this;
}

params_t& params_t::solver(const string_t& id)
{
    return solver(solver_t::all().get(id));
}

params_t& params_t::splitter(const splitter_t& splitter)
{
    m_splitter = splitter.clone();
    return *this;
}

params_t& params_t::splitter(rsplitter_t&& splitter)
{
    critical(!splitter, "params: unitialized splitter!");
    m_splitter = std::move(splitter);
    return *this;
}

params_t& params_t::splitter(const rsplitter_t& splitter)
{
    critical(!splitter, "params: unitialized splitter!");
    m_splitter = splitter->clone();
    return *this;
}

params_t& params_t::splitter(const string_t& id)
{
    return splitter(splitter_t::all().get(id));
}

params_t& params_t::logger(logger_t logger)
{
    m_logger = std::move(logger);
    return *this;
}

const tuner_t& params_t::tuner() const
{
    return *m_tuner;
}

const solver_t& params_t::solver() const
{
    return *m_solver;
}

const splitter_t& params_t::splitter() const
{
    return *m_splitter;
}

const params_t::logger_t& params_t::logger() const
{
    return m_logger;
}

void params_t::log(const result_t& result, const string_t& prefix) const
{
    if (m_logger)
    {
        m_logger(result, prefix);
    }
}
