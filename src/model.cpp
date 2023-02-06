#include <iomanip>
#include <mutex>
#include <nano/core/logger.h>
#include <nano/gboost/model.h>
#include <nano/linear/model.h>

using namespace nano;

using split_type = fit_result_t::split_type;
using value_type = fit_result_t::value_type;

model_t::model_t(string_t id)
    : clonable_t(std::move(id))
{
}

void model_t::logger(logger_t logger)
{
    m_logger = std::move(logger);
}

void model_t::log(const fit_result_t& fit_result) const
{
    if (m_logger)
    {
        m_logger(fit_result, type_id());
    }
}

model_t::logger_t model_t::make_logger_stdio(const int precision)
{
    return [=, last_trial = size_t{0U}](const fit_result_t& result, const string_t& prefix) mutable
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

            auto sum_train_losses = 0.0, sum_train_errors = 0.0;
            auto sum_valid_losses = 0.0, sum_valid_errors = 0.0;
            for (tensor_size_t fold = 0; fold < folds; ++fold)
            {
                sum_train_losses += param_result.stats(fold, split_type::train, value_type::losses).m_mean;
                sum_train_errors += param_result.stats(fold, split_type::train, value_type::errors).m_mean;
                sum_valid_losses += param_result.stats(fold, split_type::valid, value_type::losses).m_mean;
                sum_valid_errors += param_result.stats(fold, split_type::valid, value_type::errors).m_mean;
            }

            print_params(param_result.params(), "train=", sum_train_losses / norm, "/", sum_train_errors / norm, ",",
                         "valid=", sum_valid_losses / norm, "/", sum_valid_errors / norm);
        }
        last_trial = param_results.size();

        if (std::isfinite(optim_errors_stats.m_mean))
        {
            const auto& optimum_params = result.optimum().params();
            print_params(optimum_params, "refit=", optim_losses_stats.m_mean, "/", optim_errors_stats.m_mean);
        }
    };
}

factory_t<model_t>& model_t::all()
{
    static auto manager = factory_t<model_t>{};
    const auto  op      = []()
    {
        manager.add<gboost_model_t>("gradient boosting model (and variants: VadaBoost-like, TreeBoost)");
        manager.add<linear_model_t>("linear model (and variants: Ridge, Lasso, ElasticNet, VadaBoost-like)");
    };

    static std::once_flag flag;
    std::call_once(flag, op);

    return manager;
}
