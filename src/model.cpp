#include <mutex>
//#include <nano/gboost/model.h>
#include <nano/linear/model.h>

using namespace nano;

fit_result_t::cv_result_t::cv_result_t() = default;

fit_result_t::cv_result_t::cv_result_t(tensor1d_t params, tensor_size_t folds)
    : m_params(std::move(params))
    , m_train_errors(folds)
    , m_train_values(folds)
    , m_valid_errors(folds)
    , m_valid_values(folds)
{
}

model_t::model_t(string_t id)
    : clonable_t(std::move(id))
{
}

void model_t::logger(const logger_t& logger)
{
    m_logger = logger;
}

void model_t::log(const fit_result_t& result, const string_t& prefix) const
{
    if (m_logger)
    {
        m_logger(result, prefix);
    }
}

factory_t<model_t>& model_t::all()
{
    static auto manager = factory_t<model_t>{};
    const auto  op      = []()
    {
        // manager.add<gboost_model_t>("");
        manager.add<linear_model_t>("linear model (and variants: Ridge, Lasso, ElasticNet, VadaBoost-like)");
    };

    static std::once_flag flag;
    std::call_once(flag, op);

    return manager;
}
