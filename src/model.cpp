#include <mutex>
#include <nano/core/stream.h>
#include <nano/model.h>
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
    register_parameter(parameter_t::make_integer("model::folds", 2, LE, 5, LE, 100));
    register_parameter(
        parameter_t::make_integer("model::random_seed", 0, LE, 42, LE, std::numeric_limits<int64_t>::max()));
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

void model_t::compatible(const dataset_t& dataset) const
{
    const auto n_features = dataset.features();

    critical(n_features != static_cast<tensor_size_t>(m_inputs.size()), "model: mis-matching number of inputs (",
             n_features, "), expecting (", m_inputs.size(), ")!");

    for (tensor_size_t i = 0; i < n_features; ++i)
    {
        const auto  feature          = dataset.feature(i);
        const auto& expected_feature = m_inputs[static_cast<size_t>(i)];

        critical(feature != expected_feature, "model: mis-matching input [", i, "/", n_features, "] (", feature,
                 "), expecting (", expected_feature, ")!");
    }

    critical(dataset.target() != m_target, "model: mis-matching target (", dataset.target(), "), expecting (", m_target,
             ")!");
}

std::istream& model_t::read(std::istream& stream)
{
    estimator_t::read(stream);

    critical(!::nano::read(stream, m_inputs) || !::nano::read(stream, m_target), "model: failed to read from stream!");

    return stream;
}

std::ostream& model_t::write(std::ostream& stream) const
{
    estimator_t::write(stream);

    critical(!::nano::write(stream, m_inputs) || !::nano::write(stream, m_target), "model: failed to write to stream!");

    return stream;
}

fit_result_t model_t::fit(const dataset_t& dataset, const indices_t& samples, const loss_t& loss,
                          const solver_t& solver)
{
    const auto n_features = dataset.features();

    m_target = dataset.target();
    m_inputs.clear();
    m_inputs.reserve(static_cast<size_t>(n_features));
    for (tensor_size_t i = 0; i < n_features; ++i)
    {
        m_inputs.push_back(dataset.feature(i));
    }

    return do_fit(dataset, samples, loss, solver);
}

tensor4d_t model_t::predict(const dataset_t& dataset, const indices_t& samples) const
{
    compatible(dataset);

    return do_predict(dataset, samples);
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
