#include <nano/linear/enums.h>
#include <nano/linear/function.h>
#include <nano/linear/model.h>
#include <nano/linear/util.h>
#include <nano/model/util.h>
#include <nano/tensor/stream.h>

using namespace nano;
using namespace nano::linear;

static auto make_params(const configurable_t& configurable)
{
    const auto regularization = configurable.parameter("linear::regularization").value<regularization_type>();

    auto param_names  = strings_t{};
    auto param_spaces = param_spaces_t{};

    static const auto param_space =
        make_param_space(param_space_t::type::log10, 1e-9, 1e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3,
                         3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e+0, 3e+0, 1e+1, 3e+1, 1e+2, 3e+2, 1e+3, 3e+3, 1e+4, 3e+4, 1e+5,
                         3e+5, 1e+6, 3e+6, 1e+7, 3e+7, 1e+8, 1e+9);

    switch (regularization)
    {
    case regularization_type::lasso:
        param_names.emplace_back("l1reg");
        param_spaces.emplace_back(param_space);
        break;

    case regularization_type::ridge:
        param_names.emplace_back("l2reg");
        param_spaces.emplace_back(param_space);
        break;

    case regularization_type::elasticnet:
        param_names.emplace_back("l1reg");
        param_names.emplace_back("l2reg");
        param_spaces.emplace_back(param_space);
        param_spaces.emplace_back(param_space);
        break;

    case regularization_type::variance:
        param_names.emplace_back("vAreg");
        param_spaces.emplace_back(param_space);
        break;

    default: break;
    }

    return std::make_tuple(std::move(param_names), std::move(param_spaces));
}

static auto decode_params(const tensor1d_cmap_t& params, const regularization_type regularization)
{
    scalar_t l1reg = 0.0, l2reg = 0.0, vAreg = 0.0;
    switch (regularization)
    {
    case regularization_type::lasso: l1reg = params(0); break;
    case regularization_type::ridge: l2reg = params(0); break;
    case regularization_type::variance: vAreg = params(0); break;
    case regularization_type::elasticnet: l1reg = params(0), l2reg = params(1); break;
    default: break;
    }

    return std::make_tuple(l1reg, l2reg, vAreg);
}

static auto make_x0(const ::nano::linear::function_t& function, const std::any& extra)
{
    vector_t x0 = vector_t::Zero(function.size());
    if (extra.has_value())
    {
        const auto& [weights, bias]             = std::any_cast<std::tuple<tensor2d_t, tensor1d_t>>(extra);
        x0.segment(0, weights.size())           = weights.array();
        x0.segment(weights.size(), bias.size()) = bias.array();
    }
    return x0;
} // LCOV_EXCL_LINE

static auto fit(const configurable_t& configurable, const dataset_t& dataset, const indices_t& samples,
                const loss_t& loss, const solver_t& solver, const scalar_t l1reg, const scalar_t l2reg,
                const scalar_t vAreg, const std::any& extra = std::any{})
{
    auto iterator = flatten_iterator_t{dataset, samples};
    iterator.batch(configurable.parameter("linear::batch").value<tensor_size_t>());
    iterator.scaling(configurable.parameter("linear::scaling").value<scaling_type>());
    iterator.cache_flatten(std::numeric_limits<tensor_size_t>::max());
    iterator.cache_targets(std::numeric_limits<tensor_size_t>::max());

    const auto function = ::nano::linear::function_t{iterator, loss, l1reg, l2reg, vAreg};
    const auto state    = solver.minimize(function, make_x0(function, extra));

    tensor1d_t bias    = function.bias(state.x);
    tensor2d_t weights = function.weights(state.x);
    ::upscale(iterator.flatten_stats(), iterator.scaling(), iterator.targets_stats(), iterator.scaling(), weights,
              bias);

    return std::make_tuple(std::move(weights), std::move(bias));
}

linear_model_t::linear_model_t()
    : model_t("linear")
{
    register_parameter(parameter_t::make_integer("linear::batch", 10, LE, 100, LE, 10000));
    register_parameter(parameter_t::make_enum("linear::scaling", scaling_type::standard));
    register_parameter(parameter_t::make_enum("linear::regularization", regularization_type::lasso));
}

rmodel_t linear_model_t::clone() const
{
    return std::make_unique<linear_model_t>(*this);
}

std::istream& linear_model_t::read(std::istream& stream)
{
    model_t::read(stream);

    critical(!::nano::read(stream, m_bias) || !::nano::read(stream, m_weights),
             "linear model: failed to read from stream!");

    critical(m_bias.size() != m_weights.rows(), "linear model: parameter mismatch!");

    return stream;
}

std::ostream& linear_model_t::write(std::ostream& stream) const
{
    model_t::write(stream);

    critical(!::nano::write(stream, m_bias) || !::nano::write(stream, m_weights),
             "linear model: failed to write to stream!");

    return stream;
}

fit_result_t linear_model_t::fit(const dataset_t& dataset, const indices_t& samples, const loss_t& loss,
                                 const solver_t& solver, const splitter_t& splitter, const tuner_t& tuner)
{
    learner_t::fit(dataset);

    const auto batch          = parameter("linear::batch").value<tensor_size_t>();
    const auto regularization = parameter("linear::regularization").value<regularization_type>();

    // tune hyper-parameters
    auto [param_names, param_spaces] = ::make_params(*this);

    const auto evaluator =
        [&](const auto& train_samples, const auto& valid_samples, const auto& params, const auto& extra)
    {
        const auto [l1reg, l2reg, vAreg] = decode_params(params, regularization);

        auto [weights, bias]     = ::fit(*this, dataset, train_samples, loss, solver, l1reg, l2reg, vAreg, extra);
        auto train_errors_losses = evaluate(dataset, train_samples, loss, weights, bias, batch);
        auto valid_errors_losses = evaluate(dataset, valid_samples, loss, weights, bias, batch);

        return std::make_tuple(std::move(train_errors_losses), std::move(valid_errors_losses),
                               std::make_tuple(std::move(weights), std::move(bias)));
    };

    auto fit_result =
        ml::tune(samples, splitter, tuner, std::move(param_names), param_spaces, make_logger_lambda(), evaluator);

    // refit with the optimum hyper-parameters (if any) on all given samples
    {
        const auto [l1reg, l2reg, vAreg] = decode_params(fit_result.optimum().params(), regularization);

        auto [weights, bias] = ::fit(*this, dataset, samples, loss, solver, l1reg, l2reg, vAreg);
        auto errors_losses   = evaluate(dataset, samples, loss, weights, bias, batch);

        fit_result.evaluate(std::move(errors_losses));

        m_weights = std::move(weights);
        m_bias    = std::move(bias);
    }
    this->log(fit_result);

    return fit_result;
}

tensor4d_t linear_model_t::predict(const dataset_t& dataset, const indices_t& samples) const
{
    learner_t::critical_compatible(dataset);

    // TODO: no need to allocate the sample indices one more time
    // TODO: determine at runtime if worth parallelizing
    auto iterator = flatten_iterator_t(dataset, samples);
    iterator.scaling(scaling_type::none);
    iterator.batch(parameter("linear::batch").value<tensor_size_t>());

    tensor4d_t outputs(cat_dims(samples.size(), dataset.target_dims()));
    iterator.loop([&](tensor_range_t range, size_t, tensor2d_cmap_t inputs)
                  { ::nano::linear::predict(inputs, m_weights, m_bias, outputs.slice(range)); });

    return outputs;
}
