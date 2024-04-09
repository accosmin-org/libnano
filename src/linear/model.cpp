#include <nano/linear/enums.h>
#include <nano/linear/function.h>
#include <nano/linear/model.h>
#include <nano/linear/result.h>
#include <nano/linear/util.h>
#include <nano/mlearn/tune.h>
#include <nano/tensor/stream.h>

using namespace nano;
using namespace nano::linear;

namespace
{
auto make_params(const configurable_t& configurable)
{
    const auto regularization = configurable.parameter("linear::regularization").value<linear_regularization>();

    auto param_names  = strings_t{};
    auto param_spaces = param_spaces_t{};

    static const auto param_space =
        make_param_space(param_space_t::type::log10, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2,
                         3e-2, 1e-1, 3e-1, 1e+0, 3e+0, 1e+1, 3e+1, 1e+2, 3e+2, 1e+3, 3e+3, 1e+4);

    switch (regularization)
    {
    case linear_regularization::lasso:
        param_names.emplace_back("l1reg");
        param_spaces.emplace_back(param_space);
        break;

    case linear_regularization::ridge:
        param_names.emplace_back("l2reg");
        param_spaces.emplace_back(param_space);
        break;

    case linear_regularization::elasticnet:
        param_names.emplace_back("l1reg");
        param_names.emplace_back("l2reg");
        param_spaces.emplace_back(param_space);
        param_spaces.emplace_back(param_space);
        break;

    default: break;
    }

    return std::make_tuple(std::move(param_names), std::move(param_spaces));
}

auto decode_params(const tensor1d_cmap_t& params, const linear_regularization regularization)
{
    scalar_t l1reg = 0.0;
    scalar_t l2reg = 0.0;
    switch (regularization)
    {
    case linear_regularization::lasso: l1reg = params(0); break;
    case linear_regularization::ridge: l2reg = params(0); break;
    case linear_regularization::elasticnet: l1reg = params(0), l2reg = params(1); break;
    default: break;
    }

    return std::make_tuple(l1reg, l2reg);
}

auto make_x0(const ::nano::linear::function_t& function, const std::any& extra)
{
    vector_t x0 = vector_t::zero(function.size());
    if (extra.has_value())
    {
        const auto& result                      = std::any_cast<linear::result_t>(extra);
        const auto& bias                        = result.m_bias;
        const auto& weights                     = result.m_weights;
        x0.segment(0, weights.size())           = weights.array();
        x0.segment(weights.size(), bias.size()) = bias.array();
    }
    return x0;
} // LCOV_EXCL_LINE

auto fit(const configurable_t& configurable, const dataset_t& dataset, const indices_t& samples, const loss_t& loss,
         const solver_t& solver, const scalar_t l1reg, const scalar_t l2reg, const std::any& extra = std::any{})
{
    auto iterator = flatten_iterator_t{dataset, samples};
    iterator.batch(configurable.parameter("linear::batch").value<tensor_size_t>());
    iterator.scaling(configurable.parameter("linear::scaling").value<scaling_type>());
    iterator.cache_flatten(std::numeric_limits<tensor_size_t>::max());
    iterator.cache_targets(std::numeric_limits<tensor_size_t>::max());

    const auto function = ::nano::linear::function_t{iterator, loss, l1reg, l2reg};
    const auto state    = solver.minimize(function, make_x0(function, extra));

    tensor1d_t bias    = function.bias(state.x());
    tensor2d_t weights = function.weights(state.x());
    ::upscale(iterator.flatten_stats(), iterator.scaling(), iterator.targets_stats(), iterator.scaling(), weights,
              bias);

    return linear::result_t{std::move(bias), std::move(weights), state};
}
} // namespace

linear_model_t::linear_model_t()
{
    register_parameter(parameter_t::make_integer("linear::batch", 10, LE, 100, LE, 10000));
    register_parameter(parameter_t::make_enum("linear::scaling", scaling_type::standard));
    register_parameter(parameter_t::make_enum("linear::regularization", linear_regularization::lasso));
}

std::istream& linear_model_t::read(std::istream& stream)
{
    learner_t::read(stream);

    critical(!::nano::read(stream, m_bias) || !::nano::read(stream, m_weights),
             "linear model: failed to read from stream!");

    critical(m_bias.size() != m_weights.rows(), "linear model: parameter mismatch!");

    return stream;
}

std::ostream& linear_model_t::write(std::ostream& stream) const
{
    learner_t::write(stream);

    critical(!::nano::write(stream, m_bias) || !::nano::write(stream, m_weights),
             "linear model: failed to write to stream!");

    return stream;
}

ml::result_t linear_model_t::fit(const dataset_t& dataset, const indices_t& samples, const loss_t& loss,
                                 const ml::params_t& fit_params)
{
    learner_t::fit_dataset(dataset);

    const auto batch          = parameter("linear::batch").value<tensor_size_t>();
    const auto regularization = parameter("linear::regularization").value<linear_regularization>();

    // tune hyper-parameters
    auto [param_names, param_spaces] = ::make_params(*this);

    const auto evaluator =
        [&](const auto& train_samples, const auto& valid_samples, const auto& params, const auto& extra)
    {
        const auto [l1reg, l2reg] = decode_params(params, regularization);

        auto result    = ::fit(*this, dataset, train_samples, loss, fit_params.solver(), l1reg, l2reg, extra);
        auto tr_values = ::nano::linear::evaluate(dataset, train_samples, loss, result.m_weights, result.m_bias, batch);
        auto vd_values = ::nano::linear::evaluate(dataset, valid_samples, loss, result.m_weights, result.m_bias, batch);

        return std::make_tuple(std::move(tr_values), std::move(vd_values), std::move(result));
    };

    auto fit_result = ml::tune("linear", samples, fit_params, std::move(param_names), param_spaces, evaluator);

    // refit with the optimum hyper-parameters (if any) on all given samples
    {
        const auto [l1reg, l2reg] = decode_params(fit_result.optimum().params(), regularization);

        auto result = ::fit(*this, dataset, samples, loss, fit_params.solver(), l1reg, l2reg);
        auto values = ::nano::linear::evaluate(dataset, samples, loss, result.m_weights, result.m_bias, batch);

        fit_result.evaluate(std::move(values));

        m_bias    = std::move(result.m_bias);
        m_weights = std::move(result.m_weights);
    }
    fit_params.log(fit_result, "linear");

    return fit_result;
}

void linear_model_t::do_predict(const dataset_t& dataset, indices_cmap_t samples, tensor4d_map_t outputs) const
{
    // TODO: no need to allocate the sample indices one more time
    auto iterator = flatten_iterator_t(dataset, samples);
    iterator.scaling(scaling_type::none);
    iterator.batch(parameter("linear::batch").value<tensor_size_t>());

    iterator.loop([&](tensor_range_t range, size_t, tensor2d_cmap_t inputs)
                  { ::nano::linear::predict(inputs, m_weights, m_bias, outputs.slice(range)); });
}
