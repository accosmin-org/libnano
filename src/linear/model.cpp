#include <iomanip>
#include <nano/core/parallel.h>
#include <nano/linear/function.h>
#include <nano/linear/model.h>
#include <nano/linear/regularization.h>
#include <nano/linear/util.h>
#include <nano/model/kfold.h>
#include <nano/model/tuner.h>
#include <nano/tensor/stream.h>

using namespace nano;
using namespace nano::linear;

static auto make_param_space()
{
    return param_space_t{param_space_t::type::log10,
                         make_tensor<scalar_t>(make_dims(31), 1e-9, 1e-8, 1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4,
                                               1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e+0, 3e+0, 1e+1, 3e+1, 1e+2, 3e+2,
                                               1e+3, 3e+3, 1e+4, 3e+4, 1e+5, 3e+5, 1e+6, 1e+7, 1e+8, 1e+9)};
}

static auto decode_params(const tensor1d_t& params, regularization_type regularization)
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

static auto evaluate(const estimator_t& estimator, const dataset_generator_t& dataset, const indices_t& samples,
                     const loss_t& loss, const tensor2d_t& weights, const tensor1d_t& bias, size_t threads)
{
    auto iterator = flatten_iterator_t{dataset, samples, threads};
    iterator.scaling(scaling_type::none);
    iterator.batch(estimator.parameter("model::linear::batch").value<tensor_size_t>());

    tensor1d_t errors(samples.size());
    tensor1d_t values(samples.size());
    tensor4d_t outputs(cat_dims(samples.size(), dataset.target_dims()));
    iterator.loop(
        [&](tensor_range_t range, size_t, tensor2d_cmap_t inputs, tensor4d_cmap_t targets)
        {
            ::nano::linear::predict(inputs, weights, bias, outputs.slice(range));
            loss.error(targets, outputs.slice(range), errors.slice(range));
            loss.value(targets, outputs.slice(range), values.slice(range));
        });

    return std::make_tuple(errors.mean(), values.mean());
}

static auto fit(const estimator_t& estimator, const dataset_generator_t& dataset, const indices_t& samples,
                const loss_t& loss, const solver_t& solver, scalar_t l1reg, scalar_t l2reg, scalar_t vAreg,
                size_t threads)
{
    auto iterator = flatten_iterator_t{dataset, samples, threads};
    iterator.batch(estimator.parameter("model::linear::batch").value<tensor_size_t>());
    iterator.scaling(estimator.parameter("model::linear::scaling").value<scaling_type>());
    iterator.cache_flatten(std::numeric_limits<tensor_size_t>::max());
    iterator.cache_targets(std::numeric_limits<tensor_size_t>::max());

    // TODO: fit from the optimum found at the closest parameter values!!!
    const auto function = ::nano::linear::function_t{iterator, loss, l1reg, l2reg, vAreg};
    const auto state    = solver.minimize(function, vector_t::Zero(function.size()));

    tensor1d_t bias    = function.bias(state.x);
    tensor2d_t weights = function.weights(state.x);
    ::upscale(iterator.flatten_stats(), iterator.scaling(), iterator.targets_stats(), iterator.scaling(), weights,
              bias);

    return std::make_tuple(std::move(weights), std::move(bias));
}

linear_model_t::linear_model_t()
{
    register_parameter(parameter_t::make_integer("model::linear::batch", 10, LE, 100, LE, 10000));
    register_parameter(parameter_t::make_enum("model::linear::scaling", scaling_type::standard));
    register_parameter(parameter_t::make_enum("model::linear::regularization", regularization_type::lasso));
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

fit_result_t linear_model_t::do_fit(const dataset_generator_t& dataset, const indices_t& samples, const loss_t& loss,
                                    const solver_t& solver)
{
    const auto folds          = parameter("model::folds").value<tensor_size_t>();
    const auto random_seed    = parameter("model::random_seed").value<uint64_t>();
    const auto regularization = parameter("model::linear::regularization").value<regularization_type>();

    parallel::pool_t pool{static_cast<size_t>(folds)};

    const auto cv = kfold_t{samples, folds, random_seed};
    const auto th = (parallel::pool_t::max_size() + pool.size() - 1U) / pool.size();

    fit_result_t result;

    // TODO: allocate the iterators once: (train, valid)xfolds + refit

    const auto tuner_callback = [&](const tensor1d_t& params)
    {
        const auto [l1reg, l2reg, vAreg] = decode_params(params, regularization);

        fit_result_t::cv_result_t cv_result{params, folds};

        const auto fold_callback = [&, l1reg = l1reg, l2reg = l2reg, vAreg = vAreg](auto fold, size_t)
        {
            const auto [train_samples, valid_samples] = cv.split(fold);

            const auto [weights, bias] = ::fit(*this, dataset, train_samples, loss, solver, l1reg, l2reg, vAreg, th);
            const auto [train_error, train_value] = ::evaluate(*this, dataset, train_samples, loss, weights, bias, th);
            const auto [valid_error, valid_value] = ::evaluate(*this, dataset, valid_samples, loss, weights, bias, th);

            cv_result.m_train_errors(fold) = train_error;
            cv_result.m_train_values(fold) = train_value;
            cv_result.m_valid_errors(fold) = valid_error;
            cv_result.m_valid_values(fold) = valid_value;
        };
        pool.loopi(folds, fold_callback);

        const auto goodness = std::log(cv_result.m_train_errors.mean() + std::numeric_limits<scalar_t>::epsilon());

        result.m_cv_results.emplace_back(std::move(cv_result));

        this->log(result, "linear");

        return goodness;
    };

    const auto refit = [&](const tensor1d_t& params)
    {
        const auto [l1reg, l2reg, vAreg] = decode_params(params, regularization);

        const auto [weights, bias]            = ::fit(*this, dataset, samples, loss, solver, l1reg, l2reg, vAreg, th);
        const auto [refit_error, refit_value] = ::evaluate(*this, dataset, samples, loss, weights, bias, th);

        result.m_refit_params = params;
        result.m_refit_error  = refit_error;
        result.m_refit_value  = refit_value;

        this->log(result, "linear");

        m_weights = weights;
        m_bias    = bias;
    };

    switch (parameter("model::linear::regularization").value<regularization_type>())
    {
    case regularization_type::none: refit(tensor1d_t{}); break;

    case regularization_type::lasso:
        result.m_param_names = {"l1reg"};
        {
            const auto tuner = tuner_t{param_spaces_t{make_param_space()}, tuner_callback};
            const auto steps =
                tuner.optimize(make_tensor<scalar_t>(make_dims(6, 1), 1e-4, 1e-2, 1e+0, 1e+2, 1e+4, 1e+6));

            refit(steps.rbegin()->m_opt_param);
        }
        break;

    case regularization_type::ridge:
        result.m_param_names = {"l2reg"};
        {
            const auto tuner = tuner_t{param_spaces_t{make_param_space()}, tuner_callback};
            const auto steps =
                tuner.optimize(make_tensor<scalar_t>(make_dims(6, 1), 1e-4, 1e-2, 1e+0, 1e+2, 1e+4, 1e+6));

            refit(steps.rbegin()->m_opt_param);
        }
        break;

    case regularization_type::variance:
        result.m_param_names = {"vAreg"};
        {
            const auto tuner = tuner_t{param_spaces_t{make_param_space()}, tuner_callback};
            const auto steps =
                tuner.optimize(make_tensor<scalar_t>(make_dims(6, 1), 1e-4, 1e-2, 1e+0, 1e+2, 1e+4, 1e+6));

            refit(steps.rbegin()->m_opt_param);
        }
        break;

    case regularization_type::elasticnet:
        result.m_param_names = {"l1reg", "l2reg"};
        {
            const auto tuner = tuner_t{
                param_spaces_t{make_param_space(), make_param_space()},
                tuner_callback
            };
            const auto steps = tuner.optimize(make_tensor<scalar_t>(
                make_dims(15, 2), 1e-2, 1e-2, 1e-2, 1e+0, 1e-2, 1e+2, 1e-2, 1e+4, 1e-2, 1e+6, 1e+1, 1e-2, 1e+1, 1e+0,
                1e+1, 1e+2, 1e+1, 1e+4, 1e+1, 1e+6, 1e+4, 1e-2, 1e+4, 1e+0, 1e+4, 1e+2, 1e+4, 1e+4, 1e+4, 1e+6));

            refit(steps.rbegin()->m_opt_param);
        }
        break;
    }

    return result;
}

tensor4d_t linear_model_t::do_predict(const dataset_generator_t& dataset, const indices_t& samples) const
{
    // TODO: no need to allocate the sample indices one more time
    // TODO: determine at runtime if worth parallelizing
    auto iterator = flatten_iterator_t(dataset, samples, 1U);
    iterator.scaling(scaling_type::none);
    iterator.batch(parameter("model::linear::batch").value<tensor_size_t>());

    tensor4d_t outputs(cat_dims(samples.size(), dataset.target_dims()));
    iterator.loop([&](tensor_range_t range, size_t, tensor2d_cmap_t inputs)
                  { ::nano::linear::predict(inputs, m_weights, m_bias, outputs.slice(range)); });

    return outputs;
}
