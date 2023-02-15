#include <nano/core/sampling.h>
#include <nano/gboost/enums.h>
#include <nano/gboost/function.h>
#include <nano/gboost/model.h>
#include <nano/gboost/result.h>
#include <nano/gboost/util.h>
#include <nano/model/util.h>
#include <nano/tensor/stream.h>
#include <nano/wlearner/util.h>
#include <set>

using namespace nano;
using namespace nano::gboost;

static auto make_params(const configurable_t& configurable)
{
    const auto regularization = configurable.parameter("gboost::regularization").value<regularization_type>();
    const auto subsample      = configurable.parameter("gboost::subsample").value<subsample_type>();
    const auto shrinkage      = configurable.parameter("gboost::shrinkage").value<shrinkage_type>();

    auto param_names  = strings_t{};
    auto param_spaces = param_spaces_t{};

    if (regularization == regularization_type::variance)
    {
        param_names.emplace_back("vAreg");
        param_spaces.emplace_back(param_space_t::type::log10, 1e-9, 1e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4,
                                  3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e+0, 3e+0, 1e+1, 3e+1, 1e+2, 3e+2, 1e+3,
                                  3e+3, 1e+4, 3e+4, 1e+5, 3e+5, 1e+6, 3e+6, 1e+7, 3e+7, 1e+8, 1e+9);
    }
    if (subsample == subsample_type::on)
    {
        param_names.emplace_back("subsample");
        param_spaces.emplace_back(param_space_t::type::linear, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0);
    }
    if (shrinkage == shrinkage_type::on)
    {
        param_names.emplace_back("shrinkage");
        param_spaces.emplace_back(param_space_t::type::linear, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0);
    }

    return std::make_tuple(std::move(param_names), std::move(param_spaces));
}

static auto decode_params(const tensor1d_cmap_t& params, const regularization_type regularization,
                          const subsample_type subsample, const shrinkage_type shrinkage)
{
    scalar_t vAreg = 0.0, subsample_ratio = 1.0, shrinkage_ratio = 1.0;

    tensor_size_t index = 0;
    if (regularization == regularization_type::variance)
    {
        vAreg = params(index);
        index++;
    }
    if (subsample == subsample_type::on)
    {
        subsample_ratio = params(index);
        index++;
    }
    if (shrinkage == shrinkage_type::on)
    {
        shrinkage_ratio = params(index);
        index++;
    }

    return std::make_tuple(vAreg, subsample_ratio, shrinkage_ratio);
}

static auto selected(const tensor2d_t& values, const indices_t& samples)
{
    auto selected = tensor2d_t{2, samples.size()};
    values.tensor(0).indexed(samples, selected.tensor(0));
    values.tensor(1).indexed(samples, selected.tensor(1));
    return selected;
}

static auto make_cluster(const dataset_t& dataset, const indices_t& samples, const wlearner_t& wlearner,
                         const wscale_type wscale)
{
    switch (wscale)
    {
    case wscale_type::tboost: return wlearner.split(dataset, samples);

    default:
    {
        cluster_t cluster{dataset.samples(), 1};
        for (tensor_size_t i = 0; i < samples.size(); ++i)
        {
            cluster.assign(samples(i), 0);
        }
        return cluster;
    }
    }
}

static auto make_samples(indices_t samples, const scalar_t subsample_ratio, const bootstrap_type bootstrap, rng_t& rng)
{
    if (subsample_ratio < 1.0)
    {
        const auto ssize = static_cast<scalar_t>(samples.size());
        const auto count = static_cast<tensor_size_t>(std::lround(subsample_ratio * ssize));
        samples          = sample_without_replacement(samples, count, rng);
    }
    if (bootstrap == bootstrap_type::on)
    {
        const auto count = samples.size();
        samples          = sample_with_replacement(samples, count, rng);
    }
    return samples;
}

static auto fit(const configurable_t& configurable, const dataset_t& dataset, const indices_t& train_samples,
                const indices_t& valid_samples, const loss_t& loss, const solver_t& solver,
                const rwlearners_t& prototypes, const tensor1d_t& params)
{
    const auto seed           = configurable.parameter("gboost::seed").value<uint64_t>();
    const auto batch          = configurable.parameter("gboost::batch").value<tensor_size_t>();
    const auto epsilon        = configurable.parameter("gboost::epsilon").value<scalar_t>();
    const auto patience       = configurable.parameter("gboost::patience").value<size_t>();
    auto       max_rounds     = configurable.parameter("gboost::max_rounds").value<tensor_size_t>();
    const auto wscale         = configurable.parameter("gboost::wscale").value<wscale_type>();
    const auto bootstrap      = configurable.parameter("gboost::bootstrap").value<bootstrap_type>();
    const auto regularization = configurable.parameter("gboost::regularization").value<regularization_type>();
    const auto subsample      = configurable.parameter("gboost::subsample").value<subsample_type>();
    const auto shrinkage      = configurable.parameter("gboost::shrinkage").value<shrinkage_type>();

    const auto [vAreg, subsample_ratio, shrinkage_ratio] = decode_params(params, regularization, subsample, shrinkage);

    assert(vAreg >= 0.0);
    assert(subsample_ratio > 0.0 && subsample_ratio <= 1.0);
    assert(shrinkage_ratio > 0.0 && shrinkage_ratio <= 1.0);

    const auto samples = arange(0, dataset.samples());

    auto targets_iterator = targets_iterator_t{dataset, samples};
    targets_iterator.batch(batch);
    targets_iterator.scaling(scaling_type::none);

    auto train_targets_iterator = targets_iterator_t{dataset, train_samples};
    train_targets_iterator.batch(batch);
    train_targets_iterator.scaling(scaling_type::none);

    auto rng      = make_rng(seed);
    auto values   = tensor2d_t{2, samples.size()};
    auto outputs  = tensor4d_t{cat_dims(samples.size(), dataset.target_dims())};
    auto woutputs = tensor4d_t{cat_dims(samples.size(), dataset.target_dims())};

    const auto gfunction = grads_function_t{targets_iterator, loss, vAreg};
    const auto bfunction = bias_function_t{train_targets_iterator, loss, vAreg};

    // estimate bias
    const auto bstate = solver.minimize(bfunction, vector_t::Zero(bfunction.size()));

    outputs.reshape(samples.size(), -1).matrix().rowwise() = bstate.x.transpose();
    evaluate(targets_iterator, loss, outputs, values);

    auto result   = gboost::fit_result_t{max_rounds + 1};
    result.m_bias = map_tensor(bstate.x.data(), make_dims(bstate.x.size()));
    result.update(0, values, train_samples, valid_samples, bstate);

    // keep track of the optimum boosting round using the validation error
    auto optimum_round  = size_t{0};
    auto optimum_value  = std::numeric_limits<scalar_t>::max();
    auto optimum_values = values;

    if (done(values, train_samples, valid_samples, result.m_wlearners, epsilon, patience, optimum_round, optimum_value,
             optimum_values))
    {
        max_rounds = 0;
    }

    // construct the model one boosting round at a time
    for (tensor_size_t round = 0; round < max_rounds; ++round)
    {
        const auto& gradients   = gfunction.gradients(outputs);
        const auto  fit_samples = make_samples(train_samples, subsample_ratio, bootstrap, rng);

        // choose the weak learner that aligns the best with the current residuals
        auto best_score    = wlearner_t::no_fit_score();
        auto best_wlearner = rwlearner_t{};
        for (const auto& prototype : prototypes)
        {
            auto       wlearner = prototype->clone();
            const auto score    = wlearner->fit(dataset, fit_samples, gradients);
            if (score < best_score)
            {
                best_score    = score;
                best_wlearner = std::move(wlearner);
            }
        }
        if (!best_wlearner)
        {
            break;
        }

        // scale the chosen weak learner
        woutputs.zero();
        best_wlearner->predict(dataset, samples, woutputs.tensor());

        const auto cluster  = make_cluster(dataset, samples, *best_wlearner, wscale);
        const auto function = scale_function_t{train_targets_iterator, loss, vAreg, cluster, outputs, woutputs};

        auto gstate = solver.minimize(function, vector_t::Ones(function.size()));
        if (gstate.x.minCoeff() < std::numeric_limits<scalar_t>::epsilon())
        {
            // NB: scaling fails (optimization fails or convergence on training loss)
            result.update(round + 1, values, train_samples, valid_samples, gstate, std::move(best_wlearner));
            break;
        }

        gstate.x *= shrinkage_ratio;
        best_wlearner->scale(gstate.x);

        // update predictions
        woutputs.zero();
        best_wlearner->predict(dataset, samples, woutputs.tensor());

        outputs.vector() += woutputs.vector();
        evaluate(targets_iterator, loss, outputs, values);
        result.update(round + 1, values, train_samples, valid_samples, gstate, std::move(best_wlearner));

        // early stopping
        if (done(values, train_samples, valid_samples, result.m_wlearners, epsilon, patience, optimum_round,
                 optimum_value, optimum_values))
        {
            break;
        }
    }

    result.done(static_cast<tensor_size_t>(optimum_round));

    return std::make_tuple(std::move(result), selected(optimum_values, train_samples),
                           selected(optimum_values, valid_samples));
}

gboost_model_t::gboost_model_t()
    : model_t("gboost")
{
    register_parameter(parameter_t::make_scalar("gboost::epsilon", 1e-12, LE, 1e-6, LE, 1.0));

    register_parameter(parameter_t::make_integer("gboost::seed", 0, LE, 42, LE, 1024));
    register_parameter(parameter_t::make_integer("gboost::batch", 10, LE, 100, LE, 10000));
    register_parameter(parameter_t::make_integer("gboost::patience", 1, LE, 10, LE, 1'000));
    register_parameter(parameter_t::make_integer("gboost::max_rounds", 10, LE, 10'000, LE, 1'000'000));

    register_parameter(parameter_t::make_enum("gboost::wscale", wscale_type::gboost));
    register_parameter(parameter_t::make_enum("gboost::shrinkage", shrinkage_type::off));
    register_parameter(parameter_t::make_enum("gboost::subsample", subsample_type::off));
    register_parameter(parameter_t::make_enum("gboost::bootstrap", bootstrap_type::off));
    register_parameter(parameter_t::make_enum("gboost::regularization", regularization_type::none));
}

gboost_model_t::gboost_model_t(gboost_model_t&&) noexcept = default;

gboost_model_t& gboost_model_t::operator=(gboost_model_t&&) noexcept = default;

gboost_model_t::gboost_model_t(const gboost_model_t& other)
    : model_t(other)
    , m_protos(wlearner::clone(other.m_protos))
    , m_wlearners(wlearner::clone(other.m_wlearners))
{
}

gboost_model_t& gboost_model_t::operator=(const gboost_model_t& other)
{
    if (this != &other)
    {
        model_t::operator=(other);
        m_protos    = wlearner::clone(other.m_protos);
        m_wlearners = wlearner::clone(other.m_wlearners);
    }
    return *this;
}

gboost_model_t::~gboost_model_t() = default;

rmodel_t gboost_model_t::clone() const
{
    return std::make_unique<gboost_model_t>(*this);
}

void gboost_model_t::add(const wlearner_t& wlearner)
{
    m_protos.emplace_back(wlearner.clone());
}

void gboost_model_t::add(const string_t& wlearner_id)
{
    auto wlearner = wlearner_t::all().get(wlearner_id);

    critical(!wlearner, "gboost: invalid weak learner id (", wlearner_id, ")!");

    m_protos.emplace_back(std::move(wlearner));
}

std::istream& gboost_model_t::read(std::istream& stream)
{
    model_t::read(stream);

    critical(!::nano::read(stream, m_protos) || !::nano::read(stream, m_wlearners) || !::nano::read(stream, m_bias),
             "gboost: failed to read from stream!");

    return stream;
}

std::ostream& gboost_model_t::write(std::ostream& stream) const
{
    model_t::write(stream);

    critical(!::nano::write(stream, m_protos) || !::nano::write(stream, m_wlearners) || !::nano::write(stream, m_bias),
             "gboost: failed to write to stream!");

    return stream;
}

::nano::fit_result_t gboost_model_t::fit(const dataset_t& dataset, const indices_t& samples, const loss_t& loss,
                                         const solver_t& solver, const splitter_t& splitter, const tuner_t& tuner)
{
    critical(m_protos.empty(), "gboost: cannot fit without any weak learner!");

    // tune hyper-parameters (if any)
    auto [param_names, param_spaces] = ::make_params(*this);

    const auto evaluator = [&](const auto& train_samples, const auto& valid_samples, const auto& params, const auto&)
    {
        auto [gboost, train_errors_losses, valid_errors_losses] =
            ::fit(*this, dataset, train_samples, valid_samples, loss, solver, m_protos, params);

        return std::make_tuple(std::move(train_errors_losses), std::move(valid_errors_losses), std::move(gboost));
    };

    auto fit_result =
        ml::tune(samples, splitter, tuner, std::move(param_names), param_spaces, make_logger_lambda(), evaluator);

    // choose the optimum hyper-parameters and merge the boosters fitted for each fold
    {
        const auto& optimum_params = fit_result.optimum();
        const auto  folds          = optimum_params.folds();

        m_bias = make_full_tensor<scalar_t>(make_dims(::nano::size(dataset.target_dims())), 0.0);
        m_wlearners.clear();

        for (tensor_size_t fold = 0; fold < folds; ++fold)
        {
            const auto* const pgboost = std::any_cast<gboost::fit_result_t>(&optimum_params.extra(fold));
            assert(pgboost != nullptr);
            m_bias.vector() += pgboost->m_bias.vector();
            for (const auto& wlearner : pgboost->m_wlearners)
            {
                m_wlearners.emplace_back(wlearner->clone());
            }
        }

        ::nano::wlearner::merge(m_wlearners);

        const auto denom  = 1.0 / static_cast<scalar_t>(folds);
        const auto vdenom = make_vector<scalar_t>(denom);
        m_bias.vector() *= denom;
        for (const auto& wlearner : m_wlearners)
        {
            wlearner->scale(vdenom);
        }

        learner_t::fit(dataset);

        const auto all_samples = arange(0, dataset.samples());
        const auto outputs     = predict(dataset, all_samples);
        auto       values      = tensor2d_t{2, all_samples.size()};
        const auto batch       = parameter("gboost::batch").value<tensor_size_t>();

        auto targets_iterator = targets_iterator_t{dataset, all_samples};
        targets_iterator.batch(batch);
        targets_iterator.scaling(scaling_type::none);
        evaluate(targets_iterator, loss, outputs, values);

        fit_result.evaluate(::selected(values, samples));
    }
    this->log(fit_result);

    return fit_result;
}

tensor4d_t gboost_model_t::predict(const dataset_t& dataset, const indices_t& samples) const
{
    learner_t::critical_compatible(dataset);

    tensor4d_t outputs(cat_dims(samples.size(), dataset.target_dims()));
    outputs.reshape(samples.size(), -1).matrix().rowwise() = m_bias.vector().transpose();

    for (const auto& wlearner : m_wlearners)
    {
        wlearner->predict(dataset, samples, outputs.tensor());
    }

    return outputs;
}

indices_t gboost_model_t::features() const
{
    std::set<tensor_size_t> ufeatures;
    for (const auto& wlearner : m_wlearners)
    {
        for (const auto feature : wlearner->features())
        {
            ufeatures.emplace(feature);
        }
    }

    indices_t features(static_cast<tensor_size_t>(ufeatures.size()));
    std::copy(ufeatures.begin(), ufeatures.end(), features.begin());

    return features;
}
