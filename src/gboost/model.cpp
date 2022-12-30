#include <iomanip>
#include <nano/gboost/function.h>
#include <nano/gboost/model.h>
#include <nano/tensor/stream.h>

using namespace nano;

gboost_model_t::gboost_model_t()
{
    model_t::register_param(sparam1_t{"gboost::vAreg", 0, LE, 0, LE, 1e+10});
    model_t::register_param(iparam1_t{"gboost::batch", 1, LE, 32, LE, 4096});
    model_t::register_param(iparam1_t{"gboost::rounds", 1, LE, 1000, LE, 10000});
    model_t::register_param(sparam1_t{"gboost::epsilon", 0.0, LT, 1e-4, LT, 1e-1});
    model_t::register_param(sparam1_t{"gboost::shrinkage", 0.0, LE, 1.0, LE, 1.0});
    model_t::register_param(sparam1_t{"gboost::subsample", 0.0, LE, 1.0, LE, 1.0});
    model_t::register_param(eparam1_t{"gboost::wscale", ::nano::wscale::tboost});
}

rmodel_t gboost_model_t::clone() const
{
    return std::make_unique<gboost_model_t>(*this);
}

void gboost_model_t::add(const string_t& id)
{
    add(id, wlearner_t::all().get(id));
}

void gboost_model_t::add(string_t id, rwlearner_t&& prototype)
{
    critical(prototype == nullptr, "gboost model: invalid prototype weak learner!");

    m_protos.emplace_back(std::move(id), std::move(prototype));
}

scalar_t gboost_model_t::fit(const loss_t& loss, const dataset_t& dataset, const indices_t& samples,
                             const solver_t& solver)
{
    log_info() << string_t(8, '-') << ::nano::align(" gboost model ", 112U, alignment::left, '-') << string_t(8, '-');
    for (const auto& param : params())
    {
        log_info() << "gboost model: fit using " << std::fixed << std::setprecision(8) << param;
    }
    for (const auto& proto : m_protos)
    {
        log_info() << "gboost model: fit using weak learner (" << proto.id() << ")";
    }
    log_info() << string_t(128, '-');

    critical(m_protos.empty(), "gboost model: no prototype weak learners to use!");

    m_iwlearners.clear();

    const auto tdims = dataset.tdims();

    tensor4d_t outputs(cat_dims(samples.size(), tdims));
    tensor4d_t woutputs(cat_dims(samples.size(), tdims));
    tensor4d_t fit_vgrads(cat_dims(dataset.samples(), tdims)); // NB: gradients for ALL samples, to index with samples!
    fit_vgrads.full(std::numeric_limits<scalar_t>::quiet_NaN());

    // estimate bias
    auto bias_function = gboost_bias_function_t{loss, dataset, samples};
    bias_function.vAreg(vAreg());
    bias_function.batch(batch());

    const auto state = solver.minimize(bias_function, vector_t::Zero(bias_function.size()));
    m_bias.resize(state.x.size());
    m_bias.vector() = state.x;

    // update predictions
    outputs.reshape(samples.size(), -1).matrix().rowwise() = state.x.transpose();
    auto errors                                            = evaluate(dataset, samples, loss, outputs);
    if (done(0, errors, state, indices_t{}))
    {
        return errors.mean();
    }

    auto grads_function = gboost_grads_function_t{loss, dataset, samples};
    grads_function.vAreg(vAreg());
    grads_function.batch(batch());

    // construct the model one boosting round at a time
    for (tensor_size_t round = 0; round < rounds(); ++round)
    {
        const auto& vgrads = grads_function.gradients(outputs);

        const auto fit_indices_in_samples = make_indices(samples);
        const auto fit_indices            = samples.indexed<tensor_size_t>(fit_indices_in_samples);

        for (tensor_size_t i = 0; i < fit_indices.size(); ++i)
        {
            fit_vgrads.vector(fit_indices(i)) = vgrads.vector(fit_indices_in_samples(i));
        }

        // choose the weak learner that aligns the best with the current residuals
        auto best_id       = std::string{};
        auto best_score    = wlearner_t::no_fit_score();
        auto best_wlearner = rwlearner_t{};
        for (const auto& prototype : m_protos)
        {
            auto wlearner = prototype.get().clone();
            assert(wlearner);

            const auto score = wlearner->fit(dataset, fit_indices, fit_vgrads);
            if (score < best_score)
            {
                best_id       = prototype.id();
                best_score    = score;
                best_wlearner = std::move(wlearner);
            }
        }

        if (!best_wlearner)
        {
            log_warning() << "gboost model: cannot fit any new weak learner, stopping.";
            break;
        }

        // scale the chosen weak learner
        const auto cluster = make_cluster(dataset, samples, *best_wlearner);

        woutputs.zero();
        best_wlearner->predict(dataset, samples, woutputs.tensor());

        auto function = gboost_scale_function_t(loss, dataset, samples, cluster, outputs, woutputs);
        function.vAreg(vAreg());
        function.batch(batch());

        auto state = solver.minimize(function, vector_t::Zero(function.size()));
        if (state.x.minCoeff() < 0.0)
        {
            log_warning() << "gboost model: invalid scale factor(s): [" << state.x.transpose() << "], stopping.";
            break;
        }

        state.x *= shrinkage();
        best_wlearner->scale(state.x);
        scale(cluster, samples, state.x, woutputs);

        // update predictions
        outputs.vector() += woutputs.vector();
        errors = evaluate(dataset, samples, loss, outputs);
        if (done(round + 1, errors, state, best_wlearner->features()))
        {
            break;
        }

        // update model
        m_iwlearners.emplace_back(std::move(best_id), std::move(best_wlearner));
    }

    return errors.mean();
}

bool gboost_model_t::done(tensor_size_t round, const tensor1d_t& errors, const solver_state_t& state,
                          const indices_t& features) const
{
    const auto cwidth = static_cast<int>(std::log10(rounds())) + 1;

    const auto value = state.f;
    const auto error = errors.mean();

    log_info() << std::setprecision(8) << std::fixed << std::setw(cwidth) << std::setfill('0') << round << "/"
               << std::setw(cwidth) << std::setfill('0') << rounds() << ":tr=" << value << "|" << error
               << std::setprecision(8) << std::fixed << ",vAreg=" << vAreg() << "," << state << ",feat=["
               << features.array() << "].";

    if (!std::isfinite(value) || !std::isfinite(error) || state.m_status != solver_state_t::status::converged)
    {
        log_warning() << "gboost model: training failed (check inputs and parameters), stopping.";
        return true;
    }

    if (error < epsilon())
    {
        log_warning() << "gboost model: training converged, stopping.";
        return true;
    }

    return false;
}

indices_t gboost_model_t::make_indices(const indices_t& samples) const
{
    const auto count = static_cast<tensor_size_t>(llround(subsample() * static_cast<scalar_t>(samples.size())));
    if (count >= samples.size())
    {
        return arange(0, samples.size());
    }
    else
    {
        return ::nano::sample_without_replacement(samples.size(), count);
    }
}

cluster_t gboost_model_t::make_cluster(const dataset_t& dataset, const indices_t& samples,
                                       const wlearner_t& wlearner) const
{
    switch (wscale())
    {
    case wscale::tboost: return wlearner.split(dataset, samples);

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

void gboost_model_t::scale(const cluster_t& cluster, const indices_t& samples, const vector_t& scales,
                           tensor4d_t& woutputs) const
{
    assert(samples.size() == woutputs.size<0>());

    loopi(samples.size(),
          [&](tensor_size_t i, size_t)
          {
              const auto group = cluster.group(samples(i));
              if (group >= 0)
              {
                  assert(group < scales.size());
                  woutputs.array(i) *= scales(group);
              }
          });
}

tensor1d_t gboost_model_t::evaluate(const dataset_t& dataset, const indices_t& samples, const loss_t& loss,
                                    const tensor4d_t& outputs) const
{
    assert(outputs.dims() == cat_dims(samples.size(), dataset.tdims()));

    tensor1d_t errors(samples.size());
    loopr(samples.size(), batch(),
          [&](tensor_size_t begin, tensor_size_t end, size_t)
          {
              const auto range   = make_range(begin, end);
              const auto targets = dataset.targets(samples.slice(range));
              loss.error(targets, outputs.slice(range), errors.slice(range));
          });

    return errors;
}

tensor4d_t gboost_model_t::predict(const dataset_t& dataset, const indices_t& samples) const
{
    critical(m_bias.size() != ::nano::size(dataset.tdims()) && m_iwlearners.empty(),
             "gboost model: cannot predict without a trained model!");

    tensor4d_t outputs(cat_dims(samples.size(), dataset.tdims()));
    outputs.reshape(samples.size(), -1).matrix().rowwise() = m_bias.vector().transpose();

    loopr(samples.size(), batch(),
          [&](tensor_size_t begin, tensor_size_t end, size_t)
          {
              const auto range    = make_range(begin, end);
              const auto wsamples = samples.slice(range);
              const auto woutputs = outputs.slice(range);
              for (const auto& iwlearner : m_iwlearners)
              {
                  iwlearner.get().predict(dataset, wsamples, woutputs);
              }
          });

    return outputs;
}

void gboost_model_t::read(std::istream& stream)
{
    model_t::read(stream);

    ::nano::read(stream, m_protos);
    ::nano::read(stream, m_iwlearners);

    critical(!::nano::read(stream, m_bias), "gboost model: failed to read from stream!");
}

void gboost_model_t::write(std::ostream& stream) const
{
    model_t::write(stream);

    ::nano::write(stream, m_protos);
    ::nano::write(stream, m_iwlearners);

    critical(!::nano::write(stream, m_bias), "gboost model: failed to write to stream!");
}

features_t gboost_model_t::features() const
{
    std::map<tensor_size_t, tensor_size_t> fcounts;
    for (const auto& iwlearner : m_iwlearners)
    {
        for (const auto feature : iwlearner.get().features())
        {
            fcounts[feature]++;
        }
    }

    feature_infos_t infos;
    infos.reserve(fcounts.size());
    for (const auto& [feature, count] : fcounts)
    {
        infos.emplace_back(feature, count, 0.0);
    }
    feature_info_t::sort_by_importance(infos);
    return infos;
}
