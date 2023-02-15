#include <nano/dataset.h>
#include <nano/gboost/util.h>

using namespace nano;

void gboost::evaluate(const targets_iterator_t& iterator, const loss_t& loss, const tensor4d_t& outputs,
                      tensor2d_t& values)
{
    assert(2 == values.size<0>());
    assert(outputs.size<0>() == values.size<1>());
    assert(outputs.size<0>() == iterator.samples().size());
    assert(outputs.dims() == cat_dims(values.size<1>(), iterator.dataset().target_dims()));

    iterator.loop(
        [&](const auto range, const auto, const tensor4d_cmap_t targets)
        {
            loss.error(targets, outputs.slice(range), values.tensor(0).slice(range));
            loss.value(targets, outputs.slice(range), values.tensor(1).slice(range));
        });
}

scalar_t gboost::mean_loss(const tensor2d_t& errors_losses, const indices_t& samples)
{
    const auto opsum = [&](const scalar_t sum, const tensor_size_t sample) { return sum + errors_losses(1, sample); };
    const auto denom = static_cast<scalar_t>(std::max(samples.size(), tensor_size_t{1}));
    return std::accumulate(begin(samples), end(samples), 0.0, opsum) / denom;
}

scalar_t gboost::mean_error(const tensor2d_t& errors_losses, const indices_t& samples)
{
    const auto opsum = [&](const scalar_t sum, const tensor_size_t sample) { return sum + errors_losses(0, sample); };
    const auto denom = static_cast<scalar_t>(std::max(samples.size(), tensor_size_t{1}));
    return std::accumulate(begin(samples), end(samples), 0.0, opsum) / denom;
}

bool gboost::done(const tensor2d_t& errors_values, const indices_t& train_samples, const indices_t& valid_samples,
                  const rwlearners_t& wlearners, const scalar_t epsilon, const size_t patience, size_t& optimum_round,
                  scalar_t& optimum_value, tensor2d_t& optimum_values)
{
    const auto train_value = mean_error(errors_values, train_samples);
    const auto valid_value = mean_error(errors_values, valid_samples);

    // training error is too small, stop
    if (train_value < epsilon)
    {
        optimum_value  = valid_value;
        optimum_round  = wlearners.size();
        optimum_values = errors_values;
        return true;
    }

    // significant improvement, continue
    // OR refitting step when no validation samples are given, so don't stop until the optimum number of boosting rounds
    else if (train_value < epsilon || valid_value < optimum_value - epsilon || valid_samples.size() == 0)
    {
        optimum_value  = valid_value;
        optimum_round  = wlearners.size();
        optimum_values = errors_values;
        return false;
    }

    // no significant improvement, but can wait a bit more
    else if (wlearners.size() < optimum_round + patience)
    {
        return false;
    }

    // no significant improvement in awhile, stop
    else
    {
        return true;
    }
}
