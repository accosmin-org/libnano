#include <nano/dataset.h>
#include <nano/gboost/util.h>
#include <numeric>

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

scalar_t gboost::tune_shrinkage(const targets_iterator_t& iterator, const loss_t& loss, const tensor4d_t& outputs,
                                const tensor4d_t& woutputs)
{
    assert(outputs.dims() == woutputs.dims());
    assert(outputs.dims() == cat_dims(iterator.dataset().samples(), iterator.dataset().target_dims()));

    auto values            = tensor1d_t{iterator.samples().size()};
    auto selected_outputs  = outputs.indexed(iterator.samples());
    auto selected_woutputs = woutputs.indexed(iterator.samples());

    auto best_shrinkage = 0.0;
    auto best_value     = std::numeric_limits<scalar_t>::max();

    for (const auto shrinkage : {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0})
    {
        selected_outputs.array() += shrinkage * selected_woutputs.array();

        iterator.loop([&](const auto range, const auto, const tensor4d_cmap_t targets)
                      { loss.value(targets, selected_outputs.slice(range), values.slice(range)); });

        const auto value = values.mean();
        if (value < best_value)
        {
            best_value     = value;
            best_shrinkage = shrinkage;
        }

        selected_outputs.array() -= shrinkage * selected_woutputs.array();
    }

    return best_shrinkage;
}

scalar_t gboost::mean_loss(const tensor2d_t& errors_losses, const indices_t& samples)
{
    const auto opsum = [&](const scalar_t sum, const tensor_size_t sample) { return sum + errors_losses(1, sample); };
    const auto denom = static_cast<scalar_t>(std::max(samples.size(), tensor_size_t{1}));
    return std::accumulate(std::begin(samples), std::end(samples), 0.0, opsum) / denom;
}

scalar_t gboost::mean_error(const tensor2d_t& errors_losses, const indices_t& samples)
{
    const auto opsum = [&](const scalar_t sum, const tensor_size_t sample) { return sum + errors_losses(0, sample); };
    const auto denom = static_cast<scalar_t>(std::max(samples.size(), tensor_size_t{1}));
    return std::accumulate(std::begin(samples), std::end(samples), 0.0, opsum) / denom;
}
