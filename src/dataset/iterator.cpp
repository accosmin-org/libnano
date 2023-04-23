#include <nano/dataset/iterator.h>

using namespace nano;

namespace
{
auto features_per_thread(const indices_cmap_t& features, const size_t concurrency)
{
    return std::max(tensor_size_t{1}, idiv(features.size(), concurrency));
}
} // namespace

base_dataset_iterator_t::base_dataset_iterator_t(const dataset_t& dataset)
    : m_dataset(dataset)
{
}

size_t base_dataset_iterator_t::concurrency() const
{
    return m_dataset.concurrency();
}

targets_iterator_t::targets_iterator_t(const dataset_t& dataset, indices_cmap_t samples)
    : base_dataset_iterator_t(dataset)
    , m_samples(samples)
    , m_targets_stats(dataset.target().valid() ? scalar_stats_t::make_targets_stats(dataset, samples)
                                               : scalar_stats_t{})
    , m_targets_buffers(concurrency())
{
}

tensor4d_cmap_t targets_iterator_t::targets(tensor4d_map_t data) const
{
    m_targets_stats.scale(m_scaling, data);
    return data;
}

tensor4d_cmap_t targets_iterator_t::targets(size_t tnum, const tensor_range_t& range) const
{
    if (m_targets.size<0>() == m_samples.size())
    {
        return m_targets.slice(range);
    }
    else
    {
        assert(tnum < m_targets_buffers.size());
        return targets(dataset().targets(m_samples.slice(range), m_targets_buffers[tnum]));
    }
}

bool targets_iterator_t::cache_targets(tensor_size_t max_bytes)
{
    auto cached = false;
    if (const auto tdims = dataset().target_dims();
        static_cast<tensor_size_t>(sizeof(scalar_t)) * m_samples.size() * nano::size(tdims) <= max_bytes)
    {
        try
        {
            m_targets.resize(cat_dims(m_samples.size(), tdims));
            map(m_samples.size(), batch(),
                [&](tensor_size_t begin, tensor_size_t end, size_t tnum)
                {
                    assert(tnum < m_targets_buffers.size());
                    const auto range       = make_range(begin, end);
                    const auto samples     = m_samples.slice(range);
                    m_targets.slice(range) = targets(dataset().targets(samples, m_targets_buffers[tnum]));
                });
            cached = true;
        }
        catch (...)
        {
        } // LCOV_EXCL_LINE
    }

    return cached;
}

void targets_iterator_t::batch(tensor_size_t batch)
{
    m_batch = batch;
}

void targets_iterator_t::scaling(scaling_type scaling)
{
    m_scaling = scaling;
}

flatten_iterator_t::flatten_iterator_t(const dataset_t& dataset, indices_cmap_t samples)
    : targets_iterator_t(dataset, samples)
    , m_flatten_stats(scalar_stats_t::make_flatten_stats(dataset, samples))
    , m_flatten_buffers(concurrency())
{
}

tensor2d_cmap_t flatten_iterator_t::flatten(tensor2d_map_t data) const
{
    m_flatten_stats.scale(scaling(), data);
    return data;
}

tensor2d_cmap_t flatten_iterator_t::flatten(size_t tnum, const tensor_range_t& range) const
{
    const auto& samples = this->samples();
    const auto& dataset = this->dataset();

    if (m_flatten.size<0>() == samples.size())
    {
        return m_flatten.slice(range);
    }
    else
    {
        assert(tnum < m_flatten_buffers.size());
        return flatten(dataset.flatten(samples.slice(range), m_flatten_buffers[tnum]));
    }
}

bool flatten_iterator_t::cache_flatten(tensor_size_t max_bytes)
{
    const auto& samples = this->samples();
    const auto& dataset = this->dataset();

    auto cached = false;
    if (const auto isize = dataset.columns();
        static_cast<tensor_size_t>(sizeof(scalar_t)) * samples.size() * isize <= max_bytes)
    {
        try
        {
            m_flatten.resize(samples.size(), isize);
            map(samples.size(), batch(),
                [&](tensor_size_t begin, tensor_size_t end, size_t tnum)
                {
                    assert(tnum < m_flatten_buffers.size());
                    const auto range       = make_range(begin, end);
                    m_flatten.slice(range) = flatten(dataset.flatten(samples.slice(range), m_flatten_buffers[tnum]));
                });
            cached = true;
        }
        catch (...)
        {
        } // LCOV_EXCL_LINE
    }

    return cached;
}

void flatten_iterator_t::loop(const flatten_targets_callback_t& callback) const
{
    map(samples().size(), batch(),
        [&](tensor_size_t begin, tensor_size_t end, size_t tnum)
        {
            const auto range = make_range(begin, end);

            callback(range, tnum, flatten(tnum, range), targets(tnum, range));
        });
}

void flatten_iterator_t::loop(const flatten_callback_t& callback) const
{
    map(samples().size(), batch(),
        [&](tensor_size_t begin, tensor_size_t end, size_t tnum)
        {
            const auto range = make_range(begin, end);

            callback(range, tnum, flatten(tnum, range));
        });
}

void targets_iterator_t::loop(const targets_callback_t& callback) const
{
    map(samples().size(), batch(),
        [&](tensor_size_t begin, tensor_size_t end, size_t tnum)
        {
            const auto range = make_range(begin, end);

            callback(range, tnum, targets(tnum, range));
        });
}

select_iterator_t::select_iterator_t(const dataset_t& dataset)
    : base_dataset_iterator_t(dataset)
    , m_buffers(concurrency())
    , m_sclass_features(make_sclass_features(dataset))
    , m_mclass_features(make_mclass_features(dataset))
    , m_scalar_features(make_scalar_features(dataset))
    , m_struct_features(make_struct_features(dataset))
{
}

void select_iterator_t::loop(indices_cmap_t samples, const sclass_callback_t& callback) const
{
    return loop(samples, m_sclass_features, callback);
}

void select_iterator_t::loop(indices_cmap_t samples, const mclass_callback_t& callback) const
{
    return loop(samples, m_mclass_features, callback);
}

void select_iterator_t::loop(indices_cmap_t samples, const scalar_callback_t& callback) const
{
    return loop(samples, m_scalar_features, callback);
}

void select_iterator_t::loop(indices_cmap_t samples, const struct_callback_t& callback) const
{
    return loop(samples, m_struct_features, callback);
}

void select_iterator_t::loop(indices_cmap_t samples, tensor_size_t ifeature, const sclass_callback_t& callback) const
{
    const auto tnum = size_t{0U};
    callback(ifeature, tnum, dataset().select(samples, ifeature, m_buffers[tnum].m_sclass));
}

void select_iterator_t::loop(indices_cmap_t samples, tensor_size_t ifeature, const mclass_callback_t& callback) const
{
    const auto tnum = size_t{0U};
    callback(ifeature, tnum, dataset().select(samples, ifeature, m_buffers[tnum].m_mclass));
}

void select_iterator_t::loop(indices_cmap_t samples, tensor_size_t ifeature, const scalar_callback_t& callback) const
{
    const auto tnum = size_t{0U};
    callback(ifeature, tnum, dataset().select(samples, ifeature, m_buffers[tnum].m_scalar));
}

void select_iterator_t::loop(indices_cmap_t samples, tensor_size_t ifeature, const struct_callback_t& callback) const
{
    const auto tnum = size_t{0U};
    callback(ifeature, tnum, dataset().select(samples, ifeature, m_buffers[tnum].m_struct));
}

void select_iterator_t::loop(indices_cmap_t samples, indices_cmap_t features, const sclass_callback_t& callback) const
{
    map(features.size(), features_per_thread(features, concurrency()),
        [&](const tensor_size_t begin, const tensor_size_t end, const size_t tnum)
        {
            assert(tnum < m_buffers.size());
            for (tensor_size_t index = begin; index < end; ++index)
            {
                const auto ifeature = features(index);
                callback(ifeature, tnum, dataset().select(samples, ifeature, m_buffers[tnum].m_sclass));
            }
        });
}

void select_iterator_t::loop(indices_cmap_t samples, indices_cmap_t features, const mclass_callback_t& callback) const
{
    map(features.size(), features_per_thread(features, concurrency()),
        [&](const tensor_size_t begin, const tensor_size_t end, const size_t tnum)
        {
            assert(tnum < m_buffers.size());
            for (tensor_size_t index = begin; index < end; ++index)
            {
                const auto ifeature = features(index);
                callback(ifeature, tnum, dataset().select(samples, ifeature, m_buffers[tnum].m_mclass));
            }
        });
}

void select_iterator_t::loop(indices_cmap_t samples, indices_cmap_t features, const scalar_callback_t& callback) const
{
    map(features.size(), features_per_thread(features, concurrency()),
        [&](const tensor_size_t begin, const tensor_size_t end, const size_t tnum)
        {
            assert(tnum < m_buffers.size());
            for (tensor_size_t index = begin; index < end; ++index)
            {
                const auto ifeature = features(index);
                callback(ifeature, tnum, dataset().select(samples, ifeature, m_buffers[tnum].m_scalar));
            }
        });
}

void select_iterator_t::loop(indices_cmap_t samples, indices_cmap_t features, const struct_callback_t& callback) const
{
    map(features.size(), features_per_thread(features, concurrency()),
        [&](const tensor_size_t begin, const tensor_size_t end, const size_t tnum)
        {
            assert(tnum < m_buffers.size());
            for (tensor_size_t index = begin; index < end; ++index)
            {
                const auto ifeature = features(index);
                callback(ifeature, tnum, dataset().select(samples, ifeature, m_buffers[tnum].m_struct));
            }
        });
}
