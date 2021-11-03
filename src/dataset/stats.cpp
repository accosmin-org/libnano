#include <nano/dataset/stats.h>

using namespace nano;

template <typename tvalues>
static void nan2zero(tvalues& values)
{
    // replace missing scalar values (represented as NaN) with zeros,
    // to train and evaluate dense ML models (e.g. linear models).
    for (tensor_size_t i = 0, size = values.size(); i < size; ++ i)
    {
        auto& value = values(i);
        if (!std::isfinite(value))
        {
            value = 0.0;
        }
    }
}

scalar_stats_t::scalar_stats_t() = default;

scalar_stats_t::scalar_stats_t(tensor_size_t dims) :
    m_samples(make_full_tensor<tensor_size_t>(make_dims(dims), 0)),
    m_min(make_full_tensor<scalar_t>(make_dims(dims), std::numeric_limits<scalar_t>::max())),
    m_max(make_full_tensor<scalar_t>(make_dims(dims), std::numeric_limits<scalar_t>::lowest())),
    m_mean(make_full_tensor<scalar_t>(make_dims(dims), 0.0)),
    m_stdev(make_full_tensor<scalar_t>(make_dims(dims), 0.0)),
    m_div_range(make_full_tensor<scalar_t>(make_dims(dims), 1.0)),
    m_mul_range(make_full_tensor<scalar_t>(make_dims(dims), 1.0)),
    m_div_stdev(make_full_tensor<scalar_t>(make_dims(dims), 1.0)),
    m_mul_stdev(make_full_tensor<scalar_t>(make_dims(dims), 1.0))
{
}

scalar_stats_t& scalar_stats_t::operator+=(const scalar_stats_t& other)
{
    m_samples.array() += other.m_samples.array();
    m_mean.array() += other.m_mean.array();
    m_stdev.array() += other.m_stdev.array();
    m_min.array() = m_min.array().min(other.m_min.array());
    m_max.array() = m_max.array().max(other.m_max.array());
    return *this;
}

scalar_stats_t& scalar_stats_t::done(const tensor_mem_t<uint8_t, 1>& enable_scaling)
{
    const auto epsilon = epsilon2<scalar_t>();

    for (tensor_size_t i = 0, size = m_samples.size(); i < size; ++ i)
    {
        const auto N = m_samples(i);
        if (N > 1)
        {
            m_stdev(i) = std::sqrt((m_stdev(i) - m_mean(i) * m_mean(i) / static_cast<scalar_t>(N)) / static_cast<scalar_t>(N - 1));
            m_mean(i) /= static_cast<scalar_t>(N);
            m_div_range(i) = 1.0 / std::max(m_max(i) - m_min(i), epsilon);
            m_div_stdev(i) = 1.0 / std::max(m_stdev(i), epsilon);
            m_mul_range(i) = std::max(m_max(i) - m_min(i), epsilon);
            m_mul_stdev(i) = std::max(m_stdev(i), epsilon);
        }
        else
        {
            if (N == 0)
            {
                m_min(i) = 0.0;
                m_max(i) = 0.0;
                m_mean(i) = 0.0;
            }
            m_stdev(i) = 0.0;
            m_div_range(i) = 1.0;
            m_div_stdev(i) = 1.0;
            m_mul_range(i) = 1.0;
            m_mul_stdev(i) = 1.0;
        }

        // NB: disable scaling for this dimension!
        if (i < enable_scaling.size() && enable_scaling(i) == 0x00)
        {
            m_min(i) = 0.0;
            m_max(i) = 0.0;
            m_mean(i) = 0.0;
            m_stdev(i) = 0.0;
            m_div_range(i) = 1.0;
            m_div_stdev(i) = 1.0;
            m_mul_range(i) = 1.0;
            m_mul_stdev(i) = 1.0;
        }
    }
    return *this;
}

void scalar_stats_t::scale(scaling_type scaling, tensor2d_map_t values) const
{
    assert(values.size<1>() == m_min.size());

    switch (scaling)
    {
    case scaling_type::none:
        for (tensor_size_t sample = 0, samples = values.size<0>(); sample < samples; ++ sample)
        {
            auto array = values.array(sample);
            nan2zero(array);
        }
        break;

    case scaling_type::mean:
        for (tensor_size_t sample = 0, samples = values.size<0>(); sample < samples; ++ sample)
        {
            auto array = values.array(sample);
            array = (array - m_mean.array()) * m_div_range.array();
            nan2zero(array);
        }
        break;

    case scaling_type::minmax:
        for (tensor_size_t sample = 0, samples = values.size<0>(); sample < samples; ++ sample)
        {
            auto array = values.array(sample);
            array = (array - m_min.array()) * m_div_range.array();
            nan2zero(array);
        }
        break;

    case scaling_type::standard:
        for (tensor_size_t sample = 0, samples = values.size<0>(); sample < samples; ++ sample)
        {
            auto array = values.array(sample);
            array = (array - m_mean.array()) * m_div_stdev.array();
            nan2zero(array);
        }
        break;

    default:
        throw std::runtime_error("unhandled scaling type");
        break;
    }
}

void scalar_stats_t::scale(scaling_type scaling, tensor4d_map_t values) const
{
    assert(values.size() == m_min.size() * values.size<0>());

    scale(scaling, values.reshape(values.size<0>(), -1));
}

void scalar_stats_t::upscale(scaling_type scaling, tensor2d_map_t values) const
{
    assert(values.size<1>() == m_min.size());

    switch (scaling)
    {
    case scaling_type::none:
        break;

    case scaling_type::mean:
        for (tensor_size_t sample = 0, samples = values.size<0>(); sample < samples; ++ sample)
        {
            auto array = values.array(sample);
            array = m_mean.array() + array * m_mul_range.array();
        }
        break;

    case scaling_type::minmax:
        for (tensor_size_t sample = 0, samples = values.size<0>(); sample < samples; ++ sample)
        {
            auto array = values.array(sample);
            array = m_min.array() + array * m_mul_range.array();
        }
        break;

    case scaling_type::standard:
        for (tensor_size_t sample = 0, samples = values.size<0>(); sample < samples; ++ sample)
        {
            auto array = values.array(sample);
            array = m_mean.array() + array * m_mul_stdev.array();
        }
        break;

    default:
        throw std::runtime_error("unhandled scaling type");
        break;
    }
}

void scalar_stats_t::upscale(scaling_type scaling, tensor4d_map_t values) const
{
    assert(values.size() == m_min.size() * values.size<0>());

    upscale(scaling, values.reshape(values.size<0>(), -1));
}

void nano::upscale(
    const scalar_stats_t& flatten_stats, scaling_type flatten_scaling,
    const scalar_stats_t& targets_stats, scaling_type targets_scaling,
    tensor2d_map_t weights, tensor1d_map_t bias)
{
    assert(bias.size<0>() == targets_stats.size());
    assert(weights.size<0>() == targets_stats.size());
    assert(weights.size<1>() == flatten_stats.size());

    const auto make_scaling = [] (const scalar_stats_t& stats, scaling_type scaling)
    {
        auto w = make_full_tensor<scalar_t>(make_dims(stats.size()), 1.0);
        auto b = make_full_tensor<scalar_t>(make_dims(stats.size()), 0.0);

        switch (scaling)
        {
        case scaling_type::mean:
            w = stats.div_range();
            b.array() = -stats.mean().array() * stats.div_range().array();
            break;

        case scaling_type::minmax:
            w = stats.div_range();
            b.array() = -stats.min().array() * stats.div_range().array();
            break;

        case scaling_type::standard:
            w = stats.div_stdev();
            b.array() = -stats.mean().array() * stats.div_stdev().array();
            break;

        default:
            break;
        }

        return std::make_pair(w, b);
    };

    const auto [flatten_w, flatten_b] = make_scaling(flatten_stats, flatten_scaling);
    const auto [targets_w, targets_b] = make_scaling(targets_stats, targets_scaling);

    // cppcheck-suppress unreadVariable
    bias.array() = (weights.matrix() * flatten_b.vector()).array() + bias.array() - targets_b.array();
    bias.array() /= targets_w.array();

    // cppcheck-suppress unreadVariable
    weights.matrix().array().colwise() /= targets_w.array();
    weights.matrix().array().rowwise() *= flatten_w.array().transpose();
}

sclass_stats_t::sclass_stats_t() = default;

sclass_stats_t::sclass_stats_t(tensor_size_t classes) :
    m_class_counts(classes),
    m_class_weights(classes)
{
    m_class_counts.zero();
    m_class_weights.zero();
}

sclass_stats_t& sclass_stats_t::done()
{
    m_class_weights.array() =
        static_cast<scalar_t>(m_samples) /
        static_cast<scalar_t>(m_class_counts.size()) /
        m_class_counts.array().cast<scalar_t>().max(1.0);
    return *this;
}


mclass_stats_t::mclass_stats_t() = default;

mclass_stats_t::mclass_stats_t(tensor_size_t classes) :
    m_class_counts(2 * classes),
    m_class_weights(2 * classes)
{
    m_class_counts.zero();
    m_class_weights.zero();
}

mclass_stats_t& mclass_stats_t::done()
{
    m_class_weights.array() =
        static_cast<scalar_t>(m_samples) /
        static_cast<scalar_t>(m_class_counts.size()) /
        m_class_counts.array().cast<scalar_t>().max(1.0);
    return *this;
}
