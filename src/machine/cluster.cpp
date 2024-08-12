#include <nano/machine/cluster.h>

using namespace nano;

cluster_t::cluster_t(tensor_size_t samples, tensor_size_t groups)
    : m_indices(samples)
    , m_groups(groups)
{
    m_indices.full(-1);
}

cluster_t::cluster_t(tensor_size_t samples, const indices_t& indices)
    : cluster_t(samples, 1)
{
    for (const auto index : indices)
    {
        assert(index >= 0 && index < samples);
        m_indices(index) = 0;
    }
}

indices_t cluster_t::indices(tensor_size_t group) const
{
    assert(group >= 0 && group < groups());

    indices_t indices(count(group));
    for (tensor_size_t i = 0, g = 0, size = samples(); i < size; ++i)
    {
        if (m_indices(i) == group)
        {
            indices(g++) = i;
        }
    }
    return indices;
}

tensor_size_t cluster_t::count(tensor_size_t group) const
{
    assert(group >= 0 && group < groups());

    return std::count_if(std::begin(m_indices), std::end(m_indices), [=](tensor_size_t g) { return g == group; });
}
