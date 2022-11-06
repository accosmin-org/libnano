#include <deque>
#include <iomanip>
#include <nano/core/logger.h>
#include <nano/tensor/stream.h>
#include <nano/wlearner/dtree.h>
#include <nano/wlearner/stump.h>
#include <nano/wlearner/table.h>
#include <nano/wlearner/util.h>
#include <set>

using namespace nano;

namespace
{
    class cache_t
    {
    public:
        cache_t() = default;

        explicit cache_t(indices_t samples)
            : m_samples(std::move(samples))
        {
        }

        // attributes
        indices_t     m_samples;   ///<
        tensor_size_t m_depth{0};  ///<
        tensor3d_t    m_table;     ///<
        size_t        m_parent{0}; ///<
    };
} // namespace

static void append(tensor4d_t& tables, const tensor3d_cmap_t& table)
{
    // NB: This conservative resize is not very efficient!
    const auto copy  = tables;
    const auto count = tables.size<0>();
    tables.resize(cat_dims(count + 1, table.dims()));
    if (count > 0)
    {
        tables.slice(0, count) = copy;
    }
    tables.tensor(count) = table;
}

static void update(dtree_nodes_t& nodes, indices_t& features)
{
    // gather the unique set of selected features
    std::set<tensor_size_t> ufeatures;
    for (const auto& node : nodes)
    {
        if (node.m_feature >= 0)
        {
            ufeatures.insert(node.m_feature);
        }
    }

    features.resize(static_cast<tensor_size_t>(ufeatures.size()));

    auto it = ufeatures.begin();
    for (tensor_size_t i = 0; i < features.size(); ++i)
    {
        features(i) = *(it++);
    }

    // map nodes to the unique set of selected features
    for (auto& node : nodes)
    {
        if (node.m_feature >= 0)
        {
            const auto pos = ufeatures.find(node.m_feature);
            node.m_feature = static_cast<tensor_size_t>(std::distance(ufeatures.begin(), pos));
        }
    }
}

std::istream& nano::read(std::istream& stream, dtree_node_t& node)
{
    if (!::nano::read_cast<int32_t>(stream, node.m_feature) || !::nano::read_cast<int32_t>(stream, node.m_classes) ||
        !::nano::read(stream, node.m_threshold) || !::nano::read_cast<uint32_t>(stream, node.m_next) ||
        !::nano::read_cast<int32_t>(stream, node.m_table))
    {
        stream.setstate(std::ios_base::failbit);
    }
    return stream;
}

std::ostream& nano::write(std::ostream& stream, const dtree_node_t& node)
{
    if (!::nano::write(stream, static_cast<int32_t>(node.m_feature)) ||
        !::nano::write(stream, static_cast<int32_t>(node.m_classes)) || !::nano::write(stream, node.m_threshold) ||
        !::nano::write(stream, static_cast<uint32_t>(node.m_next)) ||
        !::nano::write(stream, static_cast<int32_t>(node.m_table)))
    {
        stream.setstate(std::ios_base::failbit);
    }
    return stream;
}

std::ostream& nano::operator<<(std::ostream& stream, const dtree_node_t& node)
{
    return stream << "node: feature=" << node.m_feature << ",classes=" << node.m_classes
                  << ",threshold=" << node.m_threshold << ",next=" << node.m_next << ",table=" << node.m_table;
}

std::ostream& nano::operator<<(std::ostream& stream, const dtree_nodes_t& nodes)
{
    stream << "nodes:{\n";
    for (const auto& node : nodes)
    {
        stream << "\t" << node << "\n";
    }
    return stream << "}";
}

dtree_wlearner_t::dtree_wlearner_t()
    : wlearner_t("dtree")
{
    register_parameter(parameter_t::make_integer("wlearner::dtree::max_depth", 1, LE, 3, LE, 10));
    register_parameter(parameter_t::make_integer("wlearner::dtree::min_split", 1, LE, 5, LE, 10));
}

std::istream& dtree_wlearner_t::read(std::istream& stream)
{
    wlearner_t::read(stream);

    critical(!::nano::read(stream, m_nodes) || !::nano::read(stream, m_features) || !::nano::read(stream, m_tables),
             "dtree weak learner: failed to read from stream!");

    return stream;
}

std::ostream& dtree_wlearner_t::write(std::ostream& stream) const
{
    wlearner_t::write(stream);

    critical(!::nano::write(stream, m_nodes) || !::nano::write(stream, m_features) || !::nano::write(stream, m_tables),
             "dtree weak learner: failed to write to stream!");

    return stream;
}

rwlearner_t dtree_wlearner_t::clone() const
{
    return std::make_unique<dtree_wlearner_t>(*this);
}

void dtree_wlearner_t::scale(const vector_t& scale)
{
    ::nano::wlearner::scale(m_tables, scale);
}

scalar_t dtree_wlearner_t::fit(const dataset_t& dataset, const indices_t& samples, const tensor4d_t& gradients)
{
    learner_t::fit(dataset);

    assert(samples.min() >= 0);
    assert(samples.max() < dataset.samples());
    assert(gradients.dims() == cat_dims(dataset.samples(), dataset.target_dims()));

    const auto max_depth = parameter("wlearner::dtree::max_depth").value<tensor_size_t>();
    const auto min_split = parameter("wlearner::dtree::min_split").value<tensor_size_t>();

    scalar_t score = 0;

    m_nodes.clear();
    m_tables.resize(cat_dims(0, dataset.target_dims()));

    auto stump = stump_wlearner_t{};
    auto table = table_wlearner_t{};

    const auto min_samples_size = std::min<tensor_size_t>(10, dataset.samples() * min_split / 100);

    std::deque<cache_t> caches;
    caches.emplace_back(samples);
    while (!caches.empty())
    {
        const auto cache = caches.front();

        // split the node using both discrete and continuous features...
        log_info() << std::fixed << std::setprecision(8) << " +++ depth=" << cache.m_depth
                   << ",samples=" << cache.m_samples.size()
                   << ",score=" << (score == wlearner_t::no_fit_score() ? scat("N/A") : scat(score)) << "...";
        const auto score_stump = stump.fit(dataset, cache.m_samples, gradients);
        const auto score_table = table.fit(dataset, cache.m_samples, gradients);

        cluster_t    cluster;
        tensor4d_t   tables;
        dtree_node_t node;

        cache_t ncache;
        ncache.m_depth = cache.m_depth + 1;

        if (score_stump < score_table)
        {
            tables  = stump.tables();
            cluster = stump.split(dataset, cache.m_samples);

            node.m_feature   = stump.feature();
            node.m_threshold = stump.threshold();
        }
        else
        {
            tables  = table.tables();
            cluster = table.split(dataset, cache.m_samples);

            node.m_feature = table.feature();
            node.m_classes = tables.size<0>();
        }
        assert(cluster.groups() == tables.size<0>());

        // have the parent node point to the current terminal node (to be added)
        if (cache.m_parent < m_nodes.size())
        {
            m_nodes[cache.m_parent].m_next = m_nodes.size();
        }

        // terminal nodes...
        if (cache.m_samples.size() < min_samples_size || (cache.m_depth + 1) >= max_depth)
        {
            for (tensor_size_t i = 0, size = tables.size<0>(); i < size; ++i)
            {
                ncache.m_parent  = m_nodes.size();
                ncache.m_samples = cluster.indices(i);

                node.m_table = m_tables.size<0>();
                m_nodes.emplace_back(node);
                append(m_tables, tables.tensor(i));
            }

            // also, update the total score
            score += std::min(score_table, score_stump);
        }

        // can still split the samples
        else
        {
            for (tensor_size_t i = 0, size = tables.size<0>(); i < size; ++i)
            {
                ncache.m_parent  = m_nodes.size();
                ncache.m_samples = cluster.indices(i);

                node.m_table = -1;
                m_nodes.push_back(node);
                caches.push_back(ncache);
            }
        }

        caches.pop_front();
    }

    // OK, compact the selected features
    ::update(m_nodes, m_features);

    log_info() << std::fixed << std::setprecision(8) << " === tree(features=" << m_features.size()
               << ",nodes=" << m_nodes.size() << "), score=" << score << ".";

    return score;
}

void dtree_wlearner_t::predict(const dataset_t& dataset, const indices_cmap_t& samples, tensor4d_map_t outputs) const
{
    learner_t::critical_compatible(dataset);

    assert(outputs.dims() == cat_dims(samples.size(), dataset.target_dims()));

    const auto cluster = split(dataset, samples);
    for (tensor_size_t i = 0, size = cluster.samples(); i < size; ++i)
    {
        const auto group = cluster.group(i);
        if (group >= 0)
        {
            assert(group < m_tables.size<0>());
            outputs.vector(i) += m_tables.vector(group);
        }
    }
}

cluster_t dtree_wlearner_t::split(const dataset_t& dataset, const indices_t& samples) const
{
    learner_t::critical_compatible(dataset);

    cluster_t cluster(dataset.samples(), m_tables.size());

    std::deque<std::pair<size_t, indices_t>> splits;
    splits.emplace_back(0U, samples);

    while (!splits.empty())
    {
        const auto split = splits.front();
        assert(split.first < m_nodes.size());

        const auto& node         = m_nodes[split.first];
        const auto  node_samples = split.second;

        splits.pop_front();

        // terminal node
        if (node.m_table >= 0)
        {
            for (const auto i : node_samples)
            {
                cluster.assign(i, node.m_table);
            }
        }

        // split node
        else
        {
            const auto node_cluster =
                (node.m_classes > 0) ? table_wlearner_t::split(dataset, samples, node.m_feature, node.m_classes)
                                     : stump_wlearner_t::split(dataset, node_samples, node.m_feature, node.m_threshold);

            for (tensor_size_t group = 0; group < node_cluster.groups(); ++group)
            {
                splits.emplace_back(node.m_next + static_cast<size_t>(group), node_cluster.indices(group));
            }
        }
    }

    return cluster;
}

indices_t dtree_wlearner_t::features() const
{
    return m_features;
}