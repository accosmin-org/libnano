#include <nano/wlearner/accumulator.h>

using namespace nano;

wlearner::accumulator_t::accumulator_t(const tensor3d_dims_t& tdims)
    : m_r1(cat_dims(1, tdims))
    , m_rx(cat_dims(1, tdims))
    , m_r2(cat_dims(1, tdims))
{
    clear();
}

void wlearner::accumulator_t::clear()
{
    m_x0.zero();
    m_x1.zero();
    m_x2.zero();
    m_r1.zero();
    m_rx.zero();
    m_r2.zero();
}

void wlearner::accumulator_t::clear(const tensor_size_t bins)
{
    m_x0.resize(bins);
    m_x1.resize(bins);
    m_x2.resize(bins);
    m_r1.resize(cat_dims(bins, tdims()));
    m_rx.resize(cat_dims(bins, tdims()));
    m_r2.resize(cat_dims(bins, tdims()));

    clear();
}

std::vector<std::pair<scalar_t, tensor_size_t>> wlearner::accumulator_t::sort() const
{
    const auto bins = this->bins();

    std::vector<std::pair<scalar_t, tensor_size_t>> deltas;
    deltas.reserve(static_cast<size_t>(bins));
    for (tensor_size_t bin = 0; bin < bins; ++bin)
    {
        deltas.emplace_back(-r1(bin).square().sum() / x0(bin), bin);
    }

    std::sort(deltas.begin(), deltas.end());

    return deltas;
}

std::tuple<tensor2d_t, tensor5d_t, tensor5d_t, tensor5d_t, tensor_mem_t<tensor_size_t, 2>>
wlearner::accumulator_t::cluster() const
{
    const auto [bins, dim1, dim2, dim3] = m_r1.dims();

    auto cluster_x0 = tensor2d_t{bins, bins};
    auto cluster_r1 = tensor5d_t{bins, bins, dim1, dim2, dim3};
    auto cluster_r2 = tensor5d_t{bins, bins, dim1, dim2, dim3};
    auto cluster_rx = tensor5d_t{bins, bins, dim1, dim2, dim3};
    auto cluster_id = tensor_mem_t<tensor_size_t, 2>{bins, bins};

    // initially each bin is a separate cluster
    cluster_x0.array(0) = m_x0.array();
    cluster_r1.array(0) = m_r1.array();
    cluster_r2.array(0) = m_r2.array();

    for (tensor_size_t bin = 0; bin < bins; ++bin)
    {
        cluster_id(0, bin)       = bin;
        cluster_rx.array(0, bin) = cluster_r1.array(0, bin) / cluster_x0(0, bin);
    }

    // merge clusters until only one remaining
    for (tensor_size_t trial = 1, n_clusters = bins - trial + 1; trial < bins; ++trial, --n_clusters)
    {
        cluster_x0.array(trial) = cluster_x0.array(trial - 1);
        cluster_r1.array(trial) = cluster_r1.array(trial - 1);
        cluster_r2.array(trial) = cluster_r2.array(trial - 1);
        cluster_rx.array(trial) = cluster_rx.array(trial - 1);
        cluster_id.array(trial) = cluster_id.array(trial - 1);

        // find the closest two clusters (output-wise) to merge
        scalar_t      distance = std::numeric_limits<scalar_t>::max();
        tensor_size_t cluster1 = 0, cluster2 = 1;
        for (tensor_size_t icluster1 = 0; icluster1 + 1 < n_clusters; ++icluster1)
        {
            for (tensor_size_t icluster2 = icluster1 + 1; icluster2 < n_clusters; ++icluster2)
            {
                const auto output1 = cluster_rx.array(trial, icluster1);
                const auto output2 = cluster_rx.array(trial, icluster2);

                const auto idistance = (output1 - output2).square().sum();
                if (idistance < distance)
                {
                    distance = idistance;
                    cluster1 = icluster1;
                    cluster2 = icluster2;
                }
            }
        }

        // merge the two clusters
        assert(cluster1 < cluster2);
        cluster_x0(trial, cluster1) += cluster_x0(trial, cluster2);
        cluster_r1.array(trial, cluster1) += cluster_r1.array(trial, cluster2);
        cluster_r2.array(trial, cluster1) += cluster_r2.array(trial, cluster2);
        cluster_rx.array(trial, cluster1) = cluster_r1.array(trial, cluster1) / cluster_x0(trial, cluster1);

        for (tensor_size_t cluster = cluster2; cluster + 1 < n_clusters; ++cluster)
        {
            cluster_x0(trial, cluster)       = cluster_x0(trial, cluster + 1);
            cluster_r1.array(trial, cluster) = cluster_r1.array(trial, cluster + 1);
            cluster_r2.array(trial, cluster) = cluster_r2.array(trial, cluster + 1);
            cluster_rx.array(trial, cluster) = cluster_rx.array(trial, cluster + 1);
        }

        for (tensor_size_t bin = 0; bin < bins; ++bin)
        {
            if (cluster_id(trial, bin) == cluster2)
            {
                cluster_id(trial, bin) = cluster1;
            }
        }
        for (tensor_size_t bin = 0; bin < bins; ++bin)
        {
            if (cluster_id(trial, bin) > cluster2)
            {
                cluster_id(trial, bin) -= 1;
            }
        }
    }

    return std::make_tuple(std::move(cluster_x0), std::move(cluster_r1), std::move(cluster_r2), std::move(cluster_rx),
                           std::move(cluster_id));
}
