#include <nano/wlearner/accumulator.h>

using namespace nano;

wlearner::accumulator_t::accumulator_t(const tensor3d_dims_t& tdims)
    : m_r1(cat_dims(1, tdims))
    , m_rx(cat_dims(1, tdims))
    , m_r2(cat_dims(1, tdims))
{
    clear();
}

std::tuple<scalar_t, indices_t> wlearner::accumulator_t::kbest(const tensor_size_t kbest)
{
    const auto fvsize = fvalues();

    // NB: use x1 buffer to store score variations!
    scalar_t score = 0;
    for (tensor_size_t fv = 0; fv < fvsize; ++fv)
    {
        score += r2(fv).sum();
        x1(fv) = -r1(fv).square().sum() / x0(fv);
    }

    // sort bins by score variations and keep the mapping to the original bins
    auto mapping = arange(0, fvsize);
    for (tensor_size_t fv1 = 0; fv1 + 1 < fvsize; ++fv1)
    {
        for (tensor_size_t fv2 = fv1 + 1; fv2 < fvsize; ++fv2)
        {
            if (x1(fv1) > x1(fv2))
            {
                const auto xx = x1(fv1);
                x1(fv1)       = x1(fv2);
                x1(fv2)       = xx;

                const auto mmindex = mapping(fv1);
                mapping(fv1)       = mapping(fv2);
                mapping(fv2)       = mmindex;
            }
        }
    }

    for (tensor_size_t fv = 0; fv < std::min(kbest, fvsize); ++fv)
    {
        score += x1(fv);
    }

    return std::make_tuple(score, mapping);
}

std::tuple<scalar_t, indices_t> wlearner::accumulator_t::ksplit(const tensor_size_t ksplit)
{
    const auto fvsize = fvalues();

    // NB: use x0, r1, rx, r2 buffers to store the cluster statistics:
    //  (count, first-order momentum, output, second-order momentum)!
    for (tensor_size_t fv = 0; fv < fvsize; ++fv)
    {
        rx(fv) = r1(fv) / x0(fv);
    }

    auto clusters        = fvsize;
    auto cluster_mapping = arange(0, fvsize);
    while (clusters > ksplit)
    {
        // find the closest two clusters (output-wise) to merge
        scalar_t      distance = std::numeric_limits<scalar_t>::max();
        tensor_size_t cluster1 = 0, cluster2 = 1;
        for (tensor_size_t icluster1 = 0; icluster1 + 1 < clusters; ++icluster1)
        {
            for (tensor_size_t icluster2 = icluster1 + 1; icluster2 < clusters; ++icluster2)
            {
                const auto output1 = rx(icluster1);
                const auto output2 = rx(icluster2);

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
        x0(cluster1) += x0(cluster2);
        r1(cluster1) += r1(cluster2);
        r2(cluster1) += r2(cluster2);
        rx(cluster1) = r1(cluster1) / x0(cluster1);

        for (tensor_size_t cluster = cluster2; cluster + 1 < clusters; ++cluster)
        {
            x0(cluster) = x0(cluster + 1);
            r1(cluster) = r1(cluster + 1);
            r2(cluster) = r2(cluster + 1);
            rx(cluster) = rx(cluster + 1);
        }

        for (tensor_size_t fv = 0; fv < fvsize; ++fv)
        {
            if (cluster_mapping(fv) == cluster2)
            {
                cluster_mapping(fv) = cluster1;
            }
        }
        for (tensor_size_t fv = 0; fv < fvsize; ++fv)
        {
            if (cluster_mapping(fv) > cluster2)
            {
                cluster_mapping(fv) -= 1;
            }
        }

        --clusters;
    }

    scalar_t score = 0;
    for (tensor_size_t cluster = 0; cluster < clusters; ++cluster)
    {
        score += (r2(cluster) - r1(cluster).square() / x0(cluster)).sum();
    }

    return std::make_tuple(score, cluster_mapping);
}
