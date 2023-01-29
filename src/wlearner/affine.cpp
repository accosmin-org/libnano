#include <iomanip>
#include <nano/core/logger.h>
#include <nano/core/reduce.h>
#include <nano/wlearner/accumulator.h>
#include <nano/wlearner/affine.h>
#include <nano/wlearner/criterion.h>
#include <nano/wlearner/util.h>

using namespace nano;
using namespace nano::wlearner;

static constexpr auto bin_affine = 0;
static constexpr auto bin_missed = 1;

namespace
{
    class cache_t : public accumulator_t
    {
    public:
        explicit cache_t(const tensor3d_dims_t& tdims = tensor3d_dims_t{0, 0, 0})
            : accumulator_t(tdims)
            , m_tables(cat_dims(2, tdims))
        {
        }

        auto w() const
        {
            return (rx(bin_affine) * x0(bin_affine) - r1(bin_affine) * x1(bin_affine)) /
                   (x2(bin_affine) * x0(bin_affine) - x1(bin_affine) * x1(bin_affine));
        }

        auto b() const
        {
            return (r1(bin_affine) * x2(bin_affine) - rx(bin_affine) * x1(bin_affine)) /
                   (x2(bin_affine) * x0(bin_affine) - x1(bin_affine) * x1(bin_affine));
        }

        auto rss_affine() const
        {
            const auto w = this->w();
            const auto b = this->b();
            return (r2(bin_affine) + w.square() * x2(bin_affine) + b.square() * x0(bin_affine) -
                    2 * w * rx(bin_affine) - 2 * b * r1(bin_affine) + 2 * w * b * x1(bin_affine))
                .sum();
        }

        auto score(const criterion_type criterion) const
        {
            const auto rss = rss_affine() + rss_zero(bin_missed);
            const auto k   = 2 * ::nano::size(tdims());
            const auto n   = static_cast<tensor_size_t>(x0(bin_affine) + x0(bin_missed));
            return make_score(criterion, rss, k, n);
        }

        // attributes
        tensor4d_t    m_tables;                            ///<
        tensor_size_t m_feature{0};                        ///<
        scalar_t      m_score{wlearner_t::no_fit_score()}; ///<
    };
} // namespace

affine_wlearner_t::affine_wlearner_t()
    : single_feature_wlearner_t("affine")
{
}

rwlearner_t affine_wlearner_t::clone() const
{
    return std::make_unique<affine_wlearner_t>(*this);
}

scalar_t affine_wlearner_t::do_fit(const dataset_t& dataset, const indices_t& samples, const tensor4d_t& gradients)
{
    const auto criterion = parameter("wlearner::criterion").value<criterion_type>();

    select_iterator_t it(dataset);

    std::vector<cache_t> caches(it.concurrency(), cache_t{dataset.target_dims()});
    it.loop(samples,
            [&](const tensor_size_t feature, const size_t tnum, scalar_cmap_t fvalues)
            {
                // update accumulators
                auto& cache = caches[tnum];
                cache.clear(2);
                for (tensor_size_t i = 0; i < samples.size(); ++i)
                {
                    const auto value = fvalues(i);
                    if (std::isfinite(value))
                    {
                        cache.update(value, gradients.array(samples(i)), bin_affine);
                    }
                    else
                    {
                        cache.update(gradients.array(samples(i)), bin_missed);
                    }
                }

                // update the parameters if a better feature
                const auto score = cache.score(criterion);
                if (std::isfinite(score) && score < cache.m_score)
                {
                    cache.m_score           = score;
                    cache.m_feature         = feature;
                    cache.m_tables.array(0) = cache.w();
                    cache.m_tables.array(1) = cache.b();
                }
            });

    // OK, return and store the optimum feature across threads
    const auto& best = min_reduce(caches);

    log_info() << std::fixed << std::setprecision(8) << " === affine(feature=" << best.m_feature << "|"
               << (best.m_feature >= 0 ? dataset.feature(best.m_feature).name() : string_t("N/A"))
               << "),samples=" << samples.size()
               << ",score=" << (best.m_score == wlearner_t::no_fit_score() ? scat("N/A") : scat(best.m_score)) << ".";

    if (best.m_score != wlearner_t::no_fit_score())
    {
        set(best.m_feature, best.m_tables);
    }
    return best.m_score;
}

void affine_wlearner_t::do_predict(const dataset_t& dataset, const indices_cmap_t& samples,
                                   tensor4d_map_t outputs) const
{
    assert(tables().dims() == cat_dims(2, dataset.target_dims()));

    const auto w = vector(0);
    const auto b = vector(1);

    loop_scalar(dataset, samples, feature(),
                [&](const tensor_size_t i, const scalar_t value) { outputs.vector(i) += w * value + b; });
}

cluster_t affine_wlearner_t::do_split(const dataset_t& dataset, const indices_t& samples) const
{
    cluster_t cluster(dataset.samples(), 1);

    loop_scalar(dataset, samples, feature(),
                [&](const tensor_size_t i, const scalar_t) { cluster.assign(samples(i), 0); });

    return cluster;
} // LCOV_EXCL_LINE
