#include <iomanip>
#include <nano/core/logger.h>
#include <nano/core/stream.h>
#include <nano/wlearner/accumulator.h>
#include <nano/wlearner/hinge.h>
#include <nano/wlearner/reduce.h>
#include <nano/wlearner/util.h>

using namespace nano;
using namespace nano::wlearner;

namespace
{
    class cache_t
    {
    public:
        cache_t() = default;

        explicit cache_t(const tensor3d_dims_t& tdims)
            : m_beta0(tdims)
            , m_acc_sum(tdims)
            , m_acc_neg(tdims)
            , m_tables(cat_dims(2, tdims))
        {
            m_beta0.zero();
        }

        auto x0_neg() const { return m_acc_neg.x0(); }

        auto x1_neg() const { return m_acc_neg.x1(); }

        auto x2_neg() const { return m_acc_neg.x2(); }

        auto r1_neg() const { return m_acc_neg.r1(); }

        auto rx_neg() const { return m_acc_neg.rx(); }

        auto r2_neg() const { return m_acc_neg.r2(); }

        auto x0_pos() const { return m_acc_sum.x0() - m_acc_neg.x0(); }

        auto x1_pos() const { return m_acc_sum.x1() - m_acc_neg.x1(); }

        auto x2_pos() const { return m_acc_sum.x2() - m_acc_neg.x2(); }

        auto r1_pos() const { return m_acc_sum.r1() - m_acc_neg.r1(); }

        auto rx_pos() const { return m_acc_sum.rx() - m_acc_neg.rx(); }

        auto r2_pos() const { return m_acc_sum.r2() - m_acc_neg.r2(); }

        void clear(const tensor4d_t& gradients, const scalar_cmap_t& values, const indices_t& samples)
        {
            m_acc_sum.clear();
            m_acc_neg.clear();

            m_ivalues.clear();
            m_ivalues.reserve(static_cast<size_t>(values.size()));
            for (tensor_size_t i = 0; i < values.size(); ++i)
            {
                if (std::isfinite(values(i)))
                {
                    m_ivalues.emplace_back(values(i), samples(i));
                    m_acc_sum.update(values(i), gradients.array(samples(i)));
                }
            }
            std::sort(m_ivalues.begin(), m_ivalues.end());
        }

        template <typename tarray>
        static auto beta(scalar_t x0, scalar_t x1, scalar_t x2, const tarray& r1, const tarray& rx, scalar_t threshold)
        {
            return (rx - r1 * threshold) / (x2 + x0 * threshold * threshold - 2 * x1 * threshold);
        }

        template <typename tarray, typename tbarray>
        static auto score(scalar_t x0, scalar_t x1, scalar_t x2, const tarray& r1, const tarray& rx, const tarray& r2,
                          scalar_t threshold, const tbarray& beta)
        {
            return (r2 + beta.square() * (x2 + x0 * threshold * threshold - 2 * x1 * threshold) -
                    2 * beta * (rx - r1 * threshold))
                .sum();
        }

        auto beta0() const { return m_beta0.array(); }

        auto beta_neg(scalar_t threshold) const
        {
            return cache_t::beta(x0_neg(), x1_neg(), x2_neg(), r1_neg(), rx_neg(), threshold);
        }

        auto beta_pos(scalar_t threshold) const
        {
            return cache_t::beta(x0_pos(), x1_pos(), x2_pos(), r1_pos(), rx_pos(), threshold);
        }

        auto score_neg(scalar_t threshold) const
        {
            return cache_t::score(x0_neg(), x1_neg(), x2_neg(), r1_neg(), rx_neg(), r2_neg(), threshold,
                                  beta_neg(threshold)) +
                   cache_t::score(x0_pos(), x1_pos(), x2_pos(), r1_pos(), rx_pos(), r2_pos(), threshold, beta0());
        }

        auto score_pos(scalar_t threshold) const
        {
            return cache_t::score(x0_neg(), x1_neg(), x2_neg(), r1_neg(), rx_neg(), r2_neg(), threshold, beta0()) +
                   cache_t::score(x0_pos(), x1_pos(), x2_pos(), r1_pos(), rx_pos(), r2_pos(), threshold,
                                  beta_pos(threshold));
        }

        using ivalues_t = std::vector<std::pair<scalar_t, tensor_size_t>>;

        // attributes
        ivalues_t     m_ivalues;                           ///<
        tensor3d_t    m_beta0;                             ///<
        accumulator_t m_acc_sum, m_acc_neg;                ///<
        tensor4d_t    m_tables;                            ///<
        tensor_size_t m_feature{-1};                       ///<
        scalar_t      m_threshold{0};                      ///<
        hinge_type    m_hinge{hinge_type::left};           ///<
        scalar_t      m_score{wlearner_t::no_fit_score()}; ///<
    };
} // namespace

hinge_wlearner_t::hinge_wlearner_t()
    : single_feature_wlearner_t("hinge")
{
}

std::istream& hinge_wlearner_t::read(std::istream& stream)
{
    single_feature_wlearner_t::read(stream);

    critical(!::nano::read(stream, m_threshold) || !::nano::read_cast<uint32_t>(stream, m_hinge),
             "hinge weak learner: failed to read from stream!");

    return stream;
}

std::ostream& hinge_wlearner_t::write(std::ostream& stream) const
{
    single_feature_wlearner_t::write(stream);

    critical(!::nano::write(stream, m_threshold) || !::nano::write(stream, static_cast<uint32_t>(m_hinge)),
             "hinge weak learner: failed to write to stream!");

    return stream;
}

rwlearner_t hinge_wlearner_t::clone() const
{
    return std::make_unique<hinge_wlearner_t>(*this);
}

scalar_t hinge_wlearner_t::fit(const dataset_t& dataset, const indices_t& samples, const tensor4d_t& gradients)
{
    learner_t::fit(dataset);

    assert(samples.min() >= 0);
    assert(samples.max() < dataset.samples());
    assert(gradients.dims() == cat_dims(dataset.samples(), dataset.target_dims()));

    select_iterator_t it{dataset};

    std::vector<cache_t> caches(it.concurrency(), cache_t{dataset.target_dims()});
    it.loop(samples,
            [&](const tensor_size_t feature, const size_t tnum, scalar_cmap_t fvalues)
            {
                // update accumulators
                auto& cache = caches[tnum];
                cache.clear(gradients, fvalues, samples);
                for (size_t iv = 0, sv = cache.m_ivalues.size(); iv + 1 < sv; ++iv)
                {
                    const auto& ivalue1 = cache.m_ivalues[iv + 0];
                    const auto& ivalue2 = cache.m_ivalues[iv + 1];

                    cache.m_acc_neg.update(ivalue1.first, gradients.array(ivalue1.second));

                    if (ivalue1.first < ivalue2.first)
                    {
                        // update the parameters if a better feature
                        const auto threshold = 0.5 * (ivalue1.first + ivalue2.first);

                        // ... try the left hinge
                        const auto score_neg = cache.score_neg(threshold);
                        if (std::isfinite(score_neg) && score_neg < cache.m_score)
                        {
                            cache.m_score           = score_neg;
                            cache.m_feature         = feature;
                            cache.m_hinge           = hinge_type::left;
                            cache.m_threshold       = threshold;
                            cache.m_tables.array(0) = cache.beta_neg(threshold);
                            cache.m_tables.array(1) = -threshold * cache.m_tables.array(0);
                        }

                        // ... try the right hinge
                        const auto score_pos = cache.score_pos(threshold);
                        if (std::isfinite(score_pos) && score_pos < cache.m_score)
                        {
                            cache.m_score           = score_pos;
                            cache.m_feature         = feature;
                            cache.m_hinge           = hinge_type::right;
                            cache.m_threshold       = threshold;
                            cache.m_tables.array(0) = cache.beta_pos(threshold);
                            cache.m_tables.array(1) = -threshold * cache.m_tables.array(0);
                        }
                    }
                }
            });

    // OK, return and store the optimum feature across threads
    const auto& best = min_reduce(caches);

    log_info() << std::fixed << std::setprecision(8) << " === hinge(feature=" << best.m_feature << "|"
               << (best.m_feature >= 0 ? dataset.feature(best.m_feature).name() : string_t("N/A"))
               << ",threshold=" << best.m_threshold << "),samples=" << samples.size()
               << ",score=" << (best.m_score == wlearner_t::no_fit_score() ? scat("N/A") : scat(best.m_score)) << ".";

    set(best.m_feature, best.m_tables);
    m_threshold = best.m_threshold;
    m_hinge     = best.m_hinge;
    return best.m_score;
}

void hinge_wlearner_t::predict(const dataset_t& dataset, const indices_cmap_t& samples, tensor4d_map_t outputs) const
{
    learner_t::critical_compatible(dataset);

    assert(tables().dims() == cat_dims(2, dataset.target_dims()));
    assert(outputs.dims() == cat_dims(samples.size(), dataset.target_dims()));

    const auto w = vector(0);
    const auto b = vector(1);

    switch (m_hinge)
    {
    case hinge_type::left:
        loop_scalar(dataset, samples, feature(),
                    [&](const tensor_size_t i, const scalar_t value)
                    {
                        if (value < m_threshold)
                        {
                            outputs.vector(i) += w * value + b;
                        }
                    });
        break;
    default:
        loop_scalar(dataset, samples, feature(),
                    [&](const tensor_size_t i, const scalar_t value)
                    {
                        if (value >= m_threshold && m_hinge == hinge_type::right)
                        {
                            outputs.vector(i) += w * value + b;
                        }
                    });
        break;
    }
}

cluster_t hinge_wlearner_t::split(const dataset_t& dataset, const indices_t& samples) const
{
    learner_t::critical_compatible(dataset);

    cluster_t cluster(dataset.samples(), 2);

    loop_scalar(dataset, samples, feature(),
                [&](const tensor_size_t i, const scalar_t value)
                { cluster.assign(samples(i), value < m_threshold ? 0 : 1); });

    return cluster;
}
