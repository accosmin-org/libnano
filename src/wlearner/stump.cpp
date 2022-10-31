#include <iomanip>
#include <nano/core/logger.h>
#include <nano/core/stream.h>
#include <nano/wlearner/accumulator.h>
#include <nano/wlearner/reduce.h>
#include <nano/wlearner/stump.h>
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
            : m_acc_sum(tdims)
            , m_acc_neg(tdims)
            , m_tables(cat_dims(2, tdims))
        {
        }

        auto x0_neg() const { return m_acc_neg.x0(); }

        auto r1_neg() const { return m_acc_neg.r1(); }

        auto r2_neg() const { return m_acc_neg.r2(); }

        auto x0_pos() const { return m_acc_sum.x0() - m_acc_neg.x0(); }

        auto r1_pos() const { return m_acc_sum.r1() - m_acc_neg.r1(); }

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
                    m_acc_sum.update(gradients.array(samples(i)));
                }
            }
            std::sort(m_ivalues.begin(), m_ivalues.end());
        }

        auto output_neg() const { return r1_neg() / x0_neg(); }

        auto output_pos() const { return r1_pos() / x0_pos(); }

        template <typename tarray, typename toutputs>
        static auto score(const scalar_t r0, const tarray& r1, const tarray& r2, const toutputs& outputs)
        {
            return (r2 + outputs.square() * r0 - 2 * outputs * r1).sum();
        }

        auto score() const
        {
            return cache_t::score(x0_neg(), r1_neg(), r2_neg(), output_neg()) +
                   cache_t::score(x0_pos(), r1_pos(), r2_pos(), output_pos());
        }

        using ivalues_t = std::vector<std::pair<scalar_t, tensor_size_t>>;

        // attributes
        ivalues_t     m_ivalues;                           ///<
        accumulator_t m_acc_sum, m_acc_neg;                ///<
        tensor4d_t    m_tables;                            ///<
        tensor_size_t m_feature{-1};                       ///<
        scalar_t      m_threshold{0};                      ///<
        scalar_t      m_score{wlearner_t::no_fit_score()}; ///<
    };
} // namespace

stump_wlearner_t::stump_wlearner_t()
    : single_feature_wlearner_t("stump")
{
}

std::istream& stump_wlearner_t::read(std::istream& stream)
{
    single_feature_wlearner_t::read(stream);

    critical(!::nano::read(stream, m_threshold), "stump weak learner: failed to read from stream!");

    return stream;
}

std::ostream& stump_wlearner_t::write(std::ostream& stream) const
{
    single_feature_wlearner_t::write(stream);

    critical(!::nano::write(stream, m_threshold), "stump weak learner: failed to write to stream!");

    return stream;
}

rwlearner_t stump_wlearner_t::clone() const
{
    return std::make_unique<stump_wlearner_t>(*this);
}

scalar_t stump_wlearner_t::fit(const dataset_t& dataset, const indices_t& samples, const tensor4d_t& gradients)
{
    learner_t::fit(dataset);

    assert(samples.min() >= 0);
    assert(samples.max() < dataset.samples());
    assert(gradients.dims() == cat_dims(dataset.samples(), dataset.target_dims()));

    select_iterator_t it(dataset);

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

                    cache.m_acc_neg.update(gradients.array(ivalue1.second));

                    if (ivalue1.first < ivalue2.first)
                    {
                        // update the parameters if a better feature
                        const auto score = cache.score();
                        if (std::isfinite(score) && score < cache.m_score)
                        {
                            cache.m_score           = score;
                            cache.m_feature         = feature;
                            cache.m_threshold       = 0.5 * (ivalue1.first + ivalue2.first);
                            cache.m_tables.array(0) = cache.output_neg();
                            cache.m_tables.array(1) = cache.output_pos();
                        }
                    }
                }
            });

    // OK, return and store the optimum feature across threads
    const auto& best = min_reduce(caches);

    log_info() << std::fixed << std::setprecision(8) << " === stump(feature=" << best.m_feature << "|"
               << (best.m_feature >= 0 ? dataset.feature(best.m_feature).name() : string_t("N/A"))
               << ",threshold=" << best.m_threshold << "),samples=" << samples.size()
               << ",score=" << (best.m_score == wlearner_t::no_fit_score() ? scat("N/A") : scat(best.m_score)) << ".";

    set(best.m_feature, best.m_tables);
    m_threshold = best.m_threshold;
    return best.m_score;
}

void stump_wlearner_t::predict(const dataset_t& dataset, const indices_cmap_t& samples, tensor4d_map_t outputs) const
{
    learner_t::critical_compatible(dataset);

    assert(tables().dims() == cat_dims(2, dataset.target_dims()));
    assert(outputs.dims() == cat_dims(samples.size(), dataset.target_dims()));

    const auto lo = vector(0);
    const auto hi = vector(1);

    loop_scalar(dataset, samples, feature(),
                [&](const tensor_size_t i, const scalar_t value)
                { outputs.vector(i) += value < m_threshold ? lo : hi; });
}

cluster_t stump_wlearner_t::split(const dataset_t& dataset, const indices_t& samples) const
{
    learner_t::critical_compatible(dataset);

    return split(dataset, samples, feature(), m_threshold);
}

cluster_t stump_wlearner_t::split(const dataset_t& dataset, const indices_t& samples, const tensor_size_t feature,
                                  const scalar_t threshold)
{
    cluster_t cluster(dataset.samples(), 2);

    loop_scalar(dataset, samples, feature,
                [&](const tensor_size_t i, const scalar_t value)
                { cluster.assign(samples(i), value < threshold ? 0 : 1); });

    return cluster;
}
