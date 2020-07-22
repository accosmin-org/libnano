#include <nano/logger.h>
#include <nano/gboost/util.h>
#include <nano/tensor/stream.h>
#include <nano/gboost/wlearner_hinge.h>

using namespace nano;
using namespace nano::gboost;

namespace
{
    class cache_t
    {
    public:

        cache_t() = default;

        explicit cache_t(const tensor3d_dim_t& tdim) :
            m_acc_sum(tdim),
            m_acc_neg(tdim),
            m_tables(cat_dims(4, tdim))
        {
        }

        [[nodiscard]] auto x0_neg() const { return m_acc_neg.x0(); }
        [[nodiscard]] auto x1_neg() const { return m_acc_neg.x1(); }
        [[nodiscard]] auto x2_neg() const { return m_acc_neg.x2(); }
        [[nodiscard]] auto r1_neg() const { return m_acc_neg.r1(); }
        [[nodiscard]] auto rx_neg() const { return m_acc_neg.rx(); }
        [[nodiscard]] auto r2_neg() const { return m_acc_neg.r2(); }

        [[nodiscard]] auto x0_pos() const { return m_acc_sum.x0() - m_acc_neg.x0(); }
        [[nodiscard]] auto x1_pos() const { return m_acc_sum.x1() - m_acc_neg.x1(); }
        [[nodiscard]] auto x2_pos() const { return m_acc_sum.x2() - m_acc_neg.x2(); }
        [[nodiscard]] auto r1_pos() const { return m_acc_sum.r1() - m_acc_neg.r1(); }
        [[nodiscard]] auto rx_pos() const { return m_acc_sum.rx() - m_acc_neg.rx(); }
        [[nodiscard]] auto r2_pos() const { return m_acc_sum.r2() - m_acc_neg.r2(); }

        void clear(const tensor4d_t& gradients, const tensor1d_t& values, const indices_t& indices)
        {
            m_acc_sum.clear();
            m_acc_neg.clear();

            m_ivalues.clear();
            m_ivalues.reserve(indices.size());
            for (const auto i : indices)
            {
                if (!feature_t::missing(values(i)))
                {
                    m_ivalues.emplace_back(values(i), i);
                    m_acc_sum.update(values(i), gradients.array(i));
                }
            }
            std::sort(m_ivalues.begin(), m_ivalues.end());
        }

        template <typename tarray>
        static auto beta(scalar_t x0, scalar_t x1, scalar_t x2,
            const tarray& r1, const tarray& rx, scalar_t threshold)
        {
            return (rx - r1 * threshold) / (x2 + x0 * threshold * threshold - 2 * x1 * threshold);
        }

        template <typename tarray, typename tbarray>
        static auto score(scalar_t x0, scalar_t x1, scalar_t x2,
            const tarray& r1, const tarray& rx, const tarray& r2, scalar_t threshold, const tbarray& beta)
        {
            return (r2 + beta.square() * (x2 + x0 * threshold * threshold - 2 * x1 * threshold)
                - 2 * beta * (rx - r1 * threshold)).sum();
        }

        [[nodiscard]] auto beta_neg(scalar_t threshold) const
        {
            return cache_t::beta(x0_neg(), x1_neg(), x2_neg(), r1_neg(), rx_neg(), threshold);
        }

        [[nodiscard]] auto beta_pos(scalar_t threshold) const
        {
            return cache_t::beta(x0_pos(), x1_pos(), x2_pos(), r1_pos(), rx_pos(), threshold);
        }

        [[nodiscard]] auto score(scalar_t threshold) const
        {
            return
                cache_t::score(x0_neg(), x1_neg(), x2_neg(), r1_neg(), rx_neg(), r2_neg(), threshold, beta_neg(threshold)),
                cache_t::score(x0_pos(), x1_pos(), x2_pos(), r1_pos(), rx_pos(), r2_pos(), threshold, beta_pos(threshold));
        }

        using ivalues_t = std::vector<std::pair<scalar_t, tensor_size_t>>;

        // attributes
        ivalues_t       m_ivalues;                              ///<
        accumulator_t   m_acc_sum, m_acc_neg;                   ///<
        tensor4d_t      m_tables;                               ///<
        tensor_size_t   m_feature{-1};                          ///<
        scalar_t        m_threshold{0};                         ///<
        scalar_t        m_score{wlearner_t::no_fit_score()};    ///<
    };
}

wlearner_hinge_t::wlearner_hinge_t() = default;

void wlearner_hinge_t::read(std::istream& stream)
{
    wlearner_feature1_t::read(stream);

    critical(
        !::nano::detail::read(stream, m_threshold),
        "hinge weak learner: failed to read from stream!");
}

void wlearner_hinge_t::write(std::ostream& stream) const
{
    wlearner_feature1_t::write(stream);

    critical(
        !::nano::detail::write(stream, m_threshold),
        "hinge weak learner: failed to write to stream!");
}

rwlearner_t wlearner_hinge_t::clone() const
{
    return std::make_unique<wlearner_hinge_t>(*this);
}

scalar_t wlearner_hinge_t::fit(const dataset_t& dataset, fold_t fold, const tensor4d_t& gradients, const indices_t& indices)
{
    assert(indices.min() >= 0);
    assert(indices.max() < dataset.samples(fold));
    assert(gradients.dims() == cat_dims(dataset.samples(fold), dataset.tdim()));

    std::vector<cache_t> caches(tpool_t::size(), cache_t{dataset.tdim()});
    loopi(dataset.features(), [&] (const tensor_size_t feature, const size_t tnum)
    {
        const auto& ifeature = dataset.ifeature(feature);

        // NB: This weak learner works only with continuous features!
        if (ifeature.discrete())
        {
            return;
        }
        const auto fvalues = dataset.inputs(fold, make_range(0, dataset.samples(fold)), feature);

        // update accumulators
        auto& cache = caches[tnum];
        cache.clear(gradients, fvalues, indices);
        for (size_t iv = 0, sv = cache.m_ivalues.size(); iv + 1 < sv; ++ iv)
        {
            const auto& ivalue1 = cache.m_ivalues[iv + 0];
            const auto& ivalue2 = cache.m_ivalues[iv + 1];

            cache.m_acc_neg.update(ivalue1.first, gradients.array(ivalue1.second));

            if (ivalue1.first < ivalue2.first)
            {
                // update the parameters if a better feature
                const auto threshold = 0.5 * (ivalue1.first + ivalue2.first);
                const auto score = cache.score(threshold);
                if (std::isfinite(score) && score < cache.m_score)
                {
                    cache.m_score = score;
                    cache.m_feature = feature;
                    cache.m_threshold = threshold;
                    cache.m_tables.array(0) = cache.beta_neg(threshold);
                    cache.m_tables.array(1) = -threshold * cache.m_tables.array(0);
                    cache.m_tables.array(2) = cache.beta_pos(threshold);
                    cache.m_tables.array(3) = -threshold * cache.m_tables.array(2);
                }
            }
        }
    });

    // OK, return and store the optimum feature across threads
    const auto& best = ::nano::gboost::min_reduce(caches);

    log_info() << std::fixed << std::setprecision(8) << " === hinge(feature=" << best.m_feature << "|"
        << (best.m_feature >= 0 ? dataset.ifeature(best.m_feature).name() : string_t("N/A"))
        << ",threshold=" << best.m_threshold << "), samples=" << indices.size() << ",score=" << best.m_score << ".";

    set(best.m_feature, best.m_tables);
    m_threshold = best.m_threshold;
    return best.m_score;
}

void wlearner_hinge_t::predict(const dataset_t& dataset, fold_t fold, tensor_range_t range, tensor4d_map_t&& outputs) const
{
    wlearner_feature1_t::predict(dataset, fold, range, outputs, [&] (scalar_t x, tensor_size_t i)
    {
        outputs.vector(i) = vector(x < m_threshold ? 0 : 2) * x + vector(x < m_threshold ? 1 : 3);
    });
}

cluster_t wlearner_hinge_t::split(const dataset_t& dataset, fold_t fold, const indices_t& indices) const
{
    cluster_t cluster(dataset.samples(fold), 2);
    wlearner_feature1_t::split(dataset, fold, indices, [&] (scalar_t x, tensor_size_t i)
    {
        cluster.assign(i, x < m_threshold ? 0 : 1);
    });

    return cluster;
}
