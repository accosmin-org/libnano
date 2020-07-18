#include <nano/logger.h>
#include <nano/gboost/util.h>
#include <nano/tensor/stream.h>
#include <nano/gboost/wlearner_stump.h>

using namespace nano;

namespace
{
    class cache_t
    {
    public:

        cache_t() = default;

        explicit cache_t(const tensor3d_dim_t& tdim) :
            m_r1_sum(tdim),
            m_r2_sum(tdim),
            m_r1_neg(tdim),
            m_r2_neg(tdim),
            m_tables(cat_dims(2, tdim))
        {
        }

        auto& r0_neg() { return m_r0_neg; }
        auto& r0_sum() { return m_r0_sum; }
        auto r1_neg() { return m_r1_neg.array(); }
        auto r2_neg() { return m_r2_neg.array(); }
        auto r1_sum() { return m_r1_sum.array(); }
        auto r2_sum() { return m_r2_sum.array(); }

        [[nodiscard]] auto r0_neg() const { return m_r0_neg; }
        [[nodiscard]] auto r0_sum() const { return m_r0_sum; }
        [[nodiscard]] auto r1_neg() const { return m_r1_neg.array(); }
        [[nodiscard]] auto r2_neg() const { return m_r2_neg.array(); }
        [[nodiscard]] auto r1_sum() const { return m_r1_sum.array(); }
        [[nodiscard]] auto r2_sum() const { return m_r2_sum.array(); }

        [[nodiscard]] auto r0_pos() const { return r0_sum() - r0_neg(); }
        [[nodiscard]] auto r1_pos() const { return r1_sum() - r1_neg(); }
        [[nodiscard]] auto r2_pos() const { return r2_sum() - r2_neg(); }

        void clear(const tensor4d_t& gradients, const tensor1d_t& values, const indices_t& indices)
        {
            m_r1_sum.zero();
            m_r2_sum.zero();
            m_r1_neg.zero();
            m_r2_neg.zero();
            m_r0_neg = m_r0_sum = 0.0;

            m_ivalues.clear();
            m_ivalues.reserve(indices.size());
            for (const auto i : indices)
            {
                if (!feature_t::missing(values(i)))
                {
                    m_ivalues.emplace_back(values(i), i);
                    update_sum(gradients.array(i));
                }
            }
            std::sort(m_ivalues.begin(), m_ivalues.end());
        }

        [[nodiscard]] auto output_neg() const
        {
            return r1_neg() / r0_neg();
        }

        [[nodiscard]] auto output_pos() const
        {
            return r1_pos() / r0_pos();
        }

        template <typename tarray, typename toutputs>
        static auto score(const scalar_t r0, const tarray& r1, const tarray& r2, const toutputs& outputs)
        {
            return (r2 + outputs.square() * r0 - 2 * outputs * r1).sum();
        }

        [[nodiscard]] auto score() const
        {
            return
                cache_t::score(r0_neg(), r1_neg(), r2_neg(), output_neg()) +
                cache_t::score(r0_pos(), r1_pos(), r2_pos(), output_pos());
        }

        template <typename tarray>
        void update_sum(tarray&& vgrad)
        {
            r0_sum() += 1;
            r1_sum() -= vgrad;
            r2_sum() += vgrad * vgrad;
        }

        template <typename tarray>
        void update_neg(tarray&& vgrad)
        {
            r0_neg() += 1;
            r1_neg() -= vgrad;
            r2_neg() += vgrad * vgrad;
        }

        using ivalues_t = std::vector<std::pair<scalar_t, tensor_size_t>>;

        // attributes
        ivalues_t       m_ivalues;                                      ///<
        scalar_t        m_r0_sum{0}, m_r0_neg{0};                       ///<
        tensor3d_t      m_r1_sum, m_r2_sum, m_r1_neg, m_r2_neg;         ///<
        tensor4d_t      m_tables;                                       ///<
        tensor_size_t   m_feature{-1};                                  ///<
        scalar_t        m_threshold{0};                                 ///<
        scalar_t        m_score{std::numeric_limits<scalar_t>::max()};  ///<
    };
}

wlearner_stump_t::wlearner_stump_t() = default;

void wlearner_stump_t::read(std::istream& stream)
{
    wlearner_feature1_t::read(stream);

    critical(
        !::nano::detail::read(stream, m_threshold),
        "stump weak learner: failed to read from stream!");
}

void wlearner_stump_t::write(std::ostream& stream) const
{
    wlearner_feature1_t::write(stream);

    critical(
        !::nano::detail::write(stream, m_threshold),
        "stump weak learner: failed to write to stream!");
}

rwlearner_t wlearner_stump_t::clone() const
{
    return std::make_unique<wlearner_stump_t>(*this);
}

scalar_t wlearner_stump_t::fit(const dataset_t& dataset, fold_t fold, const tensor4d_t& gradients, const indices_t& indices)
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

            cache.update_neg(gradients.array(ivalue1.second));

            if (ivalue1.first < ivalue2.first)
            {
                // update the parameters if a better feature
                const auto score = cache.score();
                if (std::isfinite(score) && score < cache.m_score)
                {
                    cache.m_score = score;
                    cache.m_feature = feature;
                    cache.m_threshold = 0.5 * (ivalue1.first + ivalue2.first);
                    cache.m_tables.array(0) = cache.output_neg();
                    cache.m_tables.array(1) = cache.output_pos();
                }
            }
        }
    });

    // OK, return and store the optimum feature across threads
    const auto& best = ::nano::gboost::min_reduce(caches);

    log_info() << std::fixed << std::setprecision(8) << " === stump(feature=" << best.m_feature << "|"
        << (best.m_feature >= 0 ? dataset.ifeature(best.m_feature).name() : string_t("N/A"))
        << ",threshold=" << best.m_threshold << "), samples=" << indices.size() << ",score=" << best.m_score << ".";

    set(best.m_feature, best.m_tables);
    m_threshold = best.m_threshold;
    return best.m_score;
}

void wlearner_stump_t::predict(const dataset_t& dataset, fold_t fold, tensor_range_t range, tensor4d_map_t&& outputs) const
{
    wlearner_feature1_t::predict(dataset, fold, range, outputs, [&] (scalar_t x, tensor_size_t i)
    {
        outputs.vector(i) = vector(x < m_threshold ? 0 : 1);
    });
}

cluster_t wlearner_stump_t::split(const dataset_t& dataset, fold_t fold, const indices_t& indices) const
{
    cluster_t cluster(dataset.samples(fold), 2);
    wlearner_feature1_t::split(dataset, fold, indices, [&] (scalar_t x, tensor_size_t i)
    {
        cluster.assign(i, x < m_threshold ? 0 : 1);
    });

    return cluster;
}
