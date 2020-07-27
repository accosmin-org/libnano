#include <nano/logger.h>
#include <nano/gboost/util.h>
#include <nano/tensor/stream.h>
#include <nano/gboost/wlearner_stump.h>

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
            m_tables(cat_dims(2, tdim))
        {
        }

        [[nodiscard]] auto x0_neg() const { return m_acc_neg.x0(); }
        [[nodiscard]] auto r1_neg() const { return m_acc_neg.r1(); }
        [[nodiscard]] auto r2_neg() const { return m_acc_neg.r2(); }

        [[nodiscard]] auto x0_pos() const { return m_acc_sum.x0() - m_acc_neg.x0(); }
        [[nodiscard]] auto r1_pos() const { return m_acc_sum.r1() - m_acc_neg.r1(); }
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
                    m_acc_sum.update(gradients.array(i));
                }
            }
            std::sort(m_ivalues.begin(), m_ivalues.end());
        }

        [[nodiscard]] auto output_neg() const
        {
            return r1_neg() / x0_neg();
        }

        [[nodiscard]] auto output_pos() const
        {
            return r1_pos() / x0_pos();
        }

        template <typename tarray, typename toutputs>
        static auto score(const scalar_t r0, const tarray& r1, const tarray& r2, const toutputs& outputs)
        {
            return (r2 + outputs.square() * r0 - 2 * outputs * r1).sum();
        }

        [[nodiscard]] auto score() const
        {
            return
                cache_t::score(x0_neg(), r1_neg(), r2_neg(), output_neg()) +
                cache_t::score(x0_pos(), r1_pos(), r2_pos(), output_pos());
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
    wlearner_feature1_t::loopc(dataset, fold,
        [&] (tensor_size_t feature, const tensor1d_t& fvalues, size_t tnum)
    {
        // update accumulators
        auto& cache = caches[tnum];
        cache.clear(gradients, fvalues, indices);
        for (size_t iv = 0, sv = cache.m_ivalues.size(); iv + 1 < sv; ++ iv)
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
