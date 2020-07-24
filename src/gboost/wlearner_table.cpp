#include <nano/logger.h>
#include <nano/gboost/util.h>
#include <nano/tensor/stream.h>
#include <nano/gboost/wlearner_table.h>

using namespace nano;
using namespace nano::gboost;

namespace
{
    class cache_t
    {
    public:

        cache_t() = default;

        explicit cache_t(const tensor3d_dim_t& tdim) :
            m_acc(tdim)
        {
        }

        auto& x0(tensor_size_t fv) { return m_acc.x0(fv); }
        auto r1(tensor_size_t fv) { return m_acc.r1(fv); }
        auto r2(tensor_size_t fv) { return m_acc.r2(fv); }

        [[nodiscard]] auto x0(tensor_size_t fv) const { return m_acc.x0(fv); }
        [[nodiscard]] auto r1(tensor_size_t fv) const { return m_acc.r1(fv); }
        [[nodiscard]] auto r2(tensor_size_t fv) const { return m_acc.r2(fv); }

        void clear(tensor_size_t n_fvalues)
        {
            m_acc.clear(n_fvalues);
        }

        [[nodiscard]] auto output(const tensor_size_t fv) const
        {
            return r1(fv) / x0(fv);
        }

        template <typename toutputs>
        [[nodiscard]] scalar_t score(const tensor_size_t fv, const toutputs& outputs) const
        {
            return (r2(fv) + outputs.square() * x0(fv) - 2 * outputs * r1(fv)).sum();
        }

        [[nodiscard]] auto score() const
        {
            scalar_t score = 0;
            for (tensor_size_t fv = 0, n_fvalues = m_acc.fvalues(); fv < n_fvalues; ++ fv)
            {
                score += this->score(fv, output(fv));
            }
            return score;
        }

        // attributes
        accumulator_t   m_acc;                                  ///<
        tensor4d_t      m_tables;                               ///<
        tensor_size_t   m_feature{-1};                          ///<
        scalar_t        m_score{wlearner_t::no_fit_score()};    ///<
    };
}

wlearner_table_t::wlearner_table_t() = default;

rwlearner_t wlearner_table_t::clone() const
{
    return std::make_unique<wlearner_table_t>(*this);
}

scalar_t wlearner_table_t::fit(const dataset_t& dataset, fold_t fold, const tensor4d_t& gradients, const indices_t& indices)
{
    assert(indices.min() >= 0);
    assert(indices.max() < dataset.samples(fold));
    assert(gradients.dims() == cat_dims(dataset.samples(fold), dataset.tdim()));

    std::vector<cache_t> caches(tpool_t::size(), cache_t{dataset.tdim()});
    loopi(dataset.features(), [&] (tensor_size_t feature, size_t tnum)
    {
        const auto& ifeature = dataset.ifeature(feature);

        // NB: This weak learner works only with discrete features!
        if (!ifeature.discrete())
        {
            return;
        }

        const auto n_fvalues = static_cast<tensor_size_t>(ifeature.labels().size());
        const auto fvalues = dataset.inputs(fold, make_range(0, dataset.samples(fold)), feature);

        // update accumulators
        auto& cache = caches[tnum];
        cache.clear(n_fvalues);
        for (const auto i : indices)
        {
            const auto value = fvalues(i);
            if (feature_t::missing(value))
            {
                continue;
            }

            const auto fv = static_cast<tensor_size_t>(value);
            critical(fv < 0 || fv >= n_fvalues,
                scat("table weak learner: invalid feature value ", fv, ", expecting [0, ", n_fvalues, ")"));

            cache.m_acc.update(gradients.array(i), fv);
        }

        // update the parameters if a better feature
        const auto score = cache.score();
        if (std::isfinite(score) && score < cache.m_score)
        {
            cache.m_score = score;
            cache.m_feature = feature;
            cache.m_tables.resize(cat_dims(n_fvalues, dataset.tdim()));
            for (tensor_size_t fv = 0; fv < n_fvalues; ++ fv)
            {
                cache.m_tables.array(fv) = cache.output(fv);
            }
        }
    });

    // OK, return and store the optimum feature across threads
    const auto& best = ::nano::gboost::min_reduce(caches);

    log_info() << std::fixed << std::setprecision(8) << " === table(feature=" << best.m_feature << "|"
        << (best.m_feature >= 0 ? dataset.ifeature(best.m_feature).name() : string_t("N/A"))
        << ",fvalues=" << best.m_tables.size<0>() << "), samples=" << indices.size() << ",score=" << best.m_score << ".";

    set(best.m_feature, best.m_tables, static_cast<size_t>(best.m_tables.size<0>()));
    return best.m_score;
}

void wlearner_table_t::predict(const dataset_t& dataset, fold_t fold, tensor_range_t range, tensor4d_map_t&& outputs) const
{
    wlearner_feature1_t::predict(dataset, fold, range, outputs, [&] (scalar_t x, tensor_size_t i)
    {
        const auto index = static_cast<tensor_size_t>(x);
        critical(
            index < 0 || index >= n_fvalues(),
            scat("table weak learner: invalid feature value ", x, ", expecting [0, ", n_fvalues(), ")"));
        outputs.vector(i) = vector(index);
    });
}

cluster_t wlearner_table_t::split(const dataset_t& dataset, fold_t fold, const indices_t& indices) const
{
    cluster_t cluster(dataset.samples(fold), n_fvalues());
    wlearner_feature1_t::split(dataset, fold, indices, [&] (scalar_t x, tensor_size_t i)
    {
        const auto index = static_cast<tensor_size_t>(x);
        critical(
            index < 0 || index >= n_fvalues(),
            scat("table weak learner: invalid feature value ", x, ", expecting [0, ", n_fvalues(), ")"));
        cluster.assign(i, index);
    });

    return cluster;
}
