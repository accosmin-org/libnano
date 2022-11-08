#include <iomanip>
#include <nano/wlearner/accumulator.h>
#include <nano/wlearner/reduce.h>
#include <nano/wlearner/table.h>
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
            : m_acc(tdims)
        {
        }

        auto x0(tensor_size_t fv) const { return m_acc.x0(fv); }

        auto r1(tensor_size_t fv) const { return m_acc.r1(fv); }

        auto r2(tensor_size_t fv) const { return m_acc.r2(fv); }

        void clear(tensor_size_t n_fvalues) { m_acc.clear(n_fvalues); }

        auto output(const tensor_size_t fv) const { return r1(fv) / x0(fv); }

        template <typename toutputs>
        scalar_t score(const tensor_size_t fv, const toutputs& outputs) const
        {
            return (r2(fv) + outputs.square() * x0(fv) - 2 * outputs * r1(fv)).sum();
        }

        auto score() const
        {
            scalar_t score = 0;
            for (tensor_size_t fv = 0, n_fvalues = m_acc.fvalues(); fv < n_fvalues; ++fv)
            {
                score += this->score(fv, output(fv));
            }
            return score;
        }

        // attributes
        accumulator_t m_acc;                               ///<
        tensor4d_t    m_tables;                            ///<
        tensor_size_t m_feature{-1};                       ///<
        scalar_t      m_score{wlearner_t::no_fit_score()}; ///<
    };
} // namespace

table_wlearner_t::table_wlearner_t()
    : single_feature_wlearner_t("table")
{
}

rwlearner_t table_wlearner_t::clone() const
{
    return std::make_unique<table_wlearner_t>(*this);
}

scalar_t table_wlearner_t::fit(const dataset_t& dataset, const indices_t& samples, const tensor4d_t& gradients)
{
    assert(samples.min() >= 0);
    assert(samples.max() < dataset.samples());
    assert(gradients.dims() == cat_dims(dataset.samples(), dataset.target_dims()));

    select_iterator_t it{dataset};

    std::vector<cache_t> caches(it.concurrency(), cache_t{dataset.target_dims()});
    it.loop(samples,
            [&](const tensor_size_t feature, const size_t tnum, sclass_cmap_t fvalues)
            {
                const auto& ff = dataset.feature(feature);
                assert(ff.type() == feature_type::sclass);

                const auto classes = ff.classes();

                // update accumulators
                auto& cache = caches[tnum];
                cache.clear(classes);
                for (tensor_size_t i = 0; i < fvalues.size(); ++i)
                {
                    const auto value = fvalues(i);
                    if (value < 0)
                    {
                        continue;
                    }

                    assert(value < classes);
                    cache.m_acc.update(gradients.array(samples(i)), value);
                }

                // update the parameters if a better feature
                const auto score = cache.score();
                if (std::isfinite(score) && score < cache.m_score)
                {
                    cache.m_score   = score;
                    cache.m_feature = feature;
                    cache.m_tables.resize(cat_dims(classes, dataset.target_dims()));
                    for (tensor_size_t value = 0; value < classes; ++value)
                    {
                        cache.m_tables.array(value) = cache.output(value);
                    }
                }
            });

    // OK, return and store the optimum feature across threads
    const auto& best = min_reduce(caches);

    log_info() << std::fixed << std::setprecision(8) << " === table(feature=" << best.m_feature << "|"
               << (best.m_feature >= 0 ? dataset.feature(best.m_feature).name() : string_t("N/A"))
               << ",fvalues=" << best.m_tables.size<0>() << "),samples=" << samples.size()
               << ",score=" << (best.m_score == wlearner_t::no_fit_score() ? scat("N/A") : scat(best.m_score)) << ".";

    if (best.m_score != wlearner_t::no_fit_score())
    {
        learner_t::fit(dataset);
        set(best.m_feature, best.m_tables);
    }
    return best.m_score;
}

void table_wlearner_t::predict(const dataset_t& dataset, const indices_cmap_t& samples, tensor4d_map_t outputs) const
{
    learner_t::critical_compatible(dataset);

    assert(outputs.dims() == cat_dims(samples.size(), dataset.target_dims()));

    loop_sclass(dataset, samples, feature(),
                [&](const tensor_size_t i, const int32_t value)
                {
                    assert(value >= 0 && value < this->tables().size<0>());
                    outputs.vector(i) += vector(value);
                });
}

cluster_t table_wlearner_t::split(const dataset_t& dataset, const indices_t& samples) const
{
    learner_t::critical_compatible(dataset);

    return split(dataset, samples, feature(), tables().size<0>());
}

cluster_t table_wlearner_t::split(const dataset_t& dataset, const indices_t& samples, const tensor_size_t feature,
                                  const tensor_size_t classes)
{
    cluster_t cluster(dataset.samples(), classes);

    loop_sclass(dataset, samples, feature,
                [&](const tensor_size_t i, const int32_t value)
                {
                    assert(value >= 0 && value < classes);
                    cluster.assign(samples(i), value);
                });

    return cluster;
} // LCOV_EXCL_LINE
