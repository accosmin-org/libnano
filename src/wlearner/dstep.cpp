#include <iomanip>
#include <nano/core/logger.h>
#include <nano/core/stream.h>
#include <nano/wlearner/accumulator.h>
#include <nano/wlearner/dstep.h>
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
            , m_accumulator(tdims)
        {
            m_beta0.zero();
        }

        auto& x0(tensor_size_t fv) { return m_accumulator.x0(fv); }

        auto r1(tensor_size_t fv) { return m_accumulator.r1(fv); }

        auto r2(tensor_size_t fv) { return m_accumulator.r2(fv); }

        auto x0(tensor_size_t fv) const { return m_accumulator.x0(fv); }

        auto r1(tensor_size_t fv) const { return m_accumulator.r1(fv); }

        auto r2(tensor_size_t fv) const { return m_accumulator.r2(fv); }

        void clear(tensor_size_t n_fvalues)
        {
            m_accumulator.clear(n_fvalues);
            m_scores.resize(2, n_fvalues);
        }

        auto beta0() const { return m_beta0.array(); }

        auto beta(tensor_size_t fv) const { return r1(fv) / x0(fv); }

        template <typename tbarray>
        scalar_t score(const tensor_size_t fv, const tbarray& beta) const
        {
            return (r2(fv) + beta.square() * x0(fv) - 2 * beta * r1(fv)).sum();
        }

        // attributes
        tensor3d_t    m_beta0;                             ///<
        accumulator_t m_accumulator;                       ///<
        tensor4d_t    m_tables;                            ///<
        tensor_size_t m_feature{-1};                       ///<
        tensor_size_t m_fvalue{-1};                        ///<
        scalar_t      m_score{wlearner_t::no_fit_score()}; ///<
        tensor2d_t    m_scores;                            ///<
    };
} // namespace

dstep_wlearner_t::dstep_wlearner_t()
    : single_feature_wlearner_t("dstep")
{
}

rwlearner_t dstep_wlearner_t::clone() const
{
    return std::make_unique<dstep_wlearner_t>(*this);
}

std::istream& dstep_wlearner_t::read(std::istream& stream)
{
    single_feature_wlearner_t::read(stream);

    critical(!::nano::read_cast<int64_t>(stream, m_fvalue), "dstep weak learner: failed to read from stream!");

    return stream;
}

std::ostream& dstep_wlearner_t::write(std::ostream& stream) const
{
    single_feature_wlearner_t::write(stream);

    critical(!::nano::write(stream, static_cast<int64_t>(m_fvalue)), "dstep weak learner: failed to write to stream!");

    return stream;
}

scalar_t dstep_wlearner_t::fit(const dataset_t& dataset, const indices_t& samples, const tensor4d_t& gradients)
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
                    cache.m_accumulator.update(gradients.array(samples(i)), value);
                }

                // update the parameters if a better feature
                for (tensor_size_t fv = 0; fv < classes; ++fv)
                {
                    cache.m_scores(0, fv) = cache.score(fv, cache.beta0());
                    cache.m_scores(1, fv) = cache.score(fv, cache.beta(fv));
                }
                for (tensor_size_t fv = 0; fv < classes; ++fv)
                {
                    const auto score = cache.m_scores.array(0).sum() - cache.m_scores(0, fv) + cache.m_scores(1, fv);
                    if (std::isfinite(score) && score < cache.m_score)
                    {
                        cache.m_score   = score;
                        cache.m_fvalue  = fv;
                        cache.m_feature = feature;
                        cache.m_tables.resize(cat_dims(2, dataset.target_dims()));
                        cache.m_tables.zero();
                        cache.m_tables.array(0) = cache.beta(fv);
                    }
                }
            });

    // OK, return and store the optimum feature across threads
    const auto& best = min_reduce(caches);

    log_info() << std::fixed << std::setprecision(8) << " === dstep(feature=" << best.m_feature << "|"
               << (best.m_feature >= 0 ? dataset.feature(best.m_feature).name() : string_t("N/A"))
               << ",fvalues=" << best.m_tables.size<0>() << ",fvalue=" << best.m_fvalue
               << "),samples=" << samples.size()
               << ",score=" << (best.m_score == wlearner_t::no_fit_score() ? scat("N/A") : scat(best.m_score)) << ".";

    if (best.m_score != wlearner_t::no_fit_score())
    {
        learner_t::fit(dataset);
        set(best.m_feature, best.m_tables);
        m_fvalue = best.m_fvalue;
    }
    return best.m_score;
}

void dstep_wlearner_t::predict(const dataset_t& dataset, const indices_cmap_t& samples, tensor4d_map_t outputs) const
{
    learner_t::critical_compatible(dataset);

    assert(outputs.dims() == cat_dims(samples.size(), dataset.target_dims()));

    const auto output = this->vector(0);

    loop_sclass(dataset, samples, feature(),
                [&](const tensor_size_t i, const int32_t value)
                {
                    assert(value >= 0);
                    if (value == m_fvalue)
                    {
                        outputs.vector(i) += output;
                    }
                });
}

cluster_t dstep_wlearner_t::split(const dataset_t& dataset, const indices_t& samples) const
{
    learner_t::critical_compatible(dataset);

    cluster_t cluster(dataset.samples(), 2);

    loop_sclass(dataset, samples, feature(),
                [&](const tensor_size_t i, const int32_t value)
                {
                    assert(value >= 0);
                    cluster.assign(samples(i), value == m_fvalue ? 0 : 1);
                });

    return cluster;
}
