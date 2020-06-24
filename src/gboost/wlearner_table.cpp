#include <nano/logger.h>
#include <nano/gboost/util.h>
#include <nano/tensor/stream.h>
#include <nano/gboost/wlearner_table.h>

using namespace nano;

namespace
{
    class cache_t
    {
    public:

        cache_t() = default;

        void clear(const tensor_size_t n_fvalues, const tensor3d_dim_t tdim)
        {
            m_cnt.resize(n_fvalues);
            m_res1.resize(cat_dims(n_fvalues, tdim));
            m_res2.resize(cat_dims(n_fvalues, tdim));

            m_cnt.zero();
            m_res1.zero();
            m_res2.zero();
        }

        [[nodiscard]] auto outputs_real(const tensor_size_t fv) const
        {
            return m_res1.array(fv) / std::max(m_cnt(fv), scalar_t(1));
        }

        [[nodiscard]] auto outputs_discrete(const tensor_size_t fv) const
        {
            return m_res1.array(fv).sign();
        }

        template <typename toutputs>
        [[nodiscard]] scalar_t score(const tensor_size_t fv, const toutputs& outputs) const
        {
            return (m_cnt(fv) * outputs.square() - 2 * outputs * m_res1.array(fv) + m_res2.array(fv)).sum();
        }

        [[nodiscard]] auto score(const wlearner type) const
        {
            scalar_t score = 0;
            for (tensor_size_t fv = 0; fv < m_cnt.size(); ++ fv)
            {
                switch (type)
                {
                case wlearner::real:
                    score += this->score(fv, outputs_real(fv));
                    break;

                case wlearner::discrete:
                    score += this->score(fv, outputs_discrete(fv));
                    break;

                default:
                    assert(false);
                    break;
                }
            }
            return score;
        }

        // attributes
        tensor_size_t   m_feature{-1};                                  ///<
        tensor4d_t      m_tables;                                       ///<
        tensor1d_t      m_cnt;                                          ///<
        tensor4d_t      m_res1, m_res2;                                 ///<
        scalar_t        m_score{std::numeric_limits<scalar_t>::max()};  ///<
    };
}

void wlearner_table_t::read(std::istream& stream)
{
    wlearner_t::read(stream);

    critical(
        !::nano::detail::read(stream, m_feature) ||
        !::nano::read(stream, m_tables),
        "table weak learner: failed to read from stream!");
}

void wlearner_table_t::write(std::ostream& stream) const
{
    wlearner_t::write(stream);

    critical(
        !::nano::detail::write(stream, m_feature) ||
        !::nano::write(stream, m_tables),
        "table weak learner: failed to write to stream!");
}

std::ostream& wlearner_table_t::print(std::ostream& stream) const
{
    return stream << "table: feature=" << m_feature << ",classes=" << m_tables.size<0>();
}

rwlearner_t wlearner_table_t::clone() const
{
    return std::make_unique<wlearner_table_t>(*this);
}

tensor3d_dim_t wlearner_table_t::odim() const
{
    return make_dims(m_tables.size<1>(), m_tables.size<2>(), m_tables.size<3>());
}

void wlearner_table_t::scale(const vector_t& scale)
{
    wlearner_t::scale(m_tables, scale);
}

scalar_t wlearner_table_t::fit(const dataset_t& dataset, fold_t fold, const tensor4d_t& gradients, const indices_t& indices)
{
    assert(indices.min() >= 0);
    assert(indices.max() < dataset.samples(fold));
    assert(gradients.dims() == cat_dims(dataset.samples(fold), dataset.tdim()));

    switch (type())
    {
    case wlearner::real:
    case wlearner::discrete:
        break;

    default:
        critical(true, "table weak learner: unhandled wlearner");
        break;
    }

    std::vector<cache_t> caches(tpool_t::size());
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
        cache.clear(n_fvalues, dataset.tdim());
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

            cache.m_cnt(fv) ++;
            cache.m_res1.array(fv) -= gradients.array(i);
            cache.m_res2.array(fv) += gradients.array(i) * gradients.array(i);
        }

        // update the parameters if a better feature
        const auto score = cache.score(type());
        if (score < cache.m_score)
        {
            cache.m_score = score;
            cache.m_feature = feature;
            cache.m_tables.resize(cat_dims(n_fvalues, dataset.tdim()));
            for (tensor_size_t fv = 0; fv < cache.m_cnt.size(); ++ fv)
            {
                switch (type())
                {
                case wlearner::real:
                    cache.m_tables.array(fv) = cache.outputs_real(fv);
                    break;

                case wlearner::discrete:
                    cache.m_tables.array(fv) = cache.outputs_discrete(fv);
                    break;

                default:
                    assert(false);
                    break;
                }
            }
        }
    });

    // OK, return and store the optimum feature across threads
    const auto& best = ::nano::gboost::min_reduce(caches);
    m_tables = best.m_tables;
    m_feature = best.m_feature;
    return best.m_score;
}

void wlearner_table_t::compatible(const dataset_t& dataset) const
{
    critical(
        m_tables.size<0>() == 0,
        "table weak learner: empty weak learner!");

    critical(
        odim() != dataset.tdim() ||
        m_feature < 0 || m_feature >= dataset.features() ||
        !dataset.ifeature(m_feature).discrete() ||
        dataset.ifeature(m_feature).labels().size() != static_cast<size_t>(m_tables.size<0>()),
        "table weak learner: mis-matching dataset!");
}

void wlearner_table_t::predict(const dataset_t& dataset, fold_t fold, tensor_range_t range, tensor4d_map_t&& outputs) const
{
    compatible(dataset);
    check(range, outputs);

    const auto fvalues = dataset.inputs(fold, range, m_feature);
    for (tensor_size_t i = 0; i < range.size(); ++ i)
    {
        const auto x = fvalues(i);
        if (feature_t::missing(x))
        {
            outputs.vector(i).setZero();
        }
        else
        {
            const auto index = static_cast<tensor_size_t>(x);
            critical(
                index < 0 || index >= n_fvalues(),
                scat("table weak learner: invalid feature value ", x, ", expecting [0, ", n_fvalues(), ")"));
            outputs.vector(i) = m_tables.vector(index);
        }
    }
}

cluster_t wlearner_table_t::split(const dataset_t& dataset, fold_t fold, const indices_t& indices) const
{
    compatible(dataset);
    wlearner_t::check(indices);

    cluster_t cluster(dataset.samples(fold), m_tables.size<0>());
    dataset.loop(execution::par, fold, batch(), [&] (tensor_range_t range, size_t)
    {
        const auto fvalues = dataset.inputs(fold, range, m_feature);
        wlearner_t::for_each(range, indices, [&] (const tensor_size_t i)
        {
            const auto x = fvalues(i - range.begin());
            if (!feature_t::missing(x))
            {
                const auto index = static_cast<tensor_size_t>(x);
                critical(
                    index < 0 || index >= n_fvalues(),
                    scat("table weak learner: invalid feature value ", x, ", expecting [0, ", n_fvalues(), ")"));
                cluster.assign(i, index);
            }
        });
    });

    return cluster;
}

indices_t wlearner_table_t::features() const
{
    return std::array<tensor_size_t, 1>{{m_feature}};
}
