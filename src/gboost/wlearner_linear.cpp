#include <nano/logger.h>
#include <nano/gboost/util.h>
#include <nano/tensor/stream.h>
#include <nano/gboost/wlearner_linear.h>

using namespace nano;

namespace
{
    class cache_t
    {
    public:

        cache_t() = default;

        void clear(const tensor3d_dim_t& tdim)
        {
            m_x1 = 0;
            m_x2 = 0;
            m_cnt = 0;

            m_r1.resize(tdim);
            m_r2.resize(tdim);
            m_rx.resize(tdim);
            m_tables.resize(cat_dims(2, tdim));

            m_r1.zero();
            m_r2.zero();
            m_rx.zero();
        }

        [[nodiscard]] auto a() const
        {
            return (m_rx.array() * m_cnt - m_x1 * m_r1.array()) / (m_x2 * m_cnt - m_x1 * m_x1);
        }

        [[nodiscard]] auto b() const
        {
            return (m_r1.array() * m_x2 - m_x1 * m_rx.array()) / (m_x2 * m_cnt - m_x1 * m_x1);
        }

        [[nodiscard]] auto score() const
        {
            scalar_t score = 0;
            if (m_cnt > 0)
            {
                score += (a().square() * m_x2 + b().square() * m_cnt + m_r2.array()
                    + 2 * a().array() * b().array() * m_x1
                    - 2 * b().array() * m_r1.array()
                    - 2 * a().array() * m_rx.array()).sum();
            }
            return score;
        }

        // attributes
        tensor_size_t   m_feature{0};                                   ///<
        tensor4d_t      m_tables;                                       ///<
        scalar_t        m_x1{0}, m_x2{0}, m_cnt{0};                     ///<
        tensor3d_t      m_r1, m_r2, m_rx;                               ///<
        scalar_t        m_score{std::numeric_limits<scalar_t>::max()};  ///<
    };
}

void wlearner_linear_t::read(std::istream& stream)
{
    wlearner_t::read(stream);

    critical(
        !::nano::detail::read(stream, m_feature) ||
        !::nano::read(stream, m_tables),
        "linear weak learner: failed to read from stream!");
}

void wlearner_linear_t::write(std::ostream& stream) const
{
    wlearner_t::write(stream);

    critical(
        !::nano::detail::write(stream, m_feature) ||
        !::nano::write(stream, m_tables),
        "linear weak learner: failed to write to stream!");
}

std::ostream& wlearner_linear_t::print(std::ostream& stream) const
{
    return stream << "linear: feature=" << m_feature;
}

rwlearner_t wlearner_linear_t::clone() const
{
    return std::make_unique<wlearner_linear_t>(*this);
}

tensor3d_dim_t wlearner_linear_t::odim() const
{
    return make_dims(m_tables.size<1>(), m_tables.size<2>(), m_tables.size<3>());
}

void wlearner_linear_t::scale(const vector_t& scale)
{
    critical(
        scale.size() != 1,
        "linear weak learner: mis-matching scale!");

    critical(
        scale.minCoeff() < 0,
        "linear weak learner: invalid scale factors!");

    m_tables.array() *= scale(0);
}

scalar_t wlearner_linear_t::fit(const dataset_t& dataset, fold_t fold, const tensor4d_t& gradients,
    const indices_t& indices)
{
    assert(indices.min() >= 0);
    assert(indices.max() < dataset.samples(fold));
    assert(gradients.dims() == cat_dims(dataset.samples(fold), dataset.tdim()));

    switch (type())
    {
    case wlearner::real:
        break;

    default:
        critical(true, "linear weak learner: unhandled wlearner");
        break;
    }

    std::vector<cache_t> caches(tpool_t::size());
    loopi(dataset.features(), [&] (tensor_size_t feature, size_t tnum)
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
        cache.clear(dataset.tdim());
        for (const auto i : indices)
        {
            const auto value = fvalues(i);
            if (feature_t::missing(value))
            {
                continue;
            }

            ++ cache.m_cnt;
            cache.m_x1 += value;
            cache.m_x2 += value * value;
            cache.m_r1.array() -= gradients.array(i);
            cache.m_rx.array() -= value * gradients.array(i);
            cache.m_r2.array() += gradients.array(i) * gradients.array(i);
        }

        // update the parameters if a better feature
        const auto score = cache.score();
        if (score < cache.m_score)
        {
            cache.m_tables.zero();
            cache.m_score = score;
            cache.m_feature = feature;
            if (cache.m_cnt > 0)
            {
                cache.m_tables.array(0) = cache.a();
                cache.m_tables.array(1) = cache.b();
            }
        }
    });

    // OK, return and store the optimum feature across threads
    const auto& best = ::nano::gboost::min_reduce(caches);
    m_feature = best.m_feature;
    m_tables = best.m_tables;
    return best.m_score;
}

void wlearner_linear_t::compatible(const dataset_t& dataset) const
{
    critical(
        m_tables.size<0>() == 0,
        "linear weak learner: empty weak learner!");

    critical(
        odim() != dataset.tdim() ||
        m_feature < 0 || m_feature >= dataset.features() ||
        dataset.ifeature(m_feature).discrete(),
        "linear weak learner: mis-matching dataset!");
}

void wlearner_linear_t::predict(const dataset_t& dataset, fold_t fold, tensor_range_t range, tensor4d_map_t&& outputs) const
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
            outputs.vector(i) = m_tables.vector(0) * x + m_tables.vector(1);
        }
    }
}

cluster_t wlearner_linear_t::split(const dataset_t& dataset, fold_t fold, const indices_t& indices) const
{
    compatible(dataset);
    wlearner_t::check(indices);

    cluster_t cluster(dataset.samples(fold), 1);
    dataset.loop(execution::par, fold, batch(), [&] (tensor_range_t range, size_t)
    {
        const auto fvalues = dataset.inputs(fold, range, m_feature);
        wlearner_t::for_each(range, indices, [&] (tensor_size_t i)
        {
            const auto x = fvalues(i - range.begin());
            if (!feature_t::missing(x))
            {
                cluster.assign(i, 0);
            }
        });
    });

    return cluster;
}

indices_t wlearner_linear_t::features() const
{
    return std::array<tensor_size_t, 1>{{m_feature}};
}
