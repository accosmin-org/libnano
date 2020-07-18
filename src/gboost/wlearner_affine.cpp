#include <nano/logger.h>
#include <nano/gboost/util.h>
#include <nano/gboost/wlearner_affine.h>

using namespace nano;

namespace
{
    class cache_t
    {
    public:

        cache_t() = default;

        explicit cache_t(const tensor3d_dim_t& tdim) :
            m_r1(tdim),
            m_rx(tdim),
            m_r2(tdim),
            m_tables(cat_dims(2, tdim))
        {
        }

        auto r1() { return m_r1.array(); }
        auto rx() { return m_rx.array(); }
        auto r2() { return m_r2.array(); }

        [[nodiscard]] auto r1() const { return m_r1.array(); }
        [[nodiscard]] auto rx() const { return m_rx.array(); }
        [[nodiscard]] auto r2() const { return m_r2.array(); }

        void clear()
        {
            m_r1.zero();
            m_rx.zero();
            m_r2.zero();
            m_x0 = m_x1 = m_x2 = 0.0;
        }

        template <typename tarray>
        void update(scalar_t value, tarray&& vgrad)
        {
            m_x0 += 1;
            m_x1 += value;
            m_x2 += value * value;
            r1() -= vgrad;
            rx() -= vgrad * value;
            r2() += vgrad * vgrad;
        }

        [[nodiscard]] auto a() const
        {
            return (rx() * m_x0 - r1() * m_x1) / (m_x2 * m_x0 - m_x1 * m_x1);
        }

        [[nodiscard]] auto b() const
        {
            return (r1() * m_x2 - rx() * m_x1) / (m_x2 * m_x0 - m_x1 * m_x1);
        }

        [[nodiscard]] auto score() const
        {
            return (r2() + a().square() * m_x2 + b().square() * m_x0 -
                    2 * a() * rx() - 2 * b() * r1() + 2 * a() * b() * m_x1).sum();
        }

        // attributes
        tensor3d_t      m_r1, m_rx, m_r2;                               ///<
        scalar_t        m_x0{0}, m_x1{0}, m_x2{0};                      ///<
        tensor4d_t      m_tables;                                       ///<
        tensor_size_t   m_feature{0};                                   ///<
        scalar_t        m_score{std::numeric_limits<scalar_t>::max()};  ///<
    };
}

template <typename tfun1>
wlearner_affine_t<tfun1>::wlearner_affine_t() = default;

template <typename tfun1>
rwlearner_t wlearner_affine_t<tfun1>::clone() const
{
    return std::make_unique<wlearner_affine_t>(*this);
}

template <typename tfun1>
scalar_t wlearner_affine_t<tfun1>::fit(const dataset_t& dataset, fold_t fold, const tensor4d_t& gradients, const indices_t& indices)
{
    assert(indices.min() >= 0);
    assert(indices.max() < dataset.samples(fold));
    assert(gradients.dims() == cat_dims(dataset.samples(fold), dataset.tdim()));

    std::vector<cache_t> caches(tpool_t::size(), cache_t{dataset.tdim()});
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
        cache.clear();
        for (const auto i : indices)
        {
            const auto value = fvalues(i);
            if (!feature_t::missing(value))
            {
                cache.update(tfun1::get(value), gradients.array(i));
            }
        }

        // update the parameters if a better feature
        const auto score = cache.score();
        if (std::isfinite(score) && score < cache.m_score)
        {
            cache.m_score = score;
            cache.m_feature = feature;
            cache.m_tables.array(0) = cache.a();
            cache.m_tables.array(1) = cache.b();
        }
    });

    // OK, return and store the optimum feature across threads
    const auto& best = ::nano::gboost::min_reduce(caches);

    log_info() << std::fixed << std::setprecision(8) << " === affine(feature=" << best.m_feature << "|"
        << (best.m_feature >= 0 ? dataset.ifeature(best.m_feature).name() : string_t("N/A"))
        << "), samples=" << indices.size() << ",score=" << best.m_score << ".";

    set(best.m_feature, best.m_tables);
    return best.m_score;
}

template <typename tfun1>
void wlearner_affine_t<tfun1>::predict(
    const dataset_t& dataset, fold_t fold, tensor_range_t range, tensor4d_map_t&& outputs) const
{
    wlearner_feature1_t::predict(dataset, fold, range, outputs, [&] (scalar_t x, tensor_size_t i)
    {
        outputs.vector(i) = vector(0) * tfun1::get(x) + vector(1);
    });
}

template <typename tfun1>
cluster_t wlearner_affine_t<tfun1>::split(
    const dataset_t& dataset, fold_t fold, const indices_t& indices) const
{
    cluster_t cluster(dataset.samples(fold), 1);
    wlearner_feature1_t::split(dataset, fold, indices, [&] (scalar_t, tensor_size_t i)
    {
        cluster.assign(i, 0);
    });

    return cluster;
}

template class NANO_PUBLIC ::nano::wlearner_affine_t<::nano::fun1_cos_t>;
template class NANO_PUBLIC ::nano::wlearner_affine_t<::nano::fun1_lin_t>;
template class NANO_PUBLIC ::nano::wlearner_affine_t<::nano::fun1_log_t>;
template class NANO_PUBLIC ::nano::wlearner_affine_t<::nano::fun1_sin_t>;
