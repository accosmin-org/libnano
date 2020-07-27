#include <nano/logger.h>
#include <nano/gboost/util.h>
#include <nano/gboost/wlearner_affine.h>

using namespace nano;
using namespace nano::gboost;

namespace
{
    class cache_t
    {
    public:

        cache_t() = default;

        explicit cache_t(const tensor3d_dim_t& tdim) :
            m_acc(tdim),
            m_tables(cat_dims(2, tdim))
        {
        }

        [[nodiscard]] auto x0() const { return m_acc.x0(); }
        [[nodiscard]] auto x1() const { return m_acc.x1(); }
        [[nodiscard]] auto x2() const { return m_acc.x2(); }
        [[nodiscard]] auto r1() const { return m_acc.r1(); }
        [[nodiscard]] auto rx() const { return m_acc.rx(); }
        [[nodiscard]] auto r2() const { return m_acc.r2(); }

        void clear()
        {
            m_acc.clear();
        }

        [[nodiscard]] auto a() const
        {
            return (rx() * x0() - r1() * x1()) / (x2() * x0() - x1() * x1());
        }

        [[nodiscard]] auto b() const
        {
            return (r1() * x2() - rx() * x1()) / (x2() * x0() - x1() * x1());
        }

        [[nodiscard]] auto score() const
        {
            return (r2() + a().square() * x2() + b().square() * x0() -
                    2 * a() * rx() - 2 * b() * r1() + 2 * a() * b() * x1()).sum();
        }

        // attributes
        accumulator_t   m_acc;                                  ///<
        tensor4d_t      m_tables;                               ///<
        tensor_size_t   m_feature{0};                           ///<
        scalar_t        m_score{wlearner_t::no_fit_score()};    ///<
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
    wlearner_feature1_t::loopc(dataset, fold,
        [&] (tensor_size_t feature, const tensor1d_t& fvalues, size_t tnum)
    {
        // update accumulators
        auto& cache = caches[tnum];
        cache.clear();
        for (const auto i : indices)
        {
            const auto value = fvalues(i);
            if (!feature_t::missing(value))
            {
                cache.m_acc.update(tfun1::get(value), gradients.array(i));
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

template class NANO_PUBLIC ::nano::wlearner_affine_t<::nano::fun1_cos_t>;
template class NANO_PUBLIC ::nano::wlearner_affine_t<::nano::fun1_lin_t>;
template class NANO_PUBLIC ::nano::wlearner_affine_t<::nano::fun1_log_t>;
template class NANO_PUBLIC ::nano::wlearner_affine_t<::nano::fun1_sin_t>;
