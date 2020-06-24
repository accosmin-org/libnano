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

        void clear(const tensor3d_dim_t& tdim, const tensor4d_t& gradients, const tensor1d_t& values,
            const indices_t& indices)
        {
            m_res_neg1.resize(tdim);
            m_res_neg2.resize(tdim);
            m_res_sum1.resize(tdim);
            m_res_sum2.resize(tdim);

            m_cnt = 0;
            m_cnt_neg = 0;
            m_res_neg1.zero();
            m_res_neg2.zero();
            m_res_sum1.zero();
            m_res_sum2.zero();
            m_tables.resize(cat_dims(2, tdim));

            m_cnt = 0;
            m_ivalues.clear();
            m_ivalues.reserve(indices.size());
            for (const auto i : indices)
            {
                if (!feature_t::missing(values(i)))
                {
                    ++ m_cnt;
                    m_ivalues.emplace_back(values(i), i);
                    m_res_sum1.array() -= gradients.array(i);
                    m_res_sum2.array() += gradients.array(i) * gradients.array(i);
                }
            }
            std::sort(m_ivalues.begin(), m_ivalues.end());
        }

        [[nodiscard]] auto outputs_real_neg() const
        {
            return m_res_neg1.array() / std::max(m_cnt_neg, scalar_t(1));
        }

        [[nodiscard]] auto outputs_real_pos() const
        {
            return (m_res_sum1.array() - m_res_neg1.array()) / std::max(m_cnt - m_cnt_neg, scalar_t(1));
        }

        [[nodiscard]] auto outputs_discrete_neg() const
        {
            return m_res_neg1.array().sign();
        }

        [[nodiscard]] auto outputs_discrete_pos() const
        {
            return (m_res_sum1.array() - m_res_neg1.array()).sign();
        }

        template <typename tresiduals, typename toutputs>
        static auto score(
            const tresiduals& res1, const tresiduals& res2, const toutputs& outputs, const tensor_size_t cnt)
        {
            return (cnt * outputs.square() - 2 * outputs * res1 + res2).sum();
        }

        [[nodiscard]] auto score(const wlearner type) const
        {
            const auto cnt_pos = m_cnt - m_cnt_neg;
            const auto res_pos1 = m_res_sum1.array() - m_res_neg1.array();
            const auto res_pos2 = m_res_sum2.array() - m_res_neg2.array();

            scalar_t score = 0;
            switch (type)
            {
            case wlearner::real:
                score +=
                    cache_t::score(res_pos1, res_pos2, outputs_real_pos(), cnt_pos) +
                    cache_t::score(m_res_neg1.array(), m_res_neg2.array(), outputs_real_neg(), m_cnt_neg);
                break;

            case wlearner::discrete:
                score +=
                    cache_t::score(res_pos1, res_pos2, outputs_discrete_pos(), cnt_pos) +
                    cache_t::score(m_res_neg1.array(), m_res_neg2.array(), outputs_discrete_neg(), m_cnt_neg);
                break;

            default:
                assert(false);
                break;
            }
            return score;
        }

        using ivalues_t = std::vector<std::pair<scalar_t, tensor_size_t>>;

        // attributes
        tensor_size_t   m_feature{-1};                                  ///<
        scalar_t        m_threshold{0};                                 ///<
        tensor4d_t      m_tables;                                       ///<
        ivalues_t       m_ivalues;                                      ///<
        scalar_t        m_cnt{0}, m_cnt_neg{0};                         ///<
        tensor3d_t      m_res_neg1, m_res_neg2;                         ///<
        tensor3d_t      m_res_sum1, m_res_sum2;                         ///<
        scalar_t        m_score{std::numeric_limits<scalar_t>::max()};  ///<
    };
}

void wlearner_stump_t::read(std::istream& stream)
{
    wlearner_t::read(stream);

    critical(
        !::nano::detail::read(stream, m_feature) ||
        !::nano::detail::read(stream, m_threshold) ||
        !::nano::read(stream, m_tables),
        "stump weak learner: failed to read from stream!");
}

void wlearner_stump_t::write(std::ostream& stream) const
{
    wlearner_t::write(stream);

    critical(
        !::nano::detail::write(stream, m_feature) ||
        !::nano::detail::write(stream, m_threshold) ||
        !::nano::write(stream, m_tables),
        "stump weak learner: failed to write to stream!");
}

std::ostream& wlearner_stump_t::print(std::ostream& stream) const
{
    return stream << "stump: feature=" << m_feature << ",threshold=" << std::fixed << std::setprecision(6) << m_threshold;
}

rwlearner_t wlearner_stump_t::clone() const
{
    return std::make_unique<wlearner_stump_t>(*this);
}

tensor3d_dim_t wlearner_stump_t::odim() const
{
    return make_dims(m_tables.size<1>(), m_tables.size<2>(), m_tables.size<3>());
}

void wlearner_stump_t::scale(const vector_t& scale)
{
    wlearner_t::scale(m_tables, scale);
}

scalar_t wlearner_stump_t::fit(const dataset_t& dataset, fold_t fold, const tensor4d_t& gradients, const indices_t& indices)
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
        critical(true, "stump weak learner: unhandled wlearner");
        break;
    }

    std::vector<cache_t> caches(tpool_t::size());
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
        cache.clear(dataset.tdim(), gradients, fvalues, indices);
        for (size_t iv = 0, sv = cache.m_ivalues.size(); iv + 1 < sv; ++ iv)
        {
            const auto& ivalue1 = cache.m_ivalues[iv + 0];
            const auto& ivalue2 = cache.m_ivalues[iv + 1];

            ++ cache.m_cnt_neg;
            cache.m_res_neg1.array() -= gradients.array(ivalue1.second);
            cache.m_res_neg2.array() += gradients.array(ivalue1.second) * gradients.array(ivalue1.second);

            if (ivalue1.first < ivalue2.first)
            {
                // update the parameters if a better feature
                const auto score = cache.score(type());
                if (score < cache.m_score)
                {
                    cache.m_score = score;
                    cache.m_feature = feature;
                    cache.m_threshold = 0.5 * (ivalue1.first + ivalue2.first);
                    switch (type())
                    {
                    case wlearner::real:
                        cache.m_tables.array(0) = cache.outputs_real_neg();
                        cache.m_tables.array(1) = cache.outputs_real_pos();
                        break;

                    case wlearner::discrete:
                        cache.m_tables.array(0) = cache.outputs_discrete_neg();
                        cache.m_tables.array(1) = cache.outputs_discrete_pos();
                        break;

                    default:
                        assert(false);
                        break;
                    }
                }
            }
        }
    });

    // OK, return and store the optimum feature across threads
    const auto& best = ::nano::gboost::min_reduce(caches);
    m_tables = best.m_tables;
    m_feature = best.m_feature;
    m_threshold = best.m_threshold;
    return best.m_score;
}

void wlearner_stump_t::compatible(const dataset_t& dataset) const
{
    critical(
        m_tables.size<0>() == 0,
        "stump weak learner: empty weak learner!");

    critical(
        odim() != dataset.tdim() ||
        m_feature < 0 || m_feature >= dataset.features() ||
        dataset.ifeature(m_feature).discrete(),
        "stump weak learner: mis-matching dataset!");
}

void wlearner_stump_t::predict(const dataset_t& dataset, fold_t fold, tensor_range_t range, tensor4d_map_t&& outputs) const
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
            outputs.vector(i) = m_tables.vector(x < m_threshold ? 0 : 1);
        }
    }
}

cluster_t wlearner_stump_t::split(const dataset_t& dataset, fold_t fold, const indices_t& indices) const
{
    compatible(dataset);
    wlearner_t::check(indices);

    cluster_t cluster(dataset.samples(fold), 2);
    dataset.loop(execution::par, fold, batch(), [&] (tensor_range_t range, size_t)
    {
        const auto fvalues = dataset.inputs(fold, range, m_feature);
        wlearner_t::for_each(range, indices, [&] (const tensor_size_t i)
        {
            const auto x = fvalues(i - range.begin());
            if (!feature_t::missing(x))
            {
                cluster.assign(i, x < m_threshold ? 0 : 1);
            }
        });
    });

    return cluster;
}

indices_t wlearner_stump_t::features() const
{
    return std::array<tensor_size_t, 1>{{m_feature}};
}
