#pragma once

#include <variant>
#include <nano/dataset/feature.h>
#include <nano/dataset/scaling.h>
#include <nano/dataset/iterator.h>

namespace nano
{
    ///
    /// \brief
    ///
    struct select_stats_t
    {
        indices_t       m_sclass_features;  ///< indices of the single-label features
        indices_t       m_mclass_features;  ///< indices of the multi-label features
        indices_t       m_scalar_features;  ///< indices of the scalar features
        indices_t       m_struct_features;  ///< indices of structured features
    };

    ///
    /// \brief per-feature statistics for continuous feature values or flatten inputs
    ///     (e.g. useful for normalizing inputs and targets).
    ///
    /// NB: missing feature values are ignored when computing these statistics.
    ///
    class NANO_PUBLIC scalar_stats_t
    {
    public:

        scalar_stats_t();
        explicit scalar_stats_t(tensor_size_t dims);

        template <typename tarray>
        auto& operator+=(const tarray& array)
        {
            for (tensor_size_t i = 0, size = array.size(); i < size; ++ i)
            {
                const auto value = static_cast<scalar_t>(array(i));
                if (std::isfinite(value))
                {
                    m_samples(i) += 1;
                    m_mean(i) += value;
                    m_stdev(i) += value * value;
                    m_min(i) = std::min(m_min(i), value);
                    m_max(i) = std::max(m_max(i), value);
                }
            }
            return *this;
        }
        scalar_stats_t& operator+=(const scalar_stats_t& other);

        scalar_stats_t& done(const tensor_mem_t<uint8_t, 1>& enable_scaling = tensor_mem_t<uint8_t, 1>{});

        void scale(scaling_type, tensor2d_map_t values) const;
        void scale(scaling_type, tensor4d_map_t values) const;
        void upscale(scaling_type, tensor2d_map_t values) const;
        void upscale(scaling_type, tensor4d_map_t values) const;

        template <typename tscalar, size_t trank>
        static auto make(const feature_t& feature, dataset_iterator_t<tscalar, trank> it)
        {
            scalar_stats_t stats{::nano::size(feature.dims())};
            for (; it; ++ it)
            {
                if ([[maybe_unused]] const auto [index, given, values] = *it; given)
                {
                    stats += values.array().template cast<scalar_t>();
                }
            }
            return stats.done();
        }

        const auto& min() const { return m_min; }
        const auto& max() const { return m_max; }
        const auto& mean() const { return m_mean; }
        const auto& stdev() const { return m_stdev; }
        const auto& samples() const { return m_samples; }

        auto size() const { return m_min.size(); }
        const auto& div_range() const { return m_div_range; }
        const auto& div_stdev() const { return m_div_stdev; }
        const auto& mul_range() const { return m_mul_range; }
        const auto& mul_stdev() const { return m_mul_stdev; }

    private:

        // attributes
        indices_t       m_samples;                      ///< number of samples per valid component
        tensor1d_t      m_min, m_max, m_mean, m_stdev;  ///<
        tensor1d_t      m_div_range, m_mul_range;       ///<
        tensor1d_t      m_div_stdev, m_mul_stdev;       ///<
    };

    ///
    /// \brief per-feature statistics for single-label categorical feature values
    ///     (e.g. useful for handling unbalanced classification problems).
    ///
    /// NB: missing feature values are ignored when computing these statistics.
    ///
    class NANO_PUBLIC sclass_stats_t
    {
    public:

        sclass_stats_t();
        explicit sclass_stats_t(tensor_size_t classes);

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        auto& operator+=(tscalar label)
        {
            m_samples ++;
            m_class_counts(static_cast<tensor_size_t>(label)) ++;
            return *this;
        }

        sclass_stats_t& done();

        template <typename tscalar, size_t trank>
        static auto make(const feature_t& feature, dataset_iterator_t<tscalar, trank> it)
        {
            sclass_stats_t stats{feature.classes()};
            for (; it; ++ it)
            {
                if ([[maybe_unused]] const auto [index, given, label] = *it; given)
                {
                    stats += label;
                }
            }
            return stats.done();
        }

        template <typename tscalar, size_t trank>
        auto sample_weights(const feature_t& feature, dataset_iterator_t<tscalar, trank> it) const
        {
            tensor1d_t weights(it.size());
            if (feature.classes() != m_class_counts.size())
            {
                weights.zero();
            }
            else
            {
                scalar_t samples = 0;
                for (; it; ++ it)
                {
                    if (const auto [index, given, label] = *it; given)
                    {
                        samples += 1.0;
                        weights(index) = m_class_weights(static_cast<tensor_size_t>(label));
                    }
                    else
                    {
                        weights(index) = 0.0;
                    }
                }
                if (samples > 0)
                {
                    const auto scale = samples / static_cast<scalar_t>(weights.sum());
                    weights.array() *= scale;
                }
            }
            return weights;
        } // LCOV_EXCL_LINE

        auto samples() const { return m_samples; }
        auto classes() const { return m_class_counts.size(); }
        const auto& class_counts() const { return m_class_counts; }

    private:

        // attributes
        tensor_size_t   m_samples{0};       ///<
        indices_t       m_class_counts;     ///<
        tensor1d_t      m_class_weights;    ///<
    };

    ///
    /// \brief per-feature statistics for multi-label categorical feature values
    ///     (e.g. useful for handling unbalanced classification problems).
    ///
    /// NB: missing feature values are ignored when computing these statistics.
    ///
    class NANO_PUBLIC mclass_stats_t
    {
    public:

        mclass_stats_t() ;
        explicit mclass_stats_t(tensor_size_t classes);

        template <template <typename, size_t> class tstorage, typename tscalar>
        auto& operator+=(const tensor_t<tstorage, tscalar, 1>& class_hits)
        {
            m_samples ++;
            m_class_counts(hash(class_hits)) ++;
            return *this;
        }

        mclass_stats_t& done();

        template <typename tscalar, size_t trank>
        static auto make(const feature_t& feature, dataset_iterator_t<tscalar, trank> it)
        {
            mclass_stats_t stats(feature.classes());
            for (; it; ++ it)
            {
                if ([[maybe_unused]] const auto [index, given, class_hits] = *it; given)
                {
                    stats += class_hits;
                }
            }
            return stats.done();
        }

        template <typename tscalar, size_t trank>
        auto sample_weights(const feature_t& feature, dataset_iterator_t<tscalar, trank> it) const
        {
            tensor1d_t weights(it.size());
            if (feature.classes() * 2 != m_class_counts.size())
            {
                weights.zero();
            }
            else
            {
                scalar_t samples = 0.0;
                for (; it; ++ it)
                {
                    if (const auto [index, given, class_hits] = *it; given)
                    {
                        samples += 1.0;
                        weights(index) = m_class_weights(hash(class_hits));
                    }
                    else
                    {
                        weights(index) = 0.0;
                    }
                }
                if (samples > 0)
                {
                    const auto scale = samples / static_cast<scalar_t>(weights.sum());
                    weights.array() *= scale;
                }
            }
            return weights;
        } // LCOV_EXCL_LINE

        auto samples() const { return m_samples; }
        auto classes() const { return m_class_counts.size() / 2; }
        const auto& class_counts() const { return m_class_counts; }

    private:

        template <template <typename, size_t> class tstorage, typename tscalar>
        static tensor_size_t hash(const tensor_t<tstorage, tscalar, 1>& class_hits)
        {
            const auto hits = class_hits.array().template cast<tensor_size_t>().sum();
            if (hits == 0)
            {
                return 0;
            }
            else if (hits == 1)
            {
                tensor_size_t coeff = 0;
                class_hits.array().maxCoeff(&coeff);
                return 1 + coeff;
            }
            else
            {
                return class_hits.size() + hits - 1;
            }
        }

        // attributes
        tensor_size_t   m_samples{0};       ///<
        indices_t       m_class_counts;     ///<
        tensor1d_t      m_class_weights;    ///<
    };

    ///
    /// \brief per-column statistics for flatten feature values.
    ///
    using flatten_stats_t = scalar_stats_t;

    ///
    /// \brief statistics of the optional target feature values.
    ///
    using targets_stats_t = std::variant
    <
        std::monostate,
        scalar_stats_t,
        sclass_stats_t,
        mclass_stats_t
    >;

    ///
    /// \brief scale an affine transformation so that it undos precisely
    ///     the given scaling of the flatten inputs and of the targets.
    ///
    NANO_PUBLIC void upscale(
        const scalar_stats_t& flatten_stats, scaling_type flatten_scaling,
        const targets_stats_t& targets_stats, scaling_type targets_scaling,
        tensor2d_map_t weights, tensor1d_map_t bias);

    NANO_PUBLIC void upscale(
        const scalar_stats_t& flatten_stats, scaling_type flatten_scaling,
        const scalar_stats_t& targets_stats, scaling_type targets_scaling,
        tensor2d_map_t weights, tensor1d_map_t bias);
}
