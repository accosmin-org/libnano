#pragma once

#include <nano/dataset/hash.h>
#include <nano/dataset/scaling.h>
#include <nano/feature.h>
#include <nano/generator/storage.h>
#include <nano/tensor.h>

namespace nano
{
    class dataset_t;

    ///
    /// \brief returns the indices of the single-label categorical features (including the generated ones).
    ///
    NANO_PUBLIC indices_t make_sclass_features(const dataset_t&);

    ///
    /// \brief returns the indices of the multi-label categorical features (including the generated ones).
    ///
    NANO_PUBLIC indices_t make_mclass_features(const dataset_t&);

    ///
    /// \brief returns the indices of the scalar continuous features (including the generated ones).
    ///
    NANO_PUBLIC indices_t make_scalar_features(const dataset_t&);

    ///
    /// \brief returns the indices of the structured continuous features (including the generated ones).
    ///
    NANO_PUBLIC indices_t make_struct_features(const dataset_t&);

    ///
    /// \brief statistics for scalar or structured continuous feature values or flatten input features
    ///     (e.g. useful for normalizing inputs and targets).
    ///
    /// NB: missing feature values are ignored when computing these statistics.
    ///
    struct NANO_PUBLIC scalar_stats_t
    {
        ///
        /// \brief constructor
        ///
        explicit scalar_stats_t(tensor_size_t dims = 0);

        ///
        /// \brief returns the per-component statistics of the flatten input features.
        ///
        static scalar_stats_t make_flatten_stats(const dataset_t&, indices_cmap_t samples, tensor_size_t batch = 1000);

        ///
        /// \brief returns the per-component statistics of the flatten targets.
        ///
        static scalar_stats_t make_targets_stats(const dataset_t&, indices_cmap_t samples, tensor_size_t batch = 1000);

        ///
        /// \brief returns the per-component statistics of the given continuous scalar or structured feature.
        ///
        static scalar_stats_t make_feature_stats(const dataset_t&, indices_cmap_t samples, tensor_size_t feature,
                                                 tensor_size_t batch = 1000);

        ///
        /// \brief scale down (for numerical stability) the given flatten feature or target values.
        ///
        void scale(scaling_type, tensor2d_map_t values) const;
        void scale(scaling_type, tensor4d_map_t values) const;

        ///
        /// \brief scale up the given flatten feature or target values.
        ///
        void upscale(scaling_type, tensor2d_map_t values) const;
        void upscale(scaling_type, tensor4d_map_t values) const;

        // attributes
        indices_t  m_samples;                     ///< number of valid samples per component
        tensor1d_t m_min, m_max, m_mean, m_stdev; ///< (minimum, maximum, average, standard deviation) per component
        tensor1d_t m_div_range, m_mul_range;      ///< first order numerically stable denominator per component
        tensor1d_t m_div_stdev, m_mul_stdev;      ///< second order numerically stable denominator per component
    };

    ///
    /// \brief statistics for single-label and multi-label categorical feature values
    ///     (e.g. useful for handling unbalanced classification problems).
    ///
    /// NB: missing feature values are ignored when computing these statistics.
    ///
    struct NANO_PUBLIC xclass_stats_t
    {
        ///
        /// \brief returns the class statistics of the categorical target feature.
        ///
        static xclass_stats_t make_targets_stats(const dataset_t&, indices_cmap_t samples);

        ///
        /// \brief returns the class statistics of the given categorical single-label or multi-label feature.
        ///
        static xclass_stats_t make_feature_stats(const dataset_t&, indices_cmap_t samples, tensor_size_t feature);

        // attributes
        hashes_t   m_class_hashes;   ///< hashes of distinct labeling
        indices_t  m_class_samples;  ///< number of samples for each distict labeling
        indices_t  m_sample_classes; ///< class (hash) index for each sample
        tensor1d_t m_sample_weights; ///< class (hash) weight for each sample
    };

    ///
    /// \brief scale an affine transformation so that it undos precisely
    ///     the given scaling of the flatten inputs and of the targets.
    ///
    NANO_PUBLIC void upscale(const scalar_stats_t& flatten_stats, scaling_type flatten_scaling,
                             const scalar_stats_t& targets_stats, scaling_type targets_scaling, tensor2d_map_t weights,
                             tensor1d_map_t bias);
} // namespace nano
