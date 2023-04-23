#pragma once

#include <nano/dataset.h>
#include <nano/dataset/stats.h>
#include <nano/generator/storage.h>

namespace nano
{
///
/// \brief callbacks useful for dense models with the following signature:
///     (tensor_size_t sample_range, size_t thread_number, target_values)
///     (tensor_size_t sample_range, size_t thread_number, flatten_feature_values, target_values)
///
using targets_callback_t         = std::function<void(tensor_range_t, size_t, tensor4d_cmap_t)>;
using flatten_callback_t         = std::function<void(tensor_range_t, size_t, tensor2d_cmap_t)>;
using flatten_targets_callback_t = std::function<void(tensor_range_t, size_t, tensor2d_cmap_t, tensor4d_cmap_t)>;

///
/// \brief callbacks useful for feature selection-based models with the following signature:
///     (tensor_size_t feature_index, size_t thread_number, feature_values)
///
using sclass_callback_t = std::function<void(tensor_size_t, size_t, sclass_cmap_t)>;
using mclass_callback_t = std::function<void(tensor_size_t, size_t, mclass_cmap_t)>;
using scalar_callback_t = std::function<void(tensor_size_t, size_t, scalar_cmap_t)>;
using struct_callback_t = std::function<void(tensor_size_t, size_t, struct_cmap_t)>;

///
/// \brief base iterator to loop through generated input and target feature values.
///
class NANO_PUBLIC base_dataset_iterator_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit base_dataset_iterator_t(const dataset_t&);

    ///
    /// \brief returns the maximum number of threads available for processing.
    ///
    size_t concurrency() const;

    ///
    /// \brief returns the wrapped feature dataset.
    ///
    const dataset_t& dataset() const { return m_dataset; }

protected:
    template <typename toperator>
    void map(const tensor_size_t elements, const toperator& op) const
    {
        m_dataset.thread_pool().map(elements, op);
    }

    template <typename toperator>
    void map(const tensor_size_t elements, const tensor_size_t chunksize, const toperator& op) const
    {
        m_dataset.thread_pool().map(elements, chunksize, op);
    }

private:
    // attributes
    const dataset_t& m_dataset; ///<
};

///
/// \brief iterator to loop through flatten target values
///     useful for training and evaluating dense models.
///
/// the feature and the target values can be:
///     - cached to speed-up access (useful if slow to compute on the fly)
///     - scaled to speed-up training by improving the convergence rate of the solver.
///
class NANO_PUBLIC targets_iterator_t : public base_dataset_iterator_t
{
public:
    ///
    /// \brief constructor
    ///
    targets_iterator_t(const dataset_t&, indices_cmap_t samples);

    ///
    /// \brief returns true if the target values can be cached in memory in the given number of bytes.
    ///
    bool cache_targets(tensor_size_t max_bytes);

    ///
    /// \brief loop through targets with the following callback:
    ///     - op(tensor_range_t sample_range, size_t thread_number, tensor4d_cmap_t targets)
    ///
    void loop(const targets_callback_t&) const;

    ///
    /// \brief change parameters.
    ///
    void batch(tensor_size_t);
    void scaling(scaling_type);

    ///
    /// \brief returns the batch size in number of samples.
    ///
    auto batch() const { return m_batch; }

    ///
    /// \brief returns the feature value scaling method.
    ///
    auto scaling() const { return m_scaling; }

    ///
    /// \brief returns the sample indices.
    ///
    const auto& samples() const { return m_samples; }

    ///
    /// \brief returns statistics of the flatten targets.
    ///
    const auto& targets_stats() const { return m_targets_stats; }

protected:
    tensor4d_cmap_t targets(tensor4d_map_t) const;
    tensor4d_cmap_t targets(size_t tnum, const tensor_range_t& range) const;

private:
    using buffers_t = std::vector<tensor4d_t>;

    // attributes
    indices_t         m_samples;                     ///<
    tensor_size_t     m_batch{100};                  ///<
    scaling_type      m_scaling{scaling_type::none}; ///< scaling method for flatten feature values & targets
    tensor4d_t        m_targets;                     ///< cached targets values
    scalar_stats_t    m_targets_stats;               ///< statistics for targets
    mutable buffers_t m_targets_buffers;             ///< per-thread buffer
};

///
/// \brief iterator to loop through flatten feature values and target values
///     useful for training and evaluating dense models.
///
class NANO_PUBLIC flatten_iterator_t : public targets_iterator_t
{
public:
    using targets_iterator_t::loop;

    ///
    /// \brief constructor
    ///
    flatten_iterator_t(const dataset_t&, indices_cmap_t samples);

    ///
    /// \brief returns true if the flatten feature values can be cached in memory in the given number of bytes.
    ///
    bool cache_flatten(tensor_size_t max_bytes);

    ///
    /// \brief loop through flatten feature values with the following callback:
    ///     - op(tensor_range_t sample_range, size_t thread_number, tensor2d_cmap_t flatten)
    ///
    void loop(const flatten_callback_t&) const;

    ///
    /// \brief loop through flatten feature values and the associated targets with the following callback:
    ///     - op(tensor_range_t sample_range, size_t thread_number, tensor2d_cmap_t flatten, tensor4d_cmap_t
    ///     targets)
    ///
    void loop(const flatten_targets_callback_t&) const;

    ///
    /// \brief returns statistics of the flatten feature values.
    ///
    const auto& flatten_stats() const { return m_flatten_stats; }

private:
    tensor2d_cmap_t flatten(tensor2d_map_t) const;
    tensor2d_cmap_t flatten(size_t tnum, const tensor_range_t& range) const;

    using buffers_t = std::vector<tensor2d_t>;

    // attributes
    scalar_stats_t    m_flatten_stats;   ///< statistics for flatten feature values
    mutable buffers_t m_flatten_buffers; ///< per-thread buffer
    tensor2d_t        m_flatten;         ///< cached feature values
};

///
/// \brief iterator to loop through features of a particular type
///     useful for feature selection-based models.
///
class NANO_PUBLIC select_iterator_t : public base_dataset_iterator_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit select_iterator_t(const dataset_t&);

    ///
    /// \brief loop through all features of the compatible type with the following callback:
    ///     - op(tensor_size_t feature_index, size_t thread_number, ... feature_values)
    ///
    void loop(indices_cmap_t samples, const sclass_callback_t&) const;
    void loop(indices_cmap_t samples, const mclass_callback_t&) const;
    void loop(indices_cmap_t samples, const scalar_callback_t&) const;
    void loop(indices_cmap_t samples, const struct_callback_t&) const;

    ///
    /// \brief loop through the samples of the given feature compatible type with the following callback:
    ///     - op(tensor_size_t feature_index, size_t thread_number, ... feature_values)
    ///
    void loop(indices_cmap_t samples, tensor_size_t feature, const sclass_callback_t&) const;
    void loop(indices_cmap_t samples, tensor_size_t feature, const mclass_callback_t&) const;
    void loop(indices_cmap_t samples, tensor_size_t feature, const scalar_callback_t&) const;
    void loop(indices_cmap_t samples, tensor_size_t feature, const struct_callback_t&) const;

    ///
    /// \brief loop through the given features of the compatible type with the following callback:
    ///     - op(tensor_size_t feature_index, size_t thread_number, ... feature_values)
    ///
    void loop(indices_cmap_t samples, indices_cmap_t features, const sclass_callback_t&) const;
    void loop(indices_cmap_t samples, indices_cmap_t features, const mclass_callback_t&) const;
    void loop(indices_cmap_t samples, indices_cmap_t features, const scalar_callback_t&) const;
    void loop(indices_cmap_t samples, indices_cmap_t features, const struct_callback_t&) const;

private:
    struct buffer_t
    {
        sclass_mem_t m_sclass;
        mclass_mem_t m_mclass;
        scalar_mem_t m_scalar;
        struct_mem_t m_struct;
    };

    using buffers_t = std::vector<buffer_t>;

    // attributes
    mutable buffers_t m_buffers;         ///< per-thread buffer
    indices_t         m_sclass_features; ///< indices of the single-label features
    indices_t         m_mclass_features; ///< indices of the multi-label features
    indices_t         m_scalar_features; ///< indices of the scalar features
    indices_t         m_struct_features; ///< indices of structured features
};
} // namespace nano
