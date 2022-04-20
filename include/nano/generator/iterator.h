#pragma once

#include <nano/dataset/stats.h>
#include <nano/core/execution.h>
#include <nano/generator/generator.h>

namespace nano
{
    class dataset_generator_t;

    ///
    /// \brief callbacks useful for dense models with the following signature:
    ///     (tensor_size_t sample_range, size_t thread_number, target_values)
    ///     (tensor_size_t sample_range, size_t thread_number, flatten_feature_values, target_values)
    ///
    /// NB: the thread number is set to zero if the execution policy is sequential.
    using targets_callback_t = std::function<void(tensor_range_t, size_t, tensor4d_cmap_t)>;
    using flatten_callback_t = std::function<void(tensor_range_t, size_t, tensor2d_cmap_t)>;
    using flatten_targets_callback_t = std::function<void(tensor_range_t, size_t, tensor2d_cmap_t, tensor4d_cmap_t)>;

    ///
    /// \brief callbacks useful for feature selection-based models with the following signature:
    ///     (tensor_size_t feature_index, size_t thread_number, feature_values)
    ///
    /// NB: the thread number is set to zero if the execution policy is sequential.
    using sclass_callback_t = std::function<void(tensor_size_t, size_t, sclass_cmap_t)>;
    using mclass_callback_t = std::function<void(tensor_size_t, size_t, mclass_cmap_t)>;
    using scalar_callback_t = std::function<void(tensor_size_t, size_t, scalar_cmap_t)>;
    using struct_callback_t = std::function<void(tensor_size_t, size_t, struct_cmap_t)>;

    ///
    /// \brief iterator to loop through target values
    ///     in single and multi-threaded scenarious, useful for training and evaluating dense models.
    ///
    /// the feature and the target values can be:
    ///     - cached to speed-up access (useful if slow to compute on the fly)
    ///     - scaled to speed-up training by improving the convergence rate of the solver.
    ///
    class NANO_PUBLIC targets_iterator_t
    {
    public:

        ///
        /// \brief constructor
        ///
        targets_iterator_t(const dataset_generator_t&, indices_cmap_t samples);

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
        void execution(execution_type);

        ///
        /// \brief access functions.
        ///
        auto batch() const { return m_batch; }
        auto scaling() const { return m_scaling; }
        auto execution() const { return m_execution; }
        auto concurrency() const { return m_targets_buffers.size(); }
        const auto& samples() const { return m_samples; }
        const auto& generator() const { return m_generator; }
        const auto& targets_stats() const { return m_targets_stats; }

    protected:

        tensor4d_cmap_t targets(tensor4d_map_t) const;
        tensor4d_cmap_t targets(size_t tnum, const tensor_range_t& range) const;

    private:

        targets_stats_t make_targets_stats() const;

        using buffers_t = std::vector<tensor4d_t>;
        using dgenerator_t = dataset_generator_t;

        // attributes
        const dgenerator_t& m_generator;                    ///<
        indices_t           m_samples;                      ///<
        tensor_size_t       m_batch{100};                   ///<
        execution_type      m_execution{execution_type::par};///<
        scaling_type        m_scaling{scaling_type::none};  ///< scaling method for flatten feature values & targets
        tensor4d_t          m_targets;                      ///< cached targets values
        targets_stats_t     m_targets_stats;                ///< statistics for targets
        mutable buffers_t   m_targets_buffers;              ///< per-thread buffer
    };

    ///
    /// \brief iterator to loop through flatten feature values and target values
    ///     in single and multi-threaded scenarious, useful for training and evaluating dense models.
    ///
    class NANO_PUBLIC flatten_iterator_t : public targets_iterator_t
    {
    public:

        using targets_iterator_t::loop;

        ///
        /// \brief constructor
        ///
        flatten_iterator_t(const dataset_generator_t&, indices_cmap_t samples);

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
        ///     - op(tensor_range_t sample_range, size_t thread_number, tensor2d_cmap_t flatten, tensor4d_cmap_t targets)
        ///
        void loop(const flatten_targets_callback_t&) const;

        ///
        /// \brief access functions.
        ///
        const auto& flatten_stats() const { return m_flatten_stats; }

    private:

        flatten_stats_t make_flatten_stats() const;
        tensor2d_cmap_t flatten(tensor2d_map_t) const;
        tensor2d_cmap_t flatten(size_t tnum, const tensor_range_t& range) const;

        using buffers_t = std::vector<tensor2d_t>;

        // attributes
        flatten_stats_t     m_flatten_stats;                ///< statistics for flatten feature values
        mutable buffers_t   m_flatten_buffers;              ///< per-thread buffer
        tensor2d_t          m_flatten;                      ///< cached feature values
    };

    ///
    /// \brief iterator to loop through features of a particular type
    ///     in single and multi-threaded scenarious, useful for feature selection-based models.
    ///
    class NANO_PUBLIC select_iterator_t
    {
    public:

        ///
        /// \brief constructor
        ///
        explicit select_iterator_t(const dataset_generator_t&);

        ///
        /// \brief loop through all features of the same type with the following callback:
        ///     - op(tensor_size_t feature_index, size_t thread_number, ... feature_values)
        ///
        void loop(indices_cmap_t samples, const sclass_callback_t&) const;
        void loop(indices_cmap_t samples, const mclass_callback_t&) const;
        void loop(indices_cmap_t samples, const scalar_callback_t&) const;
        void loop(indices_cmap_t samples, const struct_callback_t&) const;

        ///
        /// \brief loop through the given features of the same type with the following callback:
        ///     - op(tensor_size_t feature_index, size_t thread_number, ... feature_values)
        ///
        void loop(indices_cmap_t samples, indices_cmap_t features, const sclass_callback_t&) const;
        void loop(indices_cmap_t samples, indices_cmap_t features, const mclass_callback_t&) const;
        void loop(indices_cmap_t samples, indices_cmap_t features, const scalar_callback_t&) const;
        void loop(indices_cmap_t samples, indices_cmap_t features, const struct_callback_t&) const;

        ///
        /// \brief change parameters.
        ///
        void execution(execution_type);

        ///
        /// \brief access functions.
        ///
        auto concurrency() const { return m_buffers.size(); }
        const auto& generator() const { return m_generator; }

    private:

        struct buffer_t
        {
            sclass_mem_t        m_sclass;
            mclass_mem_t        m_mclass;
            scalar_mem_t        m_scalar;
            struct_mem_t        m_struct;
        };

        using buffers_t = std::vector<buffer_t>;
        using dgenerator_t = dataset_generator_t;

        // attributes
        const dgenerator_t& m_generator;                        ///<
        execution_type      m_execution{execution_type::par};   ///<
        mutable buffers_t   m_buffers;                          ///< per-thread buffer
    };
}
