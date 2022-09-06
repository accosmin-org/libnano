#pragma once

#include <nano/datasource/stats.h>
#include <nano/generator.h>

namespace nano
{
    ///
    /// \brief wraps a collection of feature generators, potentially of different types.
    ///
    class NANO_PUBLIC dataset_t
    {
    public:
        ///
        /// \brief constructor
        ///
        explicit dataset_t(const datasource_t&);

        ///
        /// \brief register a new feature generator.
        ///
        template <typename tgenerator, typename... tgenerator_args>
        dataset_t& add(tgenerator_args... args)
        {
            static_assert(std::is_base_of_v<generator_t, tgenerator>);
            return this->add(std::make_unique<tgenerator>(args...));
        }

        ///
        /// \brief register a new feature generator.
        ///
        dataset_t& add(rgenerator_t&&);

        ///
        /// \brief returns the total number of features.
        ///
        tensor_size_t features() const;

        ///
        /// \brief returns the feature at a given index.
        ///
        feature_t feature(tensor_size_t ifeature) const;

        ///
        /// \brief returns the number of columns of flatten feature values.
        ///
        tensor_size_t columns() const;

        ///
        /// \brief returns the feature index that produces the given column index.
        ///
        tensor_size_t column2feature(tensor_size_t column) const;

        ///
        /// \brief returns the target feature.
        ///
        feature_t target() const;

        ///
        /// \brief returns the target dimensions.
        ///
        tensor3d_dims_t target_dims() const;

        ///
        /// \brief compute the sample weights from the given target statistics.
        ///
        /// NB: the targets statistics should be computed only on the training dataset,
        ///     while the samples can vary (e.g. validation, testing).
        ///
        tensor1d_t sample_weights(indices_cmap_t samples, const targets_stats_t&) const;

        ///
        /// \brief support for feature importance estimation using drop-column like methods.
        ///
        void undrop() const;
        void drop(tensor_size_t feature) const;

        ///
        /// \brief support for feature importance estimation using sample-permutation like methods.
        ///
        void      unshuffle() const;
        void      shuffle(tensor_size_t feature) const;
        indices_t shuffled(indices_cmap_t samples, tensor_size_t feature) const;

        ///
        /// \brief returns the flatten feature values for all features on a given subset of samples.
        ///
        tensor2d_map_t flatten(indices_cmap_t samples, tensor2d_t& buffer) const;

        ///
        /// \brief returns the targets on a given subset of samples.
        ///
        tensor4d_map_t targets(indices_cmap_t samples, tensor4d_t& buffer) const;

        ///
        /// \brief returns the values of a feature on a given subset of samples.
        ///
        sclass_cmap_t select(indices_cmap_t samples, tensor_size_t feature, sclass_mem_t& buffer) const;
        mclass_cmap_t select(indices_cmap_t samples, tensor_size_t feature, mclass_mem_t& buffer) const;
        scalar_cmap_t select(indices_cmap_t samples, tensor_size_t feature, scalar_mem_t& buffer) const;
        struct_cmap_t select(indices_cmap_t samples, tensor_size_t feature, struct_mem_t& buffer) const;

        // access functions
        auto type() const { return m_datasource.type(); }

        auto samples() const { return m_datasource.samples(); }

        const datasource_t& datasource() const { return m_datasource; }

        const indices_t& sclass_features() const { return m_select_stats.m_sclass_features; }

        const indices_t& mclass_features() const { return m_select_stats.m_mclass_features; }

        const indices_t& scalar_features() const { return m_select_stats.m_scalar_features; }

        const indices_t& struct_features() const { return m_select_stats.m_struct_features; }

    private:
        void                update();
        void                update_stats();
        void                check(tensor_size_t feature) const;
        void                check(indices_cmap_t samples) const;
        const rgenerator_t& byfeature(tensor_size_t feature) const;

        // per column:
        //  - 0: generator index,
        //  - 1: column index within generator,
        //  - 2: offset n_features (up to the current generator)
        using column_mapping_t = tensor_mem_t<tensor_size_t, 2>;

        // per feature:
        //  - 0: generator index,
        //  - 1: feature index within generator,
        //  - 2-4: feature dimensions (dim1, dim2, dim3)
        using feature_mapping_t = tensor_mem_t<tensor_size_t, 2>;

        // per generator:
        //  - 0: number of features
        using generator_mapping_t = tensor_mem_t<tensor_size_t, 2>;

        // attributes
        const datasource_t& m_datasource;        ///<
        rgenerators_t       m_generators;        ///<
        column_mapping_t    m_column_mapping;    ///<
        feature_mapping_t   m_feature_mapping;   ///<
        generator_mapping_t m_generator_mapping; ///<
        select_stats_t      m_select_stats;      ///<
    };
} // namespace nano
