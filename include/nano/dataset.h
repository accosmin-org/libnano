#pragma once

#include <nano/core/parallel.h>
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
    explicit dataset_t(const datasource_t&, size_t threads = parallel::pool_t::max_size());

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
    const feature_t& target() const { return m_target; }

    ///
    /// \brief returns the target dimensions.
    ///
    tensor3d_dims_t target_dims() const;

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
    indices_t shuffled(tensor_size_t feature, indices_cmap_t samples) const;

    ///
    /// \brief returns the flatten feature values for all features on a given subset of samples.
    ///
    tensor2d_map_t flatten(indices_cmap_t samples, tensor2d_t& buffer) const;

    ///
    /// \brief returns the flatten targets on a given subset of samples.
    ///
    tensor4d_map_t targets(indices_cmap_t samples, tensor4d_t& buffer) const;

    ///
    /// \brief returns the values of the target on a given subset of samples.
    ///
    sclass_cmap_t select(indices_cmap_t samples, sclass_mem_t& buffer) const;
    mclass_cmap_t select(indices_cmap_t samples, mclass_mem_t& buffer) const;
    scalar_cmap_t select(indices_cmap_t samples, scalar_mem_t& buffer) const;
    struct_cmap_t select(indices_cmap_t samples, struct_mem_t& buffer) const;

    ///
    /// \brief returns the values of a feature on a given subset of samples.
    ///
    sclass_cmap_t select(indices_cmap_t samples, tensor_size_t feature, sclass_mem_t& buffer) const;
    mclass_cmap_t select(indices_cmap_t samples, tensor_size_t feature, mclass_mem_t& buffer) const;
    scalar_cmap_t select(indices_cmap_t samples, tensor_size_t feature, scalar_mem_t& buffer) const;
    struct_cmap_t select(indices_cmap_t samples, tensor_size_t feature, struct_mem_t& buffer) const;

    ///
    /// \brief returns the appropriate mathine learning task (by inspecting the target feature).
    ///
    task_type type() const { return m_datasource.type(); }

    ///
    /// \brief returns the total number of samples.
    ///
    tensor_size_t samples() const { return m_datasource.samples(); }

    ///
    /// \brief returns the maximum number of threads available for processing.
    ///
    size_t concurrency() const { return m_pool->size(); }

    ///
    /// \brief returns the builtin thread pool.
    ///
    parallel::pool_t& thread_pool() const { return *m_pool; }

    ///
    /// \brief returns the original data source.
    ///
    const datasource_t& datasource() const { return m_datasource; }

private:
    void                update();
    void                check(tensor_size_t feature) const;
    void                check(indices_cmap_t samples) const;
    const rgenerator_t& byfeature(tensor_size_t feature) const;

    // per column:
    //  - 0: generator index,
    //  - 1: column index within generator,
    //  - 2: feature index
    using column_mapping_t = tensor_mem_t<tensor_size_t, 2>;

    // per feature:
    //  - 0: generator index,
    //  - 1: feature index within generator,
    //  - 2-4: feature dimensions (dim1, dim2, dim3),
    using feature_mapping_t = tensor_mem_t<tensor_size_t, 2>;

    // per generator:
    //  - 0: number of columns
    using generator_mapping_t = tensor_mem_t<tensor_size_t, 2>;

    using rtpool_t = std::unique_ptr<parallel::pool_t>;

    // attributes
    const datasource_t& m_datasource;        ///<
    rgenerators_t       m_generators;        ///<
    column_mapping_t    m_column_mapping;    ///<
    feature_mapping_t   m_feature_mapping;   ///<
    generator_mapping_t m_generator_mapping; ///<
    feature_t           m_target;            ///<
    rtpool_t            m_pool;              ///< thread pool to speed-up feature generation
};
} // namespace nano
