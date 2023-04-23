#pragma once

#include <nano/datasource.h>
#include <nano/datasource/iterator.h>
#include <nano/generator/storage.h>

namespace nano
{
class generator_t;
using rgenerator_t  = std::unique_ptr<generator_t>;
using rgenerators_t = std::vector<rgenerator_t>;

///
/// \brief type of generated features.
///
enum class generator_type
{
    mclass,
    sclass,
    scalar,
    structured,
};

struct generated_sclass_t
{
    static constexpr auto generated_type = generator_type::sclass;
};

struct generated_mclass_t
{
    static constexpr auto generated_type = generator_type::mclass;
};

struct generated_scalar_t
{
    static constexpr auto generated_type = generator_type::scalar;
};

struct generated_struct_t
{
    static constexpr auto generated_type = generator_type::structured;
};

///
/// \brief generate features from a given collection of samples of a dataset (e.g. the training samples).
///
/// NB: optional inputs are supported.
/// NB: the targets cannot be optional if defined.
/// NB: the inputs can be continuous (scalar), structured (3D tensors) or categorical.
/// NB: the inputs and the targets are generated on the fly by default, but they can be cached if possible.
///
/// NB: missing feature values are filled:
///     - with NaN/-1 depending if continuous/categorical respectively,
///         if accessing one feature at a time (e.g. feature selection models)
///
///     - with NaN,
///         if accessing all features at once as flatten (e.g. linear models).
///
class NANO_PUBLIC generator_t : public clonable_t<generator_t>
{
public:
    ///
    /// \brief constructor.
    ///
    explicit generator_t(string_t id);

    ///
    /// \brief returns the available implementations.
    ///
    static factory_t<generator_t>& all();

    ///
    /// \brief process the whole dataset:
    ///     - to decide which features to generate and
    ///     - to generate features fast when needed (if needed).
    ///
    virtual void fit(const datasource_t&);

    ///
    /// \brief returns the total number of generated features.
    ///
    virtual tensor_size_t features() const = 0;

    ///
    /// \brief returns the description of the given feature index.
    ///
    virtual feature_t feature(tensor_size_t feature) const = 0;

    ///
    /// \brief toggle dropping of features, useful for feature importance analysis.
    ///
    void undrop();
    void drop(tensor_size_t feature);

    ///
    /// \brief toggle sample permutation of features, useful for feature importance analysis.
    ///
    void      unshuffle();
    void      shuffle(tensor_size_t feature);
    indices_t shuffled(indices_cmap_t samples, tensor_size_t feature) const;

    ///
    /// \brief computes the values of the given feature and samples,
    ///     useful for training and evaluating ML models that perform feature selection
    ///     (e.g. gradient boosting).
    ///
    virtual void select(indices_cmap_t samples, tensor_size_t feature, sclass_map_t) const = 0;
    virtual void select(indices_cmap_t samples, tensor_size_t feature, mclass_map_t) const = 0;
    virtual void select(indices_cmap_t samples, tensor_size_t feature, scalar_map_t) const = 0;
    virtual void select(indices_cmap_t samples, tensor_size_t feature, struct_map_t) const = 0;

    ///
    /// \brief computes the values of all features for the given samples,
    ///     useful for training and evaluating ML model that map densely continuous inputs to targets
    ///     (e.g. linear models, MLPs).
    ///
    virtual void flatten(indices_cmap_t samples, tensor2d_map_t, tensor_size_t column) const = 0;

protected:
    const datasource_t& datasource() const;

    void allocate(tensor_size_t features);
    bool should_drop(tensor_size_t feature) const;
    bool should_shuffle(tensor_size_t feature) const;

    static void flatten_dropped(tensor2d_map_t storage, tensor_size_t column, tensor_size_t colsize);

    template <size_t input_rank1, typename toperator>
    void iterate(indices_cmap_t samples, const tensor_size_t ifeature, const tensor_size_t ioriginal,
                 const toperator& op) const
    {
        const auto visitor = [&](const auto&, const auto& data, const auto& mask)
        {
            if (should_shuffle(ifeature))
            {
                loop_samples<input_rank1>(data, mask, shuffled(samples, ifeature), op);
            }
            else
            {
                loop_samples<input_rank1>(data, mask, samples, op);
            }
        };
        datasource().visit_inputs(ioriginal, visitor);
    }

    template <size_t input_rank1, size_t input_rank2, typename toperator>
    void iterate(indices_cmap_t samples, const tensor_size_t ifeature, const tensor_size_t ioriginal1,
                 const tensor_size_t ioriginal2, const toperator& op) const
    {
        const auto visitor = [&](const auto& data1, const auto& mask1, const auto& data2, const auto& mask2)
        {
            if (should_shuffle(ifeature))
            {
                loop_samples<input_rank1, input_rank2>(data1, mask1, data2, mask2, shuffled(samples, ifeature), op);
            }
            else
            {
                loop_samples<input_rank1, input_rank2>(data1, mask1, data2, mask2, samples, op);
            }
        };

        const auto& ds = datasource();
        ds.visit_inputs(ioriginal1,
                        [&](const auto&, const auto& data1, const auto& mask1)
                        {
                            ds.visit_inputs(ioriginal2, [&](const auto&, const auto& data2, const auto& mask2)
                                            { visitor(data1, mask1, data2, mask2); });
                        });
    }

private:
    // per feature:
    //  - 0: flags - 0 - default, 1 - to drop, 2 - to shuffle
    using feature_infos_t = tensor_mem_t<uint8_t, 1>;

    // per feature:
    //  - random number generator to use to shuffle the given samples
    using feature_rands_t = std::vector<rng_t>;

    // attributes
    const datasource_t* m_datasource{nullptr}; ///<
    feature_infos_t     m_feature_infos;       ///<
    feature_rands_t     m_feature_rands;       ///<
};
} // namespace nano
