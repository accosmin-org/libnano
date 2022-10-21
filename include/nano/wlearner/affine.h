#pragma once

#include <nano/wlearner/single.h>

namespace nano
{
    ///
    /// \brief this weak learner is performing an element-wise transformation of the form:
    ///     affine1(x) =
    ///     {
    ///         weights[0] * x(feature) + weights[1], if x(feature) is given
    ///         zero, otherwise (if the feature is missing)
    ///     }
    ///
    ///     where `feature` is the selected continuous feature.
    ///
    /// NB: the discrete features and the missing feature values are skipped during fiting.
    ///
    class affine_wlearner_t final : public single_feature_wlearner_t
    {
    public:
        ///
        /// \brief default constructor
        ///
        affine_wlearner_t();

        ///
        /// \brief @see clonable_t
        ///
        rwlearner_t clone() const override;

        ///
        /// \brief @see wlearner_t
        ///
        cluster_t split(const dataset_t&, const indices_t&) const override;

        ///
        /// \brief @see wlearner_t
        ///
        void predict(const dataset_t&, const indices_cmap_t&, tensor4d_map_t) const override;

        ///
        /// \brief @see wlearner_t
        ///
        scalar_t fit(const dataset_t&, const indices_t&, const tensor4d_t&) override;
    };
} // namespace nano
