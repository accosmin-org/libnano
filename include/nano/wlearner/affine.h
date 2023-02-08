#pragma once

#include <nano/wlearner/single.h>

namespace nano
{
    ///
    /// \brief this weak learner is performing an element-wise transformation of the form:
    ///     affine(x) =
    ///     {
    ///         weights * x(feature) + bias, if x(feature) is given
    ///         zero, otherwise (if the feature is missing)
    ///     }
    ///
    ///     where `feature` is the selected continuous feature.
    ///
    /// NB: the discrete and the structured features are skipped during fiting.
    ///
    class NANO_PUBLIC affine_wlearner_t final : public single_feature_wlearner_t
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
        cluster_t do_split(const dataset_t&, const indices_t&) const override;

        ///
        /// \brief @see wlearner_t
        ///
        void do_predict(const dataset_t&, const indices_cmap_t&, tensor4d_map_t) const override;

        ///
        /// \brief @see wlearner_t
        ///
        scalar_t do_fit(const dataset_t&, const indices_t&, const tensor4d_t&) override;

        ///
        /// \brief @see wlearner_t
        ///
        bool try_merge(const rwlearner_t&) override;
    };
} // namespace nano
