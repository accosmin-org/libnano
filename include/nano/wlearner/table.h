#pragma once

#include <nano/wlearner/single.h>

namespace nano
{
    ///
    /// \brief a (look-up) table is a weak learner that returns a constant for each discrete feature value:
    ///     table(x) =
    ///     {
    ///         tables[x(feature)], if x(feature) is given
    ///         zero, otherwise (if the feature is missing)
    ///     }
    ///
    ///     where feature is the selected discrete feature.
    ///
    /// NB: the continuous features and the missing feature values are skipped during fiting.
    ///
    class NANO_PUBLIC table_wlearner_t final : public single_feature_wlearner_t
    {
    public:
        ///
        /// \brief default constructor
        ///
        table_wlearner_t();

        ///
        /// \brief @see clonable_t
        ///
        rwlearner_t clone() const override;

        ///
        /// \brief @see wlearner_t
        ///
        cluster_t        split(const dataset_t&, const indices_t&) const override;
        static cluster_t split(const dataset_t&, const indices_t&, tensor_size_t feature, tensor_size_t classes);

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
