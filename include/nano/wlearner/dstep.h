#pragma once

#include <nano/wlearner/single.h>

namespace nano
{
    ///
    /// \brief a discrete step weak learner that returns a constant for a chosen discrete feature value:
    ///     dstep(x) =
    ///     {
    ///         beta, if x(feature) is given and x(feature) == fvalue,
    ///         zero, if x(feature) is given and x(feature) != fvalue,
    ///         zero, otherwise (if the feature is missing)
    ///     }
    ///
    ///     where feature is the selected discrete feature.
    ///
    /// NB: the continuous features and the missing feature values are skipped during fiting.
    /// NB: this weak learner is inspired by the MARS algorithm extend to handle discrete/categorical features:
    ///     see "Multivariate adaptive regression splines", by Jerome Friedman
    ///
    class NANO_PUBLIC dstep_wlearner_t final : public single_feature_wlearner_t
    {
    public:
        ///
        /// \brief default constructor
        ///
        dstep_wlearner_t();

        ///
        /// \brief @see estimator_t
        ///
        std::istream& read(std::istream&) override;

        ///
        /// \brief @see estimator_t
        ///
        std::ostream& write(std::ostream&) const override;

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

        ///
        /// \brief returns the chosen feature value.
        ///
        auto fvalue() const { return m_fvalue; }

    private:
        // attributes
        tensor_size_t m_fvalue{-1}; ///< the chosen feature value
    };
} // namespace nano
