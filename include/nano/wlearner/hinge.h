#pragma once

#include <nano/model/enums.h>
#include <nano/wlearner/single.h>

namespace nano
{
    ///
    /// \brief hinge type (see MARS).
    ///
    /// see "Multivariate adaptive regression splines", by Jerome Friedman
    ///
    enum class hinge_type : int32_t
    {
        left = 0, ///< beta * (threshold - x(feature))+       => zero on the right, linear on the left!
        right,    ///< beta * (x(feature) - threshold)+       => zero on the left, linear on the right!
    };

    template <>
    NANO_PUBLIC enum_map_t<hinge_type> enum_string<hinge_type>();

    ///
    /// \brief a hinge is a weak learner that performs the following operation element-wise:
    ///     hinge(x) =
    ///     {
    ///         beta * (threshold - x(feature))+ or
    ///         beta * (x(feature) - threshold)+, if the feature value is given,
    ///         zero, otherwise (if the feature is missing)
    ///     }
    ///
    ///     where feature is the selected continuous feature.
    ///
    /// NB: the discrete and the structured features are skipped during fiting.
    /// NB: the threshold and the hinge direction is shared across outputs, but the predictions can be different.
    /// NB: this weak learner is inspired by the MARS algorithm:
    ///     see "Multivariate adaptive regression splines", by Jerome Friedman
    ///
    class NANO_PUBLIC hinge_wlearner_t final : public single_feature_wlearner_t
    {
    public:
        ///
        /// \brief default constructor
        ///
        hinge_wlearner_t();

        ///
        /// \brief @see configurable_t
        ///
        std::istream& read(std::istream&) override;

        ///
        /// \brief @see configurable_t
        ///
        std::ostream& write(std::ostream&) const override;

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
        /// \brief returns the chosen hinge type (left or right).
        ///
        auto hinge() const { return m_hinge; }

        ///
        /// \brief returns the chosen feature value threshold.
        ///
        auto threshold() const { return m_threshold; }

    private:
        // attributes
        scalar_t   m_threshold{0};            ///< threshold
        hinge_type m_hinge{hinge_type::left}; ///<
    };
} // namespace nano
