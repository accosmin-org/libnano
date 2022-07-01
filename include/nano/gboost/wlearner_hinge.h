#pragma once

#include <nano/gboost/wlearner_feature1.h>

namespace nano
{
    class wlearner_hinge_t;

    template <>
    struct factory_traits_t<wlearner_hinge_t>
    {
        static string_t id() { return "hinge"; }

        static string_t description() { return "hinge weak learner"; }
    };

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
    /// NB: the discrete features and the missing feature values are skipped during fiting.
    /// NB: the threshold is shared across outputs, but the predictions and the hinge directions can be different.
    /// NB: this weak learner is inspired by the MARS algorithm:
    ///     see "Multivariate adaptive regression splines", by Jerome Friedman
    ///
    class NANO_PUBLIC wlearner_hinge_t final : public wlearner_feature1_t
    {
    public:
        ///
        /// \brief default constructor
        ///
        wlearner_hinge_t();

        ///
        /// \brief @see wlearner_t
        ///
        void read(std::istream&) override;

        ///
        /// \brief @see wlearner_t
        ///
        void write(std::ostream&) const override;

        ///
        /// \brief @see wlearner_t
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
        /// \brief access functions
        ///
        auto hinge() const { return m_hinge; }

        auto threshold() const { return m_threshold; }

    private:
        // attributes
        scalar_t      m_threshold{0};               ///< threshold
        ::nano::hinge m_hinge{::nano::hinge::left}; ///<
    };
} // namespace nano
