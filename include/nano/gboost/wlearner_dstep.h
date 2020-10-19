#pragma once

#include <nano/gboost/wlearner_feature1.h>

namespace nano
{
    class wlearner_dstep_t;

    template <>
    struct factory_traits_t<wlearner_dstep_t>
    {
        static string_t id() { return "dstep"; }
        static string_t description() { return "discrete step weak learner"; }
    };

    ///
    /// \brief a discrete step weak learner that returns a constant for a chosen discrete feature value:
    ///     dstep(x) =
    ///     {
    ///         beta, if x(feature) is given and x(feature) == fvalue,
    ///         beta, if x(feature) is given and x(feature) != fvalue,
    ///         zero, otherwise (if the feature is missing)
    ///     }
    ///
    ///     where feature is the selected discrete feature.
    ///
    /// NB: the continuous features and the missing feature values are skipped during fiting.
    /// NB: this weak learner is inspired by the MARS algorithm extend to handle discrete/categorical features:
    ///     see "Multivariate adaptive regression splines", by Jerome Friedman
    ///
    class NANO_PUBLIC wlearner_dstep_t final : public wlearner_feature1_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        wlearner_dstep_t();

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
        auto fvalue() const { return m_fvalue; }
        auto fvalues() const { return tables().size<0>(); }

    private:

        // attributes
        tensor_size_t   m_fvalue{-1};    ///< the chosen feature value
    };
}
