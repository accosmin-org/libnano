#pragma once

#include <nano/wlearner/single.h>

namespace nano
{
    ///
    /// \brief a decision stump is a weak learner that compares the value of a selected feature with a threshold:
    ///     stump(x) =
    ///     {
    ///         tables[0], if x(feature) is given and x(feature) < threshold
    ///         tables[1], if x(feature) is given and x(feature) >= threshold
    ///         zero, otherwise (if the feature is missing)
    ///     }
    ///
    ///     where feature is the selected continuous feature.
    ///
    /// NB: the discrete features and the missing feature values are skipped during fiting.
    /// NB: the threshold is shared across outputs, but the predictions can be different.
    ///
    class NANO_PUBLIC stump_wlearner_t final : public single_feature_wlearner_t
    {
    public:
        using wlearner_t::split;

        ///
        /// \brief default constructor
        ///
        stump_wlearner_t();

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
        cluster_t        do_split(const dataset_t&, const indices_t&) const override;
        static cluster_t split(const dataset_t&, const indices_t&, tensor_size_t feature, scalar_t threshold);

        ///
        /// \brief @see wlearner_t
        ///
        void do_predict(const dataset_t&, const indices_cmap_t&, tensor4d_map_t) const override;

        ///
        /// \brief @see wlearner_t
        ///
        scalar_t do_fit(const dataset_t&, const indices_t&, const tensor4d_t&) override;

        ///
        /// \brief returns the chosen feature value threshold.
        ///
        auto threshold() const { return m_threshold; }

    private:
        // attributes
        scalar_t m_threshold{0}; ///< threshold
    };
} // namespace nano
