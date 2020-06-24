#pragma once

#include <nano/gboost/wlearner.h>

namespace nano
{
    class wlearner_stump_t;

    template <>
    struct factory_traits_t<wlearner_stump_t>
    {
        static string_t id() { return "stump"; }
        static string_t description() { return "decision stump weak learner"; }
    };

    ///
    ///
    /// \brief a stump is a weak learner that compares the value of a selected feature with a threshold:
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
    ///
    class NANO_PUBLIC wlearner_stump_t : public wlearner_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        wlearner_stump_t() = default;

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
        [[nodiscard]] std::ostream& print(std::ostream&) const override;

        ///
        /// \brief @see wlearner_t
        ///
        [[nodiscard]] rwlearner_t clone() const override;

        ///
        /// \brief @see wlearner_t
        ///
        [[nodiscard]] tensor3d_dim_t odim() const override;

        ///
        /// \brief @see wlearner_t
        ///
        void scale(const vector_t&) override;

        ///
        /// \brief @see wlearner_t
        ///
        void predict(const dataset_t&, fold_t, tensor_range_t, tensor4d_map_t&& outputs) const override;

        ///
        /// \brief @see wlearner_t
        ///
        [[nodiscard]] scalar_t fit(const dataset_t&, fold_t, const tensor4d_t& gradients, const indices_t&) override;

        ///
        /// \brief @see wlearner_t
        ///
        [[nodiscard]] cluster_t split(const dataset_t&, fold_t, const indices_t&) const override;

        ///
        /// \brief @see wlearner_t
        ///
        [[nodiscard]] indices_t features() const override;

        ///
        /// \brief access functions
        ///
        [[nodiscard]] auto feature() const { return m_feature; }
        [[nodiscard]] auto threshold() const { return m_threshold; }
        [[nodiscard]] const auto& tables() const { return m_tables; }

    private:

        void compatible(const dataset_t&) const;

        // attributes
        tensor_size_t   m_feature{-1};          ///< index of the selected feature
        scalar_t        m_threshold{0};         ///< threshold
        tensor4d_t      m_tables;               ///< (2, #outputs) - predictions below and above the threshold
    };
}
