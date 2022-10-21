#pragma once

#include <nano/wlearner.h>

namespace nano
{
    ///
    /// \brief interface for weak learner that are parametrized by a single feature,
    ///     that is either continuous or discrete.
    ///
    /// NB: the incompatible features and the missing feature values are skipped during fiting.
    ///
    class NANO_PUBLIC single_feature_wlearner_t : public wlearner_t
    {
    public:
        ///
        /// \brief default constructor
        ///
        explicit single_feature_wlearner_t(string_t id);

        ///
        /// \brief @see estimator_t
        ///
        std::istream& read(std::istream&) override;

        ///
        /// \brief @see estimator_t
        ///
        std::ostream& write(std::ostream&) const override;

        ///
        /// \brief @see wlearner_t
        ///
        void scale(const vector_t&) override;

        ///
        /// \brief @see wlearner_t
        ///
        indices_t features() const override;

        ///
        /// \brief returns the index of the selected feature.
        ///
        auto feature() const { return m_feature; }

        ///
        /// \brief returns the table of coefficients.
        ///
        const auto& tables() const { return m_tables; }

        ///
        /// \brief returns the coefficients at the given index.
        ///
        auto vector(tensor_size_t i) const { return m_tables.vector(i); }

    protected:
        void set(tensor_size_t feature, const tensor4d_t& tables);

    private:
        // attributes
        tensor_size_t m_feature{-1}; ///< index of the selected feature
        tensor4d_t    m_tables;      ///< coefficients (:, #outputs)
    };
} // namespace nano
