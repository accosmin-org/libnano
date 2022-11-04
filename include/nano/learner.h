#pragma once

#include <nano/core/estimator.h>
#include <nano/dataset.h>

namespace nano
{
    ///
    /// \brief base class for machine learning models (e.g. strong and weak learners) useful
    ///     for fingerprinting the fitted dataset and checking its compatibility with the evaluation dataset.
    ///
    class NANO_PUBLIC learner_t : public estimator_t
    {
    public:
        ///
        /// \brief constructor
        ///
        learner_t();

        ///
        /// \brief @see estimator_t
        ///
        std::istream& read(std::istream&) override;

        ///
        /// \brief @see estimator_t
        ///
        std::ostream& write(std::ostream&) const override;

        ///
        /// \brief fit the given dataset and store its fingerprint.
        ///
        void fit(const dataset_t&);

        ///
        /// \brief check if the fitted dataset is compatible with the given one and throws an exception if not the case.
        ///
        void critical_compatible(const dataset_t&) const;

    private:
        // attributes
        features_t m_inputs; ///< input features
        feature_t  m_target; ///< optional target feature
    };
} // namespace nano
