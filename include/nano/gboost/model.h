#pragma once

#include <nano/model.h>
#include <nano/wlearner.h>

namespace nano
{
    ///
    /// \brief (Stochastic) Gradient Boosting model.
    ///
    /// some important features:
    ///     - weak learners are selected from a configurable pool of prototypes and thus the final model
    ///         can mix different types of weak learners (e.g. like stumps with look-up-tables).
    ///     - support for variance-based regularization (like EBBoost or VadaBoost).
    ///     - builtin early stopping if the validation error doesn't decrease in a configurable number of boosting
    ///     rounds.
    ///     - support for serialization of its parameters and the selected weak learners.
    ///     - training and evaluation is performed using all available threads.
    ///     - the bias computation and the scaling of the weak learners can be solved
    ///         using any of the available builtin line-search-based solvers (e.g. lBFGS, CGD, CG_DESCENT).
    ///     - support for estimating the importance of the selected features.
    ///
    /// see "The Elements of Statistical Learning", by Trevor Hastie, Robert Tibshirani, Jerome Friedman
    /// see "Greedy Function Approximation: A Gradient Boosting Machine", by Jerome Friedman
    /// see "Stochastic Gradient Boosting", by Jerome Friedman
    ///
    /// see "Empirical Bernstein Boosting", by Pannagadatta K. Shivaswamy & Tony Jebara
    /// see "Variance Penalizing AdaBoost", by Pannagadatta K. Shivaswamy & Tony Jebara
    ///
    class NANO_PUBLIC gboost_model_t final : public model_t
    {
    public:
        ///
        /// \brief constructor
        ///
        gboost_model_t();

        ///
        /// \brief enable moving
        ///
        gboost_model_t(gboost_model_t&&) noexcept;
        gboost_model_t& operator=(gboost_model_t&&) noexcept;

        ///
        /// \brief enable copying
        ///
        gboost_model_t(const gboost_model_t&);
        gboost_model_t& operator=(const gboost_model_t&);

        ///
        /// \brief destructor
        ///
        ~gboost_model_t() override;

        ///
        /// \brief register a weak learner as a prototype.
        ///
        void add(const wlearner_t&);
        void add(const string_t& wlearner_id);

        ///
        /// \brief @see configurable_t
        ///
        std::istream& read(std::istream&) override;

        ///
        /// \brief @see configurable_t
        ///
        std::ostream& write(std::ostream&) const override;

        ///
        /// \brief @see clone_t
        ///
        rmodel_t clone() const override;

        ///
        /// \brief @see model_
        ///
        fit_result_t fit(const dataset_t&, const indices_t&, const loss_t&, const solver_t&, const splitter_t&,
                         const tuner_t&) override;

        ///
        /// \brief @see model_t
        ///
        tensor4d_t predict(const dataset_t&, const indices_t&) const override;

        ///
        /// \brief returns the selected features.
        ///
        indices_t features() const;

        ///
        /// \brief returns the fitted bias.
        ///
        const tensor1d_t& bias() const { return m_bias; }

        ///
        /// \brief returns the fitted weak learners.
        ///
        const rwlearners_t& wlearners() const { return m_wlearners; }

    private:
        // attributes
        tensor1d_t   m_bias;      ///< fitted bias
        rwlearners_t m_protos;    ///< weak learners to choose from (prototypes)
        rwlearners_t m_wlearners; ///< fitted weak learners chosen from the prototypes
    };
} // namespace nano
