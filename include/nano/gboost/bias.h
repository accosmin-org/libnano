#pragma once

#include <nano/loss.h>
#include <nano/dataset.h>
#include <nano/function.h>
#include <nano/parameter.h>

namespace nano
{
    ///
    /// \brief the criterion use for optimizing the bias of a Gradient Boosting model,
    ///     using a given loss function.
    ///
    /// NB: the ERM loss can be optionally regularized by penalizing:
    ///     - (1) the variance of the loss values - like in VadaBoost
    ///
    class NANO_PUBLIC gboost_bias_function_t final : public function_t
    {
    public:

        ///
        /// \brief constructor
        ///
        gboost_bias_function_t(const loss_t&, const dataset_t&, fold_t);

        ///
        /// \brief enable copying
        ///
        gboost_bias_function_t(const gboost_bias_function_t&) = default;
        gboost_bias_function_t& operator=(const gboost_bias_function_t&) = delete;

        ///
        /// \brief enable moving
        ///
        gboost_bias_function_t(gboost_bias_function_t&&) noexcept = default;
        gboost_bias_function_t& operator=(gboost_bias_function_t&&) noexcept = delete;

        ///
        /// \brief default destructor
        ///
        ~gboost_bias_function_t() override = default;

        ///
        /// \brief @see function_t
        ///
        scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const override;

        ///
        /// \brief change parameters
        ///
        void vAreg(const scalar_t vAreg) { m_vAreg.set(vAreg); }
        void batch(const tensor_size_t batch) { m_batch.set(batch); }

        ///
        /// \brief access functions
        ///
        [[nodiscard]] auto fold() const { return m_fold; }
        [[nodiscard]] auto vAreg() const { return m_vAreg.get(); }
        [[nodiscard]] auto batch() const { return m_batch.get(); }
        [[nodiscard]] const auto& loss() const { return m_loss; }
        [[nodiscard]] const auto& dataset() const { return m_dataset; }

    private:

        // attributes
        const loss_t&       m_loss;         ///<
        const dataset_t&    m_dataset;      ///<
        fold_t              m_fold;         ///<
        sparam1_t           m_vAreg{"linear::VA", 0, LE, 0, LE, 1e+8};  ///< regularization factor - see (4)
        iparam1_t           m_batch{"linear::batch", 1, LE, 32, LE, 4092};///< batch size in number of samples
    };
}
