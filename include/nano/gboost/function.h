#pragma once

#include <nano/dataset/iterator.h>
#include <nano/function.h>
#include <nano/loss.h>
#include <nano/model/cluster.h>

namespace nano::gboost
{
    ///
    /// \brief the criterion used for computing the gradient wrt outputs of a Gradient Boosting model,
    ///     using a given loss function:
    ///
    ///     f(outputs) = EXPECTATION[loss(target_i, output_i)] + vAreg * VARIANCE[loss(target_i, output_i)]
    ///
    /// NB: the function_t interface is used only for testing/debugging
    ///     as it computes more than needed when training a Gradient Boosting model.
    ///
    /// NB: the ERM loss can be optionally regularized by penalizing:
    ///     - (1) the variance of the loss values - like in VadaBoost
    ///
    class NANO_PUBLIC grads_function_t final : public function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        grads_function_t(const targets_iterator_t&, const loss_t&, scalar_t vAreg);

        ///
        /// \brief @see clonable_t
        ///
        rfunction_t clone() const override;

        ///
        /// \brief @see function_t
        ///
        scalar_t do_vgrad(const vector_t& x, vector_t* gx = nullptr) const override;

        ///
        /// \brief compute the gradient wrt output for each sample.
        ///
        const tensor4d_t& gradients(const tensor4d_cmap_t& outputs) const;

    private:
        // attributes
        const targets_iterator_t& m_iterator;   ///<
        const loss_t&             m_loss;       ///<
        scalar_t                  m_vAreg{0.0}; ///<
        mutable tensor1d_t        m_values;     ///<
        mutable tensor4d_t        m_vgrads;     ///<
    };

    ///
    /// \brief the criterion used for optimizing the bias of a Gradient Boosting model,
    ///     using a given loss function:
    ///
    ///     f(x) = EXPECTATION[loss(target_i, x)] + vAreg * VARIANCE[loss(target_i, x)]
    ///
    /// NB: the ERM loss can be optionally regularized by penalizing:
    ///     - (1) the variance of the loss values - like in VadaBoost
    ///
    class NANO_PUBLIC bias_function_t final : public function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        bias_function_t(const targets_iterator_t&, const loss_t&, scalar_t vAreg);

        ///
        /// \brief @see clonable_t
        ///
        rfunction_t clone() const override;

        ///
        /// \brief @see function_t
        ///
        scalar_t do_vgrad(const vector_t& x, vector_t* gx = nullptr) const override;

    private:
        // attributes
        const targets_iterator_t& m_iterator;   ///<
        const loss_t&             m_loss;       ///<
        scalar_t                  m_vAreg{0.0}; ///<
    };

    ///
    /// \brief the criterion used for optimizing the scale (aka the line-search like step) of a Gradient Boosting model,
    ///     using a given loss function.
    ///
    ///     f(x) = EXPECTATION[loss(target_i, output_i + x[cluster_i] * woutput_i)] +
    ///            vAreg * VARIANCE[loss(target_i, output_i + x[cluster_i] * woutput_i)]
    ///
    /// NB: the ERM loss can be optionally regularized by penalizing:
    ///     - (1) the variance of the loss values - like in VadaBoost
    ///
    class NANO_PUBLIC scale_function_t final : public function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        scale_function_t(const targets_iterator_t&, const loss_t&, scalar_t vAreg, const cluster_t&,
                         const tensor4d_t& outputs, const tensor4d_t& woutputs);

        ///
        /// \brief @see clonable_t
        ///
        rfunction_t clone() const override;

        ///
        /// \brief @see function_t
        ///
        scalar_t do_vgrad(const vector_t& x, vector_t* gx = nullptr) const override;

    private:
        // attributes
        const targets_iterator_t& m_iterator;   ///<
        const loss_t&             m_loss;       ///<
        scalar_t                  m_vAreg{0.0}; ///<
        const cluster_t&          m_cluster;    ///<
        const tensor4d_t&         m_outputs;    ///< predictions of the strong learner so far
        const tensor4d_t&         m_woutputs;   ///< predictions of the current weak learner
    };
} // namespace nano::gboost
