#pragma once

#include <nano/dataset.h>
#include <nano/function.h>
#include <nano/loss.h>
#include <nano/mlearn/cluster.h>
#include <nano/parameter.h>

namespace nano
{
    ///
    /// \brief base class for the optimization criteria of a Gradient Boosting model,
    ///     using a given loss function.
    ///
    /// NB: the ERM loss can be optionally regularized by penalizing:
    ///     - (1) the variance of the loss values - like in VadaBoost
    ///
    class NANO_PUBLIC gboost_function_t : public function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        explicit gboost_function_t(tensor_size_t dims);

        ///
        /// \brief change parameters
        ///
        void vAreg(scalar_t vAreg);
        void batch(tensor_size_t batch);

        ///
        /// \brief access functions
        ///
        auto vAreg() const { return m_vAreg.get(); }

        auto batch() const { return m_batch.get(); }

    private:
        // attributes
        sparam1_t m_vAreg{"gboost::VA", 0, LE, 0, LE, 1e+8};     ///< regularization factor - see (1)
        iparam1_t m_batch{"gboost::batch", 1, LE, 32, LE, 4092}; ///< batch size in number of samples
    };

    ///
    /// \brief the criterion used for computing the gradient wrt outputs of a Gradient Boosting model,
    ///     using a given loss function:
    ///
    ///     f(outputs) = EXPECTATION[loss(target_i, output_i)] + vAreg * VARIANCE[loss(target_i, output_i)]
    ///
    /// NB: the function_t interface is used only for testing/debugging
    ///     as it computes more than needed when training a Gradient Boosting model.
    ///
    class NANO_PUBLIC gboost_grads_function_t final : public gboost_function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        gboost_grads_function_t(const loss_t&, const dataset_t&, const indices_t&);

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
        const loss_t&      m_loss;    ///<
        const dataset_t&   m_dataset; ///<
        const indices_t&   m_samples; ///<
        mutable tensor1d_t m_values;  ///<
        mutable tensor4d_t m_vgrads;  ///<
    };

    ///
    /// \brief the criterion used for optimizing the bias of a Gradient Boosting model,
    ///     using a given loss function:
    ///
    ///     f(x) = EXPECTATION[loss(target_i, x)] + vAreg * VARIANCE[loss(target_i, x)]
    ///
    class NANO_PUBLIC gboost_bias_function_t final : public gboost_function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        gboost_bias_function_t(const loss_t&, const dataset_t&, const indices_t&);

        ///
        /// \brief @see function_t
        ///
        scalar_t do_vgrad(const vector_t& x, vector_t* gx = nullptr) const override;

    private:
        // attributes
        const loss_t&    m_loss;    ///<
        const dataset_t& m_dataset; ///<
        const indices_t& m_samples; ///<
    };

    ///
    /// \brief the criterion used for optimizing the scale (aka the line-search like step) of a Gradient Boosting model,
    ///     using a given loss function.
    ///
    ///     f(x) = EXPECTATION[loss(target_i, output_i + x[cluster_i] * woutput_i)] +
    ///            vAreg * VARIANCE[loss(target_i, output_i + x[cluster_i] * woutput_i)]
    ///
    class NANO_PUBLIC gboost_scale_function_t final : public gboost_function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        gboost_scale_function_t(const loss_t&, const dataset_t&, const indices_t&, const cluster_t&,
                                const tensor4d_t& outputs, const tensor4d_t& woutputs);

        ///
        /// \brief @see function_t
        ///
        scalar_t do_vgrad(const vector_t& x, vector_t* gx = nullptr) const override;

    private:
        // attributes
        const loss_t&     m_loss;     ///<
        const dataset_t&  m_dataset;  ///<
        const indices_t&  m_samples;  ///<
        const cluster_t&  m_cluster;  ///<
        const tensor4d_t& m_outputs;  ///< predictions of the strong learner so far
        const tensor4d_t& m_woutputs; ///< predictions of the current weak learner
    };
} // namespace nano
