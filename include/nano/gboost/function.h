#pragma once

#include <nano/dataset/iterator.h>
#include <nano/function.h>
#include <nano/gboost/accumulator.h>
#include <nano/loss.h>
#include <nano/model/cluster.h>

namespace nano::gboost
{
///
/// \brief the criterion used for computing the gradient wrt outputs of a Gradient Boosting model,
///     using a given loss function:
///
///     f(outputs) = EXPECTATION[loss(target_i, output_i)].
///
/// NB: the function_t interface is used only for testing/debugging
///     as it computes more than needed when training a Gradient Boosting model.
///
class NANO_PUBLIC grads_function_t final : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    grads_function_t(const targets_iterator_t&, const loss_t&);

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
    const targets_iterator_t& m_iterator; ///<
    const loss_t&             m_loss;     ///<
    mutable tensor1d_t        m_values;   ///<
    mutable tensor4d_t        m_vgrads;   ///<
};

///
/// \brief the criterion used for computing the bias of a Gradient Boosting model,
///     using a given loss function:
///
///     f(x) = EXPECTATION[loss(target_i, x)].
///
class NANO_PUBLIC bias_function_t final : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    bias_function_t(const targets_iterator_t&, const loss_t&);

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
    const targets_iterator_t& m_iterator;     ///<
    const loss_t&             m_loss;         ///<
    mutable tensor1d_t        m_values;       ///<
    mutable tensor4d_t        m_vgrads;       ///<
    mutable tensor4d_t        m_outputs;      ///<
    mutable accumulators_t    m_accumulators; ///<
};

///
/// \brief the criterion used for optimizing the scale (aka the line-search like step) of a Gradient Boosting model,
///     using a given loss function:
///
///     f(x) = EXPECTATION[loss(target_i, soutput_i + x[cluster_i] * woutput_i)].
///
class NANO_PUBLIC scale_function_t final : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    scale_function_t(const targets_iterator_t&, const loss_t&, const cluster_t&, const tensor4d_t& soutputs,
                     const tensor4d_t& woutputs);

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
    const targets_iterator_t& m_iterator;     ///<
    const loss_t&             m_loss;         ///<
    const cluster_t&          m_cluster;      ///<
    const tensor4d_t&         m_soutputs;     ///< predictions of the strong learner so far
    const tensor4d_t&         m_woutputs;     ///< predictions of the current weak learner
    mutable tensor1d_t        m_values;       ///<
    mutable tensor4d_t        m_vgrads;       ///<
    mutable tensor4d_t        m_outputs;      ///<
    mutable accumulators_t    m_accumulators; ///<
};
} // namespace nano::gboost
