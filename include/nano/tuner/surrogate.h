#pragma once

#include <nano/function.h>
#include <nano/loss.h>
#include <nano/tuner.h>

namespace nano
{
///
/// \brief quadratic surrogate function useful for tuning continuous hyper-parameters.
///
/// given a set of initial measurements (p_i, y_i),
///     where y_i is typically the validation error associated to the hyper-parameter values p_i,
///
/// then the surrogate function fits a quadratic function like:
///     f(p, y; x) = sum(loss(y_i, x.dot(quadratic_terms(p_i))), i).
///
class NANO_PUBLIC quadratic_surrogate_fit_t final : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit quadratic_surrogate_fit_t(const loss_t& loss, tensor2d_t p, tensor1d_t y);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    scalar_t do_vgrad(vector_cmap_t x, vector_map_t gx) const override;

private:
    // attributes
    const loss_t&      m_loss;         ///<
    tensor2d_t         m_p2;           ///< quadratic terms of hyper-parameter values p
    tensor1d_t         m_y;            ///< target
    mutable tensor4d_t m_loss_outputs; ///<
    mutable tensor1d_t m_loss_values;  ///<
    mutable tensor4d_t m_loss_vgrads;  ///<
};

///
/// \brief quadratic surrogate function useful for finding the optimum hyper-parameters.
///
class NANO_PUBLIC quadratic_surrogate_t final : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit quadratic_surrogate_t(vector_t model);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    scalar_t do_vgrad(vector_cmap_t x, vector_map_t gx) const override;

private:
    // attributes
    vector_t m_model; ///< coefficients of the quadratic terms of hyper-parameter values p
};

///
/// \brief optimizer hyper-parameters by iteratively building and minimizing a quadratic surrogate function
///     that maps hyper-parameter values to a scalar value function (the lower, the better).
///
class NANO_PUBLIC surrogate_tuner_t final : public tuner_t
{
public:
    ///
    /// \brief constructor
    ///
    surrogate_tuner_t();

    ///
    /// \brief @see clonable_t
    ///
    rtuner_t clone() const override;

    ///
    /// \brief @see tuner_t
    ///
    void do_optimize(const param_spaces_t&, const tuner_callback_t&, tuner_steps_t&) const override;
};
} // namespace nano
