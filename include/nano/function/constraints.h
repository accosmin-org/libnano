#pragma once

#include <nano/function.h>

namespace nano
{
    ///
    /// \brief models a hyper-ball (equality or inequality) constraint: c(x) = ||x - origin||^2 - radius^2.
    ///
    class NANO_PUBLIC ball_constraint_t final : public function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        ball_constraint_t(vector_t origin, scalar_t radius);

        ///
        /// \brief @see function_t
        ///
        scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const override;

    private:
        // attributes
        vector_t m_origin;      ///<
        scalar_t m_radius{0.0}; ///<
    };

    ///
    /// \brief models an affine (equality or inequality) constraint: c(x) = weights.dot(x) + bias.
    ///
    class NANO_PUBLIC affine_constraint_t final : public function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        affine_constraint_t(vector_t weights, scalar_t bias);

        ///
        /// \brief @see function_t
        ///
        scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const override;

    private:
        // attributes
        vector_t m_weights;   ///<
        scalar_t m_bias{0.0}; ///<
    };

    // TODO: add quadratic constraints
} // namespace nano
