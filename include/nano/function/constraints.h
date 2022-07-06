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
        scalar_t do_vgrad(const vector_t& x, vector_t* gx = nullptr) const override;

    private:
        // attributes
        vector_t m_origin;      ///<
        scalar_t m_radius{0.0}; ///<
    };

    ///
    /// \brief models an affine (equality or inequality) constraint: c(x) = q.dot(x) + r.
    ///
    class NANO_PUBLIC affine_constraint_t final : public function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        affine_constraint_t(vector_t q, scalar_t r);

        ///
        /// \brief @see function_t
        ///
        scalar_t do_vgrad(const vector_t& x, vector_t* gx = nullptr) const override;

    private:
        // attributes
        vector_t m_q;      ///<
        scalar_t m_r{0.0}; ///<
    };

    ///
    /// \brief models a quadratic (equality or inequality) constraint: c(x) = 1/2 * x.dot(P * x) + q.dot(x) + r.
    ///
    class NANO_PUBLIC quadratic_constraint_t final : public function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        quadratic_constraint_t(matrix_t P, vector_t q, scalar_t r);

        ///
        /// \brief @see function_t
        ///
        scalar_t do_vgrad(const vector_t& x, vector_t* gx = nullptr) const override;

    private:
        // attributes
        matrix_t m_P;      ///<
        vector_t m_q;      ///<
        scalar_t m_r{0.0}; ///<
    };
} // namespace nano
