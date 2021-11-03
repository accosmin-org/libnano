#pragma once

#include <nano/function.h>

namespace nano
{
    ///
    /// \brief random quadratic function: f(x) = x.dot(a) + x * A * x, where A is PD.
    ///
    class function_quadratic_t final : public function_t
    {
    public:

        explicit function_quadratic_t(tensor_size_t dims) :
            function_t("Quadratic", dims, convexity::yes), // LCOV_EXCL_LINE
            m_a(vector_t::Random(dims)) // LCOV_EXCL_LINE
        {
            // NB: generate random positive semi-definite matrix to keep the function convex
            matrix_t A = matrix_t::Random(dims, dims);
            m_A = matrix_t::Identity(dims, dims) + A * A.transpose();
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
            if (gx != nullptr)
            {
                gx->noalias() = m_a + m_A * x;
            }

            return x.dot(m_a + (m_A * x) / scalar_t(2));
        }

    private:

        // attributes
        vector_t    m_a;
        matrix_t    m_A;
    };
}
