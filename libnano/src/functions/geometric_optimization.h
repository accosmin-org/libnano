#pragma once

#include <nano/function.h>

namespace nano
{
    ///
    /// \brief generic geometric optimization function: f(x) = sum(i, exp(alpha_i + a_i.dot(x))).
    ///
    ///     see "Introductory Lectures on Convex Optimization (Applied Optimization)",
    ///     by Y. Nesterov, 2013, p.56
    ///
    ///     seee "Convex Optimization",
    ///     by S. Boyd and L. Vanderberghe, p.458 (logarithmic version)
    ///
    class function_geometric_optimization_t final : public function_t
    {
    public:

        explicit function_geometric_optimization_t(const tensor_size_t dims) :
            function_t("Geometric Optimization", dims, convexity::yes),
            m_a(vector_t::Random(16)),
            m_A(matrix_t::Random(16, dims) / dims)
        {
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
            if (gx)
            {
                gx->noalias() = m_A.transpose() * (m_a + m_A * x).array().exp().matrix();
            }

            return (m_a + m_A * x).array().exp().sum();
        }

    private:

        // attributes
        vector_t    m_a;
        matrix_t    m_A;
    };
}
