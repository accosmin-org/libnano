#pragma once

#include <nano/core/random.h>
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

        explicit function_geometric_optimization_t(tensor_size_t dims, tensor_size_t summands = 16) :
            function_t("Geometric Optimization", dims, convexity::yes), // LCOV_EXCL_LINE
            m_a(vector_t::Random(summands)), // LCOV_EXCL_LINE
            m_A(matrix_t::Random(summands, dims) / dims) // LCOV_EXCL_LINE
        {
            assert(summands > 0);
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const override
        {
            if (gx != nullptr)
            {
                gx->noalias() = m_A.transpose() * (m_a + m_A * x).array().exp().matrix();
            }

            return (m_a + m_A * x).array().exp().sum();
        }

    private:

        // attributes
        vector_t    m_a;    ///<
        matrix_t    m_A;    ///<
    };
}
