#pragma once

#include <nano/function/benchmark.h>

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
    class NANO_PUBLIC function_geometric_optimization_t final : public benchmark_function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        explicit function_geometric_optimization_t(tensor_size_t dims = 10, tensor_size_t summands = 16);

        ///
        /// \brief @see function_t
        ///
        scalar_t vgrad(const vector_t& x, vector_t* gx) const override;

        ///
        /// \brief @see benchmark_function_t
        ///
        rfunction_t make(tensor_size_t dims, tensor_size_t summands) const override;

    private:
        // attributes
        vector_t m_a; ///<
        matrix_t m_A; ///<
    };
} // namespace nano
