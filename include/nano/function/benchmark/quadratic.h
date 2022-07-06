#pragma once

#include <nano/function/benchmark.h>

namespace nano
{
    ///
    /// \brief random quadratic function: f(x) = x.dot(a) + x * A * x, where A is PD.
    ///
    class NANO_PUBLIC function_quadratic_t final : public benchmark_function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        explicit function_quadratic_t(tensor_size_t dims = 10);

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
        vector_t m_a;
        matrix_t m_A;
    };
} // namespace nano
