#pragma once

#include <nano/function/benchmark.h>

namespace nano
{
    ///
    /// \brief Sargan function: see http://infinity77.net/global_optimization/test_functions_nd_S.html.
    ///
    class NANO_PUBLIC function_sargan_t final : public benchmark_function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        explicit function_sargan_t(tensor_size_t dims = 10);

        ///
        /// \brief @see function_t
        ///
        scalar_t do_vgrad(const vector_t& x, vector_t* gx) const override;

        ///
        /// \brief @see benchmark_function_t
        ///
        rfunction_t make(tensor_size_t dims, tensor_size_t summands) const override;
    };
} // namespace nano
