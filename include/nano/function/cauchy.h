#pragma once

#include <nano/function/benchmark.h>

namespace nano
{
    ///
    /// \brief Cauchy function: f(x) = log(1 + x.dot(x)).
    ///
    class NANO_PUBLIC function_cauchy_t final : public benchmark_function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        explicit function_cauchy_t(tensor_size_t dims = 10);

        ///
        /// \brief @see function_t
        ///
        scalar_t vgrad(const vector_t& x, vector_t* gx, vgrad_config_t) const override;

        ///
        /// \brief @see benchmark_function_t
        ///
        rfunction_t make(tensor_size_t dims, tensor_size_t summands) const override;
    };
} // namespace nano
