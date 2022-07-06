#pragma once

#include <nano/function/benchmark.h>

namespace nano
{
    ///
    /// \brief Trid function: see https://www.sfu.ca/~ssurjano/trid.html.
    ///
    class NANO_PUBLIC function_trid_t final : public benchmark_function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        explicit function_trid_t(tensor_size_t dims = 10);

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
