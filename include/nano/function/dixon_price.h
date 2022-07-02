#pragma once

#include <nano/function/benchmark.h>

namespace nano
{
    ///
    /// \brief Dixon-Price function: see https://www.sfu.ca/~ssurjano/dixonpr.html.
    ///
    class NANO_PUBLIC function_dixon_price_t final : public benchmark_function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        explicit function_dixon_price_t(tensor_size_t dims = 10);

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
        vector_t m_bias; ///<
    };
} // namespace nano
