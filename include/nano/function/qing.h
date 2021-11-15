#pragma once

#include <nano/function/benchmark.h>

namespace nano
{
    ///
    /// \brief Qing function: see http://benchmarkfcns.xyz/benchmarkfcns/qingfcn.html.
    ///
    class NANO_PUBLIC function_qing_t final : public benchmark_function_t
    {
    public:

        ///
        /// \brief constructor
        ///
        explicit function_qing_t(tensor_size_t dims = 10);

        ///
        /// \brief @see function_t
        ///
        scalar_t vgrad(const vector_t& x, vector_t* gx) const override;

        ///
        /// \brief @see benchmark_function_t
        ///
        rfunction_t make(tensor_size_t dims) const override;

    private:

        // attributes
        vector_t    m_bias; ///<
    };
}
