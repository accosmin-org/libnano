#pragma once

#include <nano/function/benchmark.h>

namespace nano
{
    ///
    /// \brief axis-parallel hyper-ellipsoid function: f(x) = sum(i*x+i^2, i=1,D).
    ///
    class NANO_PUBLIC function_axis_ellipsoid_t final : public benchmark_function_t
    {
    public:

        ///
        /// \brief constructor
        ///
        explicit function_axis_ellipsoid_t(tensor_size_t dims = 10);

        ///
        /// \brief @see function_t
        ///
        scalar_t vgrad(const vector_t& x, vector_t* gx, vgrad_config_t) const override;

        ///
        /// \brief @see benchmark_function_t
        ///
        rfunction_t make(tensor_size_t dims, tensor_size_t summands) const override;

    private:

        // attributes
        vector_t    m_bias; ///<
    };
}
