#pragma once

#include <nano/function.h>

namespace nano
{
    ///
    /// \brief axis-parallel hyper-ellipsoid function: f(x) = sum(i*x+i^2, i=1,D).
    ///
    class function_axis_ellipsoid_t final : public function_t
    {
    public:

        explicit function_axis_ellipsoid_t(const tensor_size_t dims) :
            function_t("Axis Parallel Hyper-Ellipsoid", dims, 1, convexity::yes),
            m_bias(vector_t::LinSpaced(dims, scalar_t(1), scalar_t(dims)))
        {
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
            if (gx != nullptr)
            {
                *gx = 2 * x.array() * m_bias.array();
            }

            return (x.array().square() * m_bias.array()).sum();
        }

    private:

        // attributes
        vector_t    m_bias; ///<
    };
}
