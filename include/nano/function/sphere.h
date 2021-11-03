#pragma once

#include <nano/function.h>

namespace nano
{
    ///
    /// \brief sphere function: f(x) = x.dot(x).
    ///
    class function_sphere_t final : public function_t
    {
    public:

        explicit function_sphere_t(tensor_size_t dims) :
            function_t("Sphere", dims, convexity::yes) // LCOV_EXCL_LINE
        {
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
            if (gx != nullptr)
            {
                gx->noalias() = 2 * x;
            }

            return x.dot(x);
        }
    };
}
