#pragma once

#include <nano/function.h>

namespace nano
{
    ///
    /// \brief Chung-Reynolds function: f(x) = (x.dot(x))^2.
    ///
    class function_chung_reynolds_t final : public function_t
    {
    public:

        explicit function_chung_reynolds_t(const tensor_size_t dims) :
            function_t("Chung-Reynolds", dims, convexity::yes)
        {
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
            const auto u = x.dot(x);

            if (gx)
            {
                *gx = (4 * u) * x;
            }

            return u * u;
        }
    };
}
