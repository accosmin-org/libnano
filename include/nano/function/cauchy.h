#pragma once

#include <nano/function.h>

namespace nano
{
    ///
    /// \brief Cauchy function: f(x) = log(1 + x.dot(x)).
    ///
    class function_cauchy_t final : public function_t
    {
    public:

        explicit function_cauchy_t(tensor_size_t dims) :
            function_t("Cauchy", dims, convexity::no) // LCOV_EXCL_LINE
        {
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
            if (gx != nullptr)
            {
                *gx = 2 * x / (1 + x.dot(x));
            }

            return std::log1p(x.dot(x));
        }
    };
}
