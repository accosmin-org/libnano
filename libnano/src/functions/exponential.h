#pragma once

#include <nano/function.h>

namespace nano
{
    ///
    /// \brief exponential function: f(x) = exp(1 + x.dot(x) / D).
    ///
    class function_exponential_t final : public function_t
    {
    public:

        explicit function_exponential_t(const tensor_size_t dims) :
            function_t("Exponential", dims, convexity::yes)
        {
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
            const auto fx = std::exp(1 + x.dot(x) / scalar_t(size()));

            if (gx)
            {
                gx->noalias() = (2 * fx / scalar_t(size())) * x;
            };

            return fx;
        }
    };
}
