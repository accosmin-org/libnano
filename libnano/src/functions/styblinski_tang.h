#pragma once

#include <nano/function.h>

namespace nano
{
    ///
    /// \brief Styblinski-Tang function: see https://www.sfu.ca/~ssurjano/stybtang.html.
    ///
    class function_styblinski_tang_t final : public function_t
    {
    public:

        explicit function_styblinski_tang_t(const tensor_size_t dims) :
            function_t("Styblinski-Tang", dims, convexity::no)
        {
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
            if (gx)
            {
                *gx = 4 * x.array().cube() - 32 * x.array() + 5;
            }

            return (x.array().square().square() - 16 * x.array().square() + 5 * x.array()).sum();
        }
    };
}
