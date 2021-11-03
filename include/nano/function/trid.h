#pragma once

#include <nano/function.h>

namespace nano
{
    ///
    /// \brief Trid function: see https://www.sfu.ca/~ssurjano/trid.html.
    ///
    class function_trid_t final : public function_t
    {
    public:

        explicit function_trid_t(tensor_size_t dims) :
            function_t("Trid", dims, convexity::yes) // LCOV_EXCL_LINE
        {
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
            if (gx != nullptr)
            {
                *gx = 2 * (x.array() - 1);
                gx->segment(1, size() - 1) -= x.segment(0, size() - 1);
                gx->segment(0, size() - 1) -= x.segment(1, size() - 1);
            }

            return (x.array() - 1).square().sum() -
                   (x.segment(0, size() - 1).array() * x.segment(1, size() - 1).array()).sum();
        }
    };
}
