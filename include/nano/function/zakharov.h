#pragma once

#include <nano/core/numeric.h>
#include <nano/function.h>

namespace nano
{
    ///
    /// \brief Zakharov function: see https://www.sfu.ca/~ssurjano/zakharov.html.
    ///
    class function_zakharov_t final : public function_t
    {
    public:

        explicit function_zakharov_t(tensor_size_t dims) :
            function_t("Zakharov", dims, convexity::yes), // LCOV_EXCL_LINE
            m_bias(vector_t::LinSpaced(dims, scalar_t(0.5), scalar_t(dims) / scalar_t(2))) // LCOV_EXCL_LINE

        {
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
            const scalar_t u = x.dot(x);
            const scalar_t v = x.dot(m_bias);

            if (gx != nullptr)
            {
                *gx = 2 * x + (2 * v + 4 * nano::cube(v)) * m_bias;
            }

            return u + nano::square(v) + nano::quartic(v);
        }

    private:

        // attributes
        vector_t    m_bias; ///<
    };
}
