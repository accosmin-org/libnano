#pragma once

#include <nano/function.h>

namespace nano
{
    ///
    /// \brief Qing function: see http://benchmarkfcns.xyz/benchmarkfcns/qingfcn.html.
    ///
    class function_qing_t final : public function_t
    {
    public:

        explicit function_qing_t(const tensor_size_t dims) :
            function_t("Qing", dims, 1, convexity::no),
            m_bias(vector_t::LinSpaced(dims, scalar_t(1), scalar_t(dims)))
        {
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
            if (gx != nullptr)
            {
                *gx = 4 * (x.array().square() - m_bias.array()) * x.array();
            }

            return (x.array().square() - m_bias.array()).square().sum();
        }

    private:

        // attributes
        vector_t    m_bias; ///<
    };
}
