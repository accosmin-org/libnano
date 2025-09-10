#include <function/nonlinear/chained_lq.h>
#include <nano/core/numeric.h>

using namespace nano;

function_chained_lq_t::function_chained_lq_t(const tensor_size_t dims)
    : function_t("chained_lq", dims)
{
    convex(convexity::yes);
    smooth(smoothness::no);
    strong_convexity(0.0);
}

rfunction_t function_chained_lq_t::clone() const
{
    return std::make_unique<function_chained_lq_t>(*this);
}

scalar_t function_chained_lq_t::do_eval(eval_t eval) const
{
    const auto values = [&](const tensor_size_t i = 0)
    {
        const auto xi  = eval.m_x(i);
        const auto xi1 = eval.m_x(i + 1);
        const auto v1  = -xi - xi1;
        const auto v2  = v1 + square(xi) + square(xi1) - 1.0;

        return std::make_tuple(v1, v2);
    };

    if (eval.has_grad())
    {
        eval.m_gx.full(0.0);
        for (tensor_size_t i = 0, dims = size(); i + 1 < dims; ++i)
        {
            const auto [v1, v2] = values(i);

            if (v2 > v1)
            {
                eval.m_gx(i) += -1.0 + 2 * eval.m_x(i);
                eval.m_gx(i + 1) += -1.0 + 2 * eval.m_x(i + 1);
            }
            else
            {
                eval.m_gx(i) += -1.0;
                eval.m_gx(i + 1) += -1.0;
            }
        }
    }

    auto fx = 0.0;
    for (tensor_size_t i = 0, dims = size(); i + 1 < dims; ++i)
    {
        const auto [v1, v2] = values(i);
        fx += std::max(v1, v2);
    }

    return fx;
}

rfunction_t function_chained_lq_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_chained_lq_t>(dims);
}
