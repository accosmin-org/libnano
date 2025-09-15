#include <function/nonlinear/powell.h>
#include <nano/core/numeric.h>

using namespace nano;

function_powell_t::function_powell_t(const tensor_size_t dims)
    : function_t("powell", std::max(tensor_size_t(4), dims - dims % 4))
{
    convex(convexity::no);
    smooth(smoothness::yes);
}

rfunction_t function_powell_t::clone() const
{
    return std::make_unique<function_powell_t>(*this);
}

scalar_t function_powell_t::do_eval(eval_t eval) const
{
    const auto x = eval.m_x;

    scalar_t fx = 0;
    for (tensor_size_t i = 0, i4 = 0, size = this->size(); i < size / 4; ++i, i4 += 4)
    {
        fx += nano::square(x(i4 + 0) + x(i4 + 1) * 10);
        fx += nano::square(x(i4 + 2) - x(i4 + 3)) * 5;
        fx += nano::quartic(x(i4 + 1) - x(i4 + 2) * 2);
        fx += nano::quartic(x(i4 + 0) - x(i4 + 3)) * 10;
    }

    if (eval.has_grad())
    {
        for (tensor_size_t i = 0, i4 = 0; i < size() / 4; ++i, i4 += 4)
        {
            const auto gfx0 = (x(i4 + 0) + x(i4 + 1) * 10) * 2;
            const auto gfx1 = (x(i4 + 2) - x(i4 + 3)) * 5 * 2;
            const auto gfx2 = nano::cube(x(i4 + 1) - x(i4 + 2) * 2) * 4;
            const auto gfx3 = nano::cube(x(i4 + 0) - x(i4 + 3)) * 10 * 4;

            eval.m_gx(i4 + 0) = gfx0 + gfx3;
            eval.m_gx(i4 + 1) = gfx0 * 10 + gfx2;
            eval.m_gx(i4 + 2) = gfx1 - 2 * gfx2;
            eval.m_gx(i4 + 3) = -gfx1 - gfx3;
        }
    }

    if (eval.has_hess())
    {
        eval.m_Hx.full(0.0);
        for (tensor_size_t i = 0, i4 = 0; i < size() / 4; ++i, i4 += 4)
        {
            eval.m_Hx(i4 + 0, i4 + 0) += 2.0;
            eval.m_Hx(i4 + 0, i4 + 1) += 20.0;
            eval.m_Hx(i4 + 1, i4 + 0) += 20.0;
            eval.m_Hx(i4 + 1, i4 + 1) += 200.0;

            eval.m_Hx(i4 + 2, i4 + 2) += 10.0;
            eval.m_Hx(i4 + 2, i4 + 3) += -10.0;
            eval.m_Hx(i4 + 3, i4 + 2) += -10.0;
            eval.m_Hx(i4 + 3, i4 + 3) += 10.0;

            const auto f2 = nano::square(x(i4 + 1) - x(i4 + 2) * 2);
            eval.m_Hx(i4 + 1, i4 + 1) += 12.0 * f2;
            eval.m_Hx(i4 + 1, i4 + 2) += -24.0 * f2;
            eval.m_Hx(i4 + 2, i4 + 1) += -24.0 * f2;
            eval.m_Hx(i4 + 2, i4 + 2) += 48.0 * f2;

            const auto f3 = nano::square(x(i4 + 0) - x(i4 + 3));
            eval.m_Hx(i4 + 0, i4 + 0) += 120.0 * f3;
            eval.m_Hx(i4 + 0, i4 + 3) -= 120.0 * f3;
            eval.m_Hx(i4 + 3, i4 + 0) -= 120.0 * f3;
            eval.m_Hx(i4 + 3, i4 + 3) += 120.0 * f3;
        }
    }

    return fx;
}

rfunction_t function_powell_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_powell_t>(dims);
}
