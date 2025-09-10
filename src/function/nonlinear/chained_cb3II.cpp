#include <function/nonlinear/chained_cb3II.h>
#include <nano/core/numeric.h>

using namespace nano;

function_chained_cb3II_t::function_chained_cb3II_t(const tensor_size_t dims)
    : function_t("chained_cb3II", dims)
{
    convex(convexity::yes);
    smooth(smoothness::no);
    strong_convexity(0.0);
}

rfunction_t function_chained_cb3II_t::clone() const
{
    return std::make_unique<function_chained_cb3II_t>(*this);
}

scalar_t function_chained_cb3II_t::do_eval(eval_t eval) const
{
    auto fx1 = 0.0;
    auto fx2 = 0.0;
    auto fx3 = 0.0;
    for (tensor_size_t i = 0, dims = size(); i + 1 < dims; ++i)
    {
        const auto xi  = x(i);
        const auto xi1 = x(i + 1);
        const auto v1  = quartic(xi) + square(xi1);
        const auto v2  = square(2.0 - xi) + square(2 - xi1);
        const auto v3  = 2.0 * std::exp(-xi + xi1);

        fx1 += v1;
        fx2 += v2;
        fx3 += v3;
    }

    if (gx.size() == x.size())
    {
        gx.full(0.0);
        if (fx1 > std::max(fx2, fx3))
        {
            for (tensor_size_t i = 0, dims = size(); i + 1 < dims; ++i)
            {
                gx(i) += 4.0 * cube(x(i));
                gx(i + 1) += 2.0 * x(i + 1);
            }
        }
        else if (fx2 > std::max(fx1, fx3))
        {
            for (tensor_size_t i = 0, dims = size(); i + 1 < dims; ++i)
            {
                gx(i) += 2.0 * x(i) - 4.0;
                gx(i + 1) += 2.0 * x(i + 1) - 4.0;
            }
        }
        else
        {
            for (tensor_size_t i = 0, dims = size(); i + 1 < dims; ++i)
            {
                const auto e = std::exp(x(i + 1) - x(i));
                gx(i) -= 2.0 * e;
                gx(i + 1) += 2.0 * e;
            }
        }
    }

    return std::max({fx1, fx2, fx3});
}

rfunction_t function_chained_cb3II_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_chained_cb3II_t>(dims);
}
