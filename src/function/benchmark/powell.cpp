#include <nano/core/numeric.h>
#include <nano/function/benchmark/powell.h>

using namespace nano;

function_powell_t::function_powell_t(tensor_size_t dims)
    : function_t("powell", std::max(tensor_size_t(4), dims - dims % 4))
{
    convex(false);
    smooth(true);
}

rfunction_t function_powell_t::clone() const
{
    return std::make_unique<function_powell_t>(*this);
}

scalar_t function_powell_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    scalar_t fx = 0;
    for (tensor_size_t i = 0, i4 = 0; i < size() / 4; ++i, i4 += 4)
    {
        fx += nano::square(x(i4 + 0) + x(i4 + 1) * 10);
        fx += nano::square(x(i4 + 2) - x(i4 + 3)) * 5;
        fx += nano::quartic(x(i4 + 1) - x(i4 + 2) * 2);
        fx += nano::quartic(x(i4 + 0) - x(i4 + 3)) * 10;
    }

    if (gx != nullptr)
    {
        for (tensor_size_t i = 0, i4 = 0; i < size() / 4; ++i, i4 += 4)
        {
            const auto gfx0 = (x(i4 + 0) + x(i4 + 1) * 10) * 2;
            const auto gfx1 = (x(i4 + 2) - x(i4 + 3)) * 5 * 2;
            const auto gfx2 = nano::cube(x(i4 + 1) - x(i4 + 2) * 2) * 4;
            const auto gfx3 = nano::cube(x(i4 + 0) - x(i4 + 3)) * 10 * 4;

            (*gx)(i4 + 0) = gfx0 + gfx3;
            (*gx)(i4 + 1) = gfx0 * 10 + gfx2;
            (*gx)(i4 + 2) = gfx1 - 2 * gfx2;
            (*gx)(i4 + 3) = -gfx1 - gfx3;
        }
    }

    return fx;
}

rfunction_t function_powell_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_powell_t>(dims);
}
