#include <function/nonlinear/rosenbrock.h>
#include <nano/core/numeric.h>

using namespace nano;

function_rosenbrock_t::function_rosenbrock_t(const tensor_size_t dims)
    : function_t("rosenbrock", std::max(dims, tensor_size_t(2)))
{
    convex(convexity::no);
    smooth(smoothness::yes);
}

rfunction_t function_rosenbrock_t::clone() const
{
    return std::make_unique<function_rosenbrock_t>(*this);
}

scalar_t function_rosenbrock_t::do_eval(eval_t eval) const
{
    const auto x  = eval.m_x;
    const auto ct = scalar_t(100);

    scalar_t fx = 0;
    for (tensor_size_t i = 0, size = this->size(); i + 1 < size; ++i)
    {
        fx += ct * nano::square(x(i + 1) - x(i) * x(i)) + nano::square(x(i) - 1);
    }

    if (eval.has_grad())
    {
        eval.m_gx.full(0.0);
        for (tensor_size_t i = 0, size = this->size(); i + 1 < size; ++i)
        {
            const auto xi0 = x(i + 0);
            const auto xi1 = x(i + 1);
            eval.m_gx(i + 0) -= ct * 4 * (xi1 - xi0 * xi0) * xi0 - 2 * (xi0 - 1);
            eval.m_gx(i + 1) += ct * 2 * (xi1 - xi0 * xi0);
        }
    }

    if (eval.has_hess())
    {
        eval.m_hx.full(0.0);
        for (tensor_size_t i = 0, size = this->size(); i + 1 < size; ++i)
        {
            const auto xi0 = x(i + 0);
            const auto xi1 = x(i + 1);
            eval.m_hx(i + 0, i + 0) += 2 - ct * 4 * xi1 + ct * 12 * xi0 * xi0;
            eval.m_hx(i + 0, i + 1) -= ct * 4 * xi0;
            eval.m_hx(i + 1, i + 0) -= ct * 4 * xi0;
            eval.m_hx(i + 1, i + 1) += ct * 2;
        }
    }

    return fx;
}

rfunction_t function_rosenbrock_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_rosenbrock_t>(dims);
}
