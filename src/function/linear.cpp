#include <nano/core/overloaded.h>
#include <nano/function/linear.h>

using namespace nano;
using namespace constraint;

linear_program_t::linear_program_t(string_t id, vector_t c)
    : function_t(std::move(id), c.size())
    , m_c(std::move(c))
{
    smooth(smoothness::yes);
    convex(convexity::yes);
    strong_convexity(0.0);
}

rfunction_t linear_program_t::clone() const
{
    return std::make_unique<linear_program_t>(*this);
}

scalar_t linear_program_t::do_eval(eval_t eval) const
{
    if (eval.has_grad())
    {
        eval.m_gx = m_c;
    }

    if (eval.has_hess())
    {
        eval.m_Hx.full(0.0);
    }

    return m_c.dot(eval.m_x);
}

bool linear_program_t::constrain(constraint_t&& constraint)
{
    return is_linear(constraint) && function_t::constrain(std::move(constraint));
}
