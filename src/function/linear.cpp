#include <nano/function/linear.h>

using namespace nano;

linear_program_t::linear_program_t(string_t id, vector_t c)
    : function_t(std::move(id), c.size())
    : m_c(std::move(c))
{
    smooth(smoothness::yes);
    convex(convexity::yes);
    strong_convexity(0.0);
}

rfunction_t linear_program_t::clone() const
{
    return std::make_unique<linear_program_t>(*this);
}

scalar_t linear_program_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    if (gx.size() == size())
    {
        gx = m_c;
    }

    return m_c.dot(x);
}
