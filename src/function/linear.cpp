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

scalar_t linear_program_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    if (gx.size() == size())
    {
        gx = m_c;
    }

    return m_c.dot(x);
}

void linear_program_t::reset(vector_t c)
{
    assert(c.size() == size());

    m_c = std::move(c);
}

bool linear_program_t::constrain(constraint_t&& constraint)
{
    return std::visit(overloaded{[](const constant_t&) { return true; },          ///<
                                 [](const minimum_t&) { return true; },           ///<
                                 [](const maximum_t&) { return true; },           ///<
                                 [](const linear_equality_t&) { return true; },   ///<
                                 [](const linear_inequality_t&) { return true; }, ///<
                                 [](const euclidean_ball_t&) { return false; },   ///<
                                 [](const quadratic_t&) { return false; },        ///<
                                 [](const functional_t&) { return false; }},      ///<
                      constraint) &&
           function_t::constrain(std::move(constraint));
}
