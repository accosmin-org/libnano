#include <nano/function/linprog.h>

using namespace nano;

linprog_function_t::linprog_function_t(vector_t c, matrix_t A)
    : function_t("linprog", c.size())
    , m_c(std::move(c))
    , m_A(std::move(A))
{
    convex(convexity::yes);
    smooth(smoothness::yes);

    // TODO: add constraints!
}

rfunction_t linprog_function_t::clone() const
{
    return std::make_unique<linprog_function_t>(*this);
}

scalar_t linprog_function_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    assert(x.size() == size());

    if (gx != nullptr)
    {
        *gx = m_c;
    }
    return m_c.dot(x);
}
