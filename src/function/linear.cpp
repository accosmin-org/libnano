#include <nano/function/linear.h>

using namespace nano;

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

void linear_program_t::xbest(vector_t xbest)
{
    assert(xbest.size() == size());

    m_xbest = std::move(xbest);
}

std::optional<optimum_t> linear_program_t::optimum() const
{
    if (m_xbest.size() == size())
    {
        const auto fbest = do_vgrad(m_xbest, vector_map_t{});
        return optimum_t{m_xbest, fbest};
    }
    else
    {
        return {};
    }
}
