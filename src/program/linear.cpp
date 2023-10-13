#include <nano/program/linear.h>

using namespace nano;
using namespace nano::program;

linear_program_t::linear_program_t(vector_t c)
    : m_c(std::move(c))
{
    assert(m_c.size() > 0);
}

linear_program_t nano::program::operator&(const linear_program_t& program, const equality_t& eq)
{
    auto result = program;
    result.m_eq = result.m_eq & eq;
    assert(!result.m_eq || result.m_eq.m_A.cols() == result.m_c.size());
    return result;
}

linear_program_t nano::program::operator&(const linear_program_t& program, const inequality_t& ineq)
{
    auto result   = program;
    result.m_ineq = result.m_ineq & ineq;
    assert(!result.m_ineq || result.m_ineq.m_A.cols() == result.m_c.size());
    return result;
}
