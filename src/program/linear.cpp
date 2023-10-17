#include <nano/program/linear.h>

using namespace nano;
using namespace nano::program;

linear_program_t::linear_program_t(vector_t c)
    : m_c(std::move(c))
{
    assert(m_c.size() > 0);
}
