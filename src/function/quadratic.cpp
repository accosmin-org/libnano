#include <nano/core/overloaded.h>
#include <nano/function/quadratic.h>
#include <nano/function/util.h>

using namespace nano;
using namespace constraint;

namespace
{
matrix_t make_Q(const vector_t& q)
{
    const auto n = static_cast<tensor_size_t>(std::sqrt(static_cast<double>(2 * q.size())));
    assert(2 * q.size() == n * (n + 1));

    auto Q = matrix_t{n, n};
    for (tensor_size_t row = 0, idx = 0; row < n; ++row)
    {
        for (tensor_size_t col = row; col < n; ++col, ++idx)
        {
            Q(row, col) = q(idx);
            Q(col, row) = q(idx); // NOLINT(readability-suspicious-call-argument)
        }
    }

    return Q;
}
} // namespace

quadratic_program_t::quadratic_program_t(string_t id, matrix_t Q, vector_t c)
    : function_t(std::move(id), c.size())
    , m_Q(std::move(Q))
    , m_c(std::move(c))
{
    assert(m_c.size() > 0);
    assert(m_Q.rows() == m_c.size());
    assert(m_Q.cols() == m_c.size());

    smooth(smoothness::yes);
    convex(::is_convex(m_Q) ? convexity::yes : convexity::no);
    strong_convexity(::strong_convexity(m_Q));
}

quadratic_program_t::quadratic_program_t(string_t id, const vector_t& Q_upper_triangular, vector_t c)
    : quadratic_program_t(std::move(id), make_Q(Q_upper_triangular), std::move(c))
{
}

rfunction_t quadratic_program_t::clone() const
{
    return std::make_unique<quadratic_program_t>(*this);
}

scalar_t quadratic_program_t::do_eval(eval_t eval) const
{
    if (eval.has_grad())
    {
        eval.m_gx = m_Q * eval.m_x + m_c;
    }

    if (eval.has_hess())
    {
        eval.m_Hx = m_Q;
    }

    return eval.m_x.dot(0.5 * (m_Q * eval.m_x) + m_c);
}

bool quadratic_program_t::constrain(constraint_t&& constraint)
{
    return is_linear(constraint) && function_t::constrain(std::move(constraint));
}
