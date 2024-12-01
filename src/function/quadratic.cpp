#include <nano/function/quadratic.h>
#include <nano/function/util.h>

using namespace nano;

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
    convex(::convex(m_Q) ? convexity::yes : convexity::no);
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

scalar_t quadratic_program_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    if (gx.size() == size())
    {
        gx = m_Q * x + m_c;
    }

    return x.dot(0.5 * (m_Q * x) + m_c);
}

void quadratic_program_t::reset(matrix_t Q, vector_t c)
{
    assert(c.size() == size());
    assert(Q.rows() == size());
    assert(Q.cols() == size());

    m_Q = std::move(Q);
    m_c = std::move(c);
}

void quadratic_program_t::xbest(vector_t xbest)
{
    assert(xbest.size() == size());

    m_xbest = std::move(xbest);
}

std::optional<optimum_t> quadratic_program_t::optimum() const
{
    if (m_xbest.size() == size())
    {
        const auto fbest = m_xbest.dot(0.5 * m_Q * m_xbest + m_c);
        return {m_xbest, fbest};
    }
    else
    {
        return {};
    }
}
