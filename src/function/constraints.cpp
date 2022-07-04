#include <Eigen/Eigenvalues>
#include <nano/function/constraints.h>

using namespace nano;

ball_constraint_t::ball_constraint_t(vector_t origin, scalar_t radius)
    : function_t("ball", origin.size())
    , m_origin(std::move(origin))
    , m_radius(radius)
{
    smooth(true);
    convex(true);
}

scalar_t ball_constraint_t::vgrad(const vector_t& x, vector_t* gx) const
{
    if (gx != nullptr)
    {
        gx->noalias() = 2.0 * (x - m_origin);
    }

    return (x - m_origin).dot(x - m_origin) - m_radius * m_radius;
}

affine_constraint_t::affine_constraint_t(vector_t q, scalar_t r)
    : function_t("affine", q.size())
    , m_q(std::move(q))
    , m_r(r)
{
    smooth(true);
    convex(true);
}

scalar_t affine_constraint_t::vgrad(const vector_t& x, vector_t* gx) const
{
    if (gx != nullptr)
    {
        gx->noalias() = m_q;
    }

    return m_q.dot(x) + m_r;
}

quadratic_constraint_t::quadratic_constraint_t(matrix_t P, vector_t q, scalar_t r)
    : function_t("quadratic", q.size())
    , m_P(std::move(P))
    , m_q(std::move(q))
    , m_r(r)
{
    const auto eigenvalues         = m_P.eigenvalues();
    const auto positive_eigenvalue = [](const auto& eigenvalue) { return eigenvalue.real() >= 0.0; };

    smooth(true);
    convex(std::all_of(begin(eigenvalues), end(eigenvalues), positive_eigenvalue));
}

scalar_t quadratic_constraint_t::vgrad(const vector_t& x, vector_t* gx) const
{
    if (gx != nullptr)
    {
        gx->noalias() = m_P * x + m_q;
    }

    return 0.5 * x.dot(m_P * x) + m_q.dot(x) + m_r;
}
