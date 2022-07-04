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

affine_constraint_t::affine_constraint_t(vector_t weights, scalar_t bias)
    : function_t("affine", weights.size())
    , m_weights(std::move(weights))
    , m_bias(bias)
{
    smooth(true);
    convex(true);
}

scalar_t affine_constraint_t::vgrad(const vector_t& x, vector_t* gx) const
{
    if (gx != nullptr)
    {
        gx->noalias() = m_weights;
    }

    return m_weights.dot(x) + m_bias;
}
