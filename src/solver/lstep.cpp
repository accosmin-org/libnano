#include <nano/solver/lstep.h>

using namespace nano;

lsearch_step_t::lsearch_step_t(const scalar_t tt, const scalar_t ff, const scalar_t dg)
    : t(tt)
    , f(ff)
    , g(dg)
{
}

lsearch_step_t::lsearch_step_t(const solver_state_t& state, const vector_t& descent, const scalar_t step_size)
    : lsearch_step_t(step_size, state.fx(), state.dg(descent))
{
}

scalar_t lsearch_step_t::cubic(const lsearch_step_t& u, const lsearch_step_t& v)
{
    const auto d1 = u.g + v.g - 3.0 * (u.f - v.f) / (u.t - v.t);
    const auto d2 = (v.t > u.t ? +1.0 : -1.0) * std::sqrt(d1 * d1 - u.g * v.g);
    return v.t - (v.t - u.t) * (v.g + d2 - d1) / (v.g - u.g + 2.0 * d2);
}

scalar_t lsearch_step_t::quadratic(const lsearch_step_t& u, const lsearch_step_t& v, bool* convexity)
{
    const auto dt = u.t - v.t;
    const auto df = u.f - v.f;
    if (convexity != nullptr)
    {
        *convexity = (dt * u.g - df) > 0.0;
    }
    return u.t - 0.5 * u.g * dt / (u.g - df / dt);
}

scalar_t lsearch_step_t::secant(const lsearch_step_t& u, const lsearch_step_t& v)
{
    return (v.t * u.g - u.t * v.g) / (u.g - v.g);
}

scalar_t lsearch_step_t::bisection(const lsearch_step_t& u, const lsearch_step_t& v)
{
    return 0.5 * (u.t + v.t);
}

scalar_t lsearch_step_t::interpolate(const lsearch_step_t& u, const lsearch_step_t& v, interpolation_type method)
{
    const auto tc = cubic(u, v);
    const auto tq = quadratic(u, v);
    const auto tb = bisection(u, v);

    switch (method)
    {
    case interpolation_type::cubic:
        if (std::isfinite(tc))
        {
            return tc;
        }
        [[fallthrough]];

    case interpolation_type::quadratic:
        if (std::isfinite(tq))
        {
            return tq;
        }
        [[fallthrough]];

    default:
        return tb;
    }
}
