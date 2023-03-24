#include <nano/solver/lstep.h>

using namespace nano;

lsearch_step_t::lsearch_step_t(scalar_t tt, scalar_t ff, scalar_t dg)
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
    const auto d1 = u.g + v.g - 3 * (u.f - v.f) / (u.t - v.t);
    const auto d2 = (v.t > u.t ? +1 : -1) * std::sqrt(d1 * d1 - u.g * v.g);
    return v.t - (v.t - u.t) * (v.g + d2 - d1) / (v.g - u.g + 2 * d2);
}

scalar_t lsearch_step_t::quadratic(const lsearch_step_t& u, const lsearch_step_t& v, bool* convexity)
{
    const auto dt = u.t - v.t;
    const auto df = u.f - v.f;
    if (convexity != nullptr)
    {
        *convexity = (u.g - df / dt) * dt > 0;
    }
    return u.t - u.g * dt * dt / (2 * (u.g * dt - df));
}

scalar_t lsearch_step_t::secant(const lsearch_step_t& u, const lsearch_step_t& v)
{
    return (v.t * u.g - u.t * v.g) / (u.g - v.g);
}

scalar_t lsearch_step_t::bisection(const lsearch_step_t& u, const lsearch_step_t& v)
{
    return (u.t + v.t) / 2;
}

scalar_t lsearch_step_t::interpolate(const lsearch_step_t& u, const lsearch_step_t& v, interpolation_type method)
{
    const auto tc = cubic(u, v);
    const auto tq = quadratic(u, v);
    const auto tb = bisection(u, v);

    switch (method)
    {
    case interpolation_type::cubic: return std::isfinite(tc) ? tc : (std::isfinite(tq) ? tq : tb);

    case interpolation_type::quadratic: return std::isfinite(tq) ? tq : tb;

    case interpolation_type::bisection: [[fallthrough]];
    default: return tb;
    }
}
