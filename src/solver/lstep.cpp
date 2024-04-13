#include <nano/solver/lstep.h>

using namespace nano;

namespace
{
template <typename toperator>
scalar_t interpolate(const lsearch_step_t& u, const lsearch_step_t& v, const toperator& op)
{
    const auto ut = static_cast<long double>(u.t);
    const auto uf = static_cast<long double>(u.f);
    const auto ug = static_cast<long double>(u.g);

    const auto vt = static_cast<long double>(v.t);
    const auto vf = static_cast<long double>(v.f);
    const auto vg = static_cast<long double>(v.g);

    return static_cast<scalar_t>(op(ut, uf, ug, vt, vf, vg));
}
} // namespace

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
    const auto op = [](const auto ut, const auto uf, const auto ug, const auto vt, const auto vf, const auto vg)
    {
        const auto d1 = ug + vg - 3.0L * (uf - vf) / (ut - vt);
        const auto d2 = (vt > ut ? +1.0L : -1.0L) * std::sqrt(d1 * d1 - ug * vg);
        return vt - (vt - ut) * (vg + d2 - d1) / (vg - ug + 2.0L * d2);
    };
    return ::interpolate(u, v, op);
}

scalar_t lsearch_step_t::quadratic(const lsearch_step_t& u, const lsearch_step_t& v, bool* convexity)
{
    const auto op = [&](const auto ut, const auto uf, const auto ug, const auto vt, const auto vf, const auto)
    {
        const auto dt = ut - vt;
        const auto df = uf - vf;
        if (convexity != nullptr)
        {
            *convexity = (dt * ug - df) > 0.0;
        }
        return ut - 0.5L * ug * dt / (ug - df / dt);
    };
    return ::interpolate(u, v, op);
}

scalar_t lsearch_step_t::secant(const lsearch_step_t& u, const lsearch_step_t& v)
{
    const auto op = [](const auto ut, const auto, const auto ug, const auto vt, const auto, const auto vg)
    { return (vt * ug - ut * vg) / (ug - vg); };
    return ::interpolate(u, v, op);
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

    default: return tb;
    }
}
