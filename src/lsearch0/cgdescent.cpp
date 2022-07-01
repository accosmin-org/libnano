#include <nano/lsearch0/cgdescent.h>

using namespace nano;

lsearch0_cgdescent_t::lsearch0_cgdescent_t()
{
    register_parameter(parameter_t::make_scalar("lsearch0::cgdescent::phi0", 0, LT, 0.01, LT, 1));
    register_parameter(parameter_t::make_scalar("lsearch0::cgdescent::phi1", 0, LT, 0.10, LT, 1));
    register_parameter(parameter_t::make_scalar("lsearch0::cgdescent::phi2", 1, LT, 2.00, LT, 1e+6));
}

rlsearch0_t lsearch0_cgdescent_t::clone() const
{
    return std::make_unique<lsearch0_cgdescent_t>(*this);
}

scalar_t lsearch0_cgdescent_t::get(const solver_state_t& state)
{
    const auto phi0 = parameter("lsearch0::cgdescent::phi0").value<scalar_t>();
    const auto phi1 = parameter("lsearch0::cgdescent::phi1").value<scalar_t>();
    const auto phi2 = parameter("lsearch0::cgdescent::phi2").value<scalar_t>();

    scalar_t t0 = 0;

    if (state.m_iterations <= 1)
    {
        const auto xnorm = state.x.lpNorm<Eigen::Infinity>();
        const auto fnorm = std::fabs(state.f);

        if (xnorm > 0)
        {
            t0 = phi0 * xnorm / state.g.lpNorm<Eigen::Infinity>();
        }
        else if (fnorm > 0)
        {
            t0 = phi0 * fnorm / state.g.squaredNorm();
        }
        else
        {
            t0 = 1;
        }
    }
    else
    {
        lsearch_step_t step0{0, state.f, state.dg()};

        lsearch_step_t stepx{state.t * phi1,
                             // NB: the line-search length is from the previous iteration!
                             state.function->vgrad(state.x + state.t * phi1 * state.d), 0};

        bool       convexity = false;
        const auto tq        = lsearch_step_t::quadratic(step0, stepx, &convexity);
        if (stepx.f < step0.f && convexity)
        {
            t0 = tq;
        }
        else
        {
            t0 = state.t * phi2;
        }
    }

    log(state, t0);
    return t0;
}
