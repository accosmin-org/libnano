#include <nano/lsearch0/cgdescent.h>

using namespace nano;

rlsearch0_t lsearch0_cgdescent_t::clone() const
{
    return std::make_unique<lsearch0_cgdescent_t>(*this);
}

scalar_t lsearch0_cgdescent_t::get(const solver_state_t& state)
{
    scalar_t t0 = 0;

    if (state.m_iterations <= 1)
    {
        const auto xnorm = state.x.lpNorm<Eigen::Infinity>();
        const auto fnorm = std::fabs(state.f);

        if (xnorm > 0)
        {
            t0 = phi0() * xnorm / state.g.lpNorm<Eigen::Infinity>();
        }
        else if (fnorm > 0)
        {
            t0 = phi0() * fnorm / state.g.squaredNorm();
        }
        else
        {
            t0 = 1;
        }
    }
    else
    {
        lsearch_step_t step0
        {
            0, state.f, state.dg()
        };

        lsearch_step_t stepx
        {
            state.t * phi1(),
            // NB: the line-search length is from the previous iteration!
            state.function->vgrad(state.x + state.t * phi1() * state.d),
            0
        };

        bool convexity = false;
        const auto tq = lsearch_step_t::quadratic(step0, stepx, &convexity);
        if (stepx.f < step0.f && convexity)
        {
            t0 = tq;
        }
        else
        {
            t0 = state.t * phi2();
        }
    }

    log(state, t0);
    return t0;
}
