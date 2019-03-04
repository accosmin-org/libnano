#include "lsearch_init.h"

using namespace nano;

scalar_t lsearch_unit_init_t::get(const solver_state_t&, const int)
{
    return 1;
}

scalar_t lsearch_linear_init_t::get(const solver_state_t& state, const int iteration)
{
    scalar_t t0;

    const auto dg = state.dg();
    switch (iteration)
    {
    case 0:
        t0 = 1;
        break;

    default:
        // NB: the line-search length is from the previous iteration!
        t0 = state.t * m_prevdg / dg;
        break;
    }

    m_prevdg = dg;
    return t0;
}

scalar_t lsearch_quadratic_init_t::get(const solver_state_t& state, const int iteration)
{
    scalar_t t0;

    switch (iteration)
    {
    case 0:
        t0 = 1;
        break;

    default:
        t0 = scalar_t(1.01) * 2 * (state.f - m_prevf) / state.dg();
        break;
    }

    m_prevf = state.f;
    return t0;
}

scalar_t lsearch_cgdescent_init_t::get(const solver_state_t& state, const int iteration)
{
    scalar_t t0;

    const auto phi0 = scalar_t(0.01);
    const auto phi1 = scalar_t(0.1);
    const auto phi2 = scalar_t(2.0);

    switch (iteration)
    {
    case 0:
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
        break;

    default:
        {
            lsearch_step_t step0
            {
                0, state.f, state.dg()
            };

            lsearch_step_t stepx
            {
                state.t * phi1,
                // NB: the line-search length is from the previous iteration!
                state.function->vgrad(state.x + state.t * phi1 * state.d),
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
                t0 = state.t * phi2;
            }
        }
        break;
    }

    return t0;
}
