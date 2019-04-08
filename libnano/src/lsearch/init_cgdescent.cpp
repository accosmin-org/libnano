#include <nano/numeric.h>
#include "init_cgdescent.h"

using namespace nano;

json_t lsearch_cgdescent_init_t::config() const
{
    json_t json;
    json["phi0"] = strcat(m_phi0, "(0,1)");
    json["phi1"] = strcat(m_phi0, "(0,1)");
    json["phi2"] = strcat(m_phi0, "(1,inf)");
    return json;
}

void lsearch_cgdescent_init_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();
    const auto inf = 1 / eps;

    from_json_range(json, "phi0", m_phi0, eps, 1 - eps);
    from_json_range(json, "phi1", m_phi1, eps, 1 - eps);
    from_json_range(json, "phi2", m_phi2, 1 + eps, inf);
}

scalar_t lsearch_cgdescent_init_t::get(const solver_state_t& state)
{
    scalar_t t0;

    if (state.m_iterations <= 1)
    {
        const auto xnorm = state.x.lpNorm<Eigen::Infinity>();
        const auto fnorm = std::fabs(state.f);

        if (xnorm > 0)
        {
            t0 = m_phi0 * xnorm / state.g.lpNorm<Eigen::Infinity>();
        }
        else if (fnorm > 0)
        {
            t0 = m_phi0 * fnorm / state.g.squaredNorm();
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
            state.t * m_phi1,
            // NB: the line-search length is from the previous iteration!
            state.function->vgrad(state.x + state.t * m_phi1 * state.d),
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
            t0 = state.t * m_phi2;
        }
    }

    log(state, t0);
    return t0;
}
