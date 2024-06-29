#include <lsearchk/morethuente.h>
#include <nano/core/numeric.h>

using namespace nano;

namespace
{
void dcstep(scalar_t& stx, scalar_t& fx, scalar_t& dx, scalar_t& sty, scalar_t& fy, scalar_t& dy, scalar_t& stp,
            const scalar_t& fp, const scalar_t& dp, bool& brackt, const scalar_t stpmin, const scalar_t stpmax,
            const scalar_t delta)
{
    scalar_t stpc = 0, stpq = 0, stpf = 0;

    const auto sgnd = dp * (dx / std::fabs(dx));

    if (fp > fx)
    {
        stpc = lsearch_step_t::cubic({stx, fx, dx}, {stp, fp, dp});
        stpq = lsearch_step_t::quadratic({stx, fx, dx}, {stp, fp, dp});

        if (std::fabs(stpc - stx) < std::fabs(stpq - stx))
        {
            stpf = stpc;
        }
        else
        {
            stpf = stpc + (stpq - stpc) / 2;
        }
        brackt = true;
    }

    else if (sgnd < 0)
    {
        stpc = lsearch_step_t::cubic({stx, fx, dx}, {stp, fp, dp});
        stpq = lsearch_step_t::secant({stx, fx, dx}, {stp, fp, dp});

        if (std::fabs(stpc - stp) > std::fabs(stpq - stp))
        {
            stpf = stpc;
        }
        else
        {
            stpf = stpq;
        }
        brackt = true;
    }

    else if (std::fabs(dp) < std::fabs(dx))
    {
        stpc = lsearch_step_t::cubic({stx, fx, dx}, {stp, fp, dp});
        stpq = lsearch_step_t::secant({stx, fx, dx}, {stp, fp, dp});

        if (std::isfinite(stpc) && (stp - stx) * (stpc - stp) > 0)
        {
        }
        else if (stp > stx)
        {
            stpc = stpmax;
        }
        else
        {
            stpc = stpmin;
        }

        if (brackt)
        {
            if (std::fabs(stpc - stp) < std::fabs(stpq - stp))
            {
                stpf = stpc;
            }
            else
            {
                stpf = stpq;
            }
            if (stp > stx)
            {
                stpf = std::min(stpf, stp + (sty - stp) * delta);
            }
            else
            {
                stpf = std::max(stpf, stp + (sty - stp) * delta);
            }
        }
        else
        {
            if (std::fabs(stpc - stp) > std::fabs(stpq - stp))
            {
                stpf = stpc;
            }
            else
            {
                stpf = stpq;
            }
            stpf = std::min(stpmax, stpf);
            stpf = std::max(stpmin, stpf);
        }
    }

    else
    {
        if (brackt)
        {
            stpc = lsearch_step_t::cubic({stp, fp, dp}, {sty, fy, dy});

            stpf = stpc;
        }
        else if (stp > stx)
        {
            stpf = stpmax;
        }
        else
        {
            stpf = stpmin;
        }
    }

    if (fp > fx)
    {
        sty = stp;
        fy  = fp;
        dy  = dp;
    }
    else
    {
        if (sgnd < 0.)
        {
            sty = stx;
            fy  = fx;
            dy  = dx;
        }
        stx = stp;
        fx  = fp;
        dx  = dp;
    }

    stp = stpf;
}
} // namespace

lsearchk_morethuente_t::lsearchk_morethuente_t()
    : lsearchk_t("morethuente")
{
    type(lsearch_type::strong_wolfe);
    register_parameter(parameter_t::make_scalar("lsearchk::morethuente::delta", 0, LT, 0.66, LT, 1));
}

rlsearchk_t lsearchk_morethuente_t::clone() const
{
    return std::make_unique<lsearchk_morethuente_t>(*this);
}

lsearchk_t::result_t lsearchk_morethuente_t::do_get(const solver_state_t& state0, const vector_t& descent,
                                                    scalar_t step_size, solver_state_t& state) const
{
    const auto [c1, c2]       = parameter("lsearchk::tolerance").value_pair<scalar_t>();
    const auto max_iterations = parameter("lsearchk::max_iterations").value<int>();
    const auto delta          = parameter("lsearchk::morethuente::delta").value<scalar_t>();

    const auto ftol = c1;
    const auto gtol = c2;
    const auto xtol = epsilon0<scalar_t>();

    const auto finit = state0.fx();
    const auto ginit = state0.dg(descent);
    const auto gtest = ftol * ginit;

    int  stage  = 1;
    bool brackt = false;

    scalar_t stp   = step_size;
    scalar_t f     = state.fx();
    scalar_t g     = state.dg(descent);
    scalar_t stmin = 0;
    scalar_t stmax = stp + stp * 4;

    scalar_t width  = stpmax() - stpmin();
    scalar_t width1 = 2 * width;

    scalar_t stx = 0;
    scalar_t fx  = finit;
    scalar_t gx  = ginit;
    scalar_t sty = 0;
    scalar_t fy  = finit;
    scalar_t gy  = ginit;

    for (int i = 0; i < max_iterations; ++i)
    {
        const auto ftest = finit + stp * gtest;
        if (stage == 1 && f <= ftest && g >= scalar_t(0))
        {
            stage = 2;
        }

        // Check if further progress can be made
        if (brackt && (stp <= stmin || stp >= stmax))
        {
            return {true, stp};
        }
        if (brackt && (stmax - stmin) <= xtol * stmax)
        {
            return {true, stp};
        }
        if (stp >= stpmax() && f <= ftest && g <= gtest)
        {
            return {true, stp};
        }
        if (stp <= stpmin() && (f > ftest || g >= gtest))
        {
            return {true, stp};
        }

        // Check convergence
        if (f <= ftest && std::fabs(g) <= gtol * (-ginit))
        {
            return {true, stp};
        }

        // Interpolate the next point to evaluate
        if (stage == 1 && f <= fx && f > ftest)
        {
            auto fm  = f - stp * gtest;
            auto fxm = fx - stx * gtest;
            auto fym = fy - sty * gtest;
            auto gm  = g - gtest;
            auto gxm = gx - gtest;
            auto gym = gy - gtest;

            dcstep(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax, delta);

            fx = fxm + stx * gtest;
            fy = fym + sty * gtest;
            gx = gxm + gtest;
            gy = gym + gtest;
        }
        else
        {
            dcstep(stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin, stmax, delta);
        }

        // Decide if a bisection step is needed
        if (brackt)
        {
            if (std::fabs(sty - stx) >= width1 * scalar_t(.66))
            {
                stp = stx + (sty - stx) * scalar_t(0.5);
            }
            width1 = width;
            width  = std::fabs(sty - stx);

            // Set the minimum and maximum steps allowed for stp
            stmin = std::min(stx, sty);
            stmax = std::max(stx, sty);
        }
        else
        {
            // Set the minimum and maximum steps allowed for stp
            stmin = stp + (stp - stx) * scalar_t(1.1);
            stmax = stp + (stp - stx) * scalar_t(4.0);
        }

        // Force the step to be within the bounds stpmax and stpmin
        stp = std::clamp(stp, stpmin(), stpmax());

        // If further progress is not possible, let stp be the best point obtained during the search
        if ((brackt && (stp <= stmin || stp >= stmax)) || (brackt && stmax - stmin <= xtol * stmax))
        {
            stp = stx;
        }

        // Obtain another function and derivative
        if (!update(state, state0, descent, stp))
        {
            return {false, stp};
        }
        f = state.fx();
        g = state.dg(descent);
    }

    return {false, stp};
}
