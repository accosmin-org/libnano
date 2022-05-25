#include <nano/solver/quasi.h>

using namespace nano;

namespace
{
    template <typename tvector>
    auto SR1(const matrix_t& H, const tvector& dx, const tvector& dg)
    {
        return  H + (dx - H * dg) * (dx - H * dg).transpose() / (dx - H * dg).dot(dg);
    }

    template <typename tvector>
    void SR1(matrix_t& H, const tvector& dx, const tvector& dg, const scalar_t r)
    {
        const auto denom = (dx - H * dg).dot(dg);
        const auto apply = std::fabs(denom) >= r * dx.norm() * (dx - H * dg).norm();

        if (apply)
        {
            H = SR1(H, dx, dg);
        }
    }

    template <typename tvector>
    auto DFP(const matrix_t& H, const tvector& dx, const tvector& dg)
    {
        return  H + (dx * dx.transpose()) / dx.dot(dg) -
                (H * dg * dg.transpose() * H) / (dg.transpose() * H * dg);
    }

    template <typename tvector>
    auto BFGS(const matrix_t& H, const tvector& dx, const tvector& dg)
    {
        const auto I = matrix_t::Identity(H.rows(), H.cols());

        return  (I - dx * dg.transpose() / dx.dot(dg)) * H * (I - dg * dx.transpose() / dx.dot(dg)) +
                dx * dx.transpose() / dx.dot(dg);
    }

    template <typename tvector>
    auto HOSHINO(const matrix_t& H, const tvector& dx, const tvector& dg)
    {
        const auto phi = dx.dot(dg) / (dx.dot(dg) + dg.transpose() * H * dg);

        return  (1 - phi) * DFP(H, dx, dg) + phi * BFGS(H, dx, dg);
    }

    template <typename tvector>
    void FLETCHER(matrix_t& H, const tvector& dx, const tvector& dg)
    {
        const auto phi = dx.dot(dg) / (dx.dot(dg) - dg.transpose() * H * dg);

        if (phi < scalar_t(0))
        {
            H = DFP(H, dx, dg);
        }
        else if (phi > scalar_t(1))
        {
            H = BFGS(H, dx, dg);
        }
        else
        {
            H = SR1(H, dx, dg);
        }
    }
}

solver_quasi_t::solver_quasi_t()
{
    monotonic(true);
    parameter("solver::tolerance") = std::make_tuple(1e-4, 9e-1);

    register_parameter(parameter_t::make_enum("solver::quasi::initialization", initialization::identity));
}

solver_state_t solver_quasi_t::minimize(const function_t& function_, const vector_t& x0) const
{
    const auto max_evals = parameter("solver::max_evals").value<int64_t>();
    const auto epsilon = parameter("solver::epsilon").value<scalar_t>();
    const auto init = parameter("solver::quasi::initialization").value<initialization>();

    auto lsearch = make_lsearch();
    auto function = make_function(function_, x0);

    auto cstate = solver_state_t{function, x0};
    if (solver_t::done(function, cstate, true, cstate.converged(epsilon)))
    {
        return cstate;
    }

    solver_state_t pstate;

    // current approximation of the Hessian's inverse
    matrix_t H = matrix_t::Identity(function.size(), function.size());

    for (int64_t i = 0; function.fcalls() < max_evals; ++ i)
    {
        // descent direction
        cstate.d = -H * cstate.g;

        // restart:
        //  - if not a descent direction
        if (!cstate.has_descent())
        {
            cstate.d = -cstate.g;
            H.setIdentity();
        }

        // line-search
        pstate = cstate;
        const auto iter_ok = lsearch.get(cstate);
        if (solver_t::done(function, cstate, iter_ok, cstate.converged(epsilon)))
        {
            break;
        }

        // initialize the Hessian's inverse
        if (i == 0 && init == initialization::scaled)
        {
            const auto dx = cstate.x - pstate.x;
            const auto dg = cstate.g - pstate.g;
            H = matrix_t::Identity(H.rows(), H.cols()) * dx.dot(dg) / dg.dot(dg);
        }

        // update approximation of the Hessian
        update(pstate, cstate, H);
    }

    return static_cast<bool>(cstate) ? cstate : pstate;
}

solver_quasi_sr1_t::solver_quasi_sr1_t()
{
    register_parameter(parameter_t::make_scalar("solver::quasiSR1::r", 0, LT, 1e-8, LT, 1));
}

void solver_quasi_sr1_t::update(const solver_state_t& prev, const solver_state_t& curr, matrix_t& H) const
{
    const auto r = parameter("solver::quasiSR1::r").value<scalar_t>();

    ::SR1(H, curr.x - prev.x, curr.g - prev.g, r);
}

void solver_quasi_dfp_t::update(const solver_state_t& prev, const solver_state_t& curr, matrix_t& H) const
{
    H = ::DFP(H, curr.x - prev.x, curr.g - prev.g);
}

void solver_quasi_bfgs_t::update(const solver_state_t& prev, const solver_state_t& curr, matrix_t& H) const
{
    H = ::BFGS(H, curr.x - prev.x, curr.g - prev.g);
}

void solver_quasi_hoshino_t::update(const solver_state_t& prev, const solver_state_t& curr, matrix_t& H) const
{
    H = ::HOSHINO(H, curr.x - prev.x, curr.g - prev.g);
}

void solver_quasi_fletcher_t::update(const solver_state_t& prev, const solver_state_t& curr, matrix_t& H) const
{
    ::FLETCHER(H, curr.x - prev.x, curr.g - prev.g);
}
