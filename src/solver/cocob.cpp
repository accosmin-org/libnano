#include <nano/solver/cocob.h>

using namespace nano;

solver_cocob_t::solver_cocob_t()
    : solver_t("cocob")
{
    type(solver_type::non_monotonic);

    register_parameter(parameter_t::make_scalar("solver::cocob::L0", 0.0, LT, 1e-6, LE, 1e+0));
    register_parameter(parameter_t::make_integer("solver::cocob::patience", 10, LE, 100, LE, 1e+6));
}

rsolver_t solver_cocob_t::clone() const
{
    return std::make_unique<solver_cocob_t>(*this);
}

solver_state_t solver_cocob_t::do_minimize(const function_t& function, const vector_t& x0) const
{
    const auto epsilon   = parameter("solver::epsilon").value<scalar_t>();
    const auto max_evals = parameter("solver::max_evals").value<int>();
    const auto L0        = parameter("solver::cocob::L0").value<scalar_t>();
    const auto patience  = parameter("solver::cocob::patience").value<tensor_size_t>();

    auto state = solver_state_t{function, x0}; // NB: keeps track of the best state

    auto L  = vector_t{state.gx().array().abs() + L0};
    auto x  = state.x();
    auto gx = state.gx();

    auto xref   = x0;
    auto G      = L;
    auto theta  = make_full_vector<scalar_t>(x0.size(), 0.0);
    auto reward = make_full_vector<scalar_t>(x0.size(), 0.0);

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        const auto fx = function.vgrad(x, &gx);
        state.update_if_better(x, gx, fx);

        const auto iter_ok   = std::isfinite(fx);
        const auto converged = state.value_test(patience) < epsilon;
        if (solver_t::done(state, iter_ok, converged))
        {
            break;
        }

        // NB: update the estimation of the Lipschitz constant
        // NB: reset state when a gradient with a larger magnitude is found
        for (tensor_size_t i = 0, size = gx.size(); i < size; ++i)
        {
            if (gx(i) < -L(i))
            {
                xref(i)   = x(i);
                L(i)      = -gx(i) + L0;
                G(i)      = L(i);
                theta(i)  = 0.0;
                reward(i) = 0.0;
            }
            else if (gx(i) > L(i))
            {
                xref(i)   = x(i);
                L(i)      = gx(i) + L0;
                G(i)      = L(i);
                theta(i)  = 0.0;
                reward(i) = 0.0;
            }
        }

        // compute parameter update
        theta -= gx;
        G.array() += gx.array().abs();
        reward.array() -= (x - xref).array() * gx.array();

        const auto beta = (theta.array() / (G + L).array()).tanh() / L.array();
        x               = xref.array() + beta * (L + reward).array();
    }

    return state;
}
