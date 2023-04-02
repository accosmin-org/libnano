#include <nano/solver/cocob.h>

using namespace nano;

solver_cocob_t::solver_cocob_t()
    : solver_t("cocob")
{
    type(solver_type::non_monotonic);

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
    const auto patience  = parameter("solver::cocob::patience").value<tensor_size_t>();

    auto state = solver_state_t{function, x0}; // NB: keeps track of the best state

    auto L  = vector_t{state.gx().array().abs() + 1e-12};
    auto x  = state.x();
    auto gx = state.gx();

    auto G      = L;
    auto theta  = make_full_vector<scalar_t>(x0.size(), 0.0);
    auto reward = make_full_vector<scalar_t>(x0.size(), 0.0);

    while (function.fcalls() < max_evals)
    {
        const auto fx = function.vgrad(x, &gx);
        state.update_if_better(x, fx);

        const auto iter_ok   = std::isfinite(fx);
        const auto converged = state.value_test(patience) < epsilon;
        if (solver_t::done(state, iter_ok, converged))
        {
            break;
        }

        // NB: update the estimation of the Lipschitz constant
        const auto restart = (gx.array().abs() - L.array()).minCoeff() > 0;
        L.array()          = L.array().max(gx.array().abs());

        if (restart)
        {
            // reset state when a gradient with a larger magnitude is found
            G      = L;
            theta.array()  = 0;
            reward.array() = 0;
        }

        // compute parameter update
        theta -= gx;
        G.array() += gx.array().abs();
        reward.array() -= (x - x0).array() * gx.array();

        const auto beta = (theta.array() / (G + L).array()).tanh() / L.array();
        x               = x0.array() + beta * (L + reward).array();

        /*if (function.name() == "trid[3D]" || function.name() == "trid[4D]")
        {
            std::cout << std::fixed << std::setprecision(10) << "iter=" << iter << ",x=" << x.transpose()
                      << ",fx=" << fx << "\n\tbeta=" << beta.transpose() << "\n\ttheta=" << theta.transpose()
                      << "\n\treward=" << reward.transpose() << "\n\tL=" << L.transpose() //<< ",restart=" << restart
                      << std::endl;
        }*/
    }

    // NB: make sure the gradient is updated at the returned point.
    state.update(state.x());
    return state;
}
