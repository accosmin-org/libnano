#include <nano/core/random.h>
#include <nano/solver/cocob.h>

using namespace nano;

solver_cocob_t::solver_cocob_t()
{
    monotonic(false);
}

solver_state_t solver_cocob_t::minimize(const function_t& function_, const vector_t& x0) const
{
    auto function = make_function(function_, x0);

    auto state = solver_state_t{function, x0};
    auto bstate = state;

    vector_t xk = state.x, xk1, sumx(x0.size());
    scalar_t fxk = state.f, fxk1;

    vector_t L = vector_t::Constant(x0.size(), 1.0);
    const auto summands = function.summands();
    const auto max_epochs = std::max((this->max_evals() + summands - 1) / summands, int64_t{1});

    auto rng = make_rng(42);
    auto udist = make_udist<tensor_size_t>(0, summands);

    vector_t G = L;
    vector_t theta = vector_t::Zero(x0.size());
    vector_t reward = vector_t::Zero(x0.size());

    for (int64_t epoch = 0; epoch < max_epochs; ++ epoch)
    {
        sumx.setZero();
        for (int64_t t = 0; t < summands; ++ t)
        {
            function.vgrad(state.x, &state.g, {udist(rng)});
            state.d = -state.g;

            // NB: update the estimation of the Lipschitz constant
            L.array() = L.array().max(2.0 * state.g.array().abs());

            theta += state.d;
            G.array() += state.d.array().abs();
            reward.array() += (state.x.array() - x0.array()) * state.d.array();

            const auto beta = (theta.array() / (G.array() + L.array())).tanh() / L.array();

            state.x = x0.array() + beta * (L.array() + reward.array());
            sumx += state.x;
        }

        // check best state and convergence after each epoch
        xk1 = sumx / static_cast<scalar_t>(summands);
        fxk1 = function.vgrad(xk1);

        bstate.update_if_better(xk1, fxk1);

        const auto iter_ok = true;
        const auto converged = this->converged(xk, fxk, xk1, fxk1);
        if (solver_t::done(function, bstate, iter_ok, converged))
        {
            break;
        }

        xk = xk1;
        fxk = fxk1;
    }

    // NB: make sure the gradient is updated at the returned point.
    bstate.f = function.vgrad(bstate.x, &bstate.g);
    return bstate;
}
