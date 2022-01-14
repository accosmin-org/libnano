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
    vector_t& x = state.d;      // buffer to reuse
    vector_t& g = state.g;      // buffer to reuse

    scalar_t fxk = state.f;
    vector_t xk = state.x, xk1, sumx(x0.size());
    vector_t L = vector_t::Constant(x0.size(), 1.0);

    const auto summands = function.summands();
    const auto minibatch = std::min(static_cast<tensor_size_t>(m_minibatch.get()), summands);
    const auto epoch_size = (summands + minibatch - 1) / minibatch;
    const auto max_epochs = std::max((this->max_evals() + summands - 1) / summands, int64_t{1});

    auto rng = make_rng(42);
    auto udist = make_udist<tensor_size_t>(0, summands - minibatch);

    x = x0;
    vector_t G = L;
    vector_t theta = vector_t::Zero(x0.size());
    vector_t reward = vector_t::Zero(x0.size());

    for (int64_t epoch = 0; epoch < max_epochs; ++ epoch)
    {
        sumx.setZero();
        for (int64_t t = 0; t < epoch_size; ++ t)
        {
            const auto begin = udist(rng);
            function.vgrad(x, &g, vgrad_config_t{make_range(begin, begin + minibatch)});

            // NB: update the estimation of the Lipschitz constant
            L.array() = L.array().max(2.0 * g.array().abs());

            theta -= g;
            G.array() += g.array().abs();
            reward.array() -= (x - x0).array() * g.array();

            const auto beta = (theta.array() / (G + L).array()).tanh() / L.array();

            x = x0.array() + beta * (L + reward).array();
            sumx += x;
        }

        // check best state and convergence after each epoch
        xk1 = sumx / static_cast<scalar_t>(epoch_size);
        const auto fxk1 = function.vgrad(xk1);

        state.update_if_better(xk1, fxk1);

        const auto iter_ok = true;
        const auto converged = this->converged(xk, fxk, xk1, fxk1);
        if (solver_t::done(function, state, iter_ok, converged))
        {
            break;
        }

        xk = xk1;
        fxk = fxk1;
    }

    // NB: make sure the gradient is updated at the returned point.
    state.f = function.vgrad(state.x, &state.g);
    return state;
}
