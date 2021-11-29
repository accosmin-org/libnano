#include <nano/solver/osga.h>

using namespace nano;

solver_osga_t::solver_osga_t()
{
}

solver_state_t solver_osga_t::iterate(const solver_function_t& function, const lsearch_t&, const vector_t& x0) const
{
    return solver_state_t{function, x0};
}

solver_state_t solver_osga_t::minimize(const function_t& f, const vector_t& x0) const
{
    assert(f.size() == x0.size());

    auto function = solver_function_t{f};

    /*
    const auto delta = m_delta.get();
    const auto alpha_max = m_alpha_max.get();
    const auto kappa_prime = m_kappas.get1();
    const auto kappa = m_kappas.get2();

    const auto miu = f.strong_convexity() / 2.0;

    auto x = x0, xb = x0;


    scalar_t alpha = ;
    scalar_t h = ;
    scalar_t gamma = ;
    scalar_t eta = ;
    scalar_t u = ;

    const auto update() = [&] (scalar_t h_hat, scalar_t gamma_hat, scalar_t eta_hat, scalar_t u_hat)
    {
        const auto R = (eta - eta_hat) / (delta * alpha * eta);

        alpha = (R < 1.0) ?
            (alpha * std::exp(-kappa)) :
            std::min(alpha * std::exp(kappa_prime * (R - 1.0)), alpha_max);

        if (eta_hat < eta)
        {
            h = h_hat;
            u = u_hat;
            eta = eta_hat;
            gamma = gamma_hat;
        }
    };

    if (solver_t::done(function, cstate, true))
    {
        return cstate;
    }

    for (int64_t i = 0; i < max_iterations(); ++ i)
    {
        // TODO: check convergence



    }*/

    auto cstate = solver_state_t{function, x0};
    return cstate;
}
