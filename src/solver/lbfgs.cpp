#include <deque>
#include <nano/solver/lbfgs.h>

using namespace nano;

solver_lbfgs_t::solver_lbfgs_t() :
    solver_t(1e-4, 9e-1)
{
}

solver_state_t solver_lbfgs_t::minimize(const function_t& function_, const vector_t& x0) const
{
    auto lsearch = make_lsearch();
    auto function = make_function(function_, x0);

    auto cstate = solver_state_t{function, x0};

    if (solver_t::done(function, cstate, true))
    {
        return cstate;
    }

    vector_t q, r;
    solver_state_t pstate;
    std::deque<vector_t> ss, ys;

    for (int64_t i = 0; i < max_iterations(); ++ i)
    {
        // descent direction
        //      (see "Numerical optimization", Nocedal & Wright, 2nd edition, p.178)
        q = cstate.g;

        const auto hsize = ss.size();

        std::vector<scalar_t> alphas(hsize);
        for (size_t j = 0; j < hsize; ++ j)
        {
            const auto& s = ss[hsize - 1 - j];
            const auto& y = ys[hsize - 1 - j];

            const scalar_t alpha = s.dot(q) / s.dot(y);
            q.noalias() -= alpha * y;
            alphas[j] = alpha;
        }

        if (ss.empty())
        {
            r = q;
        }
        else
        {
            const auto& s = ss[hsize - 1];
            const auto& y = ys[hsize - 1];

            r = s.dot(y) / y.dot(y) * q;
        }

        for (size_t j = 0; j < hsize; ++ j)
        {
            const auto& s = ss[j];
            const auto& y = ys[j];

            const scalar_t alpha = alphas[hsize - 1 - j];
            const scalar_t beta = y.dot(r) / s.dot(y);
            r.noalias() += s * (alpha - beta);
        }

        cstate.d = -r;
        const auto has_descent = cstate.has_descent();

        // Force descent direction
        if (!has_descent)
        {
            cstate.d = -cstate.g;
        }

        // line-search
        pstate = cstate;
        const auto iter_ok = lsearch.get(cstate);
        if (solver_t::done(function, cstate, iter_ok))
        {
            break;
        }

        // Skip the update if the curvature condition is not satisfied
        //      "A Multi-Batch L-BFGS Method for Machine Learning", page 6 - the non-convex case
        if (has_descent)
        {
            ss.emplace_back(cstate.x - pstate.x);
            ys.emplace_back(cstate.g - pstate.g);
            if (ss.size() > history())
            {
                ss.pop_front();
                ys.pop_front();
            }
        }
        else
        {
            ss.clear();
            ys.clear();
        }
    }

    return cstate;
}
