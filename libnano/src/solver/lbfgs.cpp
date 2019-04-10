#include <deque>
#include "lbfgs.h"

using namespace nano;

solver_lbfgs_t::solver_lbfgs_t() :
    solver_t(1e-4, 9e-1)
{
}

json_t solver_lbfgs_t::config() const
{
    json_t json = solver_t::config();
    json["history"] = strcat(m_history_size, "(1,1000)");
    return json;
}

void solver_lbfgs_t::config(const json_t& json)
{
    solver_t::config(json);
    nano::from_json_range(json, "history", m_history_size, 1, 1000);
}

solver_state_t solver_lbfgs_t::minimize(const solver_function_t& function, const lsearch_t& lsearch,
    const vector_t& x0) const
{
    auto cstate = solver_state_t{function, x0};
    auto pstate = cstate;
    log(cstate);

    std::deque<vector_t> ss, ys;
    vector_t q, r;

    for (int i = 0; i < max_iterations(); ++ i)
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
            if (ss.size() > m_history_size)
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
