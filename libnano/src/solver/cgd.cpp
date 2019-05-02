#include "cgd.h"
#include <nano/numeric.h>

using namespace nano;

namespace nano
{
    ///
    /// \brief CGD update parameters (Hestenes and Stiefel, 1952 - see (1))
    ///
    struct cgd_step_HS
    {
        static scalar_t get(const solver_state_t& prev, const solver_state_t& curr)
        {
            return  curr.g.dot(curr.g - prev.g) /
                    prev.d.dot(curr.g - prev.g);
        }
    };

    ///
    /// \brief CGD update parameters (Fletcher and Reeves, 1964 - see (1))
    ///
    struct cgd_step_FR
    {
        static scalar_t get(const solver_state_t& prev, const solver_state_t& curr)
        {
            return  curr.g.squaredNorm() /
                    prev.g.squaredNorm();
        }
    };

    ///
    /// \brief CGD update parameters (Polak and Ribiere, 1969 - see (1))
    ///
    struct cgd_step_PRP
    {
        static scalar_t get(const solver_state_t& prev, const solver_state_t& curr)
        {
            return  std::max(scalar_t(0),            // PRP(+)
                    curr.g.dot(curr.g - prev.g) /
                    prev.g.squaredNorm());
        }
    };

    ///
    /// \brief CGD update parameters (Fletcher - Conjugate Descent, 1987 - see (1))
    ///
    struct cgd_step_CD
    {
        static scalar_t get(const solver_state_t& prev, const solver_state_t& curr)
        {
            return -curr.g.squaredNorm() /
                    prev.d.dot(prev.g);
        }
    };

    ///
    /// \brief CGD update parameters (Liu and Storey, 1991 - see (1))
    ///
    struct cgd_step_LS
    {
        static scalar_t get(const solver_state_t& prev, const solver_state_t& curr)
        {
            return -curr.g.dot(curr.g - prev.g) /
                    prev.d.dot(prev.g);
        }
    };

    ///
    /// \brief CGD update parameters (Dai and Yuan, 1999 - see (1))
    ///
    struct cgd_step_DY
    {
        static scalar_t get(const solver_state_t& prev, const solver_state_t& curr)
        {
            return  curr.g.squaredNorm() /
                    prev.d.dot(curr.g - prev.g);
        }
    };

    ///
    /// \brief CGD update parameters (Hager and Zhang, 2005 - see (1)) aka CG_DESCENT
    ///
    struct cgd_step_N
    {
        static scalar_t get(const solver_state_t& prev, const solver_state_t& curr)
        {
            const auto y = curr.g - prev.g;
            const auto div = +1 / prev.d.dot(y);

            const auto pd2 = prev.d.lpNorm<2>();
            const auto pg2 = prev.g.lpNorm<2>();
            const auto eta = -1 / (pd2 * std::min(scalar_t(0.01), pg2));

            // N+ (see modification in
            //      "A NEW CONJUGATE GRADIENT METHOD WITH GUARANTEED DESCENT AND AN EFFICIENT LINE SEARCH")
            return  std::max(eta,
                    div * (y - 2 * prev.d * y.squaredNorm() * div).dot(curr.g));
        }
    };

    ///
    /// \brief CGD update parameters (Dai and Yuan, 2001  - see (2), page 21)
    ///
    struct cgd_step_DYHS
    {
        static scalar_t get(const solver_state_t& prev, const solver_state_t& curr)
        {
            const auto dy = cgd_step_DY::get(prev, curr);
            const auto hs = cgd_step_HS::get(prev, curr);

            return std::max(scalar_t(0), std::min(dy, hs));
        }
    };

    ///
    /// \brief CGD update parameters (Dai, 2002 - see (2), page 22)
    ///
    struct cgd_step_DYCD
    {
        static scalar_t get(const solver_state_t& prev, const solver_state_t& curr)
        {
            return  curr.g.squaredNorm() /
                    std::max(prev.d.dot(curr.g - prev.g), -prev.d.dot(prev.g));
        }
    };
}

template <typename tcgd>
solver_cgd_t<tcgd>::solver_cgd_t() :
    solver_t(1e-4, 1e-1)
{
}

template <typename tcgd>
json_t solver_cgd_t<tcgd>::config() const
{
    json_t json = solver_t::config();
    json["orthotest"] = strcat(m_orthotest, "(0,1)");
    return json;
}

template <typename tcgd>
void solver_cgd_t<tcgd>::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();

    solver_t::config(json);
    nano::from_json_range(json, "orthotest", m_orthotest, eps, 1 - eps);
}

template <typename tcgd>
solver_state_t solver_cgd_t<tcgd>::minimize(const solver_function_t& function, const lsearch_t& lsearch,
    const vector_t& x0) const
{
    auto cstate = solver_state_t{function, x0};
    auto pstate = cstate;
    log(cstate);

    for (int i = 0; i < max_iterations(); ++ i)
    {
        // descent direction
        if (i == 0)
        {
            cstate.d = -cstate.g;
        }
        else
        {
            const scalar_t beta = tcgd::get(pstate, cstate);
            cstate.d = -cstate.g + beta * pstate.d;

            // restart:
            //  - if not a descent direction
            //  - or two consecutive gradients far from being orthogonal
            //      (see "Numerical optimization", Nocedal & Wright, 2nd edition, p.124-125)
            if (!cstate.has_descent())
            {
                cstate.d = -cstate.g;
            }
            else if (std::fabs(cstate.g.dot(pstate.g)) >= m_orthotest * cstate.g.dot(cstate.g))
            {
                cstate.d = -cstate.g;
            }
        }

        // line-search
        pstate = cstate;
        const auto iter_ok = lsearch.get(cstate);
        if (solver_t::done(function, cstate, iter_ok))
        {
            break;
        }
    }

    return cstate;
}

template class nano::solver_cgd_t<cgd_step_HS>;
template class nano::solver_cgd_t<cgd_step_FR>;
template class nano::solver_cgd_t<cgd_step_PRP>;
template class nano::solver_cgd_t<cgd_step_CD>;
template class nano::solver_cgd_t<cgd_step_LS>;
template class nano::solver_cgd_t<cgd_step_DY>;
template class nano::solver_cgd_t<cgd_step_N>;
template class nano::solver_cgd_t<cgd_step_DYCD>;
template class nano::solver_cgd_t<cgd_step_DYHS>;
