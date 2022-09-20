#include <nano/solver/cgd.h>

using namespace nano;

namespace
{
    scalar_t HS(const solver_state_t& prev, const solver_state_t& curr)
    {
        return curr.g.dot(curr.g - prev.g) / prev.d.dot(curr.g - prev.g);
    }

    scalar_t FR(const solver_state_t& prev, const solver_state_t& curr)
    {
        return curr.g.squaredNorm() / prev.g.squaredNorm();
    }

    scalar_t PR(const solver_state_t& prev, const solver_state_t& curr)
    {
        return curr.g.dot(curr.g - prev.g) / prev.g.squaredNorm();
    }

    scalar_t CD(const solver_state_t& prev, const solver_state_t& curr)
    {
        return -curr.g.squaredNorm() / prev.d.dot(prev.g);
    }

    scalar_t LS(const solver_state_t& prev, const solver_state_t& curr)
    {
        return -curr.g.dot(curr.g - prev.g) / prev.d.dot(prev.g);
    }

    scalar_t DY(const solver_state_t& prev, const solver_state_t& curr)
    {
        return curr.g.squaredNorm() / prev.d.dot(curr.g - prev.g);
    }

    scalar_t N(const solver_state_t& prev, const solver_state_t& curr, scalar_t eta) // N(+) - see (3)
    {
        const auto y   = curr.g - prev.g;
        const auto div = +1 / prev.d.dot(y);

        const auto pd2 = prev.d.lpNorm<2>();
        const auto pg2 = prev.g.lpNorm<2>();
        eta            = -1 / (pd2 * std::min(eta, pg2));

        return std::max(eta, div * (y - 2 * prev.d * y.squaredNorm() * div).dot(curr.g));
    }

    scalar_t DYHS(const solver_state_t& prev, const solver_state_t& curr)
    {
        return std::max(scalar_t(0), std::min(DY(prev, curr), HS(prev, curr)));
    }

    scalar_t DYCD(const solver_state_t& prev, const solver_state_t& curr)
    {
        return curr.g.squaredNorm() / std::max(prev.d.dot(curr.g - prev.g), -prev.d.dot(prev.g));
    }

    scalar_t FRPR(const solver_state_t& prev, const solver_state_t& curr)
    {
        const auto fr = ::FR(prev, curr);
        const auto pr = ::PR(prev, curr);

        return (pr < -fr) ? -fr : (std::fabs(pr) <= fr) ? pr : fr;
    }
} // namespace

solver_cgd_t::solver_cgd_t(string_t id)
    : solver_t(std::move(id))
{
    type(solver_type::line_search);
    parameter("solver::tolerance") = std::make_tuple(1e-4, 1e-1);

    register_parameter(parameter_t::make_scalar("solver::cgd::orthotest", 0, LT, 0.1, LT, 1));
}

solver_state_t solver_cgd_t::do_minimize(const function_t& function, const vector_t& x0) const
{
    const auto max_evals = parameter("solver::max_evals").value<int64_t>();
    const auto epsilon   = parameter("solver::epsilon").value<scalar_t>();
    const auto orthotest = parameter("solver::cgd::orthotest").value<scalar_t>();

    auto lsearch = make_lsearch();

    auto cstate = solver_state_t{function, x0};
    auto pstate = cstate;
    if (solver_t::done(function, cstate, true, cstate.gradient_test() < epsilon))
    {
        return cstate;
    }

    for (int64_t i = 0; function.fcalls() < max_evals; ++i)
    {
        // descent direction
        if (i == 0)
        {
            cstate.d = -cstate.g;
        }
        else
        {
            const auto beta = this->beta(pstate, cstate);
            cstate.d        = -cstate.g + beta * pstate.d;

            // restart:
            //  - if not a descent direction
            //  - or two consecutive gradients far from being orthogonal
            //      (see "Numerical optimization", Nocedal & Wright, 2nd edition, p.124-125)
            if (!cstate.has_descent() || (std::fabs(cstate.g.dot(pstate.g)) >= orthotest * cstate.g.dot(cstate.g)))
            {
                cstate.d = -cstate.g;
            }
        }

        // line-search
        pstate             = cstate;
        const auto iter_ok = lsearch.get(cstate);
        if (solver_t::done(function, cstate, iter_ok, cstate.gradient_test() < epsilon))
        {
            break;
        }
    }

    return cstate.valid() ? cstate : pstate;
}

solver_cgd_cd_t::solver_cgd_cd_t()
    : solver_cgd_t("cgd-cd")
{
}

solver_cgd_dy_t::solver_cgd_dy_t()
    : solver_cgd_t("cgd-dy")
{
}

solver_cgd_fr_t::solver_cgd_fr_t()
    : solver_cgd_t("cgd-fr")
{
}

solver_cgd_hs_t::solver_cgd_hs_t()
    : solver_cgd_t("cgd-hs")
{
}

solver_cgd_ls_t::solver_cgd_ls_t()
    : solver_cgd_t("cgd-ls")
{
}

solver_cgd_pr_t::solver_cgd_pr_t()
    : solver_cgd_t("cgd-pr")
{
}

solver_cgd_dycd_t::solver_cgd_dycd_t()
    : solver_cgd_t("cgd-dycd")
{
}

solver_cgd_dyhs_t::solver_cgd_dyhs_t()
    : solver_cgd_t("cgd-dyhs")
{
}

solver_cgd_frpr_t::solver_cgd_frpr_t()
    : solver_cgd_t("cgd-frpr")
{
}

solver_cgd_n_t::solver_cgd_n_t()
    : solver_cgd_t("cgd-n")
{
    register_parameter(parameter_t::make_scalar("solver::cgdN::eta", 0, LT, 0.01, LT, 1e+6));
}

rsolver_t solver_cgd_cd_t::clone() const
{
    return std::make_unique<solver_cgd_cd_t>(*this);
}

rsolver_t solver_cgd_dy_t::clone() const
{
    return std::make_unique<solver_cgd_dy_t>(*this);
}

rsolver_t solver_cgd_fr_t::clone() const
{
    return std::make_unique<solver_cgd_fr_t>(*this);
}

rsolver_t solver_cgd_hs_t::clone() const
{
    return std::make_unique<solver_cgd_hs_t>(*this);
}

rsolver_t solver_cgd_ls_t::clone() const
{
    return std::make_unique<solver_cgd_ls_t>(*this);
}

rsolver_t solver_cgd_pr_t::clone() const
{
    return std::make_unique<solver_cgd_pr_t>(*this);
}

rsolver_t solver_cgd_dycd_t::clone() const
{
    return std::make_unique<solver_cgd_dycd_t>(*this);
}

rsolver_t solver_cgd_dyhs_t::clone() const
{
    return std::make_unique<solver_cgd_dyhs_t>(*this);
}

rsolver_t solver_cgd_frpr_t::clone() const
{
    return std::make_unique<solver_cgd_frpr_t>(*this);
}

rsolver_t solver_cgd_n_t::clone() const
{
    return std::make_unique<solver_cgd_n_t>(*this);
}

scalar_t solver_cgd_hs_t::beta(const solver_state_t& prev, const solver_state_t& curr) const
{
    // HS+ - see (1)
    return std::max(::HS(prev, curr), scalar_t(0));
}

scalar_t solver_cgd_fr_t::beta(const solver_state_t& prev, const solver_state_t& curr) const
{
    return ::FR(prev, curr);
}

scalar_t solver_cgd_pr_t::beta(const solver_state_t& prev, const solver_state_t& curr) const
{
    // PR+ - see (1)
    return std::max(::PR(prev, curr), scalar_t(0));
}

scalar_t solver_cgd_cd_t::beta(const solver_state_t& prev, const solver_state_t& curr) const
{
    return ::CD(prev, curr);
}

scalar_t solver_cgd_ls_t::beta(const solver_state_t& prev, const solver_state_t& curr) const
{
    // LS+ - see (1)
    return std::max(::LS(prev, curr), scalar_t(0));
}

scalar_t solver_cgd_dy_t::beta(const solver_state_t& prev, const solver_state_t& curr) const
{
    return ::DY(prev, curr);
}

scalar_t solver_cgd_n_t::beta(const solver_state_t& prev, const solver_state_t& curr) const
{
    const auto eta = parameter("solver::cgdN::eta").value<scalar_t>();

    return ::N(prev, curr, eta);
}

scalar_t solver_cgd_dycd_t::beta(const solver_state_t& prev, const solver_state_t& curr) const
{
    return ::DYCD(prev, curr);
}

scalar_t solver_cgd_dyhs_t::beta(const solver_state_t& prev, const solver_state_t& curr) const
{
    return ::DYHS(prev, curr);
}

scalar_t solver_cgd_frpr_t::beta(const solver_state_t& prev, const solver_state_t& curr) const
{
    return ::FRPR(prev, curr);
}
