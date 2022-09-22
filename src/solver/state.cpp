#include <nano/function/constraint.h>
#include <nano/solver/state.h>

using namespace nano;

static auto fmax()
{
    return std::sqrt(std::numeric_limits<scalar_t>::max());
}

solver_state_t::solver_state_t() = default;

solver_state_t::solver_state_t(const function_t& ffunction, vector_t x0)
    : function(&ffunction)
    , x(std::move(x0))
    , g(vector_t::Zero(x.size()))
    , d(vector_t::Zero(x.size()))
    , f(function->vgrad(x, &g))
    , ceq(vector_t::Constant(::nano::count_equalities(function->constraints()), fmax()))
    , cineq(vector_t::Constant(::nano::count_inequalities(function->constraints()), fmax()))
{
    update_constraints();
}

bool solver_state_t::update_if_better(const vector_t& x, const vector_t& gx, scalar_t fx)
{
    if (std::isfinite(fx) && fx < f)
    {
        this->x = x;
        this->f = fx;
        this->g = gx;
        update_constraints();
        return true;
    }
    else
    {
        return false;
    }
}

bool solver_state_t::update_if_better(const vector_t& x, scalar_t fx)
{
    return update_if_better(x, g, fx);
}

bool solver_state_t::update_if_better_constrained(const solver_state_t& cstate, const scalar_t epsilon)
{
    auto updated = false;

    if (cstate.valid() && cstate.constraint_test() <= constraint_test() + epsilon)
    {
        updated = true;

        // NB: the original function value should be returned!
        update(cstate.x);
        status = cstate.status;
    }
    inner_iters += cstate.inner_iters;
    outer_iters++;

    return updated;
}

void solver_state_t::update_constraints()
{
    tensor_size_t ieq = 0, ineq = 0;
    for (const auto& constraint : function->constraints())
    {
        if (::nano::is_equality(constraint))
        {
            this->ceq(ieq++) = ::vgrad(constraint, x);
        }
        else
        {
            this->cineq(ineq++) = ::vgrad(constraint, x);
        }
    }
}

scalar_t solver_state_t::gradient_test() const
{
    return g.lpNorm<Eigen::Infinity>() / std::max(scalar_t(1), std::fabs(f));
}

scalar_t solver_state_t::constraint_test() const
{
    scalar_t test = 0;
    if (ceq.size() > 0)
    {
        test += ceq.lpNorm<Eigen::Infinity>();
    }
    if (cineq.size() > 0)
    {
        test += cineq.array().max(0.0).matrix().lpNorm<Eigen::Infinity>();
    }
    return test;
}

bool solver_state_t::valid() const
{
    return std::isfinite(t) && std::isfinite(f) && g.array().isFinite().all() && ceq.array().isFinite().all() &&
           cineq.array().isFinite().all();
}

template <>
enum_map_t<solver_status> nano::enum_string<solver_status>()
{
    return {
        {solver_status::converged, "converged"},
        {solver_status::max_iters, "max_iters"},
        {   solver_status::failed,    "failed"},
        {  solver_status::stopped,   "stopped"}
    };
}

bool nano::operator<(const solver_state_t& lhs, const solver_state_t& rhs)
{
    return (std::isfinite(lhs.f) ? lhs.f : std::numeric_limits<scalar_t>::max()) <
           (std::isfinite(rhs.f) ? rhs.f : std::numeric_limits<scalar_t>::max());
}

std::ostream& nano::operator<<(std::ostream& stream, solver_status status)
{
    return stream << scat(status);
}

std::ostream& nano::operator<<(std::ostream& stream, const solver_state_t& state)
{
    stream << "i=" << state.outer_iters << "|" << state.inner_iters;
    stream << ",calls=" << state.fcalls << "|" << state.gcalls;
    stream << ",f=" << state.f << ",g=" << state.gradient_test();
    if (state.ceq.size() + state.cineq.size() > 0)
    {
        stream << ",c=" << state.constraint_test();
    }
    return stream << "[" << state.status << "]";
}
