#include <nano/function/constraint.h>
#include <nano/solver/state.h>

using namespace nano;

solver_state_t::solver_state_t() = default;

solver_state_t::solver_state_t(const function_t& ffunction, vector_t x0)
    : function(&ffunction)
    , x(std::move(x0))
    , g(vector_t::Zero(x.size()))
    , d(vector_t::Zero(x.size()))
    , p(vector_t::Constant(static_cast<tensor_size_t>(function->constraints().size()),
                           std::sqrt(std::numeric_limits<scalar_t>::max())))
    , f(function->vgrad(x, &g))
{
    update_penalties();
}

bool solver_state_t::update_if_better(const vector_t& x, const vector_t& gx, scalar_t fx)
{
    if (std::isfinite(fx) && fx < f)
    {
        this->x = x;
        this->f = fx;
        this->g = gx;
        update_penalties();
        return true;
    }
    else
    {
        return false;
    }
}

void solver_state_t::update_penalties()
{
    const auto& constraints = function->constraints();
    for (size_t i = 0U, size = constraints.size(); i < size; ++i)
    {
        p(static_cast<tensor_size_t>(i)) = ::nano::valid(constraints[i], x);
    }
}

bool nano::operator<(const solver_state_t& lhs, const solver_state_t& rhs)
{
    return (std::isfinite(lhs.f) ? lhs.f : std::numeric_limits<scalar_t>::max()) <
           (std::isfinite(rhs.f) ? rhs.f : std::numeric_limits<scalar_t>::max());
}

std::ostream& nano::operator<<(std::ostream& os, solver_status status)
{
    return os << scat(status);
}

std::ostream& nano::operator<<(std::ostream& os, const solver_state_t& state)
{
    return os << "i=" << state.inner_iters << "|" << state.outer_iters << ",calls=" << state.fcalls << "|"
              << state.gcalls << ",f=" << state.f << ",g=" << state.convergence_criterion() << ",p=" << state.p.sum()
              << "[" << state.status << "]";
}
