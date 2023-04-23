#include <nano/solver/quasi.h>

using namespace nano;

namespace
{
template <typename tvector>
auto SR1(const matrix_t& H, const tvector& dx, const tvector& dg)
{
    return H + (dx - H * dg) * (dx - H * dg).transpose() / (dx - H * dg).dot(dg);
}

template <typename tvector>
void SR1(matrix_t& H, const tvector& dx, const tvector& dg, const scalar_t r)
{
    const auto denom = (dx - H * dg).dot(dg);
    const auto apply = std::fabs(denom) >= r * dx.norm() * (dx - H * dg).norm();

    if (apply)
    {
        H = SR1(H, dx, dg);
    }
}

template <typename tvector>
auto DFP(const matrix_t& H, const tvector& dx, const tvector& dg)
{
    return H + (dx * dx.transpose()) / dx.dot(dg) - (H * dg * dg.transpose() * H) / (dg.transpose() * H * dg);
}

template <typename tvector>
auto BFGS(const matrix_t& H, const tvector& dx, const tvector& dg)
{
    const auto I = matrix_t::Identity(H.rows(), H.cols());

    return (I - dx * dg.transpose() / dx.dot(dg)) * H * (I - dg * dx.transpose() / dx.dot(dg)) +
           dx * dx.transpose() / dx.dot(dg);
}

template <typename tvector>
auto HOSHINO(const matrix_t& H, const tvector& dx, const tvector& dg)
{
    const auto phi = dx.dot(dg) / (dx.dot(dg) + dg.transpose() * H * dg);

    return (1 - phi) * DFP(H, dx, dg) + phi * BFGS(H, dx, dg);
}

template <typename tvector>
void FLETCHER(matrix_t& H, const tvector& dx, const tvector& dg)
{
    const auto phi = dx.dot(dg) / (dx.dot(dg) - dg.transpose() * H * dg);

    if (phi < scalar_t(0))
    {
        H = DFP(H, dx, dg);
    }
    else if (phi > scalar_t(1))
    {
        H = BFGS(H, dx, dg);
    }
    else
    {
        H = SR1(H, dx, dg);
    }
}
} // namespace

solver_quasi_t::solver_quasi_t(string_t id)
    : solver_t(std::move(id))
{
    type(solver_type::line_search);
    parameter("solver::tolerance") = std::make_tuple(1e-4, 9e-1);

    register_parameter(parameter_t::make_enum("solver::quasi::initialization", initialization::identity));
}

solver_state_t solver_quasi_t::do_minimize(const function_t& function, const vector_t& x0) const
{
    const auto max_evals = parameter("solver::max_evals").value<tensor_size_t>();
    const auto epsilon   = parameter("solver::epsilon").value<scalar_t>();
    const auto init      = parameter("solver::quasi::initialization").value<initialization>();

    auto cstate = solver_state_t{function, x0}; // current state
    if (solver_t::done(cstate, true, cstate.gradient_test() < epsilon))
    {
        return cstate;
    }

    auto lsearch = make_lsearch();
    auto pstate  = solver_state_t{}; // previous state
    auto descent = vector_t{};       // descent direction

    // current approximation of the Hessian's inverse
    matrix_t H = matrix_t::Identity(function.size(), function.size());

    bool first_iteration = false;
    while (function.fcalls() + function.gcalls() < max_evals)
    {
        // descent direction
        descent = -H * cstate.gx();

        // restart:
        //  - if not a descent direction
        if (!cstate.has_descent(descent))
        {
            descent = -cstate.gx();
            H.setIdentity();
        }

        // line-search
        pstate             = cstate;
        const auto iter_ok = lsearch.get(cstate, descent);
        if (solver_t::done(cstate, iter_ok, cstate.gradient_test() < epsilon))
        {
            break;
        }

        // initialize the Hessian's inverse
        if (first_iteration && init == initialization::scaled)
        {
            const auto dx = cstate.x() - pstate.x();
            const auto dg = cstate.gx() - pstate.gx();
            H             = matrix_t::Identity(H.rows(), H.cols()) * dx.dot(dg) / dg.dot(dg);
        }
        first_iteration = false;

        // update approximation of the Hessian
        update(pstate, cstate, H);
    }

    return cstate.valid() ? cstate : pstate;
}

solver_quasi_dfp_t::solver_quasi_dfp_t()
    : solver_quasi_t("dfp")
{
}

solver_quasi_bfgs_t::solver_quasi_bfgs_t()
    : solver_quasi_t("bfgs")
{
}

solver_quasi_hoshino_t::solver_quasi_hoshino_t()
    : solver_quasi_t("hoshino")
{
}

solver_quasi_fletcher_t::solver_quasi_fletcher_t()
    : solver_quasi_t("fletcher")
{
}

solver_quasi_sr1_t::solver_quasi_sr1_t()
    : solver_quasi_t("sr1")
{
    register_parameter(parameter_t::make_scalar("solver::quasi::sr1::r", 0, LT, 1e-8, LT, 1));
}

rsolver_t solver_quasi_dfp_t::clone() const
{
    return std::make_unique<solver_quasi_dfp_t>(*this);
}

rsolver_t solver_quasi_bfgs_t::clone() const
{
    return std::make_unique<solver_quasi_bfgs_t>(*this);
}

rsolver_t solver_quasi_hoshino_t::clone() const
{
    return std::make_unique<solver_quasi_hoshino_t>(*this);
}

rsolver_t solver_quasi_fletcher_t::clone() const
{
    return std::make_unique<solver_quasi_fletcher_t>(*this);
}

rsolver_t solver_quasi_sr1_t::clone() const
{
    return std::make_unique<solver_quasi_sr1_t>(*this);
}

void solver_quasi_sr1_t::update(const solver_state_t& prev, const solver_state_t& curr, matrix_t& H) const
{
    const auto r = parameter("solver::quasi::sr1::r").value<scalar_t>();

    ::SR1(H, curr.x() - prev.x(), curr.gx() - prev.gx(), r);
}

void solver_quasi_dfp_t::update(const solver_state_t& prev, const solver_state_t& curr, matrix_t& H) const
{
    H = ::DFP(H, curr.x() - prev.x(), curr.gx() - prev.gx());
}

void solver_quasi_bfgs_t::update(const solver_state_t& prev, const solver_state_t& curr, matrix_t& H) const
{
    H = ::BFGS(H, curr.x() - prev.x(), curr.gx() - prev.gx());
}

void solver_quasi_hoshino_t::update(const solver_state_t& prev, const solver_state_t& curr, matrix_t& H) const
{
    H = ::HOSHINO(H, curr.x() - prev.x(), curr.gx() - prev.gx());
}

void solver_quasi_fletcher_t::update(const solver_state_t& prev, const solver_state_t& curr, matrix_t& H) const
{
    ::FLETCHER(H, curr.x() - prev.x(), curr.gx() - prev.gx());
}
