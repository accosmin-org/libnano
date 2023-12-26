#include <nano/program/solver.h>
#include <nano/solver/proximal.h>
#include <nano/tensor/stack.h>

#include <iomanip>
#include <iostream>

using namespace nano;

namespace
{
auto make_logger(const int stop_at_iters = -1)
{
    return [stop_at_iters = stop_at_iters](const program::solver_state_t& state)
    {
        std::cout << std::fixed << std::setprecision(16) << "i=" << state.m_iters << ",fx=" << state.m_fx
                  << ",eta=" << state.m_eta << ",rdual=" << state.m_rdual.lpNorm<Eigen::Infinity>()
                  << ",rcent=" << state.m_rcent.lpNorm<Eigen::Infinity>()
                  << ",rprim=" << state.m_rprim.lpNorm<Eigen::Infinity>() << ",rcond=" << state.m_ldlt_rcond
                  << (state.m_ldlt_positive ? "(+)" : "(-)") << "[" << state.m_status << "]" << std::endl;
        ;
        return state.m_iters != stop_at_iters;
    };
}

struct point_t
{
    point_t() {}

    template <typename tvector>
    point_t(const scalar_t f, tvector g, const scalar_t gdotz)
        : m_f(f)
        , m_g(std::move(g))
        , m_gdotz(gdotz)
    {
    }

    // attributes
    scalar_t m_f{0.0};     ///< f(z)
    vector_t m_g;          ///< f'(z)
    scalar_t m_gdotz{0.0}; ///< f'(z).dot(z)
};

struct bundle_t
{
    bundle_t()
        : m_solver(make_logger())
    {
    }

    void append(const vector_cmap_t z, const scalar_t fz, const vector_cmap_t gz)
    {
        m_points.emplace_back(fz, gz, gz.dot(z));
    }

    scalar_t value(const vector_cmap_t x) const
    {
        auto value = std::numeric_limits<scalar_t>::lowest();
        for (const auto& point : m_points)
        {
            assert(point.m_g.size() == x.size());

            value = std::max(value, point.m_f + point.m_g.dot(x) - point.m_gdotz);
        }

        return value;
    }

    void proximal(const vector_cmap_t x, const scalar_t miu, vector_t& z)
    {
        const auto n = x.size();
        const auto m = static_cast<tensor_size_t>(m_points.size());

        // objective: 0.5 * [z|w].dot(Q * [z|w]) + r.dot([z|w])
        auto Q = stack<scalar_t>(n + 1, n + 1, miu * matrix_t::identity(n, n), matrix_t::zero(n, 1),
                                 matrix_t::zero(1, n + 1));
        auto r = stack<scalar_t>(n + 1, -miu * x, vector_t::constant(1, 1.0));

        // inequality constraints: A * [z|w] <= b
        auto A = matrix_t{m, n + 1};
        auto b = vector_t{m};
        for (tensor_size_t i = 0; i < m; ++i)
        {
            const auto& point = m_points[static_cast<size_t>(i)];
            assert(point.m_g.size() == x.size());

            A.row(i).segment(0, n) = point.m_g.transpose();
            A(i, n)                = -1.0;
            b(i)                   = point.m_gdotz - point.m_f;
        }

        // solve quadratic program
        const auto inequality = program::make_inequality(std::move(A), std::move(b));
        const auto program    = program::make_quadratic(std::move(Q), std::move(r), inequality);

        const auto solution = m_solver.solve(program);
        assert(solution.m_status == solver_status::converged);
        z = solution.m_x.slice(0, n);
    }

    // FIXME: implement sugradient aggregation or selection to keep the bundle small
    // FIXME: store bundle more efficiently

    // attributes
    std::vector<point_t> m_points; ///< bundle information
    program::solver_t    m_solver; ///<
};

struct sequence_t
{
    scalar_t update() { return m_lambda = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * m_lambda * m_lambda)); }

    // attributes
    scalar_t m_lambda{1.0};
};
} // namespace

struct proximal::sequence1_t final : public sequence_t
{
    auto make_alpha_beta()
    {
        const auto curr  = m_lambda;
        const auto next  = update();
        const auto alpha = (curr - 1.0) / next;
        const auto beta  = 0.0;
        return std::make_tuple(alpha, beta);
    }
};

struct proximal::sequence2_t final : public sequence_t
{
    auto make_alpha_beta()
    {
        const auto curr  = m_lambda;
        const auto next  = update();
        const auto alpha = (curr - 1.0) / next;
        const auto beta  = curr / next;
        return std::make_tuple(alpha, beta);
    }
};

struct proximal::fpba1_type_id_t
{
    static auto str() { return "fpba1"; }
};

struct proximal::fpba2_type_id_t
{
    static auto str() { return "fpba2"; }
};

template <typename tsequence, typename ttype_id>
base_solver_fpba_t<tsequence, ttype_id>::base_solver_fpba_t()
    : solver_t(ttype_id::str())
{
    type(solver_type::non_monotonic);

    const auto basename = scat("solver::", ttype_id::str(), "::");

    register_parameter(parameter_t::make_scalar(basename + "miu", 0, LT, 1.0, LT, 1e+6));
    register_parameter(parameter_t::make_scalar(basename + "sigma", 0, LT, 0.5, LT, 1.0));
}

template <typename tsequence, typename ttype_id>
rsolver_t base_solver_fpba_t<tsequence, ttype_id>::clone() const
{
    return std::make_unique<base_solver_fpba_t<tsequence, ttype_id>>(*this);
}

template <typename tsequence, typename ttype_id>
solver_state_t base_solver_fpba_t<tsequence, ttype_id>::do_minimize(const function_t& function,
                                                                    const vector_t&   x0) const
{
    const auto basename  = scat("solver::", ttype_id::str(), "::");
    const auto max_evals = parameter("solver::max_evals").template value<tensor_size_t>();
    const auto epsilon   = parameter("solver::epsilon").template value<scalar_t>();
    const auto miu       = parameter(basename + "miu").template value<scalar_t>();
    const auto sigma     = parameter(basename + "sigma").template value<scalar_t>();

    (void)epsilon;
    (void)sigma;

    auto x  = x0;
    auto y  = x0;
    auto z  = x0;
    auto gz = vector_t{x0.size()};

    auto state = solver_state_t{function, x0};
    std::cout << std::fixed << std::setprecision(10) << "calls=" << function.fcalls() << "|" << function.gcalls()
              << ",x0=" << x0.transpose() << ",fx0=" << state.fx() << std::endl;

    auto bundle   = bundle_t{};
    auto sequence = tsequence{};

    bundle.append(state.x(), state.fx(), state.gx());

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        bundle.proximal(x, miu, z);

        const auto fz = function.vgrad(z, gz);
        const auto ek = epsilon / (2.0 * sequence.m_lambda);

        std::cout << std::fixed << std::setprecision(10) << "calls=" << function.fcalls() << "|" << function.gcalls()
                  << ",z=" << z.transpose() << ",fz=" << fz << ",bv=" << bundle.value(z) << ",ek=" << ek
                  << ",lk=" << sequence.m_lambda << ",df=" << (state.fx() - bundle.value(state.x())) << std::endl;

        if (fz - bundle.value(z) <= ek)
        {
            const auto [ak, bk] = sequence.make_alpha_beta();

            x = y + ak * (z - y) + bk * (z - x);
            y = z;
        }
        else
        {
            bundle.append(z, fz, gz);
        }

        state.update_if_better(z, gz, fz);

        // TODO: stopping criterion - best state close to the bundle value?!
    }

    return state;
}

template class nano::base_solver_fpba_t<proximal::sequence1_t, proximal::fpba1_type_id_t>;
template class nano::base_solver_fpba_t<proximal::sequence2_t, proximal::fpba2_type_id_t>;
