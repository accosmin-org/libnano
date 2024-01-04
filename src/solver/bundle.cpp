#include <nano/program/solver.h>
#include <nano/solver/bundle.h>

using namespace nano;

namespace
{
struct point_t
{
    point_t() = default;

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
    explicit bundle_t(const solver_state_t& state)
        : bundle_t(state.x().size())
    {
        append(state.x(), state.gx(), state.fx());
    }

    explicit bundle_t(const tensor_size_t n)
        : m_program(matrix_t{matrix_t::zero(n + 1, n + 1)}, vector_t{vector_t::constant(n + 1, 1.0)})
        , m_x0(n + 1)
    {
    }

    void append(const vector_cmap_t z, const vector_cmap_t gz, const scalar_t fz)
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

    vector_cmap_t proximal(const vector_cmap_t x, const scalar_t miu)
    {
        const auto n = x.size();
        const auto m = static_cast<tensor_size_t>(m_points.size());

        // objective: 0.5 * [z|w].dot(Q * [z|w]) + c.dot([z|w])
        m_program.m_Q.block(0, 0, n, n) = miu * matrix_t::identity(n, n);
        m_program.m_c.segment(0, n)     = -miu * x;

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
        m_program.m_ineq = program::make_inequality(std::move(A), std::move(b));

        // solve quadratic program
        m_x0.segment(0, n) = x.vector();
        m_x0(n)            = value(x) + 0.1;
        assert(m_program.feasible(m_x0));

        m_x0 = m_solver.solve(m_program, m_x0).m_x;
        return m_x0.slice(0, n);
    }

    // FIXME: implement sugradient aggregation or selection to keep the bundle small
    // FIXME: store bundle more efficiently

    using program_t = program::quadratic_program_t;

    // attributes
    std::vector<point_t> m_points;  ///< bundle information
    program::solver_t    m_solver;  ///< buffer: quadratic program solver
    program_t            m_program; ///< buffer: quadratic program
    vector_t             m_x0;      ///< buffer: starting point for the quadratic program
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

    auto state = solver_state_t{function, x0};

    auto x  = x0;
    auto y  = x0;
    auto z  = x0;
    auto p  = vector_t{x0.size()};
    auto gz = vector_t{x0.size()};
    auto fx = state.fx();

    auto bundle   = bundle_t{state};
    auto sequence = tsequence{};

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        // estimate proximal point
        z             = bundle.proximal(x, miu);
        const auto fz = function.vgrad(z, gz);
        const auto bz = bundle.value(z);

        // check if proximal point is approximated well enough
        const auto ek = (1.0 - sigma) * (fx - bz);
        if (fz - bz <= ek + std::numeric_limits<scalar_t>::epsilon())
        {
            // update stability center
            const auto [ak, bk] = sequence.make_alpha_beta();

            x  = z + ak * (z - y) + bk * (z - x);
            y  = z;
            fx = function.vgrad(x);

            state.update(y);

            // check convergence: small gap between `y=z` and its approximated proximal point
            p                    = bundle.proximal(z, miu);
            const auto fp        = bundle.value(p) + 0.5 * miu * (z - p).dot(z - p);
            const auto iter_ok   = std::isfinite(fz) && std::isfinite(fp);
            const auto converged = fz - fp < epsilon;
            if (solver_t::done(state, iter_ok, converged))
            {
                break;
            }
        }

        // update bundle
        bundle.append(z, gz, fz);
    }

    state.update_calls();
    return state;
}

template class nano::base_solver_fpba_t<proximal::sequence1_t, proximal::fpba1_type_id_t>;
template class nano::base_solver_fpba_t<proximal::sequence2_t, proximal::fpba2_type_id_t>;
