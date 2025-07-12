#include <Eigen/Eigenvalues>
#include <nano/core/numeric.h>
#include <nano/core/overloaded.h>
#include <nano/function/util.h>
#include <nano/tensor/stack.h>

using namespace nano;
using namespace constraint;

namespace
{
tensor_size_t reduce(matrix_t& A)
{
    // independant linear constraints
    const auto dd = A.transpose().fullPivLu();
    if (dd.rank() == A.rows())
    {
        return dd.rank();
    }

    // dependant linear constraints, use decomposition to formulate equivalent linear equality constraints
    const auto& P  = dd.permutationP();
    const auto& LU = dd.matrixLU();

    const auto n = std::min(A.rows(), A.cols());
    const auto L = LU.leftCols(n).triangularView<Eigen::UnitLower>().toDenseMatrix();
    const auto U = LU.topRows(n).triangularView<Eigen::Upper>().toDenseMatrix();

    A = U.transpose().block(0, 0, dd.rank(), U.rows()) * L.transpose() * P;
    return dd.rank();
}

template <typename tverifier>
zero_rows_stats_t remove_zero_rows(matrix_t& A, vector_t& b, const tverifier& verifier)
{
    const auto tiny = 1e-40;

    auto valid_rows   = tensor_size_t{0};
    auto removed      = tensor_size_t{0};
    auto inconsistent = tensor_size_t{0};

    for (tensor_size_t row = 0; row < A.rows(); ++row)
    {
        if (A.row(row).lpNorm<Eigen::Infinity>() < tiny)
        {
            ++removed;
            if (verifier(b(row), tiny))
            {
                ++inconsistent;
            }
        }

        else
        {
            if (row > valid_rows)
            {
                A.row(valid_rows) = A.row(row);
                b(valid_rows)     = b(row);
            }
            ++valid_rows;
        }
    }

    if (A.rows() != valid_rows)
    {
        A = matrix_t{A.slice(0, valid_rows)};
        b = vector_t{b.slice(0, valid_rows)};
    }

    return {removed, inconsistent};
}

auto is_linear_constrained(const function_t& function)
{
    auto valid  = true;
    auto neqs   = tensor_size_t{0};
    auto nineqs = tensor_size_t{0};

    for (const auto& constraint : function.constraints())
    {
        std::visit(overloaded{[&](const constant_t&) { ++neqs; },              ///<
                              [&](const minimum_t&) { ++nineqs; },             ///<
                              [&](const maximum_t&) { ++nineqs; },             ///<
                              [&](const linear_equality_t&) { ++neqs; },       ///<
                              [&](const linear_inequality_t&) { ++nineqs; },   ///<
                              [&](const euclidean_ball_t&) { valid = false; }, ///<
                              [&](const quadratic_t&) { valid = false; },      ///<
                              [&](const functional_t&) { valid = false; }},    ///<
                   constraint);
    }

    return std::make_tuple(valid, neqs, nineqs);
}

void handle(linear_constraints_t& lc, [[maybe_unused]] tensor_size_t& ieq, [[maybe_unused]] tensor_size_t& neq,
            const constant_t& c)
{
    lc.m_A.row(ieq).array()    = 0.0;
    lc.m_A(ieq, c.m_dimension) = 1.0;
    lc.m_b(ieq++)              = c.m_value;
}

void handle(linear_constraints_t& lc, [[maybe_unused]] tensor_size_t& ieq, [[maybe_unused]] tensor_size_t& ineq,
            const minimum_t& c)
{
    lc.m_G.row(ineq).array()    = 0.0;
    lc.m_G(ineq, c.m_dimension) = -1.0;
    lc.m_h(ineq++)              = -c.m_value;
}

void handle(linear_constraints_t& lc, [[maybe_unused]] tensor_size_t& ieq, [[maybe_unused]] tensor_size_t& ineq,
            const maximum_t& c)
{
    lc.m_G.row(ineq).array()    = 0.0;
    lc.m_G(ineq, c.m_dimension) = 1.0;
    lc.m_h(ineq++)              = c.m_value;
}

void handle(linear_constraints_t& lc, [[maybe_unused]] tensor_size_t& ieq, [[maybe_unused]] tensor_size_t& ineq,
            const linear_equality_t& c)
{
    lc.m_A.row(ieq) = c.m_q.transpose();
    lc.m_b(ieq++)   = -c.m_r;
}

void handle(linear_constraints_t& lc, [[maybe_unused]] tensor_size_t& ieq, [[maybe_unused]] tensor_size_t& ineq,
            const linear_inequality_t& c)
{
    lc.m_G.row(ineq) = c.m_q.transpose();
    lc.m_h(ineq++)   = -c.m_r;
}
} // namespace

scalar_t nano::grad_accuracy(const function_t& function, const vector_t& x, const scalar_t desired_accuracy)
{
    assert(x.size() == function.size());

    const auto n = function.size();

    vector_t xp(n);
    vector_t xn(n);
    vector_t gx(n);
    vector_t gx_approx(n);

    // analytical gradient
    const auto fx = function(x, gx);
    assert(gx.size() == function.size());

    // finite-difference approximated gradient
    //      see "Numerical optimization", Nocedal & Wright, 2nd edition, p.197
    auto dg = std::numeric_limits<scalar_t>::max();
    for (const auto dx : {1e-9, 3e-9, 1e-8, 3e-8, 5e-8, 8e-8, 1e-7, 3e-7, 5e-7, 8e-7, 1e-6, 3e-6})
    {
        xp = x;
        xn = x;
        for (auto i = 0; i < n; i++)
        {
            if (i > 0)
            {
                xp(i - 1) = x(i - 1);
                xn(i - 1) = x(i - 1);
            }
            xp(i) = x(i) + dx * std::max(scalar_t{1}, std::fabs(x(i)));
            xn(i) = x(i) - dx * std::max(scalar_t{1}, std::fabs(x(i)));

            const auto dfi = function(xp) - function(xn);
            const auto dxi = xp(i) - xn(i);
            gx_approx(i)   = dfi / dxi;

            assert(std::isfinite(gx(i)));
            assert(std::isfinite(gx_approx(i)));
        }

        dg = std::min(dg, (gx - gx_approx).lpNorm<Eigen::Infinity>()) / (1.0 + std::fabs(fx));
        if (dg < desired_accuracy)
        {
            break;
        }
    }

    return dg;
}

bool nano::is_convex(const function_t& function, const vector_t& x1, const vector_t& x2, const int steps,
                     const scalar_t epsilon)
{
    assert(steps > 2);
    assert(x1.size() == function.size());
    assert(x2.size() == function.size());

    const auto f1 = function(x1);
    const auto f2 = function(x2);
    const auto dx = (x1 - x2).squaredNorm();

    const auto delta = epsilon * (1.0 + 0.5 * (std::fabs(f1) + std::fabs(f2)));

    assert(std::isfinite(f1));
    assert(std::isfinite(f2));

    auto tx = vector_t{function.size()};

    for (int step = 1; step < steps; step++)
    {
        const auto t1 = static_cast<scalar_t>(step) / static_cast<scalar_t>(steps);
        const auto t2 = 1.0L - t1;

        tx = t1 * x1 + t2 * x2;
        if (function(tx) > t1 * f1 + t2 * f2 - t1 * t2 * function.strong_convexity() * 0.5 * dx + delta)
        {
            return false;
        }
    }

    return true;
}

full_rank_stats_t nano::make_full_rank(matrix_t& A, vector_t& b)
{
    assert(A.rows() == b.size());

    if (A.rows() == 0)
    {
        return {};
    }

    // NB: need to reduce [A|b] altogether!
    auto Ab = ::nano::stack<scalar_t>(A.rows(), A.cols() + 1, A.matrix(), b.vector());

    if (const auto rank = ::reduce(Ab); rank == A.rows())
    {
        return {rank, false};
    }
    else
    {
        A = Ab.block(0, 0, Ab.rows(), Ab.cols() - 1);
        b = Ab.matrix().col(Ab.cols() - 1);
        return {rank, true};
    }
}

zero_rows_stats_t nano::remove_zero_rows_equality(matrix_t& A, vector_t& b)
{
    assert(A.rows() == b.size());

    return ::remove_zero_rows(A, b, [](const scalar_t brow, const scalar_t tiny) { return std::fabs(brow) > tiny; });
}

zero_rows_stats_t nano::remove_zero_rows_inequality(matrix_t& A, vector_t& b)
{
    assert(A.rows() == b.size());

    return ::remove_zero_rows(A, b, [](const scalar_t brow, [[maybe_unused]] const scalar_t) { return brow < 0.0; });
}

bool nano::is_convex(const matrix_t& P, const scalar_t tol)
{
    const auto Q = P.matrix();
    if (!Q.isApprox(Q.transpose()))
    {
        return false;
    }
    else
    {
        const auto Qmax = std::max(1.0, Q.diagonal().array().maxCoeff());
        const auto Qtol = Q / Qmax + tol * matrix_t::identity(Q.rows(), Q.cols());
        const auto ldlt = Qtol.selfadjointView<Eigen::Upper>().ldlt();

        return ldlt.info() == Eigen::Success && ldlt.isPositive();
    }
}

scalar_t nano::strong_convexity(const matrix_t& P)
{
    const auto eigenvalues = P.matrix().eigenvalues();
    const auto peigenvalue = [](const auto& lhs, const auto& rhs) { return lhs.real() < rhs.real(); };

    const auto* const it = std::min_element(begin(eigenvalues), end(eigenvalues), peigenvalue);
    return std::max(0.0, it->real());
}

std::optional<linear_constraints_t> nano::make_linear_constraints(const function_t& function)
{
    if (const auto [valid, neqs, nineqs] = is_linear_constrained(function); !valid)
    {
        return {};
    }

    else
    {
        auto lc = linear_constraints_t{};
        lc.m_A  = matrix_t{neqs, function.size()};
        lc.m_b  = vector_t{neqs};
        lc.m_G  = matrix_t{nineqs, function.size()};
        lc.m_h  = vector_t{nineqs};

        auto ieq  = tensor_size_t{0};
        auto ineq = tensor_size_t{0};

        for (const auto& constraint : function.constraints())
        {
            std::visit(overloaded{[&](const constant_t& c) { handle(lc, ieq, ineq, c); },          ///<
                                  [&](const minimum_t& c) { handle(lc, ieq, ineq, c); },           ///<
                                  [&](const maximum_t& c) { handle(lc, ieq, ineq, c); },           ///<
                                  [&](const linear_equality_t& c) { handle(lc, ieq, ineq, c); },   ///<
                                  [&](const linear_inequality_t& c) { handle(lc, ieq, ineq, c); }, ///<
                                  [&](const euclidean_ball_t&) {},                                 ///<
                                  [&](const quadratic_t&) {},                                      ///<
                                  [&](const functional_t&) {}},                                    ///<
                       constraint);
        }

        return lc;
    }
}
