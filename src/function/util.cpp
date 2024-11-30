#include <Eigen/Eigenvalues>
#include <nano/core/numeric.h>
#include <nano/function/util.h>
#include <nano/tensor/stack.h>

using namespace nano;

namespace
{
void reduce(matrix_t& A)
{
    // independant linear constraints
    const auto dd = A.transpose().fullPivLu();
    if (dd.rank() == A.rows())
    {
        return;
    }

    // dependant linear constraints, use decomposition to formulate equivalent linear equality constraints
    const auto& P  = dd.permutationP();
    const auto& LU = dd.matrixLU();

    const auto n = std::min(A.rows(), A.cols());
    const auto L = LU.leftCols(n).triangularView<Eigen::UnitLower>().toDenseMatrix();
    const auto U = LU.topRows(n).triangularView<Eigen::Upper>().toDenseMatrix();

    A = U.transpose().block(0, 0, dd.rank(), U.rows()) * L.transpose() * P;
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

    assert(std::isfinite(f1));
    assert(std::isfinite(f2));

    auto tx = vector_t{function.size()};

    for (int step = 1; step < steps; step++)
    {
        const auto t1 = scalar_t(step) / scalar_t(steps);
        const auto t2 = 1.0 - t1;

        tx = t1 * x1 + t2 * x2;
        if (function(tx) > t1 * f1 + t2 * f2 - t1 * t2 * function.strong_convexity() * 0.5 * dx + epsilon)
        {
            return false;
        }
    }

    return true;
}

bool nano::reduce(matrix_t& A, vector_t& b)
{
    assert(A.rows() == b.size());

    if (A.rows() == 0)
    {
        return false;
    }

    // NB: need to reduce [A|b] altogether!
    auto Ab = ::nano::stack<scalar_t>(A.rows(), A.cols() + 1, A.matrix(), b.vector());
    ::reduce(Ab);

    A = Ab.block(0, 0, Ab.rows(), Ab.cols() - 1);
    b = Ab.matrix().col(Ab.cols() - 1);
    return true;
}

bool nano::convex(const matrix_t& P)
{
    const auto eigenvalues         = P.matrix().eigenvalues();
    const auto positive_eigenvalue = [](const auto& eigenvalue) { return eigenvalue.real() >= 0.0; };

    return std::all_of(begin(eigenvalues), end(eigenvalues), positive_eigenvalue);
}

scalar_t nano::strong_convexity(const matrix_t& P)
{
    const auto eigenvalues = P.matrix().eigenvalues();
    const auto peigenvalue = [](const auto& lhs, const auto& rhs) { return lhs.real() < rhs.real(); };

    const auto* const it = std::min_element(begin(eigenvalues), end(eigenvalues), peigenvalue);
    return std::max(0.0, it->real());
}

std::optional<vector_t> nano::make_strictly_feasible(const matrix_t& A, const vector_t& b)
{
    std::optional<vector_t> ret;

    if (A.rows() == b.size())
    {
        const auto ldlt = (A.transpose() * A).ldlt();
        auto       x    = vector_t{A.cols()};
        const auto eval = [&](const scalar_t y)
        {
            x.vector() = ldlt.solve(A.transpose() * (b + vector_t::constant(A.rows(), -y)));
            if ((A * x.vector() - b).maxCoeff() < 0.0)
            {
                ret = std::move(x);
                return true;
            }
            return false;
        };

        static constexpr auto gamma  = 0.3;
        static constexpr auto trials = 100;

        // NB: try both smaller and bigger distances to the edges!
        auto ym = 1.0;
        auto yM = 1.0 / gamma;
        for (auto trial = 0; trial < trials; trial += 2)
        {
            if (eval(ym) || eval(yM))
            {
                break;
            }
            ym *= gamma;
            yM /= gamma;
        }
    }

    return ret;
}
