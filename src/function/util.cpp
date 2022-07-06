#include <nano/core/numeric.h>
#include <nano/function/util.h>

using namespace nano;

scalar_t nano::grad_accuracy(const function_t& function, const vector_t& x)
{
    assert(x.size() == function.size());

    const auto n = function.size();

    vector_t gx(n);
    vector_t gx_approx(n);
    vector_t xp(n), xn(n);

    // analytical gradient
    const auto fx = function.vgrad(x, &gx);
    assert(gx.size() == function.size());

    // finite-difference approximated gradient
    //      see "Numerical optimization", Nocedal & Wright, 2nd edition, p.197
    auto dg = std::numeric_limits<scalar_t>::max();
    for (const auto dx : {1e-8, 2e-8, 5e-8, 7e-8, 1e-7, 2e-7, 5e-7, 7e-7, 1e-6})
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

            const auto dfi = function.vgrad(xp, nullptr) - function.vgrad(xn, nullptr);
            const auto dxi = xp(i) - xn(i);
            gx_approx(i)   = dfi / dxi;

            assert(std::isfinite(gx(i)));
            assert(std::isfinite(gx_approx(i)));
        }

        dg = std::min(dg, (gx - gx_approx).lpNorm<Eigen::Infinity>());
    }

    return dg / (1 + std::fabs(fx));
}

bool nano::is_convex(const function_t& function, const vector_t& x1, const vector_t& x2, const int steps)
{
    assert(steps > 2);
    assert(x1.size() == function.size());
    assert(x2.size() == function.size());

    const auto f1 = function.vgrad(x1, nullptr);
    assert(std::isfinite(f1));

    const auto f2 = function.vgrad(x2, nullptr);
    assert(std::isfinite(f2));

    const auto eps = epsilon1<scalar_t>();
    const auto dx  = (x1 - x2).dot(x1 - x2);

    for (int step = 1; step < steps; step++)
    {
        const auto t1 = scalar_t(step) / scalar_t(steps);
        const auto t2 = 1.0 - t1;

        if (function.vgrad(t1 * x1 + t2 * x2) >
            t1 * f1 + t2 * f2 - t1 * t2 * function.strong_convexity() * 0.5 * dx + eps)
        {
            return false;
        }
    }

    return true;
}
