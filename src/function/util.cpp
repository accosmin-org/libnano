#include <nano/core/numeric.h>
#include <nano/core/util.h>
#include <nano/function/util.h>

using namespace nano;

static scalar_t valid(const vector_t& x, const minimum_t& constraint)
{
    return std::max(constraint.m_value - x(constraint.m_dimension), 0.0);
}

static scalar_t valid(const vector_t& x, const maximum_t& constraint)
{
    return std::max(x(constraint.m_dimension) - constraint.m_value, 0.0);
}

static scalar_t valid(const vector_t& x, const equality_t& constraint)
{
    return std::fabs(constraint.m_function->vgrad(x));
}

static scalar_t valid(const vector_t& x, const inequality_t& constraint)
{
    return std::max(constraint.m_function->vgrad(x), 0.0);
}

static scalar_t vgrad(const vector_t& x, vector_t* gx, const minimum_t& constraint)
{
    if (gx != nullptr)
    {
        gx->noalias()                 = vector_t::Zero(x.size());
        (*gx)(constraint.m_dimension) = -1.0;
    }
    return constraint.m_value - x(constraint.m_dimension);
}

static scalar_t vgrad(const vector_t& x, vector_t* gx, const maximum_t& constraint)
{
    if (gx != nullptr)
    {
        gx->noalias()                 = vector_t::Zero(x.size());
        (*gx)(constraint.m_dimension) = +1.0;
    }
    return x(constraint.m_dimension) - constraint.m_value;
}

static scalar_t vgrad(const vector_t& x, vector_t* gx, const equality_t& constraint)
{
    return constraint.m_function->vgrad(x, gx);
}

static scalar_t vgrad(const vector_t& x, vector_t* gx, const inequality_t& constraint)
{
    return constraint.m_function->vgrad(x, gx);
}

static bool convex(const minimum_t&)
{
    return true;
}

static bool convex(const maximum_t&)
{
    return true;
}

static bool convex(const equality_t& constraint)
{
    return constraint.m_function->convex();
}

static bool convex(const inequality_t& constraint)
{
    return constraint.m_function->convex();
}

static bool smooth(const minimum_t&)
{
    return true;
}

static bool smooth(const maximum_t&)
{
    return true;
}

static bool smooth(const equality_t& constraint)
{
    return constraint.m_function->smooth();
}

static bool smooth(const inequality_t& constraint)
{
    return constraint.m_function->smooth();
}

static scalar_t strong_convexity(const minimum_t&)
{
    return 0.0;
}

static scalar_t strong_convexity(const maximum_t&)
{
    return 0.0;
}

static scalar_t strong_convexity(const equality_t& constraint)
{
    return constraint.m_function->strong_convexity();
}

static scalar_t strong_convexity(const inequality_t& constraint)
{
    return constraint.m_function->strong_convexity();
}

bool nano::convex(const constraint_t& constraint)
{
    return std::visit(overloaded{[&](const minimum_t& constraint) { return ::convex(constraint); },
                                 [&](const maximum_t& constraint) { return ::convex(constraint); },
                                 [&](const equality_t& constraint) { return ::convex(constraint); },
                                 [&](const inequality_t& constraint) { return ::convex(constraint); }},
                      constraint);
}

bool nano::smooth(const constraint_t& constraint)
{
    return std::visit(overloaded{[&](const minimum_t& constraint) { return ::smooth(constraint); },
                                 [&](const maximum_t& constraint) { return ::smooth(constraint); },
                                 [&](const equality_t& constraint) { return ::smooth(constraint); },
                                 [&](const inequality_t& constraint) { return ::smooth(constraint); }},
                      constraint);
}

scalar_t nano::strong_convexity(const constraint_t& constraint)
{
    return std::visit(overloaded{[&](const minimum_t& constraint) { return ::strong_convexity(constraint); },
                                 [&](const maximum_t& constraint) { return ::strong_convexity(constraint); },
                                 [&](const equality_t& constraint) { return ::strong_convexity(constraint); },
                                 [&](const inequality_t& constraint) { return ::strong_convexity(constraint); }},
                      constraint);
}

scalar_t nano::valid(const vector_t& x, const constraint_t& constraint)
{
    return std::visit(overloaded{[&](const minimum_t& constraint) { return ::valid(x, constraint); },
                                 [&](const maximum_t& constraint) { return ::valid(x, constraint); },
                                 [&](const equality_t& constraint) { return ::valid(x, constraint); },
                                 [&](const inequality_t& constraint) { return ::valid(x, constraint); }},
                      constraint);
}

scalar_t nano::vgrad(const constraint_t& constraint, const vector_t& x, vector_t* gx)
{
    return std::visit(overloaded{[&](const minimum_t& constraint) { return ::vgrad(x, gx, constraint); },
                                 [&](const maximum_t& constraint) { return ::vgrad(x, gx, constraint); },
                                 [&](const equality_t& constraint) { return ::vgrad(x, gx, constraint); },
                                 [&](const inequality_t& constraint) { return ::vgrad(x, gx, constraint); }},
                      constraint);
}

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
