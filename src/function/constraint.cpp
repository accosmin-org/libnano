#include <Eigen/Eigenvalues>
#include <nano/core/util.h>
#include <nano/function.h>
#include <nano/function/constraint.h>

using namespace nano;
using namespace nano::constraint;

static auto smooth(const ball_t&)
{
    return true;
}

static auto smooth(const linear_t&)
{
    return true;
}

static auto smooth(const constant_t&)
{
    return true;
}

static auto smooth(const quadratic_t&)
{
    return true;
}

static auto smooth(const functional_t& constraint)
{
    return constraint.m_function->smooth();
}

bool nano::smooth(const constraint_t& constraint)
{
    return std::visit(overloaded{[&](const ball_t& ct) { return ::smooth(ct); },
                                 [&](const linear_t& ct) { return ::smooth(ct); },
                                 [&](const constant_t& ct) { return ::smooth(ct); },
                                 [&](const quadratic_t& ct) { return ::smooth(ct); },
                                 [&](const functional_t& ct) { return ::smooth(ct); }},
                      constraint);
}

static auto strong_convexity(const ball_t&)
{
    return 2.0;
}

static auto strong_convexity(const linear_t&)
{
    return 0.0;
}

static auto strong_convexity(const constant_t&)
{
    return 0.0;
}

static auto strong_convexity(const quadratic_t& constraint)
{
    const auto eigenvalues = constraint.m_P.eigenvalues();
    const auto peigenvalue = [](const auto& lhs, const auto& rhs) { return lhs.real() < rhs.real(); };

    const auto* const it = std::min_element(begin(eigenvalues), end(eigenvalues), peigenvalue);
    return std::max(0.0, it->real());
}

static scalar_t strong_convexity(const functional_t& constraint)
{
    return constraint.m_function->strong_convexity();
}

scalar_t nano::strong_convexity(const constraint_t& constraint)
{
    return std::visit(overloaded{[&](const ball_t& ct) { return ::strong_convexity(ct); },
                                 [&](const linear_t& ct) { return ::strong_convexity(ct); },
                                 [&](const constant_t& ct) { return ::strong_convexity(ct); },
                                 [&](const quadratic_t& ct) { return ::strong_convexity(ct); },
                                 [&](const functional_t& ct) { return ::strong_convexity(ct); }},
                      constraint);
}

static auto vgrad(const ball_t& constraint, const vector_t& x, vector_t* gx = nullptr)
{
    if (gx != nullptr)
    {
        gx->noalias() = 2.0 * (x - constraint.m_origin);
    }
    return (x - constraint.m_origin).dot(x - constraint.m_origin) - constraint.m_radius * constraint.m_radius;
}

static auto vgrad(const linear_t& constraint, const vector_t& x, vector_t* gx = nullptr)
{
    if (gx != nullptr)
    {
        gx->noalias() = constraint.m_q;
    }
    return constraint.m_q.dot(x) + constraint.m_r;
}

static auto vgrad(const quadratic_t& constraint, const vector_t& x, vector_t* gx = nullptr)
{
    if (gx != nullptr)
    {
        gx->noalias() = constraint.m_P * x + constraint.m_q;
    }
    return 0.5 * x.dot(constraint.m_P * x) + constraint.m_q.dot(x) + constraint.m_r;
}

static auto vgrad(const minimum_t& constraint, const vector_t& x, vector_t* gx = nullptr)
{
    if (gx != nullptr)
    {
        gx->noalias()                 = vector_t::Zero(x.size());
        (*gx)(constraint.m_dimension) = -1.0;
    }
    return constraint.m_value - x(constraint.m_dimension);
}

static auto vgrad(const maximum_t& constraint, const vector_t& x, vector_t* gx = nullptr)
{
    if (gx != nullptr)
    {
        gx->noalias()                 = vector_t::Zero(x.size());
        (*gx)(constraint.m_dimension) = +1.0;
    }
    return x(constraint.m_dimension) - constraint.m_value;
}

static auto vgrad(const constant_t& constraint, const vector_t& x, vector_t* gx = nullptr)
{
    if (gx != nullptr)
    {
        gx->noalias()                 = vector_t::Zero(x.size());
        (*gx)(constraint.m_dimension) = +1.0;
    }
    return x(constraint.m_dimension) - constraint.m_value;
}

static auto vgrad(const functional_t& constraint, const vector_t& x, vector_t* gx = nullptr)
{
    return constraint.m_function->vgrad(x, gx);
}

scalar_t nano::vgrad(const constraint_t& constraint, const vector_t& x, vector_t* gx)
{
    return std::visit(overloaded{[&](const ball_t& ct) { return ::vgrad(ct, x, gx); },
                                 [&](const linear_t& ct) { return ::vgrad(ct, x, gx); },
                                 [&](const constant_t& ct) { return ::vgrad(ct, x, gx); },
                                 [&](const maximum_t& ct) { return ::vgrad(ct, x, gx); },
                                 [&](const minimum_t& ct) { return ::vgrad(ct, x, gx); },
                                 [&](const quadratic_t& ct) { return ::vgrad(ct, x, gx); },
                                 [&](const functional_t& ct) { return ::vgrad(ct, x, gx); }},
                      constraint);
}

static bool convex(const ball_t&)
{
    return true;
}

static bool convex(const linear_t&)
{
    return true;
}

static bool convex(const constant_t&)
{
    return true;
}

static bool convex(const quadratic_t& constraint)
{
    const auto eigenvalues         = constraint.m_P.eigenvalues();
    const auto positive_eigenvalue = [](const auto& eigenvalue) { return eigenvalue.real() >= 0.0; };

    return std::all_of(begin(eigenvalues), end(eigenvalues), positive_eigenvalue);
}

static bool convex(const functional_t& constraint)
{
    return constraint.m_function->convex();
}

bool nano::convex(const constraint_t& constraint)
{
    return std::visit(overloaded{[&](const ball_t& ct) { return ::convex(ct); },
                                 [&](const linear_t& ct) { return ::convex(ct); },
                                 [&](const constant_t& ct) { return ::convex(ct); },
                                 [&](const quadratic_t& ct) { return ::convex(ct); },
                                 [&](const functional_t& ct) { return ::convex(ct); }},
                      constraint);
}

static auto valid(const constant_t& constraint, const vector_t& x)
{
    return std::fabs(constraint.m_value - x(constraint.m_dimension));
}

static auto valid(const minimum_t& constraint, const vector_t& x)
{
    return std::max(constraint.m_value - x(constraint.m_dimension), 0.0);
}

static auto valid(const maximum_t& constraint, const vector_t& x)
{
    return std::max(x(constraint.m_dimension) - constraint.m_value, 0.0);
}

static auto valid(const ball_equality_t& constraint, const vector_t& x)
{
    return std::fabs(::vgrad(constraint, x));
}

static auto valid(const ball_inequality_t& constraint, const vector_t& x)
{
    return std::max(::vgrad(constraint, x), 0.0);
}

static auto valid(const linear_equality_t& constraint, const vector_t& x)
{
    return std::fabs(::vgrad(constraint, x));
}

static auto valid(const linear_inequality_t& constraint, const vector_t& x)
{
    return std::max(::vgrad(constraint, x), 0.0);
}

static auto valid(const quadratic_equality_t& constraint, const vector_t& x)
{
    return std::fabs(::vgrad(constraint, x));
}

static auto valid(const quadratic_inequality_t& constraint, const vector_t& x)
{
    return std::max(::vgrad(constraint, x), 0.0);
}

static auto valid(const functional_equality_t& constraint, const vector_t& x)
{
    return std::fabs(::vgrad(constraint, x));
}

static auto valid(const functional_inequality_t& constraint, const vector_t& x)
{
    return std::max(::vgrad(constraint, x), 0.0);
}

scalar_t nano::valid(const constraint_t& constraint, const vector_t& x)
{
    return std::visit(overloaded{[&](const minimum_t& ct) { return ::valid(ct, x); },
                                 [&](const maximum_t& ct) { return ::valid(ct, x); },
                                 [&](const constant_t& ct) { return ::valid(ct, x); },
                                 [&](const ball_equality_t& ct) { return ::valid(ct, x); },
                                 [&](const ball_inequality_t& ct) { return ::valid(ct, x); },
                                 [&](const linear_equality_t& ct) { return ::valid(ct, x); },
                                 [&](const linear_inequality_t& ct) { return ::valid(ct, x); },
                                 [&](const quadratic_equality_t& ct) { return ::valid(ct, x); },
                                 [&](const quadratic_inequality_t& ct) { return ::valid(ct, x); },
                                 [&](const functional_equality_t& ct) { return ::valid(ct, x); },
                                 [&](const functional_inequality_t& ct) { return ::valid(ct, x); }},
                      constraint);
}

static auto compatible(const function_t& function, const ball_t& constraint)
{
    return constraint.m_origin.size() == function.size() && constraint.m_radius > 0.0;
}

static auto compatible(const function_t& function, const linear_t& constraint)
{
    return constraint.m_q.size() == function.size();
}

static auto compatible(const function_t& function, const constant_t& constraint)
{
    return constraint.m_dimension >= 0 && constraint.m_dimension < function.size();
}

static auto compatible(const function_t& function, const quadratic_t& constraint)
{
    return constraint.m_P.rows() == function.size() && constraint.m_P.cols() == function.size() &&
           constraint.m_q.size() == function.size();
}

static auto compatible(const function_t& function, const functional_t& constraint)
{
    return static_cast<bool>(constraint.m_function) && constraint.m_function->size() == function.size();
}

bool nano::compatible(const constraint_t& constraint, const function_t& function)
{
    return std::visit(overloaded{[&](const ball_t& ct) { return ::compatible(function, ct); },
                                 [&](const linear_t& ct) { return ::compatible(function, ct); },
                                 [&](const constant_t& ct) { return ::compatible(function, ct); },
                                 [&](const quadratic_t& ct) { return ::compatible(function, ct); },
                                 [&](const functional_t& ct) { return ::compatible(function, ct); }},
                      constraint);
}