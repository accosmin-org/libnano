#include <nano/core/numeric.h>
#include <nano/core/strutil.h>
#include <nano/core/util.h>
#include <nano/function/constraints.h>

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

function_t::function_t(string_t name, tensor_size_t size)
    : m_name(std::move(name))
    , m_size(size)
{
}

void function_t::convex(bool convex)
{
    m_convex = convex;
}

void function_t::smooth(bool smooth)
{
    m_smooth = smooth;
}

void function_t::strong_convexity(scalar_t sconvexity)
{
    m_sconvexity = sconvexity;
}

scalar_t function_t::grad_accuracy(const vector_t& x) const
{
    assert(x.size() == size());

    const auto n = size();

    vector_t gx(n);
    vector_t gx_approx(n);
    vector_t xp(n), xn(n);

    // analytical gradient
    const auto fx = vgrad(x, &gx);
    assert(gx.size() == size());

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

            const auto dfi = vgrad(xp, nullptr) - vgrad(xn, nullptr);
            const auto dxi = xp(i) - xn(i);
            gx_approx(i)   = dfi / dxi;

            assert(std::isfinite(gx(i)));
            assert(std::isfinite(gx_approx(i)));
        }

        dg = std::min(dg, (gx - gx_approx).lpNorm<Eigen::Infinity>());
    }

    return dg / (1 + std::fabs(fx));
}

bool function_t::is_convex(const vector_t& x1, const vector_t& x2, const int steps) const
{
    assert(steps > 2);
    assert(x1.size() == size());
    assert(x2.size() == size());

    const auto f1 = vgrad(x1, nullptr);
    assert(std::isfinite(f1));

    const auto f2 = vgrad(x2, nullptr);
    assert(std::isfinite(f2));

    const auto eps = epsilon1<scalar_t>();
    const auto dx  = (x1 - x2).dot(x1 - x2);

    for (int step = 1; step < steps; step++)
    {
        const auto t1 = scalar_t(step) / scalar_t(steps);
        const auto t2 = 1.0 - t1;

        if (vgrad(t1 * x1 + t2 * x2) > t1 * f1 + t2 * f2 - t1 * t2 * m_sconvexity * 0.5 * dx + eps)
        {
            return false;
        }
    }

    return true;
}

string_t function_t::name(bool with_size) const
{
    return with_size ? scat(m_name, "[", size(), "D]") : m_name;
}

bool function_t::constrain_equality(rfunction_t&& constraint)
{
    if (constraint->size() != size())
    {
        return false;
    }

    m_constraints.emplace_back(equality_t{std::move(constraint)});
    return true;
}

bool function_t::constrain_inequality(rfunction_t&& constraint)
{
    if (constraint->size() != size())
    {
        return false;
    }

    m_constraints.emplace_back(inequality_t{std::move(constraint)});
    return true;
}

bool function_t::constrain_box(vector_t min, vector_t max)
{
    if (min.size() != size() || max.size() != size())
    {
        return false;
    }

    for (tensor_size_t i = 0, size = this->size(); i < size; ++i)
    {
        if (min(i) >= max(i))
        {
            return false;
        }
    }

    for (tensor_size_t i = 0, size = this->size(); i < size; ++i)
    {
        m_constraints.emplace_back(minimum_t{min(i), i});
        m_constraints.emplace_back(maximum_t{max(i), i});
    }
    return true;
}

bool function_t::constrain_box(scalar_t min, scalar_t max)
{
    if (min >= max)
    {
        return false;
    }

    for (tensor_size_t i = 0, size = this->size(); i < size; ++i)
    {
        m_constraints.emplace_back(minimum_t{min, i});
        m_constraints.emplace_back(maximum_t{max, i});
    }
    return true;
}

bool function_t::constrain_ball(vector_t origin, scalar_t radius)
{
    if (origin.size() != size() || radius <= 0.0)
    {
        return false;
    }

    return constrain_inequality(std::make_unique<ball_constraint_t>(std::move(origin), radius));
}

bool function_t::constrain_equality(vector_t weights, scalar_t bias)
{
    if (weights.size() != size())
    {
        return false;
    }

    return constrain_equality(std::make_unique<affine_constraint_t>(std::move(weights), bias));
}

bool function_t::constrain_inequality(vector_t weights, scalar_t bias)
{
    if (weights.size() != size())
    {
        return false;
    }

    return constrain_inequality(std::make_unique<affine_constraint_t>(std::move(weights), bias));
}

bool function_t::valid(const vector_t& x) const
{
    assert(x.size() == size());

    const auto op = [&](const constraint_t& constraint)
    { return ::nano::valid(x, constraint) < std::numeric_limits<scalar_t>::epsilon(); };

    return std::all_of(m_constraints.begin(), m_constraints.end(), op);
}

const constraints_t& function_t::constraints() const
{
    return m_constraints;
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
