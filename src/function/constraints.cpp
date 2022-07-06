#include <Eigen/Eigenvalues>
#include <nano/core/util.h>
#include <nano/function/constraints.h>

using namespace nano;

ball_constraint_t::ball_constraint_t(vector_t origin, scalar_t radius)
    : function_t("ball", origin.size())
    , m_origin(std::move(origin))
    , m_radius(radius)
{
    smooth(true);
    convex(true);
}

scalar_t ball_constraint_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    if (gx != nullptr)
    {
        gx->noalias() = 2.0 * (x - m_origin);
    }

    return (x - m_origin).dot(x - m_origin) - m_radius * m_radius;
}

affine_constraint_t::affine_constraint_t(vector_t q, scalar_t r)
    : function_t("affine", q.size())
    , m_q(std::move(q))
    , m_r(r)
{
    smooth(true);
    convex(true);
}

scalar_t affine_constraint_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    if (gx != nullptr)
    {
        gx->noalias() = m_q;
    }

    return m_q.dot(x) + m_r;
}

quadratic_constraint_t::quadratic_constraint_t(matrix_t P, vector_t q, scalar_t r)
    : function_t("quadratic", q.size())
    , m_P(std::move(P))
    , m_q(std::move(q))
    , m_r(r)
{
    const auto eigenvalues         = m_P.eigenvalues();
    const auto positive_eigenvalue = [](const auto& eigenvalue) { return eigenvalue.real() >= 0.0; };

    smooth(true);
    convex(std::all_of(begin(eigenvalues), end(eigenvalues), positive_eigenvalue));
}

scalar_t quadratic_constraint_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    if (gx != nullptr)
    {
        gx->noalias() = m_P * x + m_q;
    }

    return 0.5 * x.dot(m_P * x) + m_q.dot(x) + m_r;
}

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

constraint_t nano::make_equality_constraint(rfunction_t&& constraint)
{
    return equality_t{std::move(constraint)};
}

constraint_t nano::make_inequality_constraint(rfunction_t&& constraint)
{
    return inequality_t{std::move(constraint)};
}

constraint_t nano::make_affine_equality_constraint(vector_t q, scalar_t r)
{
    return equality_t{std::make_unique<affine_constraint_t>(std::move(q), r)};
}

constraint_t nano::make_affine_inequality_constraint(vector_t q, scalar_t r)
{
    return inequality_t{std::make_unique<affine_constraint_t>(std::move(q), r)};
}

constraint_t nano::make_quadratic_equality_constraint(matrix_t P, vector_t q, scalar_t r)
{
    auto constraint = constraint_t{};
    if (P.rows() == P.cols() && P.rows() == q.size())
    {
        constraint = equality_t{std::make_unique<quadratic_constraint_t>(std::move(P), std::move(q), r)};
    }

    return constraint;
}

constraint_t nano::make_quadratic_inequality_constraint(matrix_t P, vector_t q, scalar_t r)
{
    auto constraint = constraint_t{};
    if (P.rows() == P.cols() && P.rows() == q.size())
    {
        constraint = inequality_t{std::make_unique<quadratic_constraint_t>(std::move(P), std::move(q), r)};
    }

    return constraint;
}

constraints_t nano::make_box_constraints(vector_t min, vector_t max)
{
    auto constraints = constraints_t{};

    if (min.size() == max.size() && (max - min).minCoeff() > 0.0)
    {
        constraints.reserve(static_cast<size_t>(2 * min.size()));
        for (tensor_size_t i = 0, size = min.size(); i < size; ++i)
        {
            constraints.emplace_back(minimum_t{min(i), i});
            constraints.emplace_back(maximum_t{max(i), i});
        }
    }

    return constraints;
}

constraints_t nano::make_box_constraints(scalar_t min, scalar_t max, tensor_size_t size)
{
    auto constraints = constraints_t{};

    if (size > 0 && min < max)
    {
        for (tensor_size_t dimension = 0; dimension < size; ++dimension)
        {
            constraints.emplace_back(minimum_t{min, dimension});
            constraints.emplace_back(maximum_t{max, dimension});
        }
    }

    return constraints;
}

constraint_t nano::make_ball_constraint(vector_t origin, scalar_t radius)
{
    auto constraint = constraint_t{};
    if (radius > 0.0)
    {
        constraint = inequality_t{std::make_unique<ball_constraint_t>(std::move(origin), radius)};
    }

    return constraint;
}
