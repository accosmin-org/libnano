#include <nano/core/strutil.h>
#include <nano/function/constraints.h>
#include <nano/function/util.h>

using namespace nano;

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
    return origin.size() == size() && radius > 0.0 &&
           constrain_inequality(std::make_unique<ball_constraint_t>(std::move(origin), radius));
}

bool function_t::constrain_equality(vector_t q, scalar_t r)
{
    return q.size() == size() && constrain_equality(std::make_unique<affine_constraint_t>(std::move(q), r));
}

bool function_t::constrain_equality(matrix_t P, vector_t q, scalar_t r)
{
    return q.size() == size() && P.rows() == size() && P.cols() == size() &&
           constrain_equality(std::make_unique<quadratic_constraint_t>(std::move(P), std::move(q), r));
}

bool function_t::constrain_inequality(vector_t q, scalar_t r)
{
    return q.size() == size() && constrain_inequality(std::make_unique<affine_constraint_t>(std::move(q), r));
}

bool function_t::constrain_inequality(matrix_t P, vector_t q, scalar_t r)
{
    return q.size() == size() && P.rows() == size() && P.cols() == size() &&
           constrain_inequality(std::make_unique<quadratic_constraint_t>(std::move(P), std::move(q), r));
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

scalar_t function_t::vgrad(const vector_t& x, vector_t* gx) const
{
    m_fcalls += 1;
    m_gcalls += (gx != nullptr) ? 1 : 0;
    return do_vgrad(x, gx);
}

tensor_size_t function_t::fcalls() const
{
    return m_fcalls;
}

tensor_size_t function_t::gcalls() const
{
    return m_gcalls;
}

void function_t::clear_statistics() const
{
    m_fcalls = 0;
    m_gcalls = 0;
}
