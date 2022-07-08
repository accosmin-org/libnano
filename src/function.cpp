#include <nano/core/strutil.h>
#include <nano/function.h>

using namespace nano;

function_t::function_t(function_t&&) noexcept = default;

function_t& function_t::operator=(function_t&&) noexcept = default;

function_t::~function_t() = default;

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

bool function_t::constrain(constraint_t&& constraint)
{
    if (compatible(constraint, *this))
    {
        m_constraints.emplace_back(std::move(constraint));
        return true;
    }
    return false;
}

bool function_t::constrain(scalar_t min, scalar_t max, tensor_size_t dimension)
{
    if (min < max && dimension >= 0 && dimension < size())
    {
        m_constraints.emplace_back(constraint::minimum_t{min, dimension});
        m_constraints.emplace_back(constraint::maximum_t{max, dimension});
        return true;
    }
    return false;
}

bool function_t::constrain(scalar_t min, scalar_t max)
{
    if (min < max)
    {
        for (tensor_size_t i = 0, size = this->size(); i < size; ++i)
        {
            m_constraints.emplace_back(constraint::minimum_t{min, i});
            m_constraints.emplace_back(constraint::maximum_t{max, i});
        }
        return true;
    }
    return false;
}

bool function_t::constrain(const vector_t& min, const vector_t& max)
{
    if (min.size() == size() && max.size() == size() && (max - min).minCoeff() > 0.0)
    {
        for (tensor_size_t i = 0, size = this->size(); i < size; ++i)
        {
            m_constraints.emplace_back(constraint::minimum_t{min(i), i});
            m_constraints.emplace_back(constraint::maximum_t{max(i), i});
        }
        return true;
    }
    return false;
}

bool function_t::valid(const vector_t& x) const
{
    assert(x.size() == size());

    const auto op = [&](const constraint_t& constraint)
    { return ::nano::valid(constraint, x) < std::numeric_limits<scalar_t>::epsilon(); };

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
