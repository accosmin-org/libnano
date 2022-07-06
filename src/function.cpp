#include <nano/core/strutil.h>
#include <nano/core/util.h>
#include <nano/function/constraints.h>

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

bool function_t::compatible(const minimum_t& constraint) const
{
    return constraint.m_dimension >= 0 && constraint.m_dimension < size();
}

bool function_t::compatible(const maximum_t& constraint) const
{
    return constraint.m_dimension >= 0 && constraint.m_dimension < size();
}

bool function_t::compatible(const equality_t& constraint) const
{
    return static_cast<bool>(constraint.m_function) && constraint.m_function->size() == size();
}

bool function_t::compatible(const inequality_t& constraint) const
{
    return static_cast<bool>(constraint.m_function) && constraint.m_function->size() == size();
}

bool function_t::compatible(const constraint_t& constraint) const
{
    return std::visit(overloaded{[&](const minimum_t& constraint) { return this->compatible(constraint); },
                                 [&](const maximum_t& constraint) { return this->compatible(constraint); },
                                 [&](const equality_t& constraint) { return this->compatible(constraint); },
                                 [&](const inequality_t& constraint) { return this->compatible(constraint); }},
                      constraint);
}

bool function_t::constrain(constraint_t&& constraint)
{
    if (compatible(constraint))
    {
        m_constraints.emplace_back(std::move(constraint));
        return true;
    }
    return false;
}

bool function_t::constrain(constraints_t&& constraints)
{
    const auto op = [&](const constraint_t& constraint) { return compatible(constraint); };
    if (!constraints.empty() && std::all_of(std::begin(constraints), std::end(constraints), op))
    {
        for (auto& constraint : constraints)
        {
            m_constraints.emplace_back(std::move(constraint));
        }
        return true;
    }
    return false;
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
