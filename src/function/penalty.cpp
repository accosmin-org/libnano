#include <nano/function/penalty.h>

using namespace nano;

static auto is_equality(const constraint_t& constraint)
{
    return std::get_if<constraint::constant_t>(&constraint) != nullptr ||
           std::get_if<constraint::linear_equality_t>(&constraint) != nullptr ||
           std::get_if<constraint::quadratic_equality_t>(&constraint) != nullptr ||
           std::get_if<constraint::functional_equality_t>(&constraint) != nullptr ||
           std::get_if<constraint::euclidean_ball_equality_t>(&constraint) != nullptr;
}

static auto is_linear_equality(const constraint_t& constraint)
{
    return std::get_if<constraint::constant_t>(&constraint) != nullptr ||
           std::get_if<constraint::linear_equality_t>(&constraint) != nullptr;
}

static auto convex(const function_t& function)
{
    const auto op = [](const auto& ct) { return ::nano::convex(ct) && (!is_equality(ct) || is_linear_equality(ct)); };
    return function.convex() && std::all_of(function.constraints().begin(), function.constraints().end(), op);
}

static auto smooth(const function_t& function)
{
    const auto op = [](const auto& constraint) { return ::nano::smooth(constraint); };
    return function.smooth() && std::all_of(function.constraints().begin(), function.constraints().end(), op);
}

template <typename toperator>
static auto penalty_vgrad(const function_t& function, const vector_t& x, vector_t* gx, const toperator& op)
{
    scalar_t fx = function.vgrad(x, gx);
    vector_t gc{gx != nullptr ? x.size() : tensor_size_t{0}};

    for (const auto& constraint : function.constraints())
    {
        const auto fc = ::nano::vgrad(constraint, x, gc.size() == 0 ? nullptr : &gc);
        const auto eq = is_equality(constraint);

        if (eq || fc > 0.0)
        {
            fx += op(fc, gc);
        }
    }

    return fx;
}

penalty_function_t::penalty_function_t(const function_t& function)
    : function_t("penalty", function.size())
    , m_function(function)
{
    strong_convexity(0.0); // NB: cannot estimate the strong convexity coefficient in general!
}

penalty_function_t& penalty_function_t::penalty(scalar_t penalty)
{
    m_penalty = penalty;
    return *this;
}

linear_penalty_function_t::linear_penalty_function_t(const function_t& function)
    : penalty_function_t(function)
{
    convex(::convex(function));
    smooth(function.constraints().empty() ? ::smooth(function) : false);
}

rfunction_t linear_penalty_function_t::clone() const
{
    return std::make_unique<linear_penalty_function_t>(*this);
}

scalar_t linear_penalty_function_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    assert(x.size() == size());

    const auto op = [&](scalar_t fc, const auto& gc)
    {
        if (gx != nullptr)
        {
            gx->noalias() += penalty() * (fc >= 0.0 ? +1.0 : -1.0) * gc;
        }
        return penalty() * std::fabs(fc);
    };

    return penalty_vgrad(function(), x, gx, op);
}

quadratic_penalty_function_t::quadratic_penalty_function_t(const function_t& function)
    : penalty_function_t(function)
{
    convex(::convex(function));
    smooth(::smooth(function));
}

rfunction_t quadratic_penalty_function_t::clone() const
{
    return std::make_unique<quadratic_penalty_function_t>(*this);
}

scalar_t quadratic_penalty_function_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    assert(x.size() == size());

    const auto op = [&](scalar_t fc, const auto& gc)
    {
        if (gx != nullptr)
        {
            gx->noalias() += penalty() * 2.0 * fc * gc;
        }
        return penalty() * fc * fc;
    };

    return penalty_vgrad(function(), x, gx, op);
}
