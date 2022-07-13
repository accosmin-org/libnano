#include <nano/function/penalty.h>

using namespace nano;

static auto is_equality(const constraint_t& constraint)
{
    return std::get_if<constraint::constant_t>(&constraint) != nullptr ||
           std::get_if<constraint::ball_equality_t>(&constraint) != nullptr ||
           std::get_if<constraint::linear_equality_t>(&constraint) != nullptr ||
           std::get_if<constraint::quadratic_equality_t>(&constraint) != nullptr ||
           std::get_if<constraint::functional_equality_t>(&constraint) != nullptr;
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

        if (is_equality(constraint))
        {
            if (fc > 0.0)
            {
                fx += op(fc, gc);
            }
            else if (fc < 0.0)
            {
                fx += op(-fc, -gc);
            }
        }
        else if (fc > 0.0)
        {
            fx += op(fc, gc);
        }
    }

    return fx;
}

penalty_function_t::penalty_function_t(const function_t& constrained)
    : function_t("penalty", constrained.size())
    , m_constrained(constrained)
{
    strong_convexity(0.0); // NB: cannot estimate the strong convexity coefficient in general!
}

penalty_function_t& penalty_function_t::penalty_term(scalar_t penalty_term)
{
    m_penalty_term = penalty_term;
    return *this;
}

linear_penalty_function_t::linear_penalty_function_t(const function_t& constrained)
    : penalty_function_t(constrained)
{
    convex(::convex(constrained));
    smooth(constrained.constraints().empty() ? ::smooth(constrained) : false);
}

scalar_t linear_penalty_function_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    assert(x.size() == size());

    const auto op = [&](scalar_t fc, const auto& gc)
    {
        if (gx != nullptr)
        {
            gx->noalias() += penalty_term() * gc;
        }
        return penalty_term() * fc;
    };

    return penalty_vgrad(constrained(), x, gx, op);
}

quadratic_penalty_function_t::quadratic_penalty_function_t(const function_t& constrained)
    : penalty_function_t(constrained)
{
    convex(::convex(constrained));
    smooth(::smooth(constrained));
}

scalar_t quadratic_penalty_function_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    assert(x.size() == size());

    const auto op = [&](scalar_t fc, const auto& gc)
    {
        if (gx != nullptr)
        {
            gx->noalias() += penalty_term() * 2.0 * fc * gc;
        }
        return penalty_term() * fc * fc;
    };

    return penalty_vgrad(constrained(), x, gx, op);
}
