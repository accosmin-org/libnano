#include <nano/function/penalty.h>

using namespace nano;

static auto smooth(const function_t& function)
{
    return function.smooth() && std::all_of(function.constraints().begin(), function.constraints().end(),
                                            [](const auto& constraint) { return ::nano::smooth(constraint); });
}

template <typename toperator>
static auto penalty_vgrad(const function_t& function, const vector_t& x, vector_t* gx, const toperator& op)
{
    scalar_t fx = function.vgrad(x, gx);
    vector_t gc{gx != nullptr ? x.size() : tensor_size_t{0}};

    for (const auto& constraint : function.constraints())
    {
        const auto fc          = ::nano::vgrad(constraint, x, gc.size() == 0 ? nullptr : &gc);
        const auto is_equality = std::get_if<equality_t>(&constraint) != nullptr;

        if (is_equality)
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
    convex(false); // NB: cannot guarantee convexity!
    smooth(constrained.constraints().empty() ? ::smooth(constrained) : false);
}

scalar_t linear_penalty_function_t::vgrad(const vector_t& x, vector_t* gx) const
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
    convex(false); // NB: cannot guarantee convexity!
    smooth(::smooth(constrained));
}

scalar_t quadratic_penalty_function_t::vgrad(const vector_t& x, vector_t* gx) const
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
