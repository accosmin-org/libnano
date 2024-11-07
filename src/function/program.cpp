#include <nano/function/lambda.h>
#include <nano/function/program.h>

using namespace nano;

namespace
{
void add_constraints(function_t& function, const program::linear_constrained_t& program)
{
    const auto& A = program.m_eq.m_A;
    const auto& b = program.m_eq.m_b;
    const auto& G = program.m_ineq.m_A;
    const auto& h = program.m_ineq.m_b;

    for (tensor_size_t i = 0; i < A.rows(); ++i)
    {
        function.constrain(constraint::linear_equality_t{A.row(i), -b(i)});
    }
    for (tensor_size_t i = 0; i < G.rows(); ++i)
    {
        function.constrain(constraint::linear_inequality_t{G.row(i), -h(i)});
    }
}
} // namespace

rfunction_t nano::make_function(const program::linear_program_t& program)
{
    const auto& c = program.m_c;

    const auto func = [&](vector_cmap_t x, vector_map_t gx) -> scalar_t
    {
        if (gx.size() == x.size())
        {
            gx = c;
        }
        return x.dot(c);
    };

    auto function = make_function(c.size(), convexity::yes, smoothness::yes, 0.0, func);
    add_constraints(function, program);
    return function.clone();
}

rfunction_t nano::make_function(const program::quadratic_program_t& program)
{
    const auto& Q = program.m_Q;
    const auto& c = program.m_c;

    const auto func = [&](vector_cmap_t x, vector_map_t gx) -> scalar_t
    {
        if (gx.size() == x.size())
        {
            gx = Q * x + c;
        }
        return x.dot(0.5 * (Q * x) + c);
    };

    auto function = make_function(c.size(), convexity::yes, smoothness::yes, 0.0, func);
    add_constraints(function, program);
    return function.clone();
}
