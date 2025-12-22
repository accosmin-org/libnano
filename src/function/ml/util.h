#pragma once

#include <nano/critical.h>
#include <nano/enum.h>
#include <nano/function.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>

namespace nano
{
enum class lasso_type : uint8_t
{
    constrained,
    unconstrained,
};

template <>
inline enum_map_t<lasso_type> enum_string<lasso_type>()
{
    return {
        {  lasso_type::constrained,   "constrained"},
        {lasso_type::unconstrained, "unconstrained"},
    };
}

inline tensor_size_t make_size(const tensor_size_t dims, const lasso_type type = lasso_type::unconstrained)
{
    const auto size = std::max(dims, tensor_size_t{2});

    switch (type)
    {
    case lasso_type::unconstrained:
        // solve for (x,)
        return size;

    default:
        // solve for (x, z)
        return 2 * size;
    }
}

inline tensor_size_t make_inputs(const tensor_size_t dims)
{
    return std::max(dims, tensor_size_t{2});
}

inline tensor_size_t make_outputs([[maybe_unused]] const tensor_size_t dims)
{
    return tensor_size_t{1};
}

inline tensor_size_t make_samples(const tensor_size_t dims, const scalar_t sratio)
{
    return static_cast<tensor_size_t>(std::max(sratio * static_cast<scalar_t>(dims), 10.0));
}

inline void constrain_lasso(function_t& function)
{
    // min  f(x, z)
    // s.t. +x <= z
    //      -x <= z
    const auto n = function.size() / 2;

    auto A              = matrix_t{2 * n, 2 * n};
    A.block(0, 0, n, n) = matrix_t::identity(n, n);
    A.block(0, n, n, n) = -matrix_t::identity(n, n);
    A.block(n, 0, n, n) = -matrix_t::identity(n, n);
    A.block(n, n, n, n) = -matrix_t::identity(n, n);

    critical(A * function.variable() <= vector_t::zero(2 * n));
}

inline function_t::eval_t make_lasso_eval(function_t::eval_t& eval)
{
    auto       x  = eval.m_x;
    auto       gx = eval.m_gx;
    auto       hx = eval.m_hx;
    const auto n  = x.size() / 2;

    return {
        .m_x  = x.slice(0, n),
        .m_gx = eval.has_grad() ? gx.slice(0, n) : gx.tensor(),
        .m_hx = eval.has_hess() ? hx.reshape(hx.size()).slice(0, n * n).reshape(n, n).tensor() : hx.tensor(),
    };
}

inline void update_lasso_hess(function_t::eval_t& eval)
{
    auto       x  = eval.m_x;
    auto       hx = eval.m_hx;
    const auto n  = x.size() / 2;

    hx.block(n, n, n, n) = hx.reshape(4 * n * n).slice(0, n * n).reshape(n, n).matrix();
    hx.block(0, 0, n, n) = hx.block(n, n, n, n);

    hx.block(0, n, n, n) = matrix_t::zero(n, n);
    hx.block(n, 0, n, n) = matrix_t::zero(n, n);
    hx.block(n, n, n, n) = matrix_t::zero(n, n);
}
} // namespace nano
