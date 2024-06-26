#pragma once

#include <nano/program/equality.h>
#include <nano/program/inequality.h>

namespace nano::program
{
namespace detail
{
template <class tconstraint>
void update_size(tensor_size_t& rows, tensor_size_t& cols, const tconstraint& constraint)
{
    assert(constraint.m_A.rows() == constraint.m_b.size());
    assert(!cols || cols == constraint.m_A.cols());

    rows += constraint.m_A.rows();
    cols = constraint.m_A.cols();
}

template <template <class, class> class tconstraint, class tmatrixA, class tvectorb>
void update_data(matrix_t& A, vector_t& b, const tensor_size_t row, const tconstraint<tmatrixA, tvectorb>& constraint)
{
    if constexpr (is_eigen_v<tmatrixA>)
    {
        A.block(row, 0, constraint.m_A.rows(), constraint.m_A.cols()) = constraint.m_A;
    }
    else
    {
        static_assert(is_tensor_v<tmatrixA>);
        static_assert(tmatrixA::rank() == 2U);
        A.block(row, 0, constraint.m_A.rows(), constraint.m_A.cols()) = constraint.m_A.matrix();
    }
    if constexpr (is_eigen_v<tvectorb>)
    {
        b.segment(row, constraint.m_b.size()) = constraint.m_b;
    }
    else
    {
        static_assert(is_tensor_v<tvectorb>);
        static_assert(tvectorb::rank() == 1U);
        b.segment(row, constraint.m_b.size()) = constraint.m_b.vector();
    }
}

template <class tconstraint, class... tconstraints>
void make_size([[maybe_unused]] tensor_size_t& eqs, tensor_size_t& dims, [[maybe_unused]] tensor_size_t& ineqs,
               const tconstraint& constraint, const tconstraints&... constraints)
{
    if constexpr (is_equality_v<tconstraint>)
    {
        update_size(eqs, dims, constraint);
    }
    else
    {
        update_size(ineqs, dims, constraint);
    }
    if constexpr (sizeof...(constraints) > 0)
    {
        make_size(eqs, dims, ineqs, constraints...);
    }
}

template <class tconstraint, class... tconstraints>
void stack([[maybe_unused]] matrix_t& A, [[maybe_unused]] vector_t& b, [[maybe_unused]] matrix_t& G,
           [[maybe_unused]] vector_t& h, [[maybe_unused]] tensor_size_t eq, [[maybe_unused]] tensor_size_t ineq,
           const tconstraint& constraint, const tconstraints&... constraints)
{
    if constexpr (is_equality_v<tconstraint>)
    {
        update_data(A, b, eq, constraint);
        eq += constraint.m_A.rows();
    }
    else
    {
        update_data(G, h, ineq, constraint);
        ineq += constraint.m_A.rows();
    }
    if constexpr (sizeof...(constraints) > 0)
    {
        stack(A, b, G, h, eq, ineq, constraints...);
    }
}
} // namespace detail

///
/// \brief (vertically-)stack in-place the given equality and inequality constraints:
///     A * x = b and G * x <= h.
///
template <class... tconstraints>
void stack(matrix_t& A, vector_t& b, matrix_t& G, vector_t& h, const tconstraints&... constraints)
{
    auto eqs   = tensor_size_t{0};
    auto dims  = tensor_size_t{0};
    auto ineqs = tensor_size_t{0};
    detail::make_size(eqs, dims, ineqs, constraints...);

    A.resize(eqs, dims);
    b.resize(eqs);
    G.resize(ineqs, dims);
    h.resize(ineqs);
    detail::stack(A, b, G, h, 0, 0, constraints...);
}
} // namespace nano::program
