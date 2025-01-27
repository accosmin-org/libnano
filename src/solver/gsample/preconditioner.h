#pragma once

#include <solver/gsample/sampler.h>

using namespace nano;

namespace nano::gsample
{
///
/// \brief identity preconditioner: W = H = I(n, n).
///
class identity_preconditioner_t
{
public:
    using storage_t = decltype(matrix_t::identity(3, 3));

    explicit identity_preconditioner_t(tensor_size_t n);

    static auto str() { return ""; }

    void update(scalar_t alpha);

    void update(const sampler_t&, const solver_state_t&, scalar_t epsilon);

    const storage_t& W() const { return m_W; }

    const storage_t& H() const { return m_H; }

private:
    // attributes
    storage_t m_W; ///< H^-1
    storage_t m_H; ///<
};

///
/// \brief LBFGS-like preconditioner.
///
class lbfgs_preconditioner_t
{
public:
    explicit lbfgs_preconditioner_t(tensor_size_t n);

    static auto str() { return "-lbfgs"; }

    void update(scalar_t alpha);

    void update(const sampler_t& sampler, const solver_state_t& state, scalar_t epsilon);

    const matrix_t& W() const { return m_W; }

    const matrix_t& H() const { return m_H; }

private:
    // attributes
    matrix_t m_W; ///< H^-1
    matrix_t m_H; ///<
    matrix_t m_Q; ///< FIXME: is this still needed?!
    scalar_t m_miu{1.0};
};
} // namespace nano::gsample
