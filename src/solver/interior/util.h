#pragma once

#include <nano/tensor.h>

namespace nano
{
///
/// \brief return the maximum scalar factor `step` so that `u + step * du >= (1 - tau) * u` element-wise.
///
/// NB: it is assumed that the vector `u` is strictly positive element-wise.
///
template <class tvectoru, class tvectordu>
requires((is_tensor_v<tvectoru> || is_eigen_v<tvectoru>) && (is_tensor_v<tvectordu> || is_eigen_v<tvectordu>))
scalar_t make_umax(const tvectoru& u, const tvectordu& du, const scalar_t tau)
{
    assert(tau > 0.0);
    assert(tau <= 1.0);
    assert(u.size() == du.size());
    assert(u.array().allFinite());
    assert(du.array().allFinite());
    assert(u.array().minCoeff() > 0.0);

    const auto delta = 2e-16;
    const auto gamma = 0.999;

    auto step = 1.0;
    for (tensor_size_t i = 0, size = u.size(); i < size; ++i)
    {
        if (du(i) < 0.0)
        {
            step = std::min(step, -tau * u(i) / du(i));
        }
    }

    assert(step > 0.0);

    // NB: take into account numerical precision issues and make sure the post-condition holds.
    for (tensor_size_t trial = 0; trial < 10; ++trial)
    {
        if ((u + step * du - (1.0 - tau) * u).minCoeff() < delta)
        {
            step *= gamma;
        }
        else
        {
            break;
        }
    }

    assert((u + step * du - (1.0 - tau) * u).minCoeff() >= 0.0);

    return step;
}

///
/// \brief in-place modified Ruiz equilibration of the matrices involved in a linear or quadratic program:
///     min. c.dot(x)
///     s.t. G * x <= h
///          A * x = b
///
///     min. 0.5 * x.dot(Q * x) + c.dot(x)
///     s.t. G * x <= h
///          A * x = b
///
/// see (1) "A scaling algorithm to equalibrate both rows and columns norms in matrices", D. Ruiz, 2001.
/// see (2) "OSQP: an operator splitting solver for quadratic programs", B. Stellato et al., 2020.
/// see (3) "COSMO: A conic operator splitting method for convex conic problems", M. Garstka et al., 2020.
///
/// NB: the implementation follows (3).
/// NB: if Q is empty then the program is considered linear.
///
NANO_PUBLIC void modified_ruiz_equilibration(vector_t& dQ, matrix_t& Q, vector_t& c, vector_t& dG, matrix_t& G,
                                             vector_t& h, vector_t& dA, matrix_t& A, vector_t& b, scalar_t tau = 1e-12,
                                             scalar_t tolerance = 1e-12);
} // namespace nano
