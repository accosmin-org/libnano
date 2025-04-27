#pragma once

#include <nano/tensor.h>

namespace nano
{
///
/// \brief return the maximum scalar factor `step` so that `u + step * du > 0` element-wise.
///
template <class tvectoru, class tvectordu>
scalar_t make_umax(const tvectoru& u, const tvectordu& du)
{
    assert(u.size() == du.size());

    auto step = std::numeric_limits<scalar_t>::max();
    for (tensor_size_t i = 0, size = u.size(); i < size; ++i)
    {
        if (du(i) < 0.0)
        {
            step = std::min(step, -u(i) / du(i));
        }
    }

    return std::min(step, 1.0);
}

///
/// \brief return the maximum scalar factor `step` so that `G * (x + step * dx) <= h` element-wise.
///
NANO_PUBLIC scalar_t make_xmax(const vector_t& x, const vector_t& dx, const matrix_t& G, const vector_t& h);

///
/// \brief in-place modified Ruiz equilibration of the matrices involved in a quadratic program:
///     min. 0.5 * x.dot(Q * x) + c.dot(x)
///     s.t. G * x <= h
///          A * x = b
///
/// see (1) "A scaling algorithm to equalibrate both rows and columns norms in matrices", D. Ruiz, 2001
/// see (2) "OSQP: an operator splitting solver for quadratic programs", B. Stellato et al., 2020
/// see (3) "COSMO: A conic operator splitting method for convex conic problems", M. Garstka et al., 2020
///
/// NB: the implementation follows (3).
///
NANO_PUBLIC void modified_ruiz_equilibration(vector_t& dQ, matrix_t& Q, vector_t& c, vector_t& dG, matrix_t& G,
                                             vector_t& h, vector_t& dA, matrix_t& A, vector_t& b, scalar_t tau = 1e-8,
                                             scalar_t tolerance = 1e-12);
}
