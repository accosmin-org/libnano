#pragma once

#include <nano/configurable.h>
#include <nano/solver/state.h>

namespace nano
{
///
/// \brief models the proximity parameter as used by penalized (proximal) bundle algorithms.
///
/// see (1) "Numerical optimization - theoretical and practical aspects", 2nd edition, 2006
/// see (2) "Variable metric bundle methods: from conceptual to implementable forms", by Lemarechal, Sagastizabal, 1997
/// see (3) "Dynamical adjustment of the prox-parameter in bundle methods", by Rey, Sagastizabal, 2002
/// see (4) "A NU-algorithm for convex minimization", by Mifflin, Sagastizabal, 2005
///
/// NB: the proximity parameter is initialized following (4) (ch. 6).
///
class proximity_t
{
public:
    ///
    /// \brief constructor
    ///
    proximity_t(const solver_state_t& state, scalar_t miu0_min, scalar_t miu0_max);

    ///
    /// \brief setup the default configuration.
    ///
    static void config(configurable_t&, const string_t& prefix);

    ///
    /// \brief construct a proximity parameter with the given configuration.
    ///
    static proximity_t make(const solver_state_t&, const configurable_t&, const string_t& prefix);

    ///
    /// \brief return the current quasi-newton approximation of the Hessian.
    ///
    const matrix_t& M() const { return m_M; }

    ///
    /// \brief return the current quasi-newton approximation of the Hessian's inverse.
    ///
    const matrix_t& invM() const { return m_invM; }

    ///
    /// \brief update the proximity parameter given a new proximity center, see (2).
    ///
    void update(scalar_t t, const vector_t& xn, const vector_t& xn1, const vector_t& gn, const vector_t& gn1);

private:
    // attributes
    matrix_t m_M;    ///< quasi-newton approximation of the Hessian (see the M_n in (2))
    matrix_t m_invM; ///< quasi-newton approximation of the Hessian's inverse (see the M_n^-1 in (2))
};
} // namespace nano
