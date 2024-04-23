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
    /// \brief
    ///
    static void config(configurable_t&, const string_t& prefix);

    ///
    /// \brief
    ///
    static proximity_t make(const solver_state_t&, const configurable_t&, const string_t& prefix);

    ///
    /// \brief return the current proximity parameter value.
    ///
    scalar_t miu() const;

    ///
    /// \brief update the proximity parameter given a new proximity center.
    ///
    void update(scalar_t t, const vector_t& xn, const vector_t& xn1, const vector_t& gn, const vector_t& gn1);
    void update(scalar_t t, const vector_t& xn, const vector_t& xn1, const vector_t& gn, const vector_t& gn1,
                const vector_t& Gn, const vector_t& Gn1);

private:
    // attributes
    scalar_t m_miu{1.0}; ///< current proximity parameter value
};
} // namespace nano
