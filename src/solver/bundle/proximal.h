#pragma once

#include <nano/configurable.h>
#include <nano/solver/state.h>

namespace nano
{
///
/// \brief models the proximal parameter as used by penalized (proximal) bundle algorithms.
///
/// see (1) "A doubly stabilized bundle method for nonsmooth convex optimization", by Oliveira, Solodov, 2013
/// see (2) "Numerical optimization - theoretical and practical aspects", 2nd edition, 2006
/// see (3) "Variable metric bundle methods: from conceptual to implementable forms", by Lemarechal, Sagastizabal, 1997
/// see (4) "Dynamical adjustment of the prox-parameter in bundle methods", by Rey, Sagastizabal, 2002
/// see (5) "Fast proximal algorithms for nonsmooth convex optimization", by Ouorou, 2020
/// see (6) "A NU-algorithm for convex minimization", by Mifflin, Sagastizabal, 2005
///
/// NB: the implementation follows the notation and the algorithm from (1) to update `tau`.
///
/// NB: some bundle algorithms like (3) or (5) use the inverse `miu = 1/tau` convention.
///
/// TODO: implement variation from PBM-1 from (1)!
///
class NANO_PUBLIC proximal_t
{
public:
    ///
    /// \brief constructor
    ///
    proximal_t(const solver_state_t& state, scalar_t tau_min, scalar_t alpha);

    ///
    /// \brief
    ///
    static void config(configurable_t&, const string_t& prefix);

    ///
    /// \brief
    ///
    static proximal_t make(const solver_state_t&, const configurable_t&, const string_t& prefix);

    ///
    /// \brief return the current proximal parameter value (`tau` like in (1)).
    ///
    scalar_t tau() const;

    ///
    /// \brief return the current proximal parameter value (`miu = 1/tau` like in (3) or (5)).
    ///
    scalar_t miu() const;

    ///
    /// \brief update the proximal parameter following strategy PBM-1 or PBM-2 from (1).
    ///
    /// NB: the scaling factor `t` is computed following the curve search algorithm from (3), thus `miu/t = 1/tau`.
    ///
    void update(bool descent_step, scalar_t t, const vector_t& xn0, const vector_t& gn0, const vector_t& xn1,
                const vector_t& gn1);

private:
    // attributes
    scalar_t      m_tau{1.0};              ///<
    scalar_t      m_tau_min{1e-5};         ///< minimum value of the proximal parameter
    scalar_t      m_alpha{4.0};            ///< scaling factor for the proximal parameter
    tensor_size_t m_past_descent_steps{0}; ///< number of steps with consecutive descent steps
};
} // namespace nano
