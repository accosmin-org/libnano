#pragma once

#include <solver/bundle/csearch.h>

namespace nano::bundle
{
enum class quasi_type : uint8_t
{
    sr1, ///< symmetric rank one from (3)
    miu, ///< poor man's approximation (scaled identity) from (3)
};

///
/// \brief models the quasi-newton updates used by penalized (proximal) bundle algorithms.
///
/// see (1) "A doubly stabilized bundle method for nonsmooth convex optimization", by Oliveira, Solodov, 2013
/// see (2) "Numerical optimization - theoretical and practical aspects", 2nd edition, 2006
/// see (3) "Variable metric bundle methods: from conceptual to implementable forms", by Lemarechal, Sagastizabal, 1997
/// see (4) "Dynamical adjustment of the prox-parameter in bundle methods", by Rey, Sagastizabal, 2002
/// see (5) "Fast proximal algorithms for nonsmooth convex optimization", by Ouorou, 2020
/// see (6) "A NU-algorithm for convex minimization", by Mifflin, Sagastizabal, 2005
///
/// NB: the implementation follows the notation and the algorithm from (3) to update `M_n`.
///
class NANO_PUBLIC quasi_t
{
public:
    ///
    /// \brief constructor
    ///
    quasi_t(tensor_size_t dims, quasi_type);
    quasi_t(const solver_state_t&, quasi_type);

    ///
    /// \brief
    ///
    static void config(configurable_t&, const string_t& prefix);

    ///
    /// \brief
    ///
    static quasi_t make(const solver_state_t&, const configurable_t&, const string_t& prefix);

    ///
    /// \brief return the current quasi-newton approximation.
    ///
    const matrix_t& M() const { return m_M; }

    ///
    /// \brief update the quasi-newton approximaton from (3) if a descent step and return the current approximation.
    ///
    const matrix_t& update(const vector_t& x, const vector_t& g, const vector_t& G, bool is_descent_step);

private:
    // attributes
    matrix_t   m_M;                     ///<
    vector_t   m_xn, m_xn1;             ///<
    vector_t   m_gn, m_gn1;             ///<
    vector_t   m_Gn, m_Gn1;             ///<
    quasi_type m_type{quasi_type::miu}; ///< strategy to update the proximal parameter
};
} // namespace nano::bundle
