#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief generic geometric optimization function: f(x) = sum(i, exp(alpha_i + a_i.dot(x))).
///
/// see "Introductory Lectures on Convex Optimization (Applied Optimization)", by Y. Nesterov, 2013, p.56
/// see "Convex Optimization", by S. Boyd and L. Vanderberghe, p.458 (logarithmic version)
///
class NANO_PUBLIC function_geometric_optimization_t final : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit function_geometric_optimization_t(tensor_size_t dims = 10, uint64_t seed = 42, scalar_t sratio = 10.0);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    string_t do_name() const override;

    ///
    /// \brief @see function_t
    ///
    scalar_t do_eval(eval_t) const override;

    ///
    /// \brief @see function_t
    ///
    rfunction_t make(tensor_size_t dims) const override;

private:
    // attributes
    vector_t m_a; ///<
    matrix_t m_A; ///<
};
} // namespace nano
