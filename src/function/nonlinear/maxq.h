#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief convex non-smooth test function: MAXQ(x) = max(i, x_i^2).
///
/// see "New limited memory bundle method for large-scale nonsmooth optimization", by Haarala, Miettinen, Makela, 2004
///
class NANO_PUBLIC function_maxq_t final : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit function_maxq_t(tensor_size_t dims = 10);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    scalar_t do_eval(eval_t) const override;

    ///
    /// \brief @see function_t
    ///
    rfunction_t make(tensor_size_t dims) const override;
};
} // namespace nano
