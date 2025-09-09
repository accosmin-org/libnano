#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief rotated hyper-ellipsoid function: see https://www.sfu.ca/~ssurjano/rothyp.html.
///
class NANO_PUBLIC function_rotated_ellipsoid_t final : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit function_rotated_ellipsoid_t(tensor_size_t dims = 10);

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
