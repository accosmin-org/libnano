#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief Cauchy function: f(x) = log(1 + x.dot(x)).
///
class NANO_PUBLIC function_cauchy_t final : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit function_cauchy_t(tensor_size_t dims = 10);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    scalar_t do_vgrad(vector_cmap_t x, vector_map_t gx) const override;

    ///
    /// \brief @see function_t
    ///
    rfunction_t make(tensor_size_t dims) const override;
};
} // namespace nano
