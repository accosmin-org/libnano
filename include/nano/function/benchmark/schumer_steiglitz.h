#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief Schumer-Steiglitz No. 02 function: f(x) = sum(x_i^4, i=1,D).
///
class NANO_PUBLIC function_schumer_steiglitz_t final : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit function_schumer_steiglitz_t(tensor_size_t dims = 10);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    scalar_t do_vgrad(const vector_t& x, vector_t* gx) const override;

    ///
    /// \brief @see function_t
    ///
    rfunction_t make(tensor_size_t dims, tensor_size_t summands) const override;
};
} // namespace nano
