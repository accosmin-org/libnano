#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief Rosenbrock function: see https://en.wikipedia.org/wiki/Test_functions_for_optimization.
///
class NANO_PUBLIC function_rosenbrock_t final : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit function_rosenbrock_t(tensor_size_t dims = 10);

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
