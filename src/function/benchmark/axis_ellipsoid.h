#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief axis-parallel hyper-ellipsoid function: f(x) = sum(i*x+i^2, i=1,D).
///
class NANO_PUBLIC function_axis_ellipsoid_t final : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit function_axis_ellipsoid_t(tensor_size_t dims = 10);

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

private:
    // attributes
    vector_t m_bias; ///<
};
} // namespace nano
