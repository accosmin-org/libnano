#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief Trid function: see https://www.sfu.ca/~ssurjano/trid.html.
///
class NANO_PUBLIC function_trid_t final : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit function_trid_t(tensor_size_t dims = 10);

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
