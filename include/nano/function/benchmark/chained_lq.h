#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief convex non-smooth test function: chained LQ.
/// see "New limited memory bundle method for large-scale nonsmooth optimization", by Haarala, Miettinen, Makela, 2004
///
class NANO_PUBLIC function_chained_lq_t final : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit function_chained_lq_t(tensor_size_t dims = 10);

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
    rfunction_t make(tensor_size_t dims, tensor_size_t summands) const override;
};
} // namespace nano
