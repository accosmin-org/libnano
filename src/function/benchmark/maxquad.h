#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief convex non-smooth test function: MAXQUAD(x) = max(k, x.dot(A_k*x) - b_k.dot(x)),
///     where A_k is symmetric (and thus the problem convex).
///
/// see "A set of nonsmooth optimization test problems" in "Nonsmooth optimization", by Lemarechal, Mifflin, 1978
///
class NANO_PUBLIC function_maxquad_t final : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit function_maxquad_t(tensor_size_t dims = 10, tensor_size_t kdims = 5);

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

private:
    // attributes
    tensor3d_t m_Aks; ///< (5, n, n)
    tensor2d_t m_bks; ///< (5, n)
};
} // namespace nano
