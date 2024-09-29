#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief convex non-smooth test function: MAXHILB(x) = max(i, sum(j, xj / (i + j -1)).
///
/// see "New limited memory bundle method for large-scale nonsmooth optimization", by Haarala, Miettinen, Makela, 2004
///
class NANO_PUBLIC function_maxhilb_t final : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit function_maxhilb_t(tensor_size_t dims = 10);

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
    matrix_t m_weights; ///<
};
} // namespace nano
