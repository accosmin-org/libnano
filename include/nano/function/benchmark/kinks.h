#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief random kinks function: f(x) = sum(|x - k_i|, i=1,K).
///
class NANO_PUBLIC function_kinks_t final : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit function_kinks_t(tensor_size_t dims = 10);

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

private:
    // attributes
    matrix_t m_kinks;     ///<
    scalar_t m_offset{0}; ///< offset so that the global minimum is exactly zero
};
} // namespace nano
