#pragma once

#include <nano/function.h>

namespace nano
{
    ///
    /// \brief Chung-Reynolds function: f(x) = (x.dot(x))^2.
    ///
    class NANO_PUBLIC function_chung_reynolds_t final : public function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        explicit function_chung_reynolds_t(tensor_size_t dims = 10);

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
