#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief Qing function: see http://benchmarkfcns.xyz/benchmarkfcns/qingfcn.html.
///
class NANO_PUBLIC function_qing_t final : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit function_qing_t(tensor_size_t dims = 10);

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
    vector_t m_bias; ///<
};
} // namespace nano
