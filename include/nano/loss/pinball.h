#pragma once

#include <nano/loss.h>

namespace nano
{
///
/// \brief the pinball loss is used to estimate a particular quantile, see (1).
///
/// (1): "Quantile Regression", 2005, by E. Koenker
///
class NANO_PUBLIC pinball_loss_t final : public loss_t
{
public:
    ///
    /// \brief constructor
    ///
    pinball_loss_t();

    ///
    /// \brief @see clonable_t
    ///
    rloss_t clone() const override;

    ///
    /// \brief @see loss_t
    ///
    void error(const tensor4d_cmap_t& targets, const tensor4d_cmap_t& outputs, tensor1d_map_t errors) const override;

    ///
    /// \brief @see loss_t
    ///
    void value(const tensor4d_cmap_t& targets, const tensor4d_cmap_t& outputs, tensor1d_map_t values) const override;

    ///
    /// \brief @see loss_t
    ///
    void vgrad(const tensor4d_cmap_t& targets, const tensor4d_cmap_t& outputs, tensor4d_map_t vgrads) const override;
};
} // namespace nano
