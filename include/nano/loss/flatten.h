#pragma once

#include <cassert>
#include <nano/loss.h>
#include <nano/loss/cauchy.h>
#include <nano/loss/class.h>
#include <nano/loss/classnll.h>
#include <nano/loss/error.h>
#include <nano/loss/exponential.h>
#include <nano/loss/hinge.h>
#include <nano/loss/logistic.h>
#include <nano/loss/mae.h>
#include <nano/loss/mse.h>
#include <nano/loss/savage.h>
#include <nano/loss/squared_hinge.h>
#include <nano/loss/tangent.h>

namespace nano
{
///
/// \brief un-structured loss function: the 3D structure of a sample is flatten
///     and all dimensions are considered the same in computing the loss.
///
/// NB: the multi-label classification problem is handled by summing or averaging:
///     - the associated binary classification loss value per output
///     - the associated 0-1 loss error per output
///
/// see the following resources regarding loss functions for classification:
///
/// (1): "On the design of robust classifiers for computer vision",
///      2010, by H. Masnadi-Shirazi, V. Mahadevan, N. Vasconcelos
///
/// (2): "On the design of loss functions for classification: theory, robustness to outliers, and SavageBoost",
///      2008, by H. Masnadi-Shirazi, N. Vasconcelos
///
template <class tloss>
class flatten_loss_t final : public loss_t
{
public:
    ///
    /// \brief constructor
    ///
    flatten_loss_t()
        : loss_t(string_t(tloss::prefix) + string_t(tloss::basename))
    {
        convex(tloss::convex);
        smooth(tloss::smooth);
    }

    ///
    /// \brief @see clonable_t
    ///
    rloss_t clone() const override { return std::make_unique<flatten_loss_t<tloss>>(*this); }

    ///
    /// \brief @see loss_t
    ///
    void do_error(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor1d_map_t errors) const override
    {
        for (tensor_size_t i = 0, samples = targets.size<0>(); i < samples; ++i)
        {
            errors(i) = tloss::error(targets.array(i), outputs.array(i));
        }
    }

    ///
    /// \brief @see loss_t
    ///
    void do_value(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor1d_map_t values) const override
    {
        for (tensor_size_t i = 0, samples = targets.size<0>(); i < samples; ++i)
        {
            values(i) = tloss::value(targets.array(i), outputs.array(i));
        }
    }

    ///
    /// \brief @see loss_t
    ///
    void do_vgrad(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor4d_map_t vgrads) const override
    {
        for (tensor_size_t i = 0, samples = targets.size<0>(); i < samples; ++i)
        {
            tloss::vgrad(targets.array(i), outputs.array(i), vgrads.array(i));
        }
    }

    ///
    /// \brief @see loss_t
    ///
    void do_vhess(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor7d_map_t vhesss) const override
    {
        const auto tsize = targets.size<1>() * targets.size<2>() * targets.size<3>();

        for (tensor_size_t i = 0, samples = targets.size<0>(); i < samples; ++i)
        {
            tloss::vhess(targets.array(i), outputs.array(i), vhesss.tensor(i).reshape(tsize, tsize).matrix());
        }
    }
};

// regression
using mae_loss_t    = flatten_loss_t<detail::mae_t<loss::detail::absdiff_t>>;
using mse_loss_t    = flatten_loss_t<detail::mse_t<loss::detail::absdiff_t>>;
using cauchy_loss_t = flatten_loss_t<detail::cauchy_t<loss::detail::absdiff_t>>;

// single-label classification
using shinge_loss_t         = flatten_loss_t<detail::hinge_t<loss::detail::sclass_t>>;
using ssavage_loss_t        = flatten_loss_t<detail::savage_t<loss::detail::sclass_t>>;
using stangent_loss_t       = flatten_loss_t<detail::tangent_t<loss::detail::sclass_t>>;
using sclassnll_loss_t      = flatten_loss_t<detail::classnll_t<loss::detail::sclass_t>>;
using slogistic_loss_t      = flatten_loss_t<detail::logistic_t<loss::detail::sclass_t>>;
using sexponential_loss_t   = flatten_loss_t<detail::exponential_t<loss::detail::sclass_t>>;
using ssquared_hinge_loss_t = flatten_loss_t<detail::squared_hinge_t<loss::detail::sclass_t>>;

// multi-label classification
using mhinge_loss_t         = flatten_loss_t<detail::hinge_t<loss::detail::mclass_t>>;
using msavage_loss_t        = flatten_loss_t<detail::savage_t<loss::detail::mclass_t>>;
using mtangent_loss_t       = flatten_loss_t<detail::tangent_t<loss::detail::mclass_t>>;
using mlogistic_loss_t      = flatten_loss_t<detail::logistic_t<loss::detail::mclass_t>>;
using mexponential_loss_t   = flatten_loss_t<detail::exponential_t<loss::detail::mclass_t>>;
using msquared_hinge_loss_t = flatten_loss_t<detail::squared_hinge_t<loss::detail::mclass_t>>;
} // namespace nano
