#pragma once

#include <nano/arch.h>
#include <nano/tensor.h>
#include <nano/factory.h>

namespace nano
{
    class loss_t;
    using loss_factory_t = factory_t<loss_t>;
    using rloss_t = loss_factory_t::trobject;

    ///
    /// \brief generic multivariate loss function of two parameters:
    ///     - the target value to predict (ground truth, annotation) and
    ///     - the model's output (prediction).
    ///
    /// the loss function upper-bounds or approximates the true (usually non-smooth) error function to minimize.
    ///
    class NANO_PUBLIC loss_t
    {
    public:

        ///
        /// \brief returns the available implementations
        ///
        static loss_factory_t& all();

        ///
        /// \brief enable copying
        ///
        loss_t() = default;
        loss_t(const loss_t&) = default;
        loss_t& operator=(const loss_t&) = default;

        ///
        /// \brief enable moving
        ///
        loss_t(loss_t&&) noexcept = default;
        loss_t& operator=(loss_t&&) noexcept = default;

        ///
        /// \brief destructor
        ///
        virtual ~loss_t() = default;

        ///
        /// \brief compute the error value
        ///
        virtual scalar_t error(const tensor3d_cmap_t& target, const tensor3d_cmap_t& output) const = 0;

        ///
        /// \brief compute the loss value (an upper bound of the usually non-continuous error function)
        ///
        virtual scalar_t value(const tensor3d_cmap_t& target, const tensor3d_cmap_t& output) const = 0;

        ///
        /// \brief compute the loss gradient (wrt the output)
        ///
        virtual void vgrad(const tensor3d_cmap_t& target, const tensor3d_cmap_t& output, const tensor3d_map_t&) const = 0;
    };
}
