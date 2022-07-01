#pragma once

#include <nano/arch.h>
#include <nano/core/factory.h>
#include <nano/tensor.h>

namespace nano
{
    class loss_t;
    using loss_factory_t = factory_t<loss_t>;
    using rloss_t        = loss_factory_t::trobject;

    ///
    /// \brief generic multivariate loss function of two parameters:
    ///     - the target value to predict (ground truth, annotation) and
    ///     - the machine learning model's output (prediction).
    ///
    /// NB: usually the loss function upper-bounds or
    ///     approximates the true (usually non-smooth) error function to minimize.
    ///
    class NANO_PUBLIC loss_t
    {
    public:
        ///
        /// \brief returns the available implementations
        ///
        static loss_factory_t& all();

        ///
        /// \brief default constructor
        ///
        loss_t();

        ///
        /// \brief enable copying
        ///
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
        /// \brief compute the error value, the loss value and the loss' gradient wrt the output for the given samples
        ///
        /// NB: the targets and the outputs are given as 4D tensors,
        ///     where the first index is the sample index
        ///
        virtual void error(const tensor4d_cmap_t& targets, const tensor4d_cmap_t& outputs, tensor1d_map_t) const = 0;
        virtual void value(const tensor4d_cmap_t& targets, const tensor4d_cmap_t& outputs, tensor1d_map_t) const = 0;
        virtual void vgrad(const tensor4d_cmap_t& targets, const tensor4d_cmap_t& outputs, tensor4d_map_t) const = 0;

        ///
        /// \brief overloads to simplify usage.
        ///
        /// NB: the output tensors are allocated accordingly.
        ///
        void error(const tensor4d_cmap_t& targets, const tensor4d_cmap_t& outputs, tensor1d_t& errors) const
        {
            errors.resize(targets.size<0>());
            error(targets, outputs, errors.tensor());
        }

        void value(const tensor4d_cmap_t& targets, const tensor4d_cmap_t& outputs, tensor1d_t& values) const
        {
            values.resize(targets.size<0>());
            value(targets, outputs, values.tensor());
        }

        void vgrad(const tensor4d_cmap_t& targets, const tensor4d_cmap_t& outputs, tensor4d_t& vgrads) const
        {
            vgrads.resize(targets.dims());
            vgrad(targets, outputs, vgrads.tensor());
        }

        ///
        /// \brief returns whether the loss function is convex.
        ///
        bool convex() const { return m_convex; }

        ///
        /// \brief returns whether the loss function is smooth.
        ///
        /// NB: if not, then only sub-gradients are available.
        ///
        bool smooth() const { return m_smooth; }

    protected:
        void convex(bool);
        void smooth(bool);

    private:
        // attributes
        bool m_convex{false}; ///< whether the loss function is convex
        bool m_smooth{false}; ///< whether the loss function is smooth (otherwise subgradients should be used)
    };
} // namespace nano
