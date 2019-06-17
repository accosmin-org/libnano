#pragma once

#include "loss.h"
#include "cortex.h"
#include <cassert>

namespace nano
{
        ///
        /// \brief multi-class classification loss that predicts the labels with positive output.
        ///
        template <typename top>
        class mclassification_t final : public loss_t
        {
        public:
                scalar_t error(const tensor3d_cmap_t& target, const tensor3d_cmap_t& output) const override
                {
                        assert(target.dims() == output.dims());

                        const auto edges = target.array() * output.array();
                        const auto epsilon = std::numeric_limits<scalar_t>::epsilon();
                        return static_cast<scalar_t>((edges < epsilon).count());
                }

                scalar_t value(const tensor3d_cmap_t& target, const tensor3d_cmap_t& output) const override
                {
                        assert(target.dims() == output.dims());

                        return top::value(target.array(), output.array());
                }

                tensor3d_t vgrad(const tensor3d_cmap_t& target, const tensor3d_cmap_t& output) const override
                {
                        assert(target.dims() == output.dims());

                        tensor3d_t vgrad(target.dims());
                        vgrad.array() = top::vgrad(target.array(), output.array());
                        return vgrad;
                }
        };

        ///
        /// \brief single-class classification loss that predicts the label with the highest score.
        ///
        template <typename top>
        class sclassification_t final : public loss_t
        {
        public:
                scalar_t error(const tensor3d_cmap_t& target, const tensor3d_cmap_t& output) const override
                {
                        assert(target.dims() == output.dims());

                        if (target.size() > 1)
                        {
                                tensor_size_t idx;
                                output.array().maxCoeff(&idx);

                                return is_pos_target(target(idx)) ? 0 : 1;
                        }
                        else
                        {
                                const auto edges = target.array() * output.array();
                                const auto epsilon = std::numeric_limits<scalar_t>::epsilon();
                                return static_cast<scalar_t>((edges < epsilon).count());
                        }
                }

                scalar_t value(const tensor3d_cmap_t& target, const tensor3d_cmap_t& output) const override
                {
                        assert(target.dims() == output.dims());

                        return top::value(target.array(), output.array());
                }

                tensor3d_t vgrad(const tensor3d_cmap_t& target, const tensor3d_cmap_t& output) const override
                {
                        assert(target.dims() == output.dims());

                        tensor3d_t vgrad(target.dims());
                        vgrad.array() = top::vgrad(target.array(), output.array());
                        return vgrad;
                }
        };
}
