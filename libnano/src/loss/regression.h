#pragma once

#include "loss.h"
#include <cassert>

namespace nano
{
        ///
        /// \brief generic (multivariate) regression loss that upper-bounds
        ///     the L1-distance between target and output.
        ///
        template <typename top>
        class regression_t final : public loss_t
        {
        public:
                scalar_t error(const tensor3d_cmap_t& target, const tensor3d_cmap_t& output) const override
                {
                        assert(target.dims() == output.dims());

                        return (target.array() - output.array()).abs().sum();
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
