#pragma once

#include <nano/arch.h>
#include <nano/factory.h>
#include <nano/memfixed.h>

namespace nano
{
    class imclass_dataset_t;
    using imclass_dataset_factory_t = factory_t<imclass_dataset_t>;
    using rimclass_dataset_t = imclass_dataset_factory_t::trobject;

    ///
    /// \brief image recognition/classification dataset consisting of:
    ///     - classifying fixed-size RGB or grayscale images.
    ///
    class NANO_PUBLIC imclass_dataset_t : public memfixed_dataset_t<uint8_t>
    {
    public:

        ///
        /// \brief returns the available implementations
        ///
        static imclass_dataset_factory_t& all();
    };
}
