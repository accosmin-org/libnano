#pragma once

#include <nano/dataset/memfixed.h>

namespace nano
{
    ///
    /// \brief image recognition/classification dataset consisting of:
    ///     - classifying fixed-size RGB or grayscale images.
    ///
    class NANO_PUBLIC imclass_dataset_t : public memfixed_dataset_t<uint8_t>
    {
    };
}
