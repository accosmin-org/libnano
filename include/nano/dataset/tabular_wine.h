#pragma once

#include <nano/dataset/tabular.h>

namespace nano
{
    ///
    /// \brief Wine dataset: https://archive.ics.uci.edu/ml/datasets/Wine
    ///
    class wine_dataset_t final : public tabular_dataset_t
    {
    public:

        wine_dataset_t();
    };
}
