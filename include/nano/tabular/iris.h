#pragma once

#include <nano/tabular.h>

namespace nano
{
    ///
    /// \brief Iris dataset: https://archive.ics.uci.edu/ml/datasets/Iris
    ///
    class iris_dataset_t final : public tabular_dataset_t
    {
    public:

        iris_dataset_t();
        split_t make_split() const override;
    };
}
