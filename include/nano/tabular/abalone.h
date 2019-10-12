#pragma once

#include <nano/tabular.h>

namespace nano
{
    ///
    /// \brief Abalone dataset: https://archive.ics.uci.edu/ml/datasets/Abalone
    ///
    class abalone_dataset_t final : public tabular_dataset_t
    {
    public:

        abalone_dataset_t();
        split_t make_split() const override;
    };
}
