#pragma once

#include <nano/tabular.h>

namespace nano
{
    ///
    /// \brief Adult dataset: http://archive.ics.uci.edu/ml/datasets/Adult
    ///
    class adult_dataset_t final : public tabular_dataset_t
    {
    public:

        adult_dataset_t();
        split_t make_split() const override;
    };
}
