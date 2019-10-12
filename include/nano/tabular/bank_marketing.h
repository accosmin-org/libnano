#pragma once

#include <nano/tabular.h>

namespace nano
{
    ///
    /// \brief Bank Marketing dataset: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
    ///
    class bank_marketing_dataset_t final : public tabular_dataset_t
    {
    public:

        bank_marketing_dataset_t();
        split_t make_split() const override;
    };
}
