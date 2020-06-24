#pragma once

#include <nano/dataset/tabular.h>

namespace nano
{
    ///
    /// \brief Breast cancer dataset: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
    ///
    class breast_cancer_dataset_t final : public tabular_dataset_t
    {
    public:

        breast_cancer_dataset_t();
        [[nodiscard]] split_t make_split() const override;
    };
}
