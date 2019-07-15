#pragma once

#include <nano/dataset/tabular.h>

namespace nano
{
    ///
    /// \brief Iris dataset: https://archive.ics.uci.edu/ml/datasets/Iris
    ///
    class iris_dataset_t final : public tabular_dataset_t
    {
    public:

        iris_dataset_t()
        {
            // todo: configure tabular dataset
        }

        json_t config() const override;
        void config(const json_t&) override;

        void split(const tensor_size_t samples, split_t&) const override;

    private:

        // attributes
        string_t        m_dir;              ///<
        size_t          m_folds{10};        ///<
        tensor_size_t   m_train_per{80};    ///< percentage ot samples used for training (without the test samples)

        auto samples() const { return m_inputs.size<0>(); }

        bool split();

            static string_t name() { return "IRIS"; }
            static string_t home() { return string_t(std::getenv("HOME")); }
            static string_t path() { return home() + "/experiments/databases/iris/iris.data"; }
            static indices_t target_columns() { return {size_t(4)}; }
    };
    };
}
