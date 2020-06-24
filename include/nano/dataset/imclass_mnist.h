#pragma once

#include <nano/dataset/imclass.h>

namespace nano
{
    ///
    /// \brief common class for MNIST-like datasets.
    ///
    class NANO_PUBLIC base_mnist_dataset_t : public imclass_dataset_t
    {
    public:

        ///
        /// \brief constructor
        ///
        base_mnist_dataset_t(string_t dir, string_t name);

        ///
        /// \brief @see imclass_dataset_t
        ///
        bool load() override;

    private:

        bool iread(const string_t&, tensor_size_t offset, tensor_size_t expected);
        bool tread(const string_t&, tensor_size_t offset, tensor_size_t expected);

        // attributes
        string_t        m_dir;              ///< directory where to load the data from
        string_t        m_name;             ///<
    };

    ///
    /// MNIST dataset:
    ///      - classify digits
    ///      - 28x28 grayscale images as inputs
    ///      - 10 outputs (10 labels)
    ///
    /// http://yann.lecun.com/exdb/mnist/
    ///
    class NANO_PUBLIC mnist_dataset_t final : public base_mnist_dataset_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        mnist_dataset_t();

        ///
        /// \brief @see imclass_dataset_t
        ///
        [[nodiscard]] feature_t tfeature() const override;
    };

    ///
    /// fashion-MNIST dataset:
    ///      - classify fashion articles
    ///      - 28x28 grayscale images as inputs
    ///      - 10 outputs (10 labels)
    ///
    /// https://github.com/zalandoresearch/fashion-mnist
    ///
    class NANO_PUBLIC fashion_mnist_dataset_t final : public base_mnist_dataset_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        fashion_mnist_dataset_t();

        ///
        /// \brief @see imclass_dataset_t
        ///
        [[nodiscard]] feature_t tfeature() const override;
    };
}
