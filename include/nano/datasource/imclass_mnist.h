#pragma once

#include <nano/dataset.h>

namespace nano
{
///
/// \brief common class for MNIST-like datasets.
///
class NANO_PUBLIC base_mnist_datasource_t : public datasource_t
{
public:
    ///
    /// \brief constructor
    ///
    base_mnist_datasource_t(string_t id, string_t dir, feature_t target);

private:
    void do_load() override;
    bool iread(const string_t&, tensor_size_t sample, tensor_size_t expected);
    bool tread(const string_t&, tensor_size_t sample, tensor_size_t expected);

    features_t make_features() const;
    string_t   make_full_path(const string_t& path) const;

    // attributes
    string_t  m_dir;    ///< directory where to load the data from
    feature_t m_target; ///< target feature
};

///
/// MNIST dataset:
///      - classify digits
///      - 28x28 grayscale images as inputs
///      - 10 outputs (10 labels)
///
/// http://yann.lecun.com/exdb/mnist/
///
class NANO_PUBLIC mnist_datasource_t final : public base_mnist_datasource_t
{
public:
    ///
    /// \brief default constructor
    ///
    mnist_datasource_t();

    ///
    /// \brief @see clonable_t
    ///
    rdatasource_t clone() const override;
};

///
/// fashion-MNIST dataset:
///      - classify fashion articles
///      - 28x28 grayscale images as inputs
///      - 10 outputs (10 labels)
///
/// https://github.com/zalandoresearch/fashion-mnist
///
class NANO_PUBLIC fashion_mnist_datasource_t final : public base_mnist_datasource_t
{
public:
    ///
    /// \brief default constructor
    ///
    fashion_mnist_datasource_t();

    ///
    /// \brief @see clonable_t
    ///
    rdatasource_t clone() const override;
};
} // namespace nano
