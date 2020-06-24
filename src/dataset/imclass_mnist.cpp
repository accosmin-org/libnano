#include <fstream>
#include <nano/logger.h>
#include <nano/mlearn/class.h>
#include <nano/dataset/imclass_mnist.h>

using namespace nano;

base_mnist_dataset_t::base_mnist_dataset_t(string_t dir, string_t name) :
    m_dir(std::move(dir)),
    m_name(std::move(name))
{
}

bool base_mnist_dataset_t::load()
{
    const auto parts =
    {
        std::make_tuple(
            m_dir + "/train-images-idx3-ubyte",
            m_dir + "/train-labels-idx1-ubyte",
            tensor_size_t(0), tensor_size_t(60000)),
        std::make_tuple(
            m_dir + "/t10k-images-idx3-ubyte",
            m_dir + "/t10k-labels-idx1-ubyte",
            tensor_size_t(60000), tensor_size_t(10000))
    };

    resize(make_dims(70000, 28, 28, 1), make_dims(70000, 10, 1, 1));

    tensor_size_t sample = 0;
    for (const auto& part : parts)
    {
        const auto& ifile = std::get<0>(part);
        const auto& tfile = std::get<1>(part);
        const auto offset = std::get<2>(part);
        const auto expected = std::get<3>(part);

        log_info() << m_name << ": loading file <" << ifile << ">...";
        if (!iread(ifile, offset, expected))
        {
            log_error() << m_name << ": failed to load file <" << ifile << ">!";
            return false;
        }

        log_info() << m_name << ": loading file <" << tfile << ">...";
        if (!tread(tfile, offset, expected))
        {
            log_error() << m_name << ": failed to load file <" << tfile << ">!";
            return false;
        }

        sample += expected;
        log_info() << m_name << ": loaded " << sample << " samples.";
    }

    for (size_t f = 0; f < folds(); ++ f)
    {
        split(f) = {
            nano::split2(60000, train_percentage()),
            arange(60000, 70000)
        };
        assert(split(f).valid(70000));
    }

    // OK
    return true;
}

bool base_mnist_dataset_t::iread(const string_t& path, tensor_size_t offset, tensor_size_t expected)
{
    std::ifstream stream(path);

    char buffer[16];
    if (!stream.read(buffer, 16))
    {
        return false;
    }

    for ( ; expected > 0; -- expected)
    {
        auto input = this->input(offset ++);
        if (!stream.read(reinterpret_cast<char*>(input.data()), input.size())) // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        {
            return false;
        }
    }

    return expected == 0;
}

bool base_mnist_dataset_t::tread(const string_t& path, tensor_size_t offset, tensor_size_t expected)
{
    std::ifstream stream(path);

    char buffer[8];
    if (!stream.read(buffer, 8))
    {
        return false;
    }

    tensor_vector_t<char> labels(expected);
    if (!stream.read(labels.data(), labels.size()))
    {
        return false;
    }

    for (const auto label : labels)
    {
        const auto ilabel = static_cast<tensor_size_t>(static_cast<unsigned char>(label));
        if (ilabel < 0 || ilabel >= 10)
        {
            log_error() << m_name << ": invalid label!";
            return false;
        }

        auto target = this->target(offset ++);
        target.constant(neg_target());
        target(ilabel) = pos_target();
    }

    return true;
}

mnist_dataset_t::mnist_dataset_t() :
    base_mnist_dataset_t(scat(std::getenv("HOME"), "/libnano/datasets/mnist"), "MNIST")
{
}

feature_t mnist_dataset_t::tfeature() const
{
    return  feature_t("digit").labels(
            {"digit0", "digit1", "digit2", "digit3", "digit4", "digit5", "digit6", "digit7", "digit8", "digit9"});
}

fashion_mnist_dataset_t::fashion_mnist_dataset_t() :
    base_mnist_dataset_t(scat(std::getenv("HOME"), "/libnano/datasets/fashion-mnist"), "Fashion-MNIST")
{
}

feature_t fashion_mnist_dataset_t::tfeature() const
{
    return  feature_t("article").labels(
            {"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"});
}
