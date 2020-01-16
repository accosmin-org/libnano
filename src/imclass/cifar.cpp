#include <fstream>
#include <nano/logger.h>
#include <nano/mlearn/class.h>
#include <nano/imclass/cifar.h>

using namespace nano;

cifar_dataset_t::cifar_dataset_t(string_t dir, string_t name) :
    m_dir(std::move(dir)),
    m_name(std::move(name))
{
}

void cifar_dataset_t::labels(tensor_size_t labels)
{
    m_labels = labels;
}

void cifar_dataset_t::file(string_t name,
    tensor_size_t offset, tensor_size_t expected, tensor_size_t label_size, tensor_size_t label_index)
{
    m_files.emplace_back(std::move(name), offset, expected, label_size, label_index);
}

bool cifar_dataset_t::load()
{
    const auto tfeature = this->tfeature();
    if (!tfeature.discrete() || tfeature.labels().size() != static_cast<size_t>(m_labels))
    {
        log_error() << m_name << ": invalid target features!";
        return false;
    }

    resize(make_dims(60000, 32, 32, 3), make_dims(60000, m_labels, 1, 1));

    tensor_size_t sample = 0;
    for (const auto& file : m_files)
    {
        log_info() << m_name << ": loading file <" << (m_dir + file.m_filename) << "> ...";
        if (!iread(file))
        {
            log_error() << m_name << ": failed to load file <" << (m_dir + file.m_filename) << ">!";
            return false;
        }

        sample += file.m_expected;
        log_info() << m_name << ": loaded " << sample << " samples.";
    }

    for (size_t f = 0; f < folds(); ++ f)
    {
        split(f) = {
            nano::split2(50000, train_percentage()),
            indices_t::LinSpaced(10000, 50000, 60000)
        };
        assert(split(f).valid(60000));
    }

    // OK
    return true;
}

bool cifar_dataset_t::iread(const file_t& file)
{
    std::ifstream stream(m_dir + file.m_filename);

    tensor_mem_t<int8_t, 1> label(file.m_label_size);
    tensor_mem_t<uint8_t, 3> image(3, 32, 32);

    auto offset = file.m_offset;
    auto expected = file.m_expected;
    for ( ; expected > 0; -- expected)
    {
        auto&& input = this->input(offset);
        auto&& target = this->target(offset ++);

        if (!stream.read(reinterpret_cast<char*>(label.data()), label.size())) // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        {
            return false;
        }

        const tensor_size_t ilabel = label(file.m_label_index);
        if (ilabel < 0 || ilabel >= m_labels)
        {
            log_error() << m_name << ": invalid label, read " << ilabel << " expected in [0, " << m_labels << ")!";
            return false;
        }
        target.constant(neg_target());
        target(ilabel) = pos_target();

        if (!stream.read(reinterpret_cast<char*>(image.data()), image.size())) // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        {
            return false;
        }

        for (int px = 0; px < 1024; ++ px)
        {
            input.data()[px * 3 + 0] = image.data()[px + 0];
            input.data()[px * 3 + 1] = image.data()[px + 1024];
            input.data()[px * 3 + 2] = image.data()[px + 2048];
        }
    }

    return expected == 0;
}

cifar10_dataset_t::cifar10_dataset_t() :
    cifar_dataset_t(scat(std::getenv("HOME"), "/libnano/datasets/cifar10/"), "CIFAR-10")
{
    labels(10);
    file("cifar-10-batches-bin/data_batch_1.bin", 0, 10000, 1, 0);
    file("cifar-10-batches-bin/data_batch_2.bin", 10000, 10000, 1, 0);
    file("cifar-10-batches-bin/data_batch_3.bin", 20000, 10000, 1, 0);
    file("cifar-10-batches-bin/data_batch_4.bin", 30000, 10000, 1, 0);
    file("cifar-10-batches-bin/data_batch_5.bin", 40000, 10000, 1, 0);
    file("cifar-10-batches-bin/test_batch.bin", 50000, 10000, 1, 0);
}

feature_t cifar10_dataset_t::tfeature() const
{
    return  feature_t("class").labels(
           {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"});
}

cifar100c_dataset_t::cifar100c_dataset_t() :
    cifar_dataset_t(scat(std::getenv("HOME"), "/libnano/datasets/cifar100/"), "CIFAR-100")
{
    labels(20);
    file("cifar-100-binary/train.bin", 0, 50000, 2, 0);
    file("cifar-100-binary/test.bin", 50000, 10000, 2, 0);
}

feature_t cifar100c_dataset_t::tfeature() const
{
    return  feature_t("class").labels(
            {"aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables",
            "household electrical devices", "household furniture", "insects", "large carnivores",
            "large man-made outdoor things", "large natural outdoor scenes", "large omnivores and herbivores",
            "medium-sized mammals", "non-insect invertebrates", "people", "reptiles", "small mammals", "trees",
            "vehicles 1", "vehicles 2"});
}

cifar100f_dataset_t::cifar100f_dataset_t() :
    cifar_dataset_t(scat(std::getenv("HOME"), "/libnano/datasets/cifar100/"), "CIFAR-100")
{
    labels(100);
    file("cifar-100-binary/train.bin", 0, 50000, 2, 1);
    file("cifar-100-binary/test.bin", 50000, 10000, 2, 1);
}

feature_t cifar100f_dataset_t::tfeature() const
{
    return  feature_t("class").labels(
            {"apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
            "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
            "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup",
            "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house",
            "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man",
            "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid",
            "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
            "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew",
            "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper",
            "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle",
            "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"});
}
