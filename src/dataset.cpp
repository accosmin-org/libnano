#include <mutex>
#include <nano/logger.h>
#include <nano/mlearn/class.h>
#include <nano/dataset/imclass_cifar.h>
#include <nano/dataset/imclass_mnist.h>
#include <nano/dataset/tabular_iris.h>
#include <nano/dataset/tabular_wine.h>
#include <nano/dataset/tabular_adult.h>
#include <nano/dataset/tabular_abalone.h>
#include <nano/dataset/tabular_forest_fires.h>
#include <nano/dataset/tabular_breast_cancer.h>
#include <nano/dataset/tabular_bank_marketing.h>

using namespace nano;

static bool is_multi_class(const tensor3d_cmap_t& targets)
{
    return std::count_if(begin(targets), end(targets), [] (scalar_t value) { return is_pos_target(value); }) != 1;
}

task_type dataset_t::type() const
{
    const auto target = this->target();

    if (!target)
    {
        return task_type::unsupervised;
    }
    else
    {
        critical(
            target.optional(),
            scat("dataset: the target feature (", target.name(), ") cannot be optional!"));

        if (!target.discrete())
        {
            return task_type::regression;
        }
        else
        {
            // decide if single-label or multi-label
            bool multi_class = false;
            const auto batch = tensor_size_t{1024};
            for (tensor_size_t begin = 0, samples = this->samples(); begin < samples && !multi_class; begin += batch)
            {
                const auto end = std::min(begin + batch, samples);
                const auto targets = this->targets(arange(begin, end));
                for (tensor_size_t i = 0; i < targets.size<0>() && !multi_class; ++ i)
                {
                    multi_class = is_multi_class(targets.tensor(i));
                }
            }

            return multi_class ? task_type::mclassification : task_type::sclassification;
        }
    }
}

dataset_factory_t& dataset_t::all()
{
    static dataset_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        manager.add<iris_dataset_t>("iris",
            "classify flowers from physical measurements of the sepal and petal (Fisher, 1936)");
        manager.add<wine_dataset_t>("wine",
            "predict the wine type from its constituents (Aeberhard, Coomans & de Vel, 1992)");
        manager.add<adult_dataset_t>("adult",
            "predict if a person makes more than 50K per year (Kohavi & Becker, 1994)");
        manager.add<abalone_dataset_t>("abalone",
            "predict the age of abalone from physical measurements (Waugh, 1995)");
        manager.add<forest_fires_dataset_t>("forest-fires",
            "predict the burned area of the forest (Cortez & Morais, 2007)");
        manager.add<breast_cancer_dataset_t>("breast-cancer",
            "diagnostic breast cancer using measurements of cell nucleai (Street, Wolberg & Mangasarian, 1992)");
        manager.add<bank_marketing_dataset_t>("bank-marketing",
            "predict if a client has subscribed a term deposit (Moro, Laureano & Cortez, 2011)");

        manager.add<mnist_dataset_t>("mnist",
            "classify 28x28 grayscale images of hand-written digits (MNIST)");
        manager.add<cifar10_dataset_t>("cifar10",
            "classify 3x32x32 color images (CIFAR-10)");
        manager.add<cifar100c_dataset_t>("cifar100c",
            "classify 3x32x32 color images (CIFAR-100 with 20 coarse labels)");
        manager.add<cifar100f_dataset_t>("cifar100f",
            "classify 3x32x32 color images (CIFAR-100 with 100 fine labels)");
        manager.add<fashion_mnist_dataset_t>("fashion-mnist",
            "classify 28x28 grayscale images of fashion articles (Fashion-MNIST)");
    });

    return manager;
}
