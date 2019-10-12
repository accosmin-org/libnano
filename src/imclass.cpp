#include <mutex>
#include <nano/imclass/cifar.h>
#include <nano/imclass/mnist.h>

using namespace nano;

imclass_dataset_factory_t& imclass_dataset_t::all()
{
    static imclass_dataset_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
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
