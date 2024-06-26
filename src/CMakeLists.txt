if(APPLE)
    set(base @loader_path)
else()
    set(base $ORIGIN)
endif()

file(RELATIVE_PATH relDir
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
)
set(CMAKE_INSTALL_RPATH ${base} ${base}/${relDir})

add_subdirectory(function)
add_subdirectory(program)
add_subdirectory(solver)

# generic utilities
make_lib(NANO core
    ${CMAKE_SOURCE_DIR}/include/nano/arch.h
    ${CMAKE_SOURCE_DIR}/include/nano/clonable.h
    ${CMAKE_SOURCE_DIR}/include/nano/configurable.h
    ${CMAKE_SOURCE_DIR}/include/nano/critical.h
    ${CMAKE_SOURCE_DIR}/include/nano/core/chrono.h
    ${CMAKE_SOURCE_DIR}/include/nano/core/cmdline.h
    ${CMAKE_SOURCE_DIR}/include/nano/core/combinatorial.h
    ${CMAKE_SOURCE_DIR}/include/nano/core/factory_util.h
    ${CMAKE_SOURCE_DIR}/include/nano/core/hash.h
    ${CMAKE_SOURCE_DIR}/include/nano/core/histogram.h
    ${CMAKE_SOURCE_DIR}/include/nano/core/numeric.h
    ${CMAKE_SOURCE_DIR}/include/nano/core/overloaded.h
    ${CMAKE_SOURCE_DIR}/include/nano/core/parallel.h
    ${CMAKE_SOURCE_DIR}/include/nano/core/random.h
    ${CMAKE_SOURCE_DIR}/include/nano/core/reduce.h
    ${CMAKE_SOURCE_DIR}/include/nano/core/sampling.h
    ${CMAKE_SOURCE_DIR}/include/nano/core/scat.h
    ${CMAKE_SOURCE_DIR}/include/nano/core/seed.h
    ${CMAKE_SOURCE_DIR}/include/nano/core/stats.h
    ${CMAKE_SOURCE_DIR}/include/nano/core/stream.h
    ${CMAKE_SOURCE_DIR}/include/nano/core/strutil.h
    ${CMAKE_SOURCE_DIR}/include/nano/core/table.h
    ${CMAKE_SOURCE_DIR}/include/nano/core/tokenizer.h
    ${CMAKE_SOURCE_DIR}/include/nano/enum.h
    ${CMAKE_SOURCE_DIR}/include/nano/factory.h
    ${CMAKE_SOURCE_DIR}/include/nano/loggable.h
    ${CMAKE_SOURCE_DIR}/include/nano/logger.h
    ${CMAKE_SOURCE_DIR}/include/nano/main.h
    ${CMAKE_SOURCE_DIR}/include/nano/parameter.h
    ${CMAKE_SOURCE_DIR}/include/nano/scalar.h
    ${CMAKE_SOURCE_DIR}/include/nano/string.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/algorithm.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/base.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/dims.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/eigen.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/index.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/integral.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/numeric.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/pprint.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/range.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/stack.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/storage.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/stream.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/tensor.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/traits.h
    ${CMAKE_SOURCE_DIR}/include/nano/typed.h

    logger.cpp
    configurable.cpp
    parameter.cpp
    core/chrono.cpp
    core/cmdline.cpp
    core/histogram.cpp
    core/parallel.cpp
    core/random.cpp
    core/sampling.cpp
    core/strutil.cpp
    core/table.cpp
)

# machine learning utilities
make_lib(NANO machine
    ${CMAKE_SOURCE_DIR}/include/nano/dataset.h
    ${CMAKE_SOURCE_DIR}/include/nano/dataset/hash.h
    ${CMAKE_SOURCE_DIR}/include/nano/dataset/iterator.h
    ${CMAKE_SOURCE_DIR}/include/nano/dataset/scaling.h
    ${CMAKE_SOURCE_DIR}/include/nano/dataset/stats.h
    ${CMAKE_SOURCE_DIR}/include/nano/datasource.h
    ${CMAKE_SOURCE_DIR}/include/nano/datasource/csv.h
    ${CMAKE_SOURCE_DIR}/include/nano/datasource/imclass_cifar.h
    ${CMAKE_SOURCE_DIR}/include/nano/datasource/imclass_mnist.h
    ${CMAKE_SOURCE_DIR}/include/nano/datasource/iterator.h
    ${CMAKE_SOURCE_DIR}/include/nano/datasource/linear.h
    ${CMAKE_SOURCE_DIR}/include/nano/datasource/mask.h
    ${CMAKE_SOURCE_DIR}/include/nano/datasource/storage.h
    ${CMAKE_SOURCE_DIR}/include/nano/datasource/tabular.h
    ${CMAKE_SOURCE_DIR}/include/nano/feature.h
    ${CMAKE_SOURCE_DIR}/include/nano/gboost/accumulator.h
    ${CMAKE_SOURCE_DIR}/include/nano/gboost/early_stopping.h
    ${CMAKE_SOURCE_DIR}/include/nano/gboost/enums.h
    ${CMAKE_SOURCE_DIR}/include/nano/gboost/function.h
    ${CMAKE_SOURCE_DIR}/include/nano/gboost/model.h
    ${CMAKE_SOURCE_DIR}/include/nano/gboost/result.h
    ${CMAKE_SOURCE_DIR}/include/nano/gboost/sampler.h
    ${CMAKE_SOURCE_DIR}/include/nano/gboost/util.h
    ${CMAKE_SOURCE_DIR}/include/nano/generator.h
    ${CMAKE_SOURCE_DIR}/include/nano/generator/elemwise_base.h
    ${CMAKE_SOURCE_DIR}/include/nano/generator/elemwise_gradient.h
    ${CMAKE_SOURCE_DIR}/include/nano/generator/elemwise.h
    ${CMAKE_SOURCE_DIR}/include/nano/generator/elemwise_identity.h
    ${CMAKE_SOURCE_DIR}/include/nano/generator/elemwise_input.h
    ${CMAKE_SOURCE_DIR}/include/nano/generator/gradient.h
    ${CMAKE_SOURCE_DIR}/include/nano/generator/pairwise_base.h
    ${CMAKE_SOURCE_DIR}/include/nano/generator/pairwise.h
    ${CMAKE_SOURCE_DIR}/include/nano/generator/pairwise_input.h
    ${CMAKE_SOURCE_DIR}/include/nano/generator/pairwise_product.h
    ${CMAKE_SOURCE_DIR}/include/nano/generator/select.h
    ${CMAKE_SOURCE_DIR}/include/nano/generator/storage.h
    ${CMAKE_SOURCE_DIR}/include/nano/learner.h
    ${CMAKE_SOURCE_DIR}/include/nano/linear/accumulator.h
    ${CMAKE_SOURCE_DIR}/include/nano/linear/enums.h
    ${CMAKE_SOURCE_DIR}/include/nano/linear/function.h
    ${CMAKE_SOURCE_DIR}/include/nano/linear/model.h
    ${CMAKE_SOURCE_DIR}/include/nano/linear/result.h
    ${CMAKE_SOURCE_DIR}/include/nano/linear/util.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss/class.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss/error.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss/flatten.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss/pinball.h
    ${CMAKE_SOURCE_DIR}/include/nano/mlearn/cluster.h
    ${CMAKE_SOURCE_DIR}/include/nano/mlearn/enums.h
    ${CMAKE_SOURCE_DIR}/include/nano/mlearn/params.h
    ${CMAKE_SOURCE_DIR}/include/nano/mlearn/result.h
    ${CMAKE_SOURCE_DIR}/include/nano/mlearn/stats.h
    ${CMAKE_SOURCE_DIR}/include/nano/mlearn/tune.h
    ${CMAKE_SOURCE_DIR}/include/nano/splitter.h
    ${CMAKE_SOURCE_DIR}/include/nano/splitter/kfold.h
    ${CMAKE_SOURCE_DIR}/include/nano/splitter/random.h
    ${CMAKE_SOURCE_DIR}/include/nano/task.h
    ${CMAKE_SOURCE_DIR}/include/nano/tuner.h
    ${CMAKE_SOURCE_DIR}/include/nano/tuner/local.h
    ${CMAKE_SOURCE_DIR}/include/nano/tuner/space.h
    ${CMAKE_SOURCE_DIR}/include/nano/tuner/step.h
    ${CMAKE_SOURCE_DIR}/include/nano/tuner/callback.h
    ${CMAKE_SOURCE_DIR}/include/nano/tuner/surrogate.h
    ${CMAKE_SOURCE_DIR}/include/nano/tuner/util.h
    ${CMAKE_SOURCE_DIR}/include/nano/wlearner.h
    ${CMAKE_SOURCE_DIR}/include/nano/wlearner/accumulator.h
    ${CMAKE_SOURCE_DIR}/include/nano/wlearner/affine.h
    ${CMAKE_SOURCE_DIR}/include/nano/wlearner/criterion.h
    ${CMAKE_SOURCE_DIR}/include/nano/wlearner/dtree.h
    ${CMAKE_SOURCE_DIR}/include/nano/wlearner/hinge.h
    ${CMAKE_SOURCE_DIR}/include/nano/wlearner/single.h
    ${CMAKE_SOURCE_DIR}/include/nano/wlearner/stump.h
    ${CMAKE_SOURCE_DIR}/include/nano/wlearner/table.h
    ${CMAKE_SOURCE_DIR}/include/nano/wlearner/util.h

    dataset.cpp
    dataset/hash.cpp
    dataset/iterator.cpp
    dataset/stats.cpp
    datasource.cpp
    datasource/imclass_cifar.cpp
    datasource/imclass_mnist.cpp
    datasource/linear.cpp
    datasource/mask.cpp
    datasource/tabular.cpp
    feature.cpp
    gboost/accumulator.cpp
    gboost/early_stopping.cpp
    gboost/function.cpp
    gboost/model.cpp
    gboost/result.cpp
    gboost/sampler.cpp
    gboost/util.cpp
    generator.cpp
    generator/elemwise_base.cpp
    generator/elemwise_gradient.cpp
    generator/elemwise_identity.cpp
    generator/elemwise_input.cpp
    generator/pairwise_base.cpp
    generator/pairwise_input.cpp
    generator/pairwise_product.cpp
    learner.cpp
    linear/accumulator.cpp
    linear/function.cpp
    linear/model.cpp
    linear/result.cpp
    linear/util.cpp
    loss.cpp
    loss/pinball.cpp
    mlearn/cluster.cpp
    mlearn/params.cpp
    mlearn/result.cpp
    mlearn/stats.cpp
    mlearn/tune.cpp
    splitter.cpp
    splitter/kfold.cpp
    splitter/random.cpp
    tuner.cpp
    tuner/local.cpp
    tuner/space.cpp
    tuner/surrogate.cpp
    tuner/util.cpp
    wlearner.cpp
    wlearner/accumulator.cpp
    wlearner/affine.cpp
    wlearner/criterion.cpp
    wlearner/dtree.cpp
    wlearner/hinge.cpp
    wlearner/single.cpp
    wlearner/stump.cpp
    wlearner/table.cpp
    wlearner/util.cpp
)

# link libraries
target_link_libraries(core
    PUBLIC Eigen3::Eigen)

if(NOT WIN32)
    target_link_libraries(core
        PUBLIC Threads::Threads
        PRIVATE Threads::Threads)
endif()

target_link_libraries(function PUBLIC core)
target_link_libraries(program PUBLIC core)
target_link_libraries(solver PUBLIC function program)
target_link_libraries(machine PUBLIC solver)
