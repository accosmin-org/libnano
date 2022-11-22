# create library

if(NOT APPLE)
    set(CMAKE_INSTALL_RPATH $ORIGIN)
endif()

add_library(nano)

target_sources(nano PRIVATE
    core/table.cpp
    core/chrono.cpp
    core/logger.cpp
    core/random.cpp
    core/cmdline.cpp
    core/parallel.cpp
    core/histogram.cpp
    core/estimator.cpp
    core/parameter.cpp
    generator.cpp
    generator/pairwise_base.cpp
    generator/pairwise_input.cpp
    generator/elemwise_base.cpp
    generator/elemwise_input.cpp
    generator/elemwise_gradient.cpp
    generator/elemwise_identity.cpp
    dataset.cpp
    dataset/iterator.cpp
    datasource.cpp
    datasource/mask.cpp
    datasource/stats.cpp
    datasource/feature.cpp
    datasource/tabular.cpp
    datasource/imclass_cifar.cpp
    datasource/imclass_mnist.cpp
    lsearch0.cpp
    lsearch0/linear.cpp
    lsearch0/constant.cpp
    lsearch0/cgdescent.cpp
    lsearch0/quadratic.cpp
    lsearchk.cpp
    lsearchk/fletcher.cpp
    lsearchk/backtrack.cpp
    lsearchk/cgdescent.cpp
    lsearchk/lemarechal.cpp
    lsearchk/morethuente.cpp
    solver.cpp
    solver/gd.cpp
    solver/cgd.cpp
    solver/osga.cpp
    solver/lstep.cpp
    solver/lbfgs.cpp
    solver/quasi.cpp
    solver/state.cpp
    solver/penalty.cpp
    solver/augmented.cpp
    solver/ellipsoid.cpp
    function.cpp
    function/constraint.cpp
    function/penalty.cpp
    function/util.cpp
    function/benchmark/axis_ellipsoid.cpp
    function/benchmark/cauchy.cpp
    function/benchmark/chung_reynolds.cpp
    function/benchmark/dixon_price.cpp
    function/benchmark/elastic_net.cpp
    function/benchmark/exponential.cpp
    function/benchmark/geometric.cpp
    function/benchmark/kinks.cpp
    function/benchmark/linear.cpp
    function/benchmark/powell.cpp
    function/benchmark/qing.cpp
    function/benchmark/quadratic.cpp
    function/benchmark/rosenbrock.cpp
    function/benchmark/rotated_ellipsoid.cpp
    function/benchmark/sargan.cpp
    function/benchmark/schumer_steiglitz.cpp
    function/benchmark/sphere.cpp
    function/benchmark/styblinski_tang.cpp
    function/benchmark/trid.cpp
    function/benchmark/zakharov.cpp
    loss.cpp
    learner.cpp
    model.cpp
    model/util.cpp
    model/cluster.cpp
    tuner.cpp
    tuner/step.cpp
    tuner/util.cpp
    tuner/local.cpp
    tuner/space.cpp
    tuner/surrogate.cpp
    splitter.cpp
    splitter/kfold.cpp
    splitter/random.cpp
    #gboost/model.cpp
    gboost/function.cpp
    wlearner.cpp
    wlearner/util.cpp
    wlearner/single.cpp
    wlearner/affine.cpp
    wlearner/dstep.cpp
    wlearner/dtree.cpp
    wlearner/hinge.cpp
    wlearner/stump.cpp
    wlearner/table.cpp
    wlearner/mhash.cpp
    linear/util.cpp
    linear/cache.cpp
    linear/model.cpp
    linear/function.cpp
)

add_library(NANO::nano ALIAS nano)
target_compile_configure(nano)

target_include_directories(nano
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(nano
    PUBLIC Eigen3::Eigen)

if(NOT WIN32)
    target_link_libraries(nano
        PUBLIC Threads::Threads
        PRIVATE Threads::Threads)
endif()

install(TARGETS nano
    EXPORT nano-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

set_target_properties(nano PROPERTIES EXPORT_NAME NANO)
