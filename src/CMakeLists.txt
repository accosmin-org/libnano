# create library

if(NOT APPLE)
    set(CMAKE_INSTALL_RPATH $ORIGIN)
endif()

add_library(nano)

target_sources(nano PRIVATE
    core/table.cpp
    core/tpool.cpp
    core/chrono.cpp
    core/logger.cpp
    core/cmdline.cpp
    core/histogram.cpp
    core/estimator.cpp
    core/parameter.cpp
    generator.cpp
    generator/iterator.cpp
    generator/generator.cpp
    generator/pairwise_base.cpp
    generator/pairwise_input.cpp
    generator/elemwise_base.cpp
    generator/elemwise_input.cpp
    generator/elemwise_gradient.cpp
    generator/elemwise_identity.cpp
    dataset.cpp
    dataset/mask.cpp
    dataset/stats.cpp
    dataset/feature.cpp
    dataset/tabular.cpp
    dataset/imclass_cifar.cpp
    dataset/imclass_mnist.cpp
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
    solver/lbfgs.cpp
    solver/quasi.cpp
    solver/universal.cpp
    function.cpp
    function/axis_ellipsoid.cpp
    function/benchmark.cpp
    function/cauchy.cpp
    function/chung_reynolds.cpp
    function/dixon_price.cpp
    function/elastic_net.cpp
    function/exponential.cpp
    function/geometric.cpp
    function/kinks.cpp
    function/linear.cpp
    function/powell.cpp
    function/qing.cpp
    function/quadratic.cpp
    function/rosenbrock.cpp
    function/rotated_ellipsoid.cpp
    function/sargan.cpp
    function/schumer_steiglitz.cpp
    function/sphere.cpp
    function/styblinski_tang.cpp
    function/trid.cpp
    function/zakharov.cpp
    loss.cpp
    model.cpp
    model/util.cpp
    model/kfold.cpp
    model/tuner.cpp
    model/cluster.cpp
    model/surrogate.cpp
    model/param_space.cpp
    #gboost/model.cpp
    #gboost/function.cpp
    #gboost/wlearner.cpp
    #gboost/wlearner_dstep.cpp
    #gboost/wlearner_dtree.cpp
    #gboost/wlearner_hinge.cpp
    #gboost/wlearner_stump.cpp
    #gboost/wlearner_table.cpp
    #gboost/wlearner_affine.cpp
    #gboost/wlearner_feature1.cpp
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

# install library
install(TARGETS nano
    EXPORT nano-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

set_target_properties(nano PROPERTIES EXPORT_NAME NANO)
