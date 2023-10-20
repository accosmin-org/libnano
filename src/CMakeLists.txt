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

# generic utilities
make_lib(NANO core
    parameter.cpp
    configurable.cpp
    core/table.cpp
    core/chrono.cpp
    core/logger.cpp
    core/random.cpp
    core/cmdline.cpp
    core/parallel.cpp
    core/sampling.cpp
    core/histogram.cpp
    core/parameter_tracker.cpp)

# numerical optimization utilities
make_lib(NANO solver
    function.cpp
    function/constraint.cpp
    function/penalty.cpp
    function/util.cpp
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
    solver/sgm.cpp
    solver/asga.cpp
    solver/osga.cpp
    solver/cocob.cpp
    solver/lstep.cpp
    solver/lbfgs.cpp
    solver/pdsgm.cpp
    solver/quasi.cpp
    solver/state.cpp
    solver/penalty.cpp
    solver/augmented.cpp
    solver/ellipsoid.cpp
    solver/universal.cpp
    program/solver.cpp
    program/linear.cpp
    program/quadratic.cpp
    program/constraint.cpp
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
    function/benchmark/zakharov.cpp)

# machine learning utilities
make_lib(NANO machine
    feature.cpp
    dataset.cpp
    dataset/hash.cpp
    dataset/stats.cpp
    dataset/iterator.cpp
    datasource.cpp
    datasource/mask.cpp
    datasource/linear.cpp
    datasource/tabular.cpp
    datasource/imclass_cifar.cpp
    datasource/imclass_mnist.cpp
    generator.cpp
    generator/pairwise_base.cpp
    generator/pairwise_input.cpp
    generator/pairwise_product.cpp
    generator/elemwise_base.cpp
    generator/elemwise_input.cpp
    generator/elemwise_gradient.cpp
    generator/elemwise_identity.cpp
    loss.cpp
    loss/pinball.cpp
    learner.cpp
    mlearn/tune.cpp
    mlearn/stats.cpp
    mlearn/params.cpp
    mlearn/result.cpp
    mlearn/cluster.cpp
    tuner.cpp
    tuner/util.cpp
    tuner/local.cpp
    tuner/space.cpp
    tuner/surrogate.cpp
    splitter.cpp
    splitter/kfold.cpp
    splitter/random.cpp
    gboost/util.cpp
    gboost/model.cpp
    gboost/result.cpp
    gboost/sampler.cpp
    gboost/function.cpp
    gboost/accumulator.cpp
    gboost/early_stopping.cpp
    wlearner.cpp
    wlearner/util.cpp
    wlearner/single.cpp
    wlearner/affine.cpp
    wlearner/dtree.cpp
    wlearner/hinge.cpp
    wlearner/stump.cpp
    wlearner/table.cpp
    wlearner/criterion.cpp
    wlearner/accumulator.cpp
    linear/util.cpp
    linear/model.cpp
    linear/result.cpp
    linear/function.cpp
    linear/accumulator.cpp)

# link libraries
target_link_libraries(solver PUBLIC core)
target_link_libraries(machine PUBLIC solver)
