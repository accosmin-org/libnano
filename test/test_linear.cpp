#include <utest/utest.h>
#include <nano/numeric.h>
#include <nano/linear/util.h>
#include <nano/linear/model.h>
#include <nano/linear/function.h>
#include <nano/synthetic/affine.h>
#include <nano/iterator/memfixed.h>

using namespace nano;

static auto make_fold()
{
    return fold_t{0, protocol::train};
}

static auto make_loss()
{
    auto loss = loss_t::all().get("squared");
    UTEST_REQUIRE(loss);
    return loss;
}

static auto make_solver()
{
    auto solver = solver_t::all().get("lbfgs");
    UTEST_REQUIRE(solver);
    solver->epsilon(epsilon2<scalar_t>());
    solver->max_iterations(20);
    return solver;
}

static auto make_dataset(const tensor_size_t isize = 5, const tensor_size_t tsize = 3)
{
    auto dataset = synthetic_affine_dataset_t{};
    dataset.folds(2);
    dataset.noise(epsilon1<scalar_t>());
    dataset.idim(make_dims(isize, 1, 1));
    dataset.tdim(make_dims(tsize, 1, 1));
    dataset.modulo(2);
    dataset.samples(200);
    dataset.train_percentage(80);
    UTEST_CHECK_NOTHROW(dataset.load());
    return dataset;
}

UTEST_BEGIN_MODULE(test_linear)

UTEST_CASE(predict)
{
    tensor1d_t bias(3); bias.random();
    tensor2d_t weights(5, 3); weights.random();
    tensor4d_t inputs(11, 5, 1, 1); inputs.random();

    tensor4d_t outputs;
    linear::predict(inputs, weights, bias, outputs);

    for (tensor_size_t sample = 0; sample < inputs.size<0>(); ++ sample)
    {
        UTEST_CHECK_EIGEN_CLOSE(
            outputs.vector(sample),
            weights.matrix().transpose() * inputs.vector(sample) + bias.vector(),
            epsilon1<scalar_t>());
    }
}

UTEST_CASE(evaluate)
{
    auto loss = make_loss();
    auto dataset = make_dataset();
    auto iterator = memfixed_iterator_t<scalar_t>{dataset};

    auto function = linear_function_t{*loss, iterator, make_fold()};
    UTEST_REQUIRE_EQUAL(function.size(), 5 * 3 + 3);
    UTEST_REQUIRE_EQUAL(function.isize(), 5);
    UTEST_REQUIRE_EQUAL(function.tsize(), 3);

    tensor4d_t inputs, targets;
    iterator.inputs(make_fold(), 0, iterator.samples(make_fold()), inputs);
    iterator.targets(make_fold(), 0, iterator.samples(make_fold()), targets);

    tensor4d_t outputs;
    const vector_t x = vector_t::Random(function.size());
    linear::predict(inputs, function.weights(x), function.bias(x), outputs);

    tensor1d_t values;
    loss->value(targets, outputs, values);

    UTEST_CHECK_LESS(std::fabs(function.vgrad(x) - values.vector().mean()), epsilon1<scalar_t>());
}

UTEST_CASE(gradient)
{
    auto loss = make_loss();
    auto dataset = make_dataset();
    auto iterator = memfixed_iterator_t<scalar_t>{dataset};

    auto function = linear_function_t{*loss, iterator, make_fold()};
    UTEST_REQUIRE_EQUAL(function.size(), 5 * 3 + 3);
    UTEST_REQUIRE_EQUAL(function.isize(), 5);
    UTEST_REQUIRE_EQUAL(function.tsize(), 3);
    UTEST_REQUIRE_THROW(function.l1reg(-1e+0), std::invalid_argument);
    UTEST_REQUIRE_THROW(function.l1reg(+1e+9), std::invalid_argument);
    UTEST_REQUIRE_THROW(function.l2reg(-1e+0), std::invalid_argument);
    UTEST_REQUIRE_THROW(function.l2reg(+1e+9), std::invalid_argument);
    UTEST_REQUIRE_THROW(function.vAreg(-1e+0), std::invalid_argument);
    UTEST_REQUIRE_THROW(function.vAreg(+1e+9), std::invalid_argument);
    UTEST_REQUIRE_NOTHROW(function.l1reg(1e-1));
    UTEST_REQUIRE_NOTHROW(function.l2reg(1e+1));
    UTEST_REQUIRE_NOTHROW(function.vAreg(5e-1));

    const vector_t x = vector_t::Random(function.size());
    UTEST_CHECK_LESS(function.grad_accuracy(x), 10 * epsilon2<scalar_t>());
}

UTEST_CASE(minimize)
{
    auto loss = make_loss();
    auto solver = make_solver();
    auto dataset = make_dataset();
    auto iterator = memfixed_iterator_t<scalar_t>{dataset};

    auto function = linear_function_t{*loss, iterator, make_fold()};
    UTEST_REQUIRE_EQUAL(function.size(), 5 * 3 + 3);
    UTEST_REQUIRE_NOTHROW(function.l1reg(0.0));
    UTEST_REQUIRE_NOTHROW(function.l2reg(0.0));
    UTEST_REQUIRE_NOTHROW(function.vAreg(0.0));

    solver->logger([] (const solver_state_t& state)
    {
        std::cout << "i=" << state.m_iterations << ", calls=" << state.m_fcalls << "/" << state.m_gcalls
            << ", fx=" << state.f << ", gx=" << state.convergence_criterion()
            << ", status=" << state.m_status << "\n";
        return true;
    });

    const auto state = solver->minimize(function, vector_t::Zero(function.size()));
    UTEST_CHECK(state);
    UTEST_CHECK(state.converged(solver->epsilon()));

    UTEST_CHECK_EIGEN_CLOSE(function.bias(state.x).vector(), dataset.bias(), epsilon3<scalar_t>());
    UTEST_CHECK_EIGEN_CLOSE(function.weights(state.x).matrix(), dataset.weights(), epsilon3<scalar_t>());
}

UTEST_CASE(train)
{
    auto loss = make_loss();
    auto solver = make_solver();
    auto dataset = make_dataset();
    auto iterator = memfixed_iterator_t<scalar_t>{dataset};

    auto model = linear_model_t{};
    for (const auto regularization : enum_values<linear_model_t::regularization>())
    {
        linear_model_t::train_result_t training;
        UTEST_REQUIRE_NOTHROW(training = model.train(*loss, iterator, *solver, regularization, 32, 5, 1));

        UTEST_CHECK_EIGEN_CLOSE(model.bias().vector(), dataset.bias(), epsilon3<scalar_t>());
        UTEST_CHECK_EIGEN_CLOSE(model.weights().matrix(), dataset.weights(), epsilon3<scalar_t>());

        UTEST_CHECK_EQUAL(training.size(), iterator.folds());
        for (const auto& train_fold : training)
        {
            UTEST_CHECK_GREATER_EQUAL(train_fold.m_tr_error, scalar_t(0));
            UTEST_CHECK_GREATER_EQUAL(train_fold.m_vd_error, scalar_t(0));
            UTEST_CHECK_GREATER_EQUAL(train_fold.m_te_error, scalar_t(0));

            UTEST_CHECK_LESS_EQUAL(train_fold.m_tr_error, epsilon3<scalar_t>());
            UTEST_CHECK_LESS_EQUAL(train_fold.m_vd_error, epsilon3<scalar_t>());
            UTEST_CHECK_LESS_EQUAL(train_fold.m_te_error, epsilon3<scalar_t>());
        }

        tensor4d_t inputs, outputs, targets;
        iterator.inputs(make_fold(), 0, iterator.samples(make_fold()), inputs);
        iterator.targets(make_fold(), 0, iterator.samples(make_fold()), targets);

        model.predict(inputs, outputs);
        UTEST_CHECK_EIGEN_CLOSE(targets.vector(), outputs.vector(), epsilon3<scalar_t>());

        model.predict(inputs, outputs.tensor());
        UTEST_CHECK_EIGEN_CLOSE(targets.vector(), outputs.vector(), epsilon3<scalar_t>());

        const auto filepath = string_t("test_linear.model");

        UTEST_REQUIRE_NOTHROW(model.save(filepath));
        {
            auto new_model = linear_model_t{};
            UTEST_REQUIRE_NOTHROW(new_model.load(filepath));
            UTEST_CHECK_EIGEN_CLOSE(new_model.bias().vector(), dataset.bias(), epsilon3<scalar_t>());
            UTEST_CHECK_EIGEN_CLOSE(new_model.weights().matrix(), dataset.weights(), epsilon3<scalar_t>());
        }
    }

    UTEST_CHECK_THROW(model.train(*loss, iterator, *solver, static_cast<linear_model_t::regularization>(-1)), std::runtime_error);
}

UTEST_END_MODULE()
