#include <utest/utest.h>
#include <nano/solver.h>
#include <nano/numeric.h>
#include <nano/gboost/bias.h>
#include <nano/dataset/memfixed.h>

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

static auto make_solver(const char* name = "lbfgs", const scalar_t epsilon = epsilon3<scalar_t>())
{
    auto solver = solver_t::all().get(name);
    UTEST_REQUIRE(solver);
    solver->epsilon(epsilon);
    solver->max_iterations(100);
    return solver;
}

class fixture_dataset_t final : public memfixed_dataset_t<scalar_t>
{
public:

    using memfixed_dataset_t::idim;
    using memfixed_dataset_t::tdim;
    using memfixed_dataset_t::samples;

    fixture_dataset_t() = default;

    bool load() override
    {
        resize(cat_dims(m_samples, m_idim), cat_dims(m_samples, m_tdim));

        for (tensor_size_t s = 0; s < m_samples; ++ s)
        {
            input(s).random();
            target(s).constant(-1.3);
            target(s).vector() += m_noise * vector_t::Random(nano::size(m_tdim));
        }

        for (size_t f = 0; f < folds(); ++ f)
        {
            this->split(f) = split_t{nano::split3(m_samples, train_percentage(), (100 - train_percentage()) / 2)};
        }
        return true;
    }

    [[nodiscard]] feature_t tfeature() const override
    {
        return feature_t{"const+noise"};
    }

    void noise(const scalar_t noise) { m_noise = noise; }
    void idim(const tensor3d_dim_t idim) { m_idim = idim; }
    void tdim(const tensor3d_dim_t tdim) { m_tdim = tdim; }
    void samples(const tensor_size_t samples) { m_samples = samples; }

private:

    // attributes
    scalar_t            m_noise{0};         ///< noise level (relative to the [-1,+1] uniform distribution)
    tensor_size_t       m_samples{1000};    ///< total number of samples to generate (train + validation + test)
    tensor3d_dim_t      m_idim{{10, 1, 1}}; ///< dimension of an input sample
    tensor3d_dim_t      m_tdim{{3, 1, 1}};  ///< dimension of a target/output sample
};

static auto make_dataset(const tensor_size_t isize = 5, const tensor_size_t tsize = 3)
{
    auto dataset = fixture_dataset_t{};
    dataset.folds(1);
    dataset.noise(epsilon1<scalar_t>());
    dataset.idim(make_dims(isize, 1, 1));
    dataset.tdim(make_dims(tsize, 1, 1));
    dataset.samples(100);
    dataset.train_percentage(80);
    UTEST_CHECK_NOTHROW(dataset.load());
    return dataset;
}

UTEST_BEGIN_MODULE(test_gboost_bias)

UTEST_CASE(gradient)
{
    auto loss = make_loss();
    auto dataset = make_dataset();

    auto function = gboost_bias_function_t{*loss, dataset, make_fold()};
    UTEST_REQUIRE_EQUAL(function.size(), 3);
    UTEST_REQUIRE_THROW(function.vAreg(-1e+0), std::invalid_argument);
    UTEST_REQUIRE_THROW(function.vAreg(+1e+9), std::invalid_argument);
    UTEST_REQUIRE_NOTHROW(function.vAreg(5e-1));

    for (int i = 0; i < 10; ++ i)
    {
        const vector_t x = vector_t::Random(function.size());
        UTEST_CHECK_LESS(function.grad_accuracy(x), 10 * epsilon2<scalar_t>());
    }
}

UTEST_CASE(minimize)
{
    auto loss = make_loss();
    auto solver = make_solver("cgd");
    auto dataset = make_dataset(3, 2);

    auto function = gboost_bias_function_t{*loss, dataset, make_fold()};
    UTEST_REQUIRE_EQUAL(function.size(), 2);
    UTEST_REQUIRE_NOTHROW(function.vAreg(0.01));

    solver->logger([] (const solver_state_t& state)
    {
        std::cout << state << ".\n";
        return true;
    });

    const auto state = solver->minimize(function, vector_t::Zero(function.size()));
    UTEST_CHECK(state);
    UTEST_CHECK(state.converged(solver->epsilon()));
    UTEST_CHECK_EIGEN_CLOSE(state.x, vector_t::Constant(2, -1.3), 1e+1 * solver->epsilon());
}

UTEST_END_MODULE()
