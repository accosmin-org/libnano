#include <utest/utest.h>
#include <nano/solver.h>
#include <nano/numeric.h>
#include <nano/gboost/scale.h>
#include <nano/dataset/memfixed.h>

using namespace nano;

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

        m_scale = vector_t::Random(m_groups);
        m_scale.array() += 1.1;

        m_outputs.resize(all_targets().dims());
        m_woutputs.resize(all_targets().dims());

        m_outputs.random();
        m_woutputs.random();

        for (tensor_size_t s = 0; s < m_samples; ++ s)
        {
            const auto group = s % m_groups;
            input(s).random();
            target(s).vector() = m_outputs.vector(s) + m_scale(group) * m_woutputs.vector(s);
        }

        for (size_t f = 0; f < folds(); ++ f)
        {
            this->split(f) = split_t{nano::split3(m_samples, train_percentage(), (100 - train_percentage()) / 2)};
        }
        return true;
    }

    [[nodiscard]] feature_t tfeature() const override
    {
        return feature_t{"const"};
    }

    [[nodiscard]] cluster_t cluster(fold_t fold) const
    {
        const auto& indices = this->indices(fold);

        cluster_t cluster(indices.size(), m_groups);
        for (tensor_size_t i = 0; i < indices.size(); ++ i)
        {
            const auto group = indices(i) % m_groups;
            if (i % 7 > 0)
            {
                cluster.assign(i, group);
            }
        }
        return cluster;
    }

    void idim(const tensor3d_dim_t idim) { m_idim = idim; }
    void tdim(const tensor3d_dim_t tdim) { m_tdim = tdim; }
    void groups(const tensor_size_t groups) { m_groups = groups; }
    void samples(const tensor_size_t samples) { m_samples = samples; }

    [[nodiscard]] auto groups() const { return m_groups; }
    [[nodiscard]] const auto& scale() const { return m_scale; }
    [[nodiscard]] auto outputs(fold_t fold) const { return m_outputs.indexed<scalar_t>(this->indices(fold)); }
    [[nodiscard]] auto woutputs(fold_t fold) const { return m_woutputs.indexed<scalar_t>(this->indices(fold)); }

private:

    // attributes
    vector_t            m_scale;            ///<
    tensor4d_t          m_outputs;          ///<
    tensor4d_t          m_woutputs;         ///<
    tensor_size_t       m_groups{1};        ///<
    tensor_size_t       m_samples{100};     ///< total number of samples to generate (train + validation + test)
    tensor3d_dim_t      m_idim{{10, 1, 1}}; ///< dimension of an input sample
    tensor3d_dim_t      m_tdim{{3, 1, 1}};  ///< dimension of a target/output sample
};

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

static auto make_dataset(const tensor_size_t isize = 5, const tensor_size_t tsize = 3, const tensor_size_t groups = 3)
{
    auto dataset = fixture_dataset_t{};
    dataset.folds(1);
    dataset.idim(make_dims(isize, 1, 1));
    dataset.tdim(make_dims(tsize, 1, 1));
    dataset.samples(50);
    dataset.groups(groups);
    dataset.train_percentage(80);
    UTEST_CHECK_NOTHROW(dataset.load());
    return dataset;
}

UTEST_BEGIN_MODULE(test_gboost_scale)

UTEST_CASE(gradient)
{
    const auto fold = make_fold();
    const auto loss = make_loss();
    const auto dataset = make_dataset();
    const auto cluster = dataset.cluster(fold);

    tensor4d_t outputs(cat_dims(dataset.samples(fold), dataset.tdim()));
    tensor4d_t woutputs(cat_dims(dataset.samples(fold), dataset.tdim()));

    outputs.zero();
    woutputs.zero();

    auto function = gboost_scale_function_t{*loss, dataset, fold, cluster, outputs, woutputs};
    UTEST_REQUIRE_EQUAL(function.size(), dataset.groups());
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
    const auto fold = make_fold();
    const auto loss = make_loss();
    const auto solver = make_solver("cgd");
    const auto dataset = make_dataset(3, 2);
    const auto cluster = dataset.cluster(fold);

    const auto outputs = dataset.outputs(fold);
    const auto woutputs = dataset.woutputs(fold);

    auto function = gboost_scale_function_t{*loss, dataset, fold, cluster, outputs, woutputs};
    UTEST_REQUIRE_EQUAL(function.size(), dataset.groups());
    UTEST_REQUIRE_NOTHROW(function.vAreg(0.1));

    solver->logger([] (const solver_state_t& state)
    {
        std::cout << state << ".\n";
        return true;
    });

    const auto state = solver->minimize(function, vector_t::Zero(function.size()));
    UTEST_CHECK(state);
    UTEST_CHECK(state.converged(solver->epsilon()));
    UTEST_CHECK_EIGEN_CLOSE(state.x, dataset.scale(), 1e+1 * solver->epsilon());
}

UTEST_END_MODULE()
