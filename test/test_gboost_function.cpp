#include "fixture/dataset.h"
#include "fixture/function.h"
#include "fixture/loss.h"
#include "fixture/solver.h"
#include <nano/gboost/function.h>
#include <nano/generator/elemwise_identity.h>

using namespace nano;
using namespace nano::gboost;

class fixture_datasource_t final : public datasource_t
{
public:
    explicit fixture_datasource_t(const tensor_size_t samples = 100, const tensor_size_t isize = 3,
                                  const tensor_size_t tsize = 2, const tensor_size_t groups = 3)
        : datasource_t("fixture")
        , m_groups(groups)
        , m_samples(samples)
        , m_idims(make_dims(isize, 1, 1))
        , m_tdims(make_dims(tsize, 1, 1))
        , m_targets(make_dims(samples, tsize, 1, 1))
    {
    }

    rdatasource_t clone() const override { return std::make_unique<fixture_datasource_t>(*this); }

    vector_t bias(const indices_t& samples) const
    {
        const auto targets = m_targets.indexed(samples);
        const auto tmatrix = targets.reshape(targets.size<0>(), -1).matrix();
        return tmatrix.colwise().mean();
    }

    tensor4d_t targets(const indices_t& samples) const { return m_targets.indexed(samples); }

    cluster_t cluster(const indices_t& samples) const
    {
        cluster_t cluster(this->samples(), m_groups);
        for (const auto sample : samples)
        {
            if (sample % 7 > 0)
            {
                cluster.assign(sample, sample % m_groups);
            }
        }
        return cluster;
    }

    auto groups() const { return m_groups; }

    const auto& scale() const { return m_scale; }

    auto outputs(const indices_t& samples) const { return m_outputs.indexed<scalar_t>(samples); }

    auto woutputs(const indices_t& samples) const { return m_woutputs.indexed<scalar_t>(samples); }

private:
    void do_load() override
    {
        auto features = features_t{feature_t{"inputs"}.scalar(feature_type::float32, m_idims),
                                   feature_t{"target"}.scalar(feature_type::float64, m_tdims)};

        resize(m_samples, features, features.size() - 1U);

        m_scale = vector_t::Random(m_groups);
        m_scale.array() += 1.1;

        m_outputs  = make_random_tensor<scalar_t>(cat_dims(m_samples, m_tdims));
        m_woutputs = make_random_tensor<scalar_t>(cat_dims(m_samples, m_tdims));

        tensor3d_t inputs(m_idims);

        for (tensor_size_t sample = 0; sample < m_samples; ++sample)
        {
            const auto group = sample % m_groups;

            inputs.random();
            m_targets.vector(sample) = m_outputs.vector(sample) + m_scale(group) * m_woutputs.vector(sample);

            set(sample, 0, inputs);
            set(sample, 1, m_targets.tensor(sample));
        }
    }

    // attributes
    vector_t        m_scale;        ///<
    tensor4d_t      m_outputs;      ///<
    tensor4d_t      m_woutputs;     ///<
    tensor_size_t   m_groups{1};    ///<
    tensor_size_t   m_samples{100}; ///< total number of samples to generate (train + validation + test)
    tensor3d_dims_t m_idims{};      ///< dimension of an input sample
    tensor3d_dims_t m_tdims{};      ///< dimension of a target/output sample
    tensor4d_t      m_targets;      ///<
};

static auto make_samples(const tensor_size_t samples)
{
    return ::nano::arange(0, samples);
}

static auto make_datasource(const tensor_size_t isize = 3, const tensor_size_t tsize = 2,
                            const tensor_size_t groups = 3, const tensor_size_t samples = 100)
{
    auto datasource = fixture_datasource_t{samples, isize, tsize, groups};
    UTEST_CHECK_NOTHROW(datasource.load());
    UTEST_CHECK_EQUAL(datasource.samples(), samples);
    return datasource;
}

static void check_optimum(const function_t& function, const vector_t& expected_optimum)
{
    const auto solver = make_solver();
    const auto state  = check_minimize(*solver, function, vector_t::Zero(function.size()));
    UTEST_CHECK_CLOSE(state.x, expected_optimum, 1e+2 * solver->parameter("solver::epsilon").value<scalar_t>());
}

template <typename ttmatrix, typename tomatrix>
static void check_value(const function_t& function, const ttmatrix& tmatrix, const tomatrix& omatrix,
                        const scalar_t vAreg)
{
    const auto values1 = 0.5 * (tmatrix - omatrix).array().square().rowwise().sum();
    const auto values2 = values1.square();

    const auto f0 = values1.mean();
    const auto fV = f0 + vAreg * (values2.mean() - values1.mean() * values1.mean());
    UTEST_CHECK_CLOSE(function.vgrad(vector_t::Zero(function.size())), fV, 1e-6);
}

UTEST_BEGIN_MODULE(test_gboost_function)

UTEST_CASE(bias)
{
    const auto loss       = make_loss();
    const auto datasource = make_datasource();
    const auto dataset    = make_dataset(datasource);
    const auto samples    = make_samples(60);
    const auto iterator   = targets_iterator_t{dataset, samples, 1U};

    const auto bias    = datasource.bias(samples);
    const auto targets = datasource.targets(samples);
    const auto tmatrix = targets.reshape(targets.size<0>(), -1).matrix();
    const auto omatrix = matrix_t::Zero(tmatrix.rows(), tmatrix.cols());

    for (const auto vAreg : {0e-1, 1e-1, 1e+0, 1e+1})
    {
        const auto function = bias_function_t{iterator, *loss, vAreg};

        UTEST_CHECK_EQUAL(function.size(), 2);
        check_gradient(function, 10);
        check_convexity(function, 10);
        check_value(function, tmatrix, omatrix, vAreg);
        if (vAreg < std::numeric_limits<scalar_t>::epsilon())
        {
            check_optimum(function, bias);
        }
    }
}

UTEST_CASE(scale)
{
    const auto loss       = make_loss();
    const auto datasource = make_datasource();
    const auto dataset    = make_dataset(datasource);
    const auto samples    = make_samples(50);
    const auto iterator   = targets_iterator_t{dataset, samples, 1U};

    const auto& scale    = datasource.scale();
    const auto  cluster  = datasource.cluster(samples);
    const auto  outputs  = datasource.outputs(samples);
    const auto  woutputs = datasource.woutputs(samples);
    const auto  targets  = datasource.targets(samples);
    const auto  tmatrix  = targets.reshape(targets.size<0>(), -1).matrix();
    const auto  omatrix  = outputs.reshape(tmatrix.rows(), tmatrix.cols()).matrix();

    for (const auto vAreg : {0e-1, 1e-1, 1e+0, 1e+1})
    {
        const auto function = scale_function_t{iterator, *loss, vAreg, cluster, outputs, woutputs};

        UTEST_CHECK_EQUAL(function.size(), datasource.groups());
        check_gradient(function, 10);
        check_convexity(function, 10);
        check_value(function, tmatrix, omatrix, vAreg);
        if (vAreg < std::numeric_limits<scalar_t>::epsilon())
        {
            check_optimum(function, scale);
        }
    }
}

UTEST_CASE(grads)
{
    const auto loss       = make_loss();
    const auto datasource = make_datasource();
    const auto dataset    = make_dataset(datasource);
    const auto samples    = make_samples(10);
    const auto iterator   = targets_iterator_t{dataset, samples, 1U};

    const auto outputs = datasource.outputs(samples);
    const auto targets = datasource.targets(samples);
    const auto tmatrix = targets.reshape(targets.size<0>(), -1).matrix();
    const auto omatrix = matrix_t::Zero(tmatrix.rows(), tmatrix.cols());

    for (const auto vAreg : {0e-1, 1e-1, 1e+0, 1e+1})
    {
        const auto function = grads_function_t{iterator, *loss, vAreg};

        UTEST_CHECK_EQUAL(function.size(), samples.size() * 2);
        check_gradient(function, 10);
        check_convexity(function, 10);
        check_value(function, tmatrix, omatrix, vAreg);
        if (vAreg < std::numeric_limits<scalar_t>::epsilon())
        {
            check_optimum(function, targets.vector());
        }
    }
}

UTEST_END_MODULE()
