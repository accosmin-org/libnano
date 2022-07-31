#include <nano/dataset.h>
#include <nano/generator/elemwise_identity.h>
#include <nano/linear/model.h>
#include <utest/utest.h>

using namespace nano;

///
/// \brief synthetic dataset:
///     the targets is a random affine transformation of the flatten input features.
///
/// NB: uniformly-distributed noise is added to targets if noise() > 0.
/// NB: every column % modulo() is not taken into account.
///
class fixture_dataset_t : public dataset_t
{
public:
    using dataset_t::features;
    using dataset_t::samples;

    ///
    /// \brief default constructor
    ///
    fixture_dataset_t() = default;

    ///
    /// \brief change parameters
    ///
    void noise(scalar_t noise) { m_noise = noise; }

    void modulo(tensor_size_t modulo) { m_modulo = modulo; }

    void samples(tensor_size_t samples) { m_samples = samples; }

    void targets(tensor_size_t targets) { m_targets = targets; }

    void features(tensor_size_t features) { m_features = features; }

    ///
    /// \brief access functions
    ///
    auto noise() const { return m_noise; }

    const auto& bias() const { return m_bias; }

    const auto& weights() const { return m_weights; }

private:
    void do_load() override
    {
        // generate random input features & target
        features_t features;
        for (tensor_size_t ifeature = 0; ifeature < m_features; ++ifeature)
        {
            feature_t feature;
            switch (ifeature % 4)
            {
            case 0: feature = feature_t{scat("scalar", ifeature)}.scalar(); break;
            case 1: feature = feature_t{scat("sclass", ifeature)}.sclass(3U); break;
            case 2: feature = feature_t{scat("mclass", ifeature)}.mclass(4U); break;
            default:
                feature = feature_t{scat("struct", ifeature)}.scalar(feature_type::float64, make_dims(2, 1, 3));
                break;
            }
            features.push_back(feature);
        }
        features.push_back(feature_t{"Wx+b+eps"}.scalar(feature_type::float64, make_dims(m_targets, 1, 1)));

        const auto itarget = features.size() - 1U;
        resize(m_samples, features, itarget);

        // populate dataset
        for (tensor_size_t ifeature = 0; ifeature < m_features; ++ifeature)
        {
            switch (ifeature % 4)
            {
            case 0:
            {
                tensor_mem_t<scalar_t, 1> values(m_samples);
                values.random(-1.0, +1.0);
                for (tensor_size_t sample = 0; sample < m_samples; ++sample)
                {
                    set(sample, ifeature, values(sample));
                }
            }
            break;

            case 1:
            {
                tensor_mem_t<int32_t, 1> values(m_samples);
                values.random(0, 2);
                for (tensor_size_t sample = 0; sample < m_samples; ++sample)
                {
                    set(sample, ifeature, values(sample));
                }
            }
            break;

            case 2:
            {
                tensor_mem_t<int32_t, 2> values(m_samples, 4);
                values.random(0, 1);
                for (tensor_size_t sample = 0; sample < m_samples; ++sample)
                {
                    set(sample, ifeature, values.tensor(sample));
                }
            }
            break;

            default:
            {
                tensor_mem_t<scalar_t, 4> values(m_samples, 2, 1, 3);
                values.random(-1.0, +1.0);
                for (tensor_size_t sample = 0; sample < m_samples; ++sample)
                {
                    set(sample, ifeature, values.tensor(sample));
                }
            }
            break;
            }
        }

        // create samples: target = weights * input + bias + noise
        auto generator = dataset_generator_t{*this};
        generator.add<elemwise_generator_t<sclass_identity_t>>();
        generator.add<elemwise_generator_t<mclass_identity_t>>();
        generator.add<elemwise_generator_t<scalar_identity_t>>();
        generator.add<elemwise_generator_t<struct_identity_t>>();

        m_bias.resize(m_targets);
        m_weights.resize(m_targets, generator.columns());

        m_bias.random();
        m_weights.random();
        for (tensor_size_t column = m_modulo, columns = generator.columns(); column < columns; column += m_modulo)
        {
            m_weights.matrix().row(column).setConstant(0.0);
        }

        auto iterator = flatten_iterator_t{generator, arange(0, m_samples), 1U};
        iterator.loop(
            [&](tensor_range_t range, size_t, tensor2d_cmap_t inputs)
            {
                auto target  = tensor1d_t{m_targets};
                auto weights = m_weights.matrix();

                for (tensor_size_t i = 0, size = range.size(); i < size; ++i)
                {
                    target.vector() = weights * inputs.vector(i) + m_bias.vector();
                    target.vector() += m_noise * vector_t::Random(m_bias.size());
                    set(i + range.begin(), static_cast<tensor_size_t>(itarget), target);
                }
            });
    }

    // attributes
    scalar_t      m_noise{0};      ///< noise level (relative to the [-1,+1] uniform distribution)
    tensor_size_t m_modulo{3};     ///< modulo columns to exclude from creating the targets
    tensor_size_t m_targets{3};    ///< number of targets
    tensor_size_t m_features{10};  ///< total number of features to generate, of various types
    tensor_size_t m_samples{1000}; ///< total number of samples to generate (train + validation + test)
    tensor2d_t    m_weights;       ///< 2D weight matrix that maps the input to the output
    tensor1d_t    m_bias;          ///< 1D bias vector that offsets the output
};

[[maybe_unused]] static auto make_dataset(tensor_size_t samples, tensor_size_t targets, tensor_size_t features,
                                          tensor_size_t modulo = 31, scalar_t noise = 0.0)
{
    auto dataset = fixture_dataset_t{};
    dataset.noise(noise);
    dataset.modulo(modulo);
    dataset.samples(samples);
    dataset.targets(targets);
    dataset.features(features);
    UTEST_REQUIRE_NOTHROW(dataset.load());
    return dataset;
}

[[maybe_unused]] static auto make_generator(const dataset_t& dataset)
{
    auto generator = dataset_generator_t{dataset};
    generator.add<elemwise_generator_t<sclass_identity_t>>();
    generator.add<elemwise_generator_t<mclass_identity_t>>();
    generator.add<elemwise_generator_t<scalar_identity_t>>();
    generator.add<elemwise_generator_t<struct_identity_t>>();
    return generator;
}

template <typename tweights, typename tbias>
[[maybe_unused]] static void check_linear(const dataset_generator_t& generator, tweights weights, tbias bias,
                                          scalar_t epsilon)
{
    const auto samples = generator.dataset().samples();

    auto called = make_full_tensor<tensor_size_t>(make_dims(samples), 0);

    auto iterator = flatten_iterator_t{generator, arange(0, samples), 1U};
    iterator.batch(11);
    iterator.scaling(scaling_type::none);
    iterator.loop(
        [&](tensor_range_t range, size_t, tensor2d_cmap_t inputs, tensor4d_cmap_t targets)
        {
            for (tensor_size_t i = 0, size = range.size(); i < size; ++i)
            {
                UTEST_CHECK_CLOSE(targets.vector(i), weights * inputs.vector(i) + bias, epsilon);
                called(range.begin() + i) = 1;
            }
        });

    UTEST_CHECK_EQUAL(called, make_full_tensor<tensor_size_t>(make_dims(samples), 1));
}
