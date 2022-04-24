#include <utest/utest.h>
#include <nano/dataset.h>
#include <nano/linear/model.h>
#include <nano/generator/elemwise_identity.h>

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
        for (tensor_size_t ifeature = 0; ifeature < m_features; ++ ifeature)
        {
            feature_t feature;
            switch (ifeature % 4)
            {
            case 0:     feature = feature_t{scat("scalar", ifeature)}.scalar(); break;
            case 1:     feature = feature_t{scat("sclass", ifeature)}.sclass(3U); break;
            case 2:     feature = feature_t{scat("mclass", ifeature)}.mclass(4U); break;
            default:    feature = feature_t{scat("struct", ifeature)}.scalar(feature_type::float64, make_dims(2, 1, 3)); break;
            }
            features.push_back(feature);
        }
        features.push_back(feature_t{"Wx+b+eps"}.scalar(feature_type::float64, make_dims(m_targets, 1, 1)));

        const auto itarget = features.size() - 1U;
        resize(m_samples, features, itarget);

        // populate dataset
        for (tensor_size_t ifeature = 0; ifeature < m_features; ++ ifeature)
        {
            switch (ifeature % 4)
            {
            case 0:
                {
                    tensor_mem_t<scalar_t, 1> values(m_samples);
                    values.random(-1.0, +1.0);
                    for (tensor_size_t sample = 0; sample < m_samples; ++ sample)
                    {
                        set(sample, ifeature, values(sample));
                    }
                }
                break;

            case 1:
                {
                    tensor_mem_t<int32_t, 1> values(m_samples);
                    values.random(0, 2);
                    for (tensor_size_t sample = 0; sample < m_samples; ++ sample)
                    {
                        set(sample, ifeature, values(sample));
                    }
                }
                break;

            case 2:
                {
                    tensor_mem_t<int32_t, 2> values(m_samples, 4);
                    values.random(0, 1);
                    for (tensor_size_t sample = 0; sample < m_samples; ++ sample)
                    {
                        set(sample, ifeature, values.tensor(sample));
                    }
                }
                break;

            default:
                {
                    tensor_mem_t<scalar_t, 4> values(m_samples, 2, 1, 3);
                    values.random(-1.0, +1.0);
                    for (tensor_size_t sample = 0; sample < m_samples; ++ sample)
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

        auto iterator = flatten_iterator_t{generator, arange(0, m_samples)};
        iterator.loop([&] (tensor_range_t range, size_t, tensor2d_cmap_t inputs)
        {
            auto target = tensor1d_t{m_targets};
            auto weights = m_weights.matrix();

            for (tensor_size_t i = 0, size = range.size(); i < size; ++ i)
            {
                target.vector() = weights * inputs.vector(i) + m_bias.vector();
                target.vector() += m_noise * vector_t::Random(m_bias.size());
                set(i + range.begin(), static_cast<tensor_size_t>(itarget), target);
            }
        });
    }

    // attributes
    scalar_t            m_noise{0};         ///< noise level (relative to the [-1,+1] uniform distribution)
    tensor_size_t       m_modulo{3};        ///< modulo columns to exclude from creating the targets
    tensor_size_t       m_targets{3};       ///< number of targets
    tensor_size_t       m_features{10};     ///< total number of features to generate, of various types
    tensor_size_t       m_samples{1000};    ///< total number of samples to generate (train + validation + test)
    tensor2d_t          m_weights;          ///< 2D weight matrix that maps the input to the output
    tensor1d_t          m_bias;             ///< 1D bias vector that offsets the output
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

[[maybe_unused]] static auto make_iterator(const dataset_generator_t& generator,
    execution_type execution, tensor_size_t batch, scaling_type scaling)
{
    const auto samples = generator.dataset().samples();
    auto iterator = flatten_iterator_t{generator, arange(0, samples)};
    iterator.batch(batch);
    iterator.scaling(scaling);
    iterator.execution(execution);
    return iterator;
}

template <typename tweights, typename tbias>
[[maybe_unused]] static void check_linear(const dataset_generator_t& generator,
    tweights weights, tbias bias, scalar_t epsilon)
{
    const auto samples = generator.dataset().samples();

    auto called = make_full_tensor<tensor_size_t>(make_dims(samples), 0);

    auto iterator = flatten_iterator_t{generator, arange(0, samples)};
    iterator.batch(11);
    iterator.scaling(scaling_type::none);
    iterator.execution(execution_type::seq);
    iterator.loop([&] (tensor_range_t range, size_t, tensor2d_cmap_t inputs, tensor4d_cmap_t targets)
    {
        for (tensor_size_t i = 0, size = range.size(); i < size; ++ i)
        {
            UTEST_CHECK_CLOSE(targets.vector(i), weights * inputs.vector(i) + bias, epsilon);
            called(range.begin() + i) = 1;
        }
    });

    UTEST_CHECK_EQUAL(called, make_full_tensor<tensor_size_t>(make_dims(samples), 1));
}

[[maybe_unused]] static void check_outputs(const dataset_generator_t& generator,
    const indices_t& samples, const tensor4d_t& outputs, scalar_t epsilon)
{
    auto iterator = flatten_iterator_t{generator, samples};
    iterator.batch(7);
    iterator.scaling(scaling_type::none);
    iterator.execution(execution_type::seq);
    iterator.loop([&] (tensor_range_t range, size_t, tensor4d_cmap_t targets)
    {
        UTEST_CHECK_CLOSE(targets, outputs.slice(range), epsilon);
    });
}

[[maybe_unused]] static auto make_smooth_solver()
{
    auto solver = solver_t::all().get("lbfgs");
    UTEST_REQUIRE(solver);
    solver->parameter("solver::max_evals") = 1000;
    solver->parameter("solver::epsilon") = 1e-12;
    solver->lsearchk("cgdescent");
    return solver;
}

[[maybe_unused]] static auto make_nonsmooth_solver()
{
    auto solver = solver_t::all().get("osga");
    UTEST_REQUIRE(solver);
    solver->parameter("solver::max_evals") = 1500;
    solver->parameter("solver::epsilon") = 1e-5;
    return solver;
}

[[maybe_unused]] static auto make_solver(const string_t& loss_id)
{
    return loss_id == "squared" ? make_smooth_solver() : make_nonsmooth_solver();
}

[[maybe_unused]] static auto make_model()
{
    auto model = linear_model_t{};
    model.parameter("model::folds") = 2;
    model.parameter("model::linear::batch") = 10;

    model.logger([] (const fit_result_t& result, const string_t& prefix)
    {
        auto&& logger = log_info();
        logger << std::fixed << std::setprecision(9) << std::fixed << prefix << ": ";

        const auto print_params = [&] (const tensor1d_t& param_values)
        {
            assert(result.m_param_names.size() == static_cast<size_t>(param_values.size()));
            for (size_t i = 0U, size = result.m_param_names.size(); i < size; ++ i)
            {
                logger << result.m_param_names[i] << "=" << param_values(static_cast<tensor_size_t>(i)) << ",";
            }
        };

        if (std::isfinite(result.m_refit_error))
        {
            print_params(result.m_refit_params);
            logger << "refit=" << result.m_refit_value << "/" << result.m_refit_error << ".";
        }
        else if (!result.m_cv_results.empty())
        {
            const auto& cv_result = *result.m_cv_results.rbegin();
            print_params(cv_result.m_params);
            logger << "train=" << cv_result.m_train_values.mean() << "/" << cv_result.m_train_errors.mean() << ",";
            logger << "valid=" << cv_result.m_valid_values.mean() << "/" << cv_result.m_valid_errors.mean() << ".";
        }
    });

    return model;
}

[[maybe_unused]] static void check_result(const fit_result_t& result,
    const strings_t& expected_param_names, size_t min_cv_results_size, scalar_t epsilon)
{
    UTEST_CHECK_CLOSE(result.m_refit_value, 0.0, epsilon);
    UTEST_CHECK_CLOSE(result.m_refit_error, 0.0, epsilon);
    UTEST_CHECK_EQUAL(result.m_param_names, expected_param_names);

    UTEST_REQUIRE_GREATER_EQUAL(result.m_cv_results.size(), min_cv_results_size);

    const auto opt_values = make_full_tensor<scalar_t>(make_dims(2), 0.0);
    const auto opt_errors = make_full_tensor<scalar_t>(make_dims(2), 0.0);

    tensor_size_t hits = 0;
    for (const auto& cv_result : result.m_cv_results)
    {
        UTEST_CHECK_GREATER(cv_result.m_params.min(), 0.0);
        UTEST_CHECK_EQUAL(cv_result.m_params.size(), static_cast<tensor_size_t>(expected_param_names.size()));
        if (close(cv_result.m_train_errors, opt_errors, epsilon))
        {
            ++ hits;
            UTEST_CHECK_CLOSE(cv_result.m_train_values, opt_values, 1.0 * epsilon);
            UTEST_CHECK_CLOSE(cv_result.m_train_errors, opt_errors, 1.0 * epsilon);
            UTEST_CHECK_CLOSE(cv_result.m_valid_values, opt_values, 5.0 * epsilon);
            UTEST_CHECK_CLOSE(cv_result.m_valid_errors, opt_errors, 5.0 * epsilon);
        }
    }
    if (!expected_param_names.empty())
    {
        UTEST_CHECK_GREATER(hits, 0);
    }
    else
    {
        UTEST_CHECK(result.m_cv_results.empty());
    }
}

[[maybe_unused]] static void check_model(const linear_model_t& model,
    const dataset_generator_t& dataset, const indices_t& samples, scalar_t epsilon)
{
    const auto outputs = model.predict(dataset, samples);
    check_outputs(dataset, samples, outputs, epsilon);

    string_t str;
    {
        std::ostringstream stream;
        UTEST_REQUIRE_NOTHROW(model.write(stream));
        str = stream.str();
    }
    {
        auto new_model = linear_model_t{};
        std::istringstream stream(str);
        UTEST_REQUIRE_NOTHROW(new_model.read(stream));
        const auto new_outputs = model.predict(dataset, samples);
        UTEST_CHECK_CLOSE(outputs, new_outputs, epsilon0<scalar_t>());
    }
}
