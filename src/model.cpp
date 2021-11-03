#include <mutex>
#include <nano/mlearn/kfold.h>
#include <nano/gboost/model.h>
#include <nano/linear/model.h>
#include <nano/model/grid_search.h>

using namespace nano;

void model_config_t::add(string_t name, int32_t value)
{
    add(std::move(name), static_cast<int64_t>(value));
}

void model_config_t::add(string_t name, int64_t value)
{
    m_values.emplace_back(std::move(name), value);
}

void model_config_t::add(string_t name, scalar_t value)
{
    m_values.emplace_back(std::move(name), value);
}

void model_config_t::evaluate(scalar_t error)
{
    m_error = error;
}

std::ostream& ::nano::operator<<(std::ostream& stream, const model_config_t& config)
{
    const auto& values = config.values();
    for (size_t i = 0; i < values.size(); ++ i)
    {
        const auto& value = values[i];
        stream << value.first << "=";
        if (std::holds_alternative<int64_t>(value.second))
        {
            stream << std::get<int64_t>(value.second);
        }
        else
        {
            stream << std::get<scalar_t>(value.second);
        }

        if (i + 1 < values.size())
        {
            stream << ',';
        }
    }
    return stream;
}

model_factory_t& model_t::all()
{
    static model_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [&] ()
    {
        manager.add_by_type<gboost_model_t>();
        manager.add_by_type<linear_model_t>();
        manager.add_by_type<grid_search_model_t>();
    });

    return manager;
}

model_t::model_t() = default;

void model_t::compatible(const dataset_generator_t& dataset) const
{
    const auto n_features = dataset.features();

    critical(
        n_features != static_cast<tensor_size_t>(m_inputs.size()),
        "model: mis-matching number of inputs (", n_features, "), expecting (", m_inputs.size(), ")!");

    for (tensor_size_t i = 0; i < n_features; ++ i)
    {
        const auto feature = dataset.feature(i);
        const auto& expected_feature = m_inputs[static_cast<size_t>(i)];

        critical(
            feature != expected_feature,
            "model: mis-matching input [", i, "/", n_features, "] (", feature, "), expecting (", expected_feature, ")!");
    }

    critical(
        dataset.target() != m_target,
        "model: mis-matching target (", dataset.target(), "), expecting (", m_target, ")!");
}

void model_t::read(std::istream& stream)
{
    serializable_t::read(stream);

    // TODO: also serialize the stored inputs & target to check compatibility
}

void model_t::write(std::ostream& stream) const
{
    serializable_t::write(stream);

    // TODO: also serialize the stored inputs & target to check compatibility
}

void model_t::set(const model_config_t& config)
{
    for (const auto& value : config.values())
    {
        if (std::holds_alternative<int64_t>(value.second))
        {
            set(value.first, std::get<int64_t>(value.second));
        }
        else
        {
            set(value.first, std::get<scalar_t>(value.second));
        }
    }
}

model_config_t model_t::config() const
{
    model_config_t config;
    for (const auto& param : m_params)
    {
        if (param.is_evalue())
        {
            config.add(param.eparam().name(), param.eparam().get());
        }
        else if (param.is_ivalue())
        {
            config.add(param.iparam().name(), param.iparam().get());
        }
        else
        {
            config.add(param.sparam().name(), param.sparam().get());
        }
    }

    return config;
}

void model_t::fit(const dataset_generator_t& dataset, const indices_t& samples, const loss_t& loss, const solver_& solver)
{
    const auto n_features = dataset.features();

    m_target = dataset.target();
    m_features.reserve(static_cast<size_t>(n_features));
    for (tensor_size_t i = 0; i < n_features; ++ i)
    {
        m_features.push_back(dataset.feature(i));
    }

    m_selected = do_fit(dataset, samples, loss, solver);
}

tensor1d_t model_t::evaluate(const dataset_generator_t& dataset, const indices_t& samples, const loss_t& loss) const
{
    const auto outputs = predict(dataset, samples);

    tensor1d_t errors(samples.size());
    loopr(samples.size(), tensor_size_t{1024}, [&] (tensor_size_t begin, tensor_size_t end, size_t)
    {
        const auto range = make_range(begin, end);
        const auto targets = dataset.targets(samples.slice(range));
        loss.error(targets, outputs.slice(range), errors.slice(range));
    });

    return errors;
}

tensor4d_t model_t::predict(const dataset_generator_t& dataset, const indices_t& samples) const
{
    compatible(dataset);

    return do_predict(dataset, samples);
}

kfold_result_t::kfold_result_t(tensor_size_t folds) :
    m_train_errors(folds),
    m_valid_errors(folds),
    m_models(static_cast<size_t>(folds))
{
}

kfold_result_t nano::kfold(const model_t& model_,
    const dataset_generator_t& dataset, const indices_t& samples, const loss_t& loss, const solver_t& solver,
    tensor_size_t folds, tensor_size_t repetitions)
{
    const auto min_folds = 3;
    const auto min_repetitions = 1;

    critical(
        folds < min_folds,
        "kfold: the number of folds (", folds, ") should be greater than ", min_folds, "!");

    critical(
        repetitions < min_repetitions,
        "kfold: the number of repetitions (", repetitions, ") should be greater than ", min_repetitions, "!");

    kfold_result_t result{folds * repetitions};
    for (tensor_size_t repetition = 0, index = 0; repetition < repetitions; ++ repetition)
    {
        const auto kfold = kfold_t{samples, folds};

        for (tensor_size_t fold = 0; fold < folds; ++ fold, ++ index)
        {
            const auto [train_samples, valid_samples] = kfold.split(fold);

            auto model = model_.clone();
            result.m_train_errors(index) = model->fit(loss, dataset, train_samples, solver);
            result.m_valid_errors(index) = model->evaluate(loss, dataset, valid_samples).mean();
            result.m_models[static_cast<size_t>(index)] = std::move(model);
        }
    }

    return result;
}
