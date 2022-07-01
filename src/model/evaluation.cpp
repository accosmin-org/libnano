#include <nano/generator/iterator.h>
#include <nano/model/kfold.h>

using namespace nano;

tensor1d_t model_t::evaluate(const dataset_generator_t& dataset, const indices_t& samples, const loss_t& loss) const
{
    const auto outputs = predict(dataset, samples);

    tensor1d_t errors(samples.size());

    auto iterator = targets_iterator_t{dataset, samples};
    iterator.loop([&](tensor_range_t range, size_t, tensor4d_cmap_t targets)
                  { loss.error(targets, outputs.slice(range), errors.slice(range)); });

    return errors;
}

kfold_result_t::kfold_result_t(tensor_size_t folds)
    : m_train_errors(folds)
    , m_valid_errors(folds)
    , m_models(static_cast<size_t>(folds))
{
}

kfold_result_t nano::kfold(const model_t& model_, const dataset_generator_t& dataset, const indices_t& samples,
                           const loss_t& loss, const solver_t& solver, tensor_size_t folds, tensor_size_t repetitions)
{
    const auto min_folds       = 3;
    const auto min_repetitions = 1;

    critical(folds < min_folds, "kfold: the number of folds (", folds, ") should be greater than ", min_folds, "!");

    critical(repetitions < min_repetitions, "kfold: the number of repetitions (", repetitions,
             ") should be greater than ", min_repetitions, "!");

    kfold_result_t result{folds * repetitions};
    for (tensor_size_t repetition = 0, index = 0; repetition < repetitions; ++repetition)
    {
        const auto kfold = kfold_t{samples, folds};

        for (tensor_size_t fold = 0; fold < folds; ++fold, ++index)
        {
            const auto [train_samples, valid_samples] = kfold.split(fold);

            auto model                                  = model_.clone();
            result.m_train_errors(index)                = model->fit(loss, dataset, train_samples, solver);
            result.m_valid_errors(index)                = model->evaluate(loss, dataset, valid_samples).mean();
            result.m_models[static_cast<size_t>(index)] = std::move(model);
        }
    }

    return result;
}
