#pragma once

#include <nano/loss.h>
#include <nano/model.h>
#include <nano/generator.h>

namespace nano
{
    ///
    /// \brief evaluate the trained model and returns the error for each of the given samples.
    ///
    tensor1d_t evaluate(const dataset_generator_t&, const indices_t&, const loss_t&) const;

    ///
    /// \brief gather the results of k-fold cross-validation.
    ///
    struct kfold_result_t
    {
        kfold_result_t() = default;
        explicit kfold_result_t(tensor_size_t folds);

        tensor1d_t      m_train_errors; ///<
        tensor1d_t      m_valid_errors; ///<
        rmodels_t       m_models;       ///<
    };

    ///
    /// \brief (repeated) k-fold cross-validation
    ///     using the given model as currently setup in terms of (hyper-)parameters.
    ///
    NANO_PUBLIC kfold_result_t kfold(
        const model_t&, const dataset_generator_t&, const indices_t&, const loss_t& loss, const solver_t&,
        tensor_size_t folds = 5, tensor_size_t repetitions = 1);
}
