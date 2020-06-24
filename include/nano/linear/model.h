#pragma once

#include <nano/loss.h>
#include <nano/solver.h>
#include <nano/dataset.h>
#include <nano/mlearn/train.h>

namespace nano
{
    ///
    /// \brief a linear model is an affine transformation of the flatten input features x:
    ///     y(x) = weights * x + bias.
    ///
    ///
    /// NB: the inputs should be normalized during training to speed-up convergence (@see nano::normalization).
    /// NB: the regularization factors are tuned during training on the validation dataset (@see nano::regularization).
    ///
    class NANO_PUBLIC linear_model_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        linear_model_t() = default;

        ///
        /// \brief save the trained model to disk.
        ///
        void save(const string_t& filepath) const;

        ///
        /// \brief load the trained model from disk.
        ///
        void load(const string_t& filepath);

        ///
        /// \brief train the linear model on the given samples.
        ///
        train_result_t train(const loss_t&, const dataset_t&, const solver_t&);

        ///
        /// \brief compute the predictions for all samples in the given fold.
        ///
        void predict(const dataset_t&, fold_t, tensor4d_t& outputs) const;
        void predict(const dataset_t&, fold_t, tensor4d_map_t&& outputs) const;

        ///
        /// \brief change parameters
        ///
        void normalization(const normalization n) { m_normalization = n; }
        void regularization(const regularization r) { m_regularization = r; }
        void batch(const tensor_size_t batch) { m_batch = batch; }
        void tune_steps(const int steps) { m_tune_steps = steps; }
        void tune_trials(const int trials) { m_tune_trials = trials; }

        ///
        /// \brief access functions
        ///
        [[nodiscard]] auto batch() const { return m_batch.get(); }
        [[nodiscard]] auto tune_steps() const { return m_tune_steps.get(); }
        [[nodiscard]] auto normalization() const { return m_normalization; }
        [[nodiscard]] auto regularization() const { return m_regularization; }
        [[nodiscard]] auto tune_trials() const { return m_tune_trials.get(); }

        [[nodiscard]] const auto& bias() const { return m_bias; }
        [[nodiscard]] const auto& weights() const { return m_weights; }

    private:

        using nnormalization = ::nano::normalization;
        using nregularization = ::nano::regularization;

        // attributes
        tensor2d_t      m_weights;                                              ///< weight matrix (#inputs, #outputs)
        tensor1d_t      m_bias;                                                 ///< bias vector (#outputs)
        iparam1_t       m_batch{"linear::batch", 1, LE, 32, LE, 4096};          ///< #samples to use at once (= minibatch)
        iparam1_t       m_tune_trials{"linear::tune_trials", 4, LE, 7, LE, 10}; ///<
        iparam1_t       m_tune_steps{"linear::tune_steps", 1, LE, 2, LE, 10};   ///<
        nregularization m_regularization{nregularization::none};                ///<
        nnormalization  m_normalization{nnormalization::standard};              ///<
    };
}
