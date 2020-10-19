#pragma once

#include <nano/loss.h>
#include <nano/dataset.h>
#include <nano/function.h>
#include <nano/parameter.h>

namespace nano
{
    ///
    /// \brief function to optimize the weights of the models following the stacking ensemble method.
    ///
    /// see "Stacked Regressions", by Leo Breiman
    ///
    class NANO_PUBLIC stacking_function_t : public function_t
    {
    public:

        ///
        /// \brief constructor
        ///
        stacking_function_t(const loss_t&, const tensor4d_t& targets, const tensor5d_t& models_outputs);

        ///
        /// \brief @see function_t
        ///
        scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const override;

        ///
        /// \brief map the given values to weights
        ///
        static vector_t as_weights(const vector_t& x);

        ///
        /// \brief change parameters
        ///
        void batch(tensor_size_t batch);

        ///
        /// \brief access functions
        ///
        auto batch() const { return m_batch.get(); }

    private:

        // attributes
        const loss_t&       m_loss;         ///<
        const tensor4d_t&   m_targets;      ///< (#samples, ...) targets
        const tensor5d_t&   m_outputs;      ///< (#models, #samples, ...) predictions with all models
        iparam1_t           m_batch{"stacking::batch", 1, LE, 32, LE, 4092};    ///< batch size in number of samples
    };
}
