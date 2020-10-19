#pragma once

#include <nano/arch.h>
#include <nano/tensor.h>

namespace nano
{
    ///
    /// \brief generates splits for k-fold cross-validation.
    ///
    class NANO_PUBLIC kfold_t
    {
    public:

        ///
        /// \brief constructor
        ///
        kfold_t(indices_t samples, tensor_size_t folds);

        ///
        /// \brief generate the (training, validation) split of the given fold index.
        ///
        std::pair<indices_t, indices_t> split(tensor_size_t fold) const;

    private:

        // attributes
        indices_t       m_samples;      ///<
        tensor_size_t   m_folds{10};    ///<
    };
}
