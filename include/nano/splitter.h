#pragma once

#include <nano/configurable.h>
#include <nano/core/seed.h>
#include <nano/factory.h>
#include <nano/tensor/tensor.h>

namespace nano
{
class splitter_t;
using rsplitter_t = std::unique_ptr<splitter_t>;

// TODO: have the splitter use the dataset, so that more complicated strategies can be implemented
// (e.g. stratified cross-validation, oversampling/undersampling to achieve class balance).

///
/// \brief generates (training, validation) splits for training, tuning and evaluating machine learning models.
///
class NANO_PUBLIC splitter_t : public typed_t, public configurable_t, public clonable_t<splitter_t>
{
public:
    ///< split with (training, validation) sample indices
    using split_t  = std::pair<indices_t, indices_t>;
    using splits_t = std::vector<split_t>;

    ///
    /// \brief constructor
    ///
    explicit splitter_t(string_t id);

    ///
    /// \brief returns the available implementations
    ///
    static factory_t<splitter_t>& all();

    ///
    /// \brief generate the (training, validation) splits for the given sample indices.
    ///
    virtual splits_t split(indices_t samples) const = 0;
};
} // namespace nano
