#pragma once

#include <nano/splitter.h>

namespace nano
{
///
/// \brief k-fold cross-validation.
///
class NANO_PUBLIC kfold_splitter_t final : public splitter_t
{
public:
    ///
    /// \brief constructor
    ///
    kfold_splitter_t();

    ///
    /// \brief @see clonable_t
    ///
    rsplitter_t clone() const override;

    ///
    /// \brief @see splitter_t
    ///
    splits_t split(indices_t samples) const override;
};
} // namespace nano
