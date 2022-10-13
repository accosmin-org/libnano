#pragma once

#include <nano/splitter.h>

namespace nano
{
    ///
    /// \brief repeated random sub-sampling validation.
    ///
    /// NB: the percentage of training samples is configurable.
    ///
    class NANO_PUBLIC random_splitter_t final : public splitter_t
    {
    public:
        ///
        /// \brief constructor
        ///
        random_splitter_t();

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
