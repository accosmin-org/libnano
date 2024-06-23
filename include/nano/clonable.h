#pragma once

#include <memory>

namespace nano
{
///
/// \brief interface for objects that can be cloned (copied exactly).
///
template <class tobject>
class clonable_t
{
public:
    ///
    /// \brief default contructor
    ///
    clonable_t() = default;

    ///
    /// \brief enable copying.
    ///
    clonable_t(const clonable_t&)            = default;
    clonable_t& operator=(const clonable_t&) = default;

    ///
    /// \brief enable moving.
    ///
    clonable_t(clonable_t&&) noexcept            = default;
    clonable_t& operator=(clonable_t&&) noexcept = default;

    ///
    /// \brief destructor
    ///
    virtual ~clonable_t() = default;

    ///
    /// \brief returns a copy of the current object.
    ///
    virtual std::unique_ptr<tobject> clone() const = 0;
};
} // namespace nano
