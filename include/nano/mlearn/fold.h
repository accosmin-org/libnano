#pragma once

#include <nano/mlearn/enums.h>

namespace nano
{
    ///
    /// \brief dataset splitting fold.
    ///
    struct fold_t
    {
        size_t          m_index;        ///< fold index
        protocol        m_protocol;     ///<
    };

    inline bool operator==(const fold_t& f1, const fold_t& f2)
    {
        return f1.m_index == f2.m_index && f1.m_protocol == f2.m_protocol;
    }

    inline bool operator<(const fold_t& f1, const fold_t& f2)
    {
        return f1.m_index < f2.m_index || (f1.m_index == f2.m_index && f1.m_protocol < f2.m_protocol);
    }
}
