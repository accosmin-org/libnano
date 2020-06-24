#pragma once

#include <nano/string.h>

namespace nano
{
    struct csv_t;
    using csvs_t = std::vector<csv_t>;

    ///
    /// \brief describes how a CSV (comma-separated values) file should be read.
    ///
    struct csv_t
    {
        csv_t() = default;
        explicit csv_t(string_t path) : m_path(std::move(path)) {}

        auto& skip(const char skip) { m_skip = skip; return *this; }
        auto& header(const bool header) { m_header = header; return *this; }
        auto& delim(string_t delim) { m_delim = std::move(delim); return *this; }
        auto& expected(const int expected) { m_expected = expected; return *this; }

        // attributes
        string_t    m_path;             ///<
        string_t    m_delim{", \r"};    ///< delimiting characters
        char        m_skip{'#'};        ///< skip lines starting with this character
        bool        m_header{false};    ///< skip the first line with the header
        int         m_expected{-1};     ///< expected number of lines to read (excepting skipped lines and the header)
    };
}
