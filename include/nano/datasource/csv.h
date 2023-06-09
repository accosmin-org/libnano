#pragma once

#include <fstream>
#include <nano/string.h>
#include <nano/tensor/index.h>

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

    explicit csv_t(string_t path)
        : m_path(std::move(path))
    {
    }

    ///
    /// \brief configure.
    ///
    auto& skip(char skip)
    {
        m_skip = skip;
        return *this;
    }

    auto& header(bool header)
    {
        m_header = header;
        return *this;
    }

    auto& expected(int expected)
    {
        m_expected = expected;
        return *this;
    }

    auto& delim(string_t delim)
    {
        m_delim = std::move(delim);
        return *this;
    }

    auto& testing(tensor_range_t testing)
    {
        m_testing = testing;
        return *this;
    }

    auto& testing(tensor_size_t begin, tensor_size_t end)
    {
        m_testing = make_range(begin, end);
        return *this;
    }

    auto& placeholder(string_t placeholder)
    {
        m_placeholder = std::move(placeholder);
        return *this;
    }

    ///
    /// \brief parse the current configured CSV and call the given operator for each line.
    /// NB: optionally a base directory path can be given as a prefix.
    ///
    template <typename toperator>
    auto parse(const string_t& basedir, const toperator& op) const
    {
        const auto path = basedir.empty() ? m_path : (basedir + "/" + m_path);

        string_t line;
        auto     header     = m_header;
        auto     line_index = tensor_size_t{0};
        for (std::ifstream stream(path); std::getline(stream, line); ++line_index)
        {
            if (header && line_index == 0)
            {
                header = false;
            }
            else if (!line.empty() && line[0] != m_skip)
            {
                if (!op(line, line_index))
                {
                    return false;
                }
            }
        }

        return true;
    }

    template <typename toperator>
    auto parse(const toperator& op) const
    {
        return parse(string_t{}, op);
    }

    // attributes
    string_t       m_path;          ///<
    string_t       m_delim{", \r"}; ///< delimiting characters
    char           m_skip{'#'};     ///< skip lines starting with this character
    bool           m_header{false}; ///< skip the first line with the header
    int            m_expected{-1};  ///< expected number of lines to read (excepting skipped lines and the header)
    tensor_range_t m_testing;       ///< optional range of samples (relative to the file) to be used for testing
    string_t       m_placeholder;   ///< placeholder string used if its value is missing
};
} // namespace nano
