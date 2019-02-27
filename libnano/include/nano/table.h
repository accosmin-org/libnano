#pragma once

#include <cassert>
#include <algorithm>
#include <nano/arch.h>
#include <nano/scalar.h>
#include <nano/string_utils.h>

namespace nano
{
    ///
    /// \brief cell in a table.
    ///
    struct NANO_PUBLIC cell_t
    {
        cell_t() = default;
        cell_t(string_t data, const size_t span, const alignment align, const char fill, const int precision) :
            m_data(std::move(data)),
            m_span(span),
            m_fill(fill),
            m_alignment(align),
            m_precision(precision)
        {
        }

        string_t format() const;
        cell_t& precision(const int precision) { m_precision = precision; return *this; }

        // attributes
        string_t    m_data;         ///<
        string_t    m_mark;         ///<
        size_t      m_span{1};          ///< column spanning
        char        m_fill{' '};        ///< filling character for aligning cells
        alignment   m_alignment{alignment::left};    ///<
        int         m_precision{0};     ///< #digits to display for floating point values
    };

    ///
    /// \brief control column spanning.
    ///
    struct colspan_t
    {
        size_t      m_span{1};
    };

    ///
    /// \brief control filling for aligning text in a cell.
    ///
    struct colfill_t
    {
        char        m_fill{' '};
    };

    ///
    /// \brief control precision for floating point cell values.
    ///
    struct precision_t
    {
        int         m_precision{0};
    };

    inline colspan_t colspan(const size_t span) { return {span}; }
    inline colfill_t colfill(const char fill) { return {fill}; }
    inline precision_t precision(const int precision) { return {precision}; }

    ///
    /// \brief row in a table.
    ///
    class NANO_PUBLIC row_t
    {
    public:

        enum class mode
        {
            data,       ///<
            delim,      ///< delimiting row
            header,     ///< header (not considered for operations like sorting or marking)
        };

        explicit row_t(const mode t = mode::data);

        ///
        /// \brief insert cells into the row or change its formatting
        ///
        template <typename tscalar>
        row_t& operator<<(const tscalar value)
        {
            m_cells.emplace_back(to_string(value), colspan(), align(), colfill(), precision());
            return colspan(1).align(alignment::left).colfill(' ').precision(0);
        }
        template <typename tscalar>
        row_t& operator<<(const std::vector<tscalar>& values)
        {
            for (const auto& value : values)
            {
                operator<<(value);
            }
            return *this;
        }
        row_t& operator<<(const alignment a)
        {
            return align(a);
        }
        row_t& operator<<(const colspan_t c)
        {
            return colspan(c.m_span);
        }
        row_t& operator<<(const colfill_t c)
        {
            return colfill(c.m_fill);
        }
        row_t& operator<<(const precision_t c)
        {
            return precision(c.m_precision);
        }

        ///
        /// \brief return the number of columns taking into account column spanning
        ///
        size_t cols() const;

        ///
        /// \brief find the a cell taking into account column spanning
        ///
        cell_t* find(const size_t col);
        const cell_t* find(const size_t col) const;

        ///
        /// \brief change a column's mark or data (finds the right cell taking into account column spanning)
        ///
        void data(const size_t col, const string_t&);
        void mark(const size_t col, const string_t&);

        ///
        /// \brief collect the columns as scalar values using nano::from_string<tscalar>
        ///
        template <typename tscalar>
        std::vector<std::pair<size_t, tscalar>> collect() const;

        ///
        /// \brief select the columns that satisfy the given operator
        ///
        template <typename tscalar, typename toperator>
        indices_t select(const toperator& op) const;

        ///
        /// \brief access functions
        ///
        const auto& cells() const { return m_cells; }
        auto& cell(const size_t icell) { assert(icell < m_cells.size()); return m_cells[icell]; }
        const auto& cell(const size_t icell) const { assert(icell < m_cells.size()); return m_cells[icell]; }

        string_t data(const size_t col) const;
        string_t mark(const size_t col) const;

        auto type() const { return m_type; }
        char colfill() const { return m_colfill; }
        size_t colspan() const { return m_colspan; }
        int precision() const { return m_precision; }
        alignment align() const { return m_alignment; }

        row_t& colfill(const char fill) { m_colfill = fill; return *this; }
        row_t& colspan(const size_t span) { m_colspan = span; return *this; }
        row_t& align(const alignment align) { m_alignment = align; return *this; }
        row_t& precision(const int precision) { m_precision = precision; return *this; }

    private:

        // attributes
        mode            m_type{mode::data};         ///< row type
        char            m_colfill{' '};         ///< current cell fill character
        size_t          m_colspan{1};           ///< current cell column span
        int             m_precision{0};         ///< current floating point precision
        alignment       m_alignment{alignment::left};   ///< current cell alignment
        std::vector<cell_t>     m_cells;
    };

    class table_t;

    ///
    /// \brief streaming operators.
    ///
    NANO_PUBLIC std::ostream& operator<<(std::ostream&, const table_t&);

    ///
    /// \brief comparison operators.
    ///
    NANO_PUBLIC bool operator==(const row_t&, const row_t&);
    NANO_PUBLIC bool operator==(const cell_t&, const cell_t&);
    NANO_PUBLIC bool operator==(const table_t&, const table_t&);

    ///
    /// \brief collects & formats tabular data for ASCII display.
    ///
    class NANO_PUBLIC table_t
    {
    public:

        table_t() = default;

        ///
        /// \brief remove all rows, but keeps the header
        ///
        void clear();

        ///
        /// \brief append a row as a header, as a data or as a delimeter row
        ///
        row_t& delim();
        row_t& header();
        row_t& append();

        ///
        /// \brief print table
        ///
        std::ostream& print(std::ostream&) const;

        ///
        /// \brief check if equal with another table
        ///
        bool equals(const table_t&) const;

        ///
        /// \brief (stable) sort the table using the given operator and columns
        /// e.g. toperator can be nano::make_[less|greater]_from_string<tscalar>
        ///
        template <typename toperator>
        void sort(const toperator&, const indices_t& columns);

        ///
        /// \brief mark row-wise the selected columns with the given operator
        ///
        template <typename tmarker>
        void mark(const tmarker& marker, const char* marker_string = " (*)");

        ///
        /// \brief save/load to/from CSV files using the given separator
        /// the header is always written/read
        ///
        bool save(const string_t& path, const char* delim = ";") const;
        bool load(const string_t& path, const char* delim = ";", const bool load_header = true);

        ///
        /// \brief access functions
        ///
        size_t cols() const;
        size_t rows() const;
        row_t& row(const size_t r) { assert(r < rows()); return m_rows[r]; }
        const row_t& row(const size_t r) const { assert(r < rows()); return m_rows[r]; }

    private:

        // attributes
        std::vector<row_t>      m_rows;
    };

    template <typename tscalar>
    std::vector<std::pair<size_t, tscalar>> row_t::collect() const
    {
        std::vector<std::pair<size_t, tscalar>> values;
        for (size_t col = 0, cols = this->cols(); type() == row_t::mode::data && col < cols; ++ col)
        {
            try
            {
                const cell_t* cell = find(col);
                assert(cell);
                values.emplace_back(col, nano::from_string<tscalar>(cell->m_data));
            }
            catch (std::exception&) {}
        }
        return values;
    }

    template <typename tscalar, typename toperator>
    indices_t row_t::select(const toperator& op) const
    {
        indices_t indices;
        for (const auto& cv : collect<tscalar>())
        {
            if (op(cv.second))
            {
                indices.emplace_back(cv.first);
            }
        }
        return indices;
    }

    template <typename toperator>
    void table_t::sort(const toperator& comp, const indices_t& columns)
    {
        std::stable_sort(m_rows.begin(), m_rows.end(), [&] (const row_t& row1, const row_t& row2)
        {
            if (row1.type() == row_t::mode::data && row2.type() == row_t::mode::data)
            {
                assert(row1.cols() == row2.cols());
                for (const auto col : columns)
                {
                    assert(row1.find(col) && row2.find(col));
                    const auto* cell1 = row1.find(col);
                    const auto* cell2 = row2.find(col);
                    if (comp(cell1->m_data, cell2->m_data))
                    {
                        return true;
                    }
                    else if (comp(cell2->m_data, cell1->m_data))
                    {
                        return false;
                    }
                }
                return true;
            }
            else
            {
                return false;
            }
        });
    }

    template <typename tmarker>
    void table_t::mark(const tmarker& marker, const char* marker_string)
    {
        for (auto& row : m_rows)
        {
            for (const auto col : marker(row))
            {
                row.mark(col, marker_string);
            }
        }
    }

    namespace detail
    {
        template <typename tscalar>
        auto min_element(const std::vector<std::pair<size_t, tscalar>>& values)
        {
            const auto comp = [] (const auto& cv1, const auto& cv2) { return cv1.second < cv2.second; };
            return std::min_element(values.begin(), values.end(), comp);
        }

        template <typename tscalar>
        auto max_element(const std::vector<std::pair<size_t, tscalar>>& values)
        {
            const auto comp = [] (const auto& cv1, const auto& cv2) { return cv1.second < cv2.second; };
            return std::max_element(values.begin(), values.end(), comp);
        }

        template <typename tscalar, typename toperator>
        indices_t filter(const std::vector<std::pair<size_t, tscalar>>& values, const toperator& op)
        {
            indices_t indices;
            std::for_each(values.begin(), values.end(), [&] (const auto& cv)
            {
                if (op(cv.second))
                {
                    indices.emplace_back(cv.first);
                }
            });
            return indices;
        }

        template <typename tscalar>
        indices_t filter_less(const std::vector<std::pair<size_t, tscalar>>& values, const tscalar threshold)
        {
            return filter(values, [threshold = threshold] (const auto& cv) { return cv < threshold; });
        }

        template <typename tscalar>
        indices_t filter_greater(const std::vector<std::pair<size_t, tscalar>>& values, const tscalar threshold)
        {
            return filter(values, [threshold = threshold] (const auto& cv) { return cv > threshold; });
        }
    }

    ///
    /// \brief select the column with the minimum value
    ///
    template <typename tscalar>
    auto make_marker_minimum_col()
    {
        return [=] (const row_t& row)
        {
            const auto values = row.collect<tscalar>();
            const auto it = detail::min_element(values);
            return (it == values.end()) ? indices_t{} : indices_t{it->first};
        };
    }

    ///
    /// \brief select the column with the maximum value
    ///
    template <typename tscalar>
    auto make_marker_maximum_col()
    {
        return [=] (const row_t& row) -> indices_t
        {
            const auto values = row.collect<tscalar>();
            const auto it = detail::max_element(values);
            return (it == values.end()) ? indices_t{} : indices_t{it->first};
        };
    }

    ///
    /// \brief select the columns within [0, epsilon] from the maximum value
    ///
    template <typename tscalar>
    auto make_marker_maximum_epsilon_cols(const tscalar epsilon)
    {
        return [=] (const row_t& row)
        {
            const auto values = row.collect<tscalar>();
            const auto it = detail::max_element(values);
            return (it == values.end()) ? indices_t{} : detail::filter_greater(values, it->second - epsilon);
        };
    }

    ///
    /// \brief select the columns within [0, epsilon] from the minimum value
    ///
    template <typename tscalar>
    auto make_marker_minimum_epsilon_cols(const tscalar epsilon)
    {
        return [=] (const row_t& row)
        {
            const auto values = row.collect<tscalar>();
            const auto it = detail::min_element(values);
            return (it == values.end()) ? indices_t{} : detail::filter_less(values, it->second + epsilon);
        };
    }

    ///
    /// \brief select the columns within [0, percentage]% from the maximum value
    ///
    template <typename tscalar>
    auto make_marker_maximum_percentage_cols(const tscalar percentage)
    {
        return [=] (const row_t& row)
        {
            assert(percentage >= tscalar(1));
            assert(percentage <= tscalar(99));
            const auto values = row.collect<tscalar>();
            const auto it = detail::max_element(values);
            return  (it == values.end()) ? indices_t{} : detail::filter_greater(values,
                    it->second - percentage * (it->second < 0 ? -it->second : +it->second) / tscalar(100));
        };
    }

    ///
    /// \brief select the columns within [0, percentage]% from the minimum value
    ///
    template <typename tscalar>
    auto make_marker_minimum_percentage_cols(const tscalar percentage)
    {
        return [=] (const row_t& row)
        {
            assert(percentage >= tscalar(1));
            assert(percentage <= tscalar(99));
            const auto values = row.collect<tscalar>();
            const auto it = detail::min_element(values);
            return  (it == values.end()) ? indices_t{} : detail::filter_less(values,
                    it->second + percentage * (it->second < 0 ? -it->second : +it->second) / tscalar(100));
        };
    }
}
