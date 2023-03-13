#include <nano/core/table.h>

using namespace nano;

cell_t::cell_t() = default;

cell_t::cell_t(string_t data, const size_t span, const alignment align, const char fill)
    : m_data(std::move(data))
    , m_span(span)
    , m_fill(fill)
    , m_alignment(align)
{
}

row_t::row_t() = default;

row_t::row_t(const mode t)
    : m_type(t)
{
}

row_t& row_t::operator<<(const alignment align)
{
    m_alignment = align;
    return *this;
}

row_t& row_t::operator<<(const colfill_t colfill)
{
    m_colfill = colfill.m_fill;
    return *this;
}

row_t& row_t::operator<<(const colspan_t colspan)
{
    m_colspan = colspan.m_span;
    return *this;
}

row_t& row_t::operator<<(const char* string)
{
    m_cells.emplace_back(string, m_colspan, m_alignment, m_colfill);
    m_cols += m_colspan;
    return (*this) << colspan(1) << alignment::left << colfill(' ');
}

row_t& row_t::operator<<(const string_t& string)
{
    m_cells.emplace_back(string, m_colspan, m_alignment, m_colfill);
    m_cols += m_colspan;
    return (*this) << colspan(1) << alignment::left << colfill(' ');
}

cell_t* row_t::find(const size_t col)
{
    size_t     icol = 0;
    const auto it   = std::find_if(m_cells.begin(), m_cells.end(),
                                   [&](const auto& cell)
                                   {
                                     icol += cell.m_span;
                                     return icol > col;
                                 });
    return (it == m_cells.end()) ? nullptr : &*it;
}

const cell_t* row_t::find(const size_t col) const
{
    size_t     icol = 0;
    const auto it   = std::find_if(m_cells.begin(), m_cells.end(),
                                   [&](const auto& cell)
                                   {
                                     icol += cell.m_span;
                                     return icol > col;
                                 });
    return (it == m_cells.end()) ? nullptr : &*it;
}

void row_t::data(const size_t col, const string_t& str)
{
    auto* cell = find(col);
    assert(cell);
    cell->m_data = str;
}

void row_t::mark(const size_t col, const string_t& str)
{
    auto* cell = find(col);
    assert(cell);
    cell->m_mark = str;
}

const string_t& row_t::data(const size_t col) const
{
    const auto* cell = find(col);
    assert(cell);
    return cell->m_data;
}

const string_t& row_t::mark(const size_t col) const
{
    const auto* cell = find(col);
    assert(cell);
    return cell->m_mark;
}

table_t::table_t() = default;

row_t& table_t::delim()
{
    m_rows.emplace_back(row_t::mode::delim);
    return *m_rows.rbegin();
}

row_t& table_t::header()
{
    m_rows.emplace_back(row_t::mode::header);
    return *m_rows.rbegin();
}

row_t& table_t::append()
{
    m_rows.emplace_back(row_t::mode::data);
    return *m_rows.rbegin();
}

size_t table_t::cols() const
{
    const auto op = [](const row_t& row1, const row_t& row2) { return row1.cols() < row2.cols(); };
    const auto it = std::max_element(m_rows.begin(), m_rows.end(), op);
    return (it == m_rows.end()) ? size_t(0) : it->cols();
}

row_t& table_t::row(const size_t r)
{
    assert(r < rows());
    return m_rows[r];
}

const row_t& table_t::row(const size_t r) const
{
    assert(r < rows());
    return m_rows[r];
}

std::ostream& nano::operator<<(std::ostream& os, const table_t& table)
{
    std::vector<size_t> colsizes(table.cols(), 0);

    // size of the value columns (in characters) - step1: single column cells
    for (const auto& row : table.content())
    {
        size_t icol = 0;
        for (const auto& cell : row.cells())
        {
            const auto span = cell.m_span;
            if (span == 1)
            {
                const auto size = cell.m_data.size() + cell.m_mark.size();
                for (size_t c = 0; c < span; ++c, ++icol)
                {
                    colsizes[icol] = std::max(colsizes[icol], idiv(size, span));
                }
            }
            else
            {
                icol += span;
            }
        }
    }

    // size of the value columns (in characters) - step2: make room for large multi column cells
    for (const auto& row : table.content())
    {
        size_t icol = 0;
        for (const auto& cell : row.cells())
        {
            const auto span = cell.m_span;
            if (span > 1)
            {
                const auto size = cell.m_data.size() + cell.m_mark.size();
                if (std::accumulate(colsizes.begin() + static_cast<int>(icol),
                                    colsizes.begin() + static_cast<int>(icol + span), size_t(0)) < size)
                {
                    for (size_t c = 0; c < span; ++c, ++icol)
                    {
                        colsizes[icol] = std::max(colsizes[icol], idiv(size, span));
                    }
                }
                else
                {
                    icol += span;
                }
            }
            else
            {
                icol += span;
            }
        }
    }

    //
    const auto print_row_delim = [&]()
    {
        for (const auto colsize : colsizes)
        {
            os << "|" << string_t(colsize + 2, '-');
        }
        os << "|" << std::endl;
    };

    // display rows
    print_row_delim();
    for (const auto& row : table.content())
    {
        auto it = colsizes.begin();
        if (row.type() == row_t::mode::delim)
        {
            print_row_delim();
        }
        else
        {
            for (const auto& cell : row.cells())
            {
                const auto colspan = static_cast<std::ptrdiff_t>(cell.m_span);
                const auto colsize = std::accumulate(it, it + colspan, size_t(0));
                const auto extsize = (cell.m_span - 1) * 3;
                const auto coltext = cell.m_data + cell.m_mark;
                os << "| " << align(coltext, colsize + extsize, cell.m_alignment, cell.m_fill) << " ";
                std::advance(it, colspan);
            }
            os << "|" << std::endl;
        }
    }
    print_row_delim();

    return os;
}
