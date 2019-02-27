#include <sstream>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <nano/table.h>
#include <nano/numeric.h>

using namespace nano;

string_t cell_t::format() const
{
    try
    {
        if (m_precision > 0)
        {
            std::stringstream stream;
            stream << std::fixed << std::setprecision(m_precision) << from_string<double>(m_data);
            return stream.str();
        }
        else
        {
            return m_data;
        }
    }
    catch (std::exception&)
    {
        return m_data;
    }
}

row_t::row_t(const mode t) :
    m_type(t)
{
}

size_t row_t::cols() const
{
    return  std::accumulate(m_cells.begin(), m_cells.end(), size_t(0),
            [] (const size_t size, const cell_t& cell) { return size + cell.m_span; });
}

cell_t* row_t::find(const size_t col)
{
    for (size_t icell = 0, icol = 0; icell < m_cells.size(); ++ icell)
    {
        if (icol + m_cells[icell].m_span > col)
        {
            return &m_cells[icell];
        }
        icol += m_cells[icell].m_span;
    }
    return nullptr;
}

const cell_t* row_t::find(const size_t col) const
{
    for (size_t icell = 0, icol = 0; icell < m_cells.size(); ++ icell)
    {
        if (icol + m_cells[icell].m_span > col)
        {
            return &m_cells[icell];
        }
        icol += m_cells[icell].m_span;
    }
    return nullptr;
}

void row_t::data(const size_t col, const string_t& str)
{
    cell_t* cell = find(col);
    if (cell)
    {
        cell->m_data = str;
    }
}

void row_t::mark(const size_t col, const string_t& str)
{
    cell_t* cell = find(col);
    if (cell)
    {
        cell->m_mark = str;
    }
}

string_t row_t::data(const size_t col) const
{
    const cell_t* cell = find(col);
    return cell ? cell->m_data : string_t();
}

string_t row_t::mark(const size_t col) const
{
    const cell_t* cell = find(col);
    return cell ? cell->m_mark : string_t();
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

row_t& table_t::delim()
{
    m_rows.emplace_back(row_t::mode::delim);
    return *m_rows.rbegin();
}

std::size_t table_t::cols() const
{
    const auto op = [] (const row_t& row1, const row_t& row2) { return row1.cols() < row2.cols(); };
    const auto it = std::max_element(m_rows.begin(), m_rows.end(), op);
    return (it == m_rows.end()) ? size_t(0) : it->cols();
}

std::size_t table_t::rows() const
{
    return m_rows.size();
}

bool table_t::save(const string_t& path, const char* delimiter) const
{
    std::ofstream os(path.c_str(), std::ios::trunc);
    if (!os.is_open())
    {
        return false;
    }

    for (const auto& row : m_rows)
    {
        for (const auto& cell : row.cells())
        {
            os << cell.m_data;
            for (size_t i = 0; i < cell.m_span; ++ i)
            {
                os << delimiter;
            }
        }

        if (!row.cells().empty())
        {
            os << std::endl;
        }
    }

    return true;
}

bool table_t::load(const string_t& path, const char* delimiter, const bool load_header)
{
    std::ifstream is(path.c_str());
    if (!is.is_open())
    {
        return false;
    }

    m_rows.clear();

    // todo: this does not handle missing values
    // todo: this does not handle delimiting rows

    size_t count = 0, cols = 0;
    for (string_t line; std::getline(is, line); ++ count)
    {
        const auto tokens = nano::split(line, delimiter);
        if (!tokens.empty() && !line.empty())
        {
            if (!count && load_header)
            {
                header() << tokens;
                delim();
            }
            else if (tokens.size() != cols && cols)
            {
                return false;
            }
            else
            {
                append() << tokens;
            }

            if (cols == 0)
            {
                cols = this->cols();
            }
        }
    }

    return is.eof();
}

std::ostream& table_t::print(std::ostream& os) const
{
    sizes_t colsizes(this->cols(), 0);

    // size of the value columns (in characters) - step1: single column cells
    for (const auto& row : m_rows)
    {
        size_t icol = 0;
        for (const auto& cell : row.cells())
        {
            const auto span = cell.m_span;
            if (span == 1)
            {
                const auto size = cell.format().size() + cell.m_mark.size();
                for (size_t c = 0; c < span; ++ c, ++ icol)
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
    for (const auto& row : m_rows)
    {
        size_t icol = 0;
        for (const auto& cell : row.cells())
        {
            const auto span = cell.m_span;
            if (span > 1)
            {
                const auto size = cell.format().size() + cell.m_mark.size();
                if (std::accumulate(colsizes.begin() + icol, colsizes.begin() + (icol + span), size_t(0)) < size)
                {
                    for (size_t c = 0; c < span; ++ c, ++ icol)
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
    const auto print_row_delim = [&] ()
    {
        for (const auto colsize : colsizes)
        {
            os << "|" << string_t(colsize + 2, '-');
        }
        os << "|" << std::endl;
    };

    // display rows
    print_row_delim();
    for (const auto& row : m_rows)
    {
        auto it = colsizes.begin();
        switch (row.type())
        {
        case row_t::mode::delim:
            print_row_delim();
            break;

        default:
            for (const auto& cell : row.cells())
            {
                const auto colspan = static_cast<std::ptrdiff_t>(cell.m_span);
                const auto colsize = std::accumulate(it, it + colspan, size_t(0));
                const auto extsize = (cell.m_span - 1) * 3;
                const auto coltext = cell.format() + cell.m_mark;
                os << "| " << align(coltext, colsize + extsize, cell.m_alignment, cell.m_fill) << " ";
                std::advance(it, colspan);
            }
            os << "|" << std::endl;
            break;
        }
    }
    print_row_delim();

    return os;
}

bool table_t::equals(const table_t& other) const
{
    return std::operator==(m_rows, other.m_rows);
}

std::ostream& nano::operator<<(std::ostream& os, const table_t& table)
{
    return table.print(os);
}

bool nano::operator==(const cell_t& c1, const cell_t& c2)
{
    return  c1.m_data == c2.m_data &&
            c1.m_span == c2.m_span &&
            c1.m_alignment == c2.m_alignment;
}

bool nano::operator==(const row_t& r1, const row_t& r2)
{
    return  r1.type() == r2.type() &&
            std::operator==(r1.cells(), r2.cells());
}

bool nano::operator==(const table_t& t1, const table_t& t2)
{
    return t1.equals(t2);
}
