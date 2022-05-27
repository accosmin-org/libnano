#pragma once

#include <nano/core/table.h>
#include <nano/core/factory.h>

namespace nano
{
    ///
    /// \brief organize the registered objects in a factory in a tabular form.
    ///
    template <typename tobject>
    table_t make_table(const string_t& name, const factory_t<tobject>& factory, const string_t& regex = ".+")
    {
        table_t table;
        append_table(table, name, factory, regex);
        return table;
    } // LCOV_EXCL_LINE

    template <typename tobject>
    void append_table(table_t& table, const string_t& name, const factory_t<tobject>& factory, const string_t& regex = ".+")
    {
        table.header() << name << "description";
        table.delim();
        for (const auto& id : factory.ids(std::regex(regex)))
        {
            table.append() << id << factory.description(id);
        }
    } // LCOV_EXCL_LINE
}
