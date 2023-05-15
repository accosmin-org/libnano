#pragma once

#include <nano/core/table.h>
#include <nano/factory.h>

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
    const auto ids = factory.ids(std::regex(regex));

    table.header() << name << "description";
    table.delim();
    for (const auto& id : ids)
    {
        table.append() << id << factory.description(id);
    }
} // LCOV_EXCL_LINE

///
/// \brief organize the registered configurable objects in a factory in a tabular form.
///
template <typename tobject>
table_t make_table_with_params(const string_t& name, const factory_t<tobject>& factory, const string_t& regex = ".+")
{
    table_t table;
    append_table_with_params(table, name, factory, regex);
    return table;
} // LCOV_EXCL_LINE

template <typename tobject>
void append_table_with_params(table_t& table, const string_t& name, const factory_t<tobject>& factory,
                              const string_t& regex = ".+")
{
    const auto ids = factory.ids(std::regex(regex));

    table.header() << name << "parameter"
                   << "value"
                   << "domain";
    for (const auto& id : ids)
    {
        table.delim();
        table.append() << id << colspan(3) << factory.description(id);
        table.delim();
        const auto configurable = factory.get(id);
        for (const auto& param : configurable->parameters())
        {
            table.append() << id << param.name() << param.value() << param.domain();
        }
    }
} // LCOV_EXCL_LINE
} // namespace nano
