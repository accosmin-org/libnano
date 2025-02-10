#pragma once

#include <nano/core/table.h>
#include <nano/factory.h>

namespace nano
{
///
/// \brief organize the registered objects in a factory in a tabular form.
///
template <class tobject>
table_t make_table(const string_t& name, const factory_t<tobject>& factory, const string_t& regex = ".+")
{
    const auto ids = factory.ids(std::regex(regex));

    table_t table;
    table.header() << name << "description";
    table.delim();

    for (const auto& id : ids)
    {
        table.append() << id << factory.description(id);
    }
    return table;
} // LCOV_EXCL_LINE

///
/// \brief organize the registered configurable objects in a factory in a tabular form.
///
template <class tobject>
table_t make_table_with_params(const string_t& name, const factory_t<tobject>& factory, const string_t& regex = ".+")
{
    const auto ids = factory.ids(std::regex(regex));

    table_t table;
    table.header() << name << "parameter" << "value" << "domain";

    for (const auto& id : ids)
    {
        table.delim();
        table.append() << id << colspan(3) << factory.description(id);

        const auto configurable = factory.get(id);
        if (!configurable->parameters().empty())
        {
            table.delim();
        }

        for (const auto& param : configurable->parameters())
        {
            table.append() << "|... " << param.name() << param.value() << param.domain();
        }
    }
    return table;
} // LCOV_EXCL_LINE
} // namespace nano
