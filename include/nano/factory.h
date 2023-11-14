#pragma once

#include <map>
#include <nano/clonable.h>
#include <regex>

namespace nano
{
///
/// \brief implements the factory pattern: create objects of similar type.
///
template <typename tobject>
class factory_t
{
public:
    using trobject = std::unique_ptr<tobject>;

    ///
    /// \brief register a new object with the given ID.
    ///
    template <typename tobject_impl, typename... targs>
    bool add(const string_t& description, targs&&... args)
    {
        static_assert(std::is_base_of_v<tobject, tobject_impl>);

        auto prototype = std::make_unique<tobject_impl>(std::forward<targs>(args)...);
        auto type_id   = prototype->type_id();

        return m_protos.emplace(std::move(type_id), proto_t{std::move(prototype), description}).second;
    }

    ///
    /// \brief check if an object was registered with the given ID.
    ///
    bool has(const string_t& type_id) const { return m_protos.find(type_id) != m_protos.end(); }

    ///
    /// \brief retrieve a new object with the given ID.
    ///
    trobject get(const string_t& type_id) const
    {
        const auto it = m_protos.find(type_id);
        return (it == m_protos.end()) ? nullptr : it->second.m_prototype->clone();
    } // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)

    ///
    /// \brief get the IDs of the registered objects matching the ID regex.
    ///
    strings_t ids(const std::regex& id_regex = std::regex(".+")) const
    {
        strings_t ret;
        for (const auto& proto : m_protos)
        {
            if (std::regex_match(proto.first, id_regex))
            {
                ret.push_back(proto.first);
            }
        }
        return ret;
    } // LCOV_EXCL_LINE

    ///
    /// \brief returns the number of registered objects.
    ///
    size_t size() const { return m_protos.size(); }

    ///
    /// \brief get the description of the object with the given ID.
    ///
    string_t description(const string_t& type_id) const
    {
        const auto it = m_protos.find(type_id);
        return (it == m_protos.end()) ? string_t() : m_protos.at(type_id).m_description;
    }

private:
    struct proto_t
    {
        trobject m_prototype;
        string_t m_description;
    };

    using protos_t = std::map<string_t, proto_t>;

    // attributes
    protos_t m_protos; ///< registered object instances
};
} // namespace nano
