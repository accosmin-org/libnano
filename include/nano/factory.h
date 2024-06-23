#pragma once

#include <nano/clonable.h>
#include <nano/typed.h>
#include <regex>

namespace nano
{
///
/// \brief implements the factory pattern: create objects of similar type.
///
template <class tobject>
class factory_t
{
public:
    using trobject = std::unique_ptr<tobject>;

    static_assert(std::is_base_of_v<typed_t, tobject>);
    static_assert(std::is_base_of_v<clonable_t<tobject>, tobject>);

    ///
    /// \brief register a new object with the given ID and return true if possible.
    ///
    template <class tobject_impl, class... targs>
    bool add(string_t description, targs&&... args)
    {
        static_assert(std::is_base_of_v<tobject, tobject_impl>);

        auto prototype = std::make_unique<tobject_impl>(std::forward<targs>(args)...);
        auto type_id   = prototype->type_id();

        const auto duplicate = find(type_id) != m_protos.end();
        if (!duplicate)
        {
            m_protos.emplace_back(std::move(type_id), proto_t{std::move(prototype), std::move(description)});
        }
        return !duplicate;
    }

    ///
    /// \brief return true if an object was registered with the given ID.
    ///
    bool has(const std::string_view type_id) const { return find(type_id) != m_protos.end(); }

    ///
    /// \brief retrieve a new object with the given ID.
    ///
    trobject get(const std::string_view type_id) const
    {
        const auto it = find(type_id);
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
    string_t description(const std::string_view type_id) const
    {
        const auto it = find(type_id);
        return (it == m_protos.end()) ? string_t() : it->second.m_description;
    }

private:
    struct proto_t
    {
        trobject m_prototype;
        string_t m_description;
    };

    auto find(const std::string_view type_id) const
    {
        const auto op = [&](const auto& proto) { return proto.first == type_id; };
        return std::find_if(m_protos.begin(), m_protos.end(), op);
    }

    using protos_t = std::vector<std::pair<string_t, proto_t>>;

    // attributes
    protos_t m_protos; ///< registered object instances
};
} // namespace nano
