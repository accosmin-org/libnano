#pragma once

#include <nano/string.h>
#include <nlohmann/json.hpp>

namespace nano
{
    using json_t = nlohmann::json;
    using jsons_t = std::vector<json_t>;

    ///
    /// \brief serialize attributes to JSON.
    ///
    inline void to_json(json_t&)
    {
    }

    template <typename tvalue, typename... tothers>
    void to_json(json_t& json, const char* name, const tvalue value, tothers&&... nvs)
    {
        json[name] = to_string(value);
        to_json(json, nvs...);
    }

    template <typename... tattributes>
    json_t to_json(tattributes&&... attributes)
    {
        json_t json;
        to_json(json, attributes...);
        return json;
    }

    ///
    /// \brief deserialize attributes from JSON (if present).
    ///
    inline size_t from_json(const json_t&)
    {
        return 0u;
    }

    template <typename tvalue, typename... tothers>
    size_t from_json(const json_t& json, const char* name, tvalue& value, tothers&&... nvs)
    {
        size_t count = 0;
        if (json.count(name))
        {
            auto&& json_token = json[name];
            if (json_token.is_string())
            {
                value = from_string<tvalue>(json_token.get<string_t>());
            }
            else
            {
                value = json_token.get<tvalue>();
            }
            ++ count;
        }
        return count + from_json(json, nvs...);
    }

    ///
    /// \brief retrieve the attribute with the given name and
    ///     checks that is within the [min, max] range and throws an exception otherwise.
    ///
    template <typename tscalar, typename tscalar_min, typename tscalar_max>
    void from_json_range(const json_t& json, const char* name,
        tscalar& value, const tscalar_min min, const tscalar_max max)
    {
        const auto count = from_json(json, name, value);
        if (count && (value < static_cast<tscalar>(min) || value > static_cast<tscalar>(max)))
        {
            throw std::invalid_argument(
                strcat("invalid JSON attribute '", name, "', expected to be in the range [", min, ",", max, "]"));
        }
    }

    ///
    /// \brief checks the presence of the attribute with the given name and throws an exception otherwise.
    ///
    inline void require_json(const json_t& json, const char* name)
    {
        if (!json.count(name))
        {
            throw std::invalid_argument(
                strcat("missing JSON attribute '", name, "'"));
        }
    }

    ///
    /// \brief interface for JSON-based configurable objects.
    ///
    class json_configurable_t
    {
    public:

        json_configurable_t() = default;
        virtual ~json_configurable_t() noexcept = default;

        virtual json_t config() const = 0;
        virtual void config(const json_t&) = 0;

        json_t config_with_id(const string_t& id) const
        {
            auto json = config();
            json["id"] = id;
            return json;
        }
    };
}
