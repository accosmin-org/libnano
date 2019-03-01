#pragma once

#include <nlohmann/json.hpp>
#include <nano/string_utils.h>

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
    /// \brief deserialize attributes from JSON if present.
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
    /// \brief interface for JSON-based configurable objects.
    ///
    class json_configurable_t
    {
    public:

        virtual ~json_configurable_t() noexcept = default;

        virtual void to_json(json_t&) const = 0;
        virtual void from_json(const json_t&) = 0;
    };
}
