#pragma once

#include <regex>
#include <vector>
#include <string>
#include <sstream>
#include <utility>
#include <typeinfo>
#include <algorithm>
#include <stdexcept>

namespace nano
{
    using string_t = std::string;
    using strings_t = std::vector<string_t>;

    ///
    /// \brief text alignment options
    ///
    enum class alignment : int
    {
        left,
        center,
        right
    };

    template <typename tenum>
    using enum_map_t = std::vector<std::pair<tenum, string_t>>;

    ///
    /// \brief maps all possible values of an enum to string.
    /// NB: to specialize it, such that nano::to_string & nano::from_string work on a particular enum.
    ///
    template <typename tenum>
    enum_map_t<tenum> enum_string();

    ///
    /// \brief collect all the values for an enum type.
    ///
    template <typename tenum>
    auto enum_values(const std::regex& enum_regex = std::regex(".+"))
    {
        std::vector<tenum> enums;
        for (const auto& elem : enum_string<tenum>())
        {
            if (std::regex_match(elem.second, enum_regex))
            {
                enums.push_back(elem.first);
            }
        }
        return enums;
    }

    namespace detail
    {
        template <typename tvalue, typename = void>
        struct to_string_t
        {
            static string_t cast(const tvalue value)
            {
                std::ostringstream ss;
                ss.precision(16);
                ss << value;
                return ss.str();
            }
        };

        template <>
        struct to_string_t<string_t, void>
        {
            static string_t cast(const string_t& value)
            {
                return value;
            }
        };

        template <>
        struct to_string_t<const char*, void>
        {
            static string_t cast(const char* value)
            {
                return value;
            }
        };

        template <typename tenum>
        struct to_string_t<tenum, typename std::enable_if<std::is_enum<tenum>::value>::type>
        {
            static string_t cast(const tenum value)
            {
                for (const auto& elem : enum_string<tenum>())
                {
                    if (elem.first == value)
                    {
                        return elem.second;
                    }
                }

                const auto str = std::to_string(static_cast<int>(value));
                const auto msg = string_t("missing mapping for enumeration ") + typeid(tenum).name() + " <" + str + ">!";
                throw std::invalid_argument(msg);
            }
        };

        template <typename, typename = void>
        struct from_string_t;

        template <>
        struct from_string_t<short>
        {
            static short cast(const string_t& str)
            {
                return static_cast<short>(std::stoi(str));
            }
        };

        template <>
        struct from_string_t<int>
        {
            static int cast(const string_t& str)
            {
                return std::stoi(str);
            }
        };

        template <>
        struct from_string_t<long>
        {
            static long cast(const string_t& str)
            {
                return std::stol(str);
            }
        };

        template <>
        struct from_string_t<long long>
        {
            static long long cast(const string_t& str)
            {
                return std::stoll(str);
            }
        };

        template <>
        struct from_string_t<unsigned long>
        {
            static unsigned long cast(const string_t& str)
            {
                return std::stoul(str);
            }
        };

        template <>
        struct from_string_t<unsigned long long>
        {
            static unsigned long long cast(const string_t& str)
            {
                return std::stoull(str);
            }
        };

        template <>
        struct from_string_t<float>
        {
            static float cast(const string_t& str)
            {
                return std::stof(str);
            }
        };

        template <>
        struct from_string_t<double>
        {
            static double cast(const string_t& str)
            {
                return std::stod(str);
            }
        };

        template <>
        struct from_string_t<long double>
        {
            static long double cast(const string_t& str)
            {
                return std::stold(str);
            }
        };

        template <>
        struct from_string_t<string_t>
        {
            static string_t cast(const string_t& str)
            {
                return str;
            }
        };

        template <typename tenum>
        struct from_string_t<tenum, typename std::enable_if<std::is_enum<tenum>::value>::type>
        {
            static tenum cast(const string_t& str)
            {
                for (const auto& elem : enum_string<tenum>())
                {
                    if (elem.second == str)
                    {
                        return elem.first;
                    }
                }

                for (const auto& elem : enum_string<tenum>())
                {
                    if (str.find(elem.second) == 0)
                    {
                        return elem.first;
                    }
                }

                const auto msg = string_t("invalid ") + typeid(tenum).name() + " <" + str + ">!";
                throw std::invalid_argument(msg);
            }
        };
    }

    ///
    /// \brief cast value to string.
    ///
    template <typename tvalue>
    string_t to_string(const tvalue value)
    {
        /// todo: replace this with "if constepr" in c++17
        return detail::to_string_t<tvalue>::cast(value);
    }

    ///
    /// \brief cast string to value.
    ///
    template <typename tvalue>
    tvalue from_string(const string_t& str)
    {
        /// todo: replace this with "if constexpr" in c++17
        return detail::from_string_t<tvalue>::cast(str);
    }

    ///
    /// \brief cast string to value and use the given default value if casting fails.
    ///
    template <typename tvalue>
    tvalue from_string(const string_t& str, const tvalue& default_value)
    {
        try
        {
            return from_string<tvalue>(str);
        }
        catch (std::exception&)
        {
            return default_value;
        }
    }

    namespace detail
    {
        template <typename tvalue>
        void strcat(string_t& str, const tvalue& value)
        {
            str += to_string(value);
        }

        template <>
        inline void strcat<string_t>(string_t& str, const string_t& value)
        {
            str += value;
        }

        template <>
        inline void strcat<char>(string_t& str, const char& value)
        {
            str += value;
        }

        inline void strcat(string_t& str, const char* value)
        {
            str += value;
        }

        template <typename tvalue, typename... tvalues>
        void strcat(string_t& str, const tvalue& value, const tvalues&... values)
        {
            strcat(str, value);
            strcat(str, values...);
        }
    }

    ///
    /// \brief concatenate a list of potentially heterogeneous values into a string
    ///
    template <typename... tvalues>
    string_t strcat(const tvalues&... values)
    {
        string_t str;
        detail::strcat(str, values...);
        return str;
    }

    ///
    /// \brief compact a list of values into a string using the given "glue" string.
    ///
    template <typename titerator>
    string_t join(titerator begin, const titerator end, const char* glue = ",",
        const char* prefix = "[", const char* suffix = "]")
    {
        string_t ret;
        if (prefix)
        {
            ret += prefix;
        }
        for (; begin != end; )
        {
            detail::strcat(ret, *begin);
            if (++ begin != end)
            {
                ret += glue;
            }
        }
        if (suffix)
        {
            ret += suffix;
        }

        return ret;
    }

    template <typename tcontainer>
    string_t join(const tcontainer& values, const char* glue = ",",
        const char* prefix = "[", const char* suffix = "]")
    {
        return join(values.begin(), values.end(), glue, prefix, suffix);
    }

    ///
    /// \brief returns the lower case string
    ///
    inline string_t lower(string_t str)
    {
        std::transform(str.begin(), str.end(), str.begin(),
                       [] (const unsigned char c) { return std::tolower(c); });
        return str;
    }

    ///
    /// \brief returns the upper case string
    ///
    inline string_t upper(string_t str)
    {
        std::transform(str.begin(), str.end(), str.begin(),
                       [] (const unsigned char c) { return std::toupper(c); });
        return str;
    }

    ///
    /// \brief replace all occurencies of a character with another one
    ///
    inline string_t replace(string_t str, const char token, const char newtoken)
    {
        std::transform(str.begin(), str.end(), str.begin(),
                       [=] (const char c) { return (c == token) ? newtoken : c; });
        return str;
    }

    ///
    /// \brief replace all occurencies of a string with another one
    ///
    inline string_t replace(string_t str, const string_t& token, const string_t& newtoken)
    {
        for (size_t index = 0;;)
        {
            index = str.find(token, index);
            if (index == string_t::npos)
            {
                break;
            }
            str.replace(index, token.size(), newtoken);
            index += newtoken.size();
        }
        return str;
    }

    ///
    /// \brief check if two characters are equal case-insensitively
    ///
    inline bool iequal(const unsigned char c1, const unsigned char c2)
    {
        return std::tolower(c1) == std::tolower(c2);
    }

    ///
    /// \brief check if a string contains a given character
    ///
    inline bool contains(const string_t& str, const char token)
    {
        return std::find(str.begin(), str.end(), token) != str.end();
    }

    ///
    /// \brief check if two strings are equal (case sensitive)
    ///
    inline bool equals(const string_t& str1, const string_t& str2)
    {
        return str1.size() == str2.size() && std::equal(str1.begin(), str1.end(), str2.begin());
    }

    ///
    /// \brief check if two strings are equal (case insensitive)
    ///
    inline bool iequals(const string_t& str1, const string_t& str2)
    {
        return str1.size() == str2.size() && std::equal(str1.begin(), str1.end(), str2.begin(), iequal);
    }

    ///
    /// \brief check if a string starts with a token (case sensitive)
    ///
    inline bool starts_with(const string_t& str, const string_t& token)
    {
        return str.size() >= token.size() && std::equal(token.begin(), token.end(), str.begin());
    }

    ///
    /// \brief check if a string starts with a token (case insensitive)
    ///
    inline bool istarts_with(const string_t& str, const string_t& token)
    {
        return str.size() >= token.size() && std::equal(token.begin(), token.end(), str.begin(), iequal);
    }

    ///
    /// \brief check if a string ends with a token (case sensitive)
    ///
    inline bool ends_with(const string_t& str, const string_t& token)
    {
        return str.size() >= token.size() && std::equal(token.rbegin(), token.rend(), str.rbegin());
    }

    ///
    /// \brief check if a string ends with a token (case insensitive)
    ///
    inline bool iends_with(const string_t& str, const string_t& token)
    {
        return str.size() >= token.size() && std::equal(token.rbegin(), token.rend(), str.rbegin(), iequal);
    }

    ///
    /// \brief align a string to fill the given size (if possible).
    ///
    inline string_t align(const string_t& str, const size_t str_size,
        const alignment mode = alignment::left, const char fill_char = ' ')
    {
        const auto fill_size = (str.size() > str_size) ? (0) : (str_size - str.size());

        string_t ret;
        switch (mode)
        {
        case alignment::center:
            ret.append(fill_size / 2, fill_char);
            ret.append(str);
            ret.append(fill_size - fill_size / 2, fill_char);
            break;

        case alignment::right:
            ret.append(fill_size, fill_char);
            ret.append(str);
            break;

        case alignment::left:
        default:
            ret.append(str);
            ret.append(fill_size, fill_char);
            break;
        }

        return ret;
    }

    ///
    /// \brief iterator for splitting strings by delimiting characters.
    ///
    class tokenizer_t
    {
    public:

        ///
        /// \brief constructor
        ///
        tokenizer_t(const string_t& str, const char* delims, const size_t pos = 0) :
            m_str(str), m_delims(delims),
            m_pos(pos), m_end(pos)
        {
            next();
        }

        ///
        /// \brief enable copying and moving
        ///
        tokenizer_t(tokenizer_t&&) = default;
        tokenizer_t(const tokenizer_t&) = default;
        tokenizer_t& operator=(tokenizer_t&&) = default;
        tokenizer_t& operator=(const tokenizer_t&) = default;

        ///
        /// \brief returns true if parsing didn't finished
        ///
        operator bool() const
        {
            return (m_pos != string_t::npos) && (m_pos < m_end);
        }

        ///
        /// \brief move to the next token
        ///
        tokenizer_t& operator++()
        {
            next();
            return *this;
        }

        ///
        /// \brief move to the next token
        ///
        tokenizer_t operator++(int)
        {
            tokenizer_t tmp(*this);
            next();
            return tmp;
        }

        ///
        /// \brief returns the current token
        ///
        auto get() const
        {
            // todo: return a std::string_view when moving to C++17
            return m_str.substr(m_pos, m_end - m_pos);
        }

        ///
        /// \brief returns the begining of the current token
        ///
        auto pos() const { return m_pos; }

        ///
        /// \brief returns the number of tokens found so far
        ///
        auto count() const { return m_count; }

    private:

        void next()
        {
            m_pos = m_str.find_first_not_of(m_delims, m_end);
            if ((m_pos == string_t::npos) ||
                ((m_end = m_str.find_first_of(m_delims, m_pos + 1)) == string_t::npos))
            {
                m_end = m_str.size();
            }

            if (this->operator bool()) ++ m_count;
        }

        // attributes
        const string_t& m_str;      ///< string to parse
        const char*     m_delims;   ///< delimiting characters
        size_t          m_pos{0};   ///< the begining of the current token
        size_t          m_end{0};   ///< the end of the current token
        size_t          m_count{0}; ///< the number of tokens found so far
    };
}
