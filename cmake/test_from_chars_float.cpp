#include <string>
#include <charconv>

int main()
{
    static constexpr auto str = std::string_view{"2.0"};
    auto value = 0.0;
    const auto [ptr, err]       = std::from_chars(str.data(), str.data() + str.size(), value);
    const auto ok = ptr == str.data() + str.size() && err == std::errc() && value == 2.0;
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
