#pragma once

///
/// \brief helper class to create a visitor by overloading the call operator for the given lambdas associated for each
/// supported type.
///
/// NB: see https://en.cppreference.com/w/cpp/utility/variant/visit.
///
template <class... Ts>
struct overloaded : Ts...
{
    using Ts::operator()...;
};
