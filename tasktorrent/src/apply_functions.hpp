#ifndef __TTOR_SRC_APPLY_FUNCTIONS_HPP__
#define __TTOR_SRC_APPLY_FUNCTIONS_HPP__

#include <stddef.h>
#include <utility>

namespace ttor {

// Allows calling a function f(A... a) where the data for a is stored in tuple t
template <typename T, typename F, size_t... Is>
constexpr auto apply_impl(F &f, T &t, std::index_sequence<Is...>)
{
    return f(std::get<Is>(t)...);
}

template <typename T, typename F>
constexpr auto apply(F &f, T &t)
{
    return apply_impl(f, t, std::make_index_sequence<std::tuple_size<std::decay_t<T>>::value>{});
}

template <typename T, typename F>
auto apply_fun(F &&fn, T &t)
{
    return apply(fn, t);
}

} // namespace ttor

#endif